from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import logit
from sklearn.metrics import log_loss, roc_auc_score


SEED = 42
EPS = 1e-3
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def ll(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]))


def split_masks(df: pd.DataFrame):
    train = np.zeros(len(df), bool)
    test = np.zeros(len(df), bool)
    fit = np.zeros(len(df), bool)
    val = np.zeros(len(df), bool)
    for _, g in df.groupby("sub"):
        pos = g[g["mw"] == 1].sample(frac=1, random_state=SEED)
        neg = g[g["mw"] == 0].sample(frac=1, random_state=SEED)
        hpos = len(pos) // 2
        hneg = len(neg) // 2
        trp = list(pos.index[:hpos])
        tep = list(pos.index[hpos:])
        trn = list(neg.index[:hneg])
        ten = list(neg.index[hneg:])
        train[trp + trn] = True
        test[tep + ten] = True
        fhp = max(1, int(round(0.75 * len(trp)))) if len(trp) > 1 else len(trp)
        fhn = max(1, int(round(0.75 * len(trn)))) if len(trn) > 1 else len(trn)
        fit[trp[:fhp] + trn[:fhn]] = True
        val[trp[fhp:] + trn[fhn:]] = True
    return train, test, fit, val


def load_all_to_vram(device: torch.device) -> torch.Tensor:
    meta = json.load(open(RESULTS / "reve_base_features_10s_balanced_f16_meta.json"))
    shape = tuple(meta["shape"])
    mm = np.memmap(
        RESULTS / "reve_base_features_10s_balanced_f16.dat",
        dtype="float16",
        mode="r",
        shape=shape,
    )
    log(f"loading full 10s feature tensor to VRAM shape={shape} disk_gb={np.prod(shape) * 2 / 1e9:.2f}")
    x = torch.from_numpy(np.array(mm, dtype=np.float16, copy=True)).to(device, non_blocking=True)
    torch.cuda.synchronize()
    log(f"loaded x={tuple(x.shape)} flat_dim={int(np.prod(shape[1:]))} vram_gb={torch.cuda.memory_allocated() / 1e9:.2f}")
    return x


def flat_batch(x4: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x4.index_select(0, idx).reshape(idx.numel(), -1).float()


def compute_stats(
    x4: torch.Tensor,
    idx_fit: torch.Tensor,
    z_eye: torch.Tensor | None,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = int(np.prod(x4.shape[1:]))
    offset = 1 if z_eye is not None else 0
    sums = torch.zeros(d + offset, device=x4.device, dtype=torch.float64)
    sums2 = torch.zeros_like(sums)
    n = int(idx_fit.numel())
    for start in range(0, n, batch_size):
        idx = idx_fit[start : start + batch_size]
        xb = flat_batch(x4, idx)
        if z_eye is not None:
            zb = z_eye.index_select(0, idx).view(-1, 1)
            xb = torch.cat([zb, xb], dim=1)
        sums += xb.sum(dim=0, dtype=torch.float64)
        sums2 += (xb * xb).sum(dim=0, dtype=torch.float64)
        del xb
    mu64 = sums / n
    var64 = torch.clamp((sums2 / n) - (mu64 * mu64), min=1e-12)
    sd64 = torch.sqrt(var64)
    mu = mu64.float()
    sd = torch.where(sd64.float() < 1e-6, torch.ones_like(mu), sd64.float())
    del sums, sums2, mu64, var64, sd64
    torch.cuda.empty_cache()
    return mu, sd


def make_standardized_batch(
    x4: torch.Tensor,
    idx: torch.Tensor,
    mu: torch.Tensor,
    sd: torch.Tensor,
    z_eye: torch.Tensor | None,
) -> torch.Tensor:
    xb = flat_batch(x4, idx)
    if z_eye is None:
        return (xb - mu) / sd
    zb = z_eye.index_select(0, idx).view(-1, 1)
    xb = (xb - mu[1:]) / sd[1:]
    zb = (zb - mu[:1]) / sd[:1]
    return torch.cat([zb, xb], dim=1)


def predict_batches(
    model: nn.Module,
    x4: torch.Tensor,
    indices: torch.Tensor,
    mu: torch.Tensor,
    sd: torch.Tensor,
    z_eye: torch.Tensor | None,
    batch_size: int,
) -> np.ndarray:
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(indices.numel()), batch_size):
            idx = indices[start : start + batch_size]
            xb = make_standardized_batch(x4, idx, mu, sd, z_eye)
            pred = torch.sigmoid(model(xb)).detach().cpu().numpy().ravel()
            out.append(pred)
            del xb
    return np.concatenate(out)


def train_select(
    x4: torch.Tensor,
    y: np.ndarray,
    idx_fit: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
    z_eye: torch.Tensor | None,
    label: str,
    wds: list[float],
    lr: float,
    epochs: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    device = x4.device
    y_t = torch.tensor(y.astype("float32"), device=device).view(-1, 1)
    y_val = y[idx_val.detach().cpu().numpy()]
    dim = int(np.prod(x4.shape[1:])) + (1 if z_eye is not None else 0)
    log(f"{label} computing fit stats dim={dim} batch={batch_size}")
    mu, sd = compute_stats(x4, idx_fit, z_eye, batch_size)
    torch.cuda.synchronize()
    log(f"{label} stats ready vram_gb={torch.cuda.memory_allocated() / 1e9:.2f}")

    best: dict[str, object] | None = None
    gen = torch.Generator(device=device)
    for wd in wds:
        torch.manual_seed(SEED)
        gen.manual_seed(SEED)
        model = nn.Linear(dim, 1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.BCEWithLogitsLoss()
        local: dict[str, object] | None = None
        for ep in range(epochs):
            perm = idx_fit[torch.randperm(int(idx_fit.numel()), generator=gen, device=device)]
            for start in range(0, int(perm.numel()), batch_size):
                idx = perm[start : start + batch_size]
                xb = make_standardized_batch(x4, idx, mu, sd, z_eye)
                loss = loss_fn(model(xb), y_t.index_select(0, idx))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                del xb, loss
            if ep % 5 == 0 or ep == epochs - 1:
                pv = predict_batches(model, x4, idx_val, mu, sd, z_eye, batch_size * 2)
                vll = ll(y_val, pv)
                if local is None or vll < float(local["val_ll"]):
                    local = {
                        "val_ll": vll,
                        "epoch": ep + 1,
                        "wd": wd,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    }
        assert local is not None
        log(f"{label} wd={wd:g} best_val={float(local['val_ll']):.4f} epoch={local['epoch']}")
        if best is None or float(local["val_ll"]) < float(best["val_ll"]):
            best = local
        del model, opt
        torch.cuda.empty_cache()

    assert best is not None
    model = nn.Linear(dim, 1).to(device)
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})  # type: ignore[index]
    pred = predict_batches(model, x4, idx_test, mu, sd, z_eye, batch_size * 2)
    del model, mu, sd
    torch.cuda.empty_cache()
    return pred, best


def main() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    df = pd.read_csv(RESULTS / "balanced_epoch_index_10s.csv").reset_index(drop=True)
    y = df["mw"].astype(int).to_numpy()
    orig = df["orig_epoch_id"].to_numpy(dtype=int)
    p_eye = np.load(RESULTS / "p_eye_personal_cal.npy")[orig]
    _, test, fit, val = split_masks(df)
    eye_ll = ll(y[test], p_eye[test])
    eye_auc = roc_auc_score(y[test], p_eye[test])
    log(f"10s baseline eye ll={eye_ll:.4f} auc={eye_auc:.3f} n={len(df)} fit={fit.sum()} val={val.sum()} test={test.sum()}")

    idx_fit = torch.tensor(np.flatnonzero(fit), device=device)
    idx_val = torch.tensor(np.flatnonzero(val), device=device)
    idx_test = torch.tensor(np.flatnonzero(test), device=device)
    z_eye = torch.tensor(logit(np.clip(p_eye, EPS, 1 - EPS)).astype("float32"), device=device)
    x4 = load_all_to_vram(device)

    configs = [
        ("reve10s_full_stream", None, [1.0, 10.0, 50.0, 200.0, 500.0], 0.003, 45),
        ("eye_reve10s_full_stream", z_eye, [1.0, 10.0, 50.0, 200.0, 500.0], 0.003, 45),
    ]
    rows = []
    for label, z, wds, lr, epochs in configs:
        pred, best = train_select(
            x4=x4,
            y=y,
            idx_fit=idx_fit,
            idx_val=idx_val,
            idx_test=idx_test,
            z_eye=z,
            label=label,
            wds=wds,
            lr=lr,
            epochs=epochs,
            batch_size=512,
        )
        test_ll = ll(y[test], pred)
        test_auc = roc_auc_score(y[test], pred)
        row = {
            "model": label,
            "best_wd": best["wd"],
            "best_epoch": best["epoch"],
            "val_ll": best["val_ll"],
            "test_ll": test_ll,
            "test_auc": test_auc,
            "ll_gain_over_eye": eye_ll - test_ll,
        }
        rows.append(row)
        np.save(RESULTS / f"p_{label}.npy", pred)
        pd.DataFrame(rows).to_csv(RESULTS / "probe_reve_10s_full_stream_summary.csv", index=False)
        log(f"{label}: val={float(best['val_ll']):.4f} test_ll={test_ll:.4f} auc={test_auc:.3f} gain={eye_ll - test_ll:+.4f} wd={best['wd']}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "probe_reve_10s_full_stream_summary.csv", index=False)
    log("DONE 10s full stream probe")
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
