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
RESULTS = Path(os.getenv("ROAMM_RESULTS_DIR", ROOT.parent / "work" / "results")).resolve()
OUT_SUMMARY = RESULTS / "probe_reve_10s_patchpca_perm_summary.csv"
OUT_PERMS = RESULTS / "probe_reve_10s_patchpca_perm_null.csv"
OUT_README = RESULTS / "probe_reve_10s_patchpca_perm_README.md"


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


def load_patchmean_to_vram(device: torch.device) -> tuple[torch.Tensor, list[str]]:
    meta = json.load(open(RESULTS / "reve_base_features_10s_balanced_f16_meta.json"))
    shape = tuple(meta["shape"])
    channels = list(meta["channels"])
    mm = np.memmap(
        RESULTS / "reve_base_features_10s_balanced_f16.dat",
        dtype="float16",
        mode="r",
        shape=shape,
    )
    log(f"loading 10s REVE tensor to VRAM shape={shape} disk_gb={np.prod(shape) * 2 / 1e9:.2f}")
    x4 = torch.from_numpy(np.array(mm, dtype=np.float16, copy=True)).to(device, non_blocking=True)
    torch.cuda.synchronize()
    log(f"loaded x4={tuple(x4.shape)} vram_gb={torch.cuda.memory_allocated() / 1e9:.2f}")
    log("computing channel-resolved patchmean view on GPU: mean over 11 patches -> 64*512 features")
    x = torch.mean(x4, dim=2, dtype=torch.float32).reshape(shape[0], -1).contiguous()
    del x4
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log(f"patchmean X={tuple(x.shape)} vram_gb={torch.cuda.memory_allocated() / 1e9:.2f}")
    return x, channels


def fit_pca_train_only(
    x: torch.Tensor,
    idx_fit: torch.Tensor,
    n_components: int,
) -> torch.Tensor:
    log("standardizing patchmean features using fit split only")
    xf = x.index_select(0, idx_fit)
    mu = xf.mean(0)
    sd = xf.std(0)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    del xf
    x.sub_(mu).div_(sd)
    del mu, sd
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log(f"standardized X in VRAM vram_gb={torch.cuda.memory_allocated() / 1e9:.2f}")

    xf = x.index_select(0, idx_fit)
    q = min(n_components + 20, min(xf.shape) - 1)
    log(f"fitting randomized PCA on fit split only xf={tuple(xf.shape)} q={q}")
    # X was already centered by fit mean, so center=False prevents using all rows for centering.
    _, s, v = torch.pca_lowrank(xf, q=q, center=False, niter=4)
    log(f"PCA fitted top_singular={float(s[0]):.4f} vram_gb={torch.cuda.memory_allocated() / 1e9:.2f}")
    comps = v[:, :n_components].contiguous()
    del xf, s, v
    torch.cuda.empty_cache()
    log(f"projecting all rows to {n_components} PCs on GPU")
    z = x @ comps
    del x, comps
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log(f"PCA scores Z={tuple(z.shape)} vram_gb={torch.cuda.memory_allocated() / 1e9:.2f}")
    return z


def standardize_design(xraw: torch.Tensor, idx_fit: torch.Tensor) -> torch.Tensor:
    mu = xraw.index_select(0, idx_fit).mean(0)
    sd = xraw.index_select(0, idx_fit).std(0)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    return (xraw - mu) / sd


def make_design(z: torch.Tensor, k: int, z_eye: torch.Tensor | None, idx_fit: torch.Tensor) -> torch.Tensor:
    cols = [z[:, :k]]
    if z_eye is not None:
        cols.insert(0, z_eye.view(-1, 1))
    xraw = torch.cat(cols, dim=1)
    return standardize_design(xraw, idx_fit)


def train_select(
    xs: torch.Tensor,
    y: np.ndarray,
    idx_fit: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
    wds: list[float],
    lr: float,
    epochs: int,
    tag: str,
) -> tuple[np.ndarray, dict[str, object]]:
    device = xs.device
    yt = torch.tensor(y.astype("float32"), device=device).view(-1, 1)
    y_val = y[idx_val.detach().cpu().numpy()]
    best: dict[str, object] | None = None
    for wd in wds:
        torch.manual_seed(SEED)
        model = nn.Linear(xs.shape[1], 1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.BCEWithLogitsLoss()
        local: dict[str, object] | None = None
        for ep in range(epochs):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xs.index_select(0, idx_fit)), yt.index_select(0, idx_fit))
            loss.backward()
            opt.step()
            if ep % 2 == 0 or ep == epochs - 1:
                with torch.no_grad():
                    pv = torch.sigmoid(model(xs.index_select(0, idx_val))).detach().cpu().numpy().ravel()
                vll = ll(y_val, pv)
                if local is None or vll < float(local["val_ll"]):
                    local = {
                        "val_ll": vll,
                        "epoch": ep + 1,
                        "wd": wd,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    }
        assert local is not None
        log(f"{tag} wd={wd:g} best_val={float(local['val_ll']):.6f} epoch={local['epoch']}")
        if best is None or float(local["val_ll"]) < float(best["val_ll"]):
            best = local
        del model, opt
        torch.cuda.empty_cache()
    assert best is not None
    model = nn.Linear(xs.shape[1], 1).to(device)
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})  # type: ignore[index]
    with torch.no_grad():
        pred = torch.sigmoid(model(xs.index_select(0, idx_test))).detach().cpu().numpy().ravel()
    del model
    torch.cuda.empty_cache()
    return pred, best


def train_fixed(
    xs: torch.Tensor,
    y: np.ndarray,
    idx_fit: torch.Tensor,
    idx_test: torch.Tensor,
    wd: float,
    lr: float,
    epochs: int,
) -> np.ndarray:
    device = xs.device
    yt = torch.tensor(y.astype("float32"), device=device).view(-1, 1)
    torch.manual_seed(SEED)
    model = nn.Linear(xs.shape[1], 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xs.index_select(0, idx_fit)), yt.index_select(0, idx_fit))
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = torch.sigmoid(model(xs.index_select(0, idx_test))).detach().cpu().numpy().ravel()
    del model, opt
    torch.cuda.empty_cache()
    return pred


def subject_permutation_indices(df: pd.DataFrame, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    perm = torch.arange(len(df), device=device)
    for _, g in df.groupby("sub"):
        idx = torch.tensor(g.index.to_numpy(), device=device, dtype=torch.long)
        if idx.numel() > 1:
            perm.index_copy_(0, idx, idx[torch.randperm(idx.numel(), generator=gen, device=device)])
    return perm


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is required for this run"
    device = torch.device("cuda")
    df = pd.read_csv(RESULTS / "balanced_epoch_index_10s.csv").reset_index(drop=True)
    y = df["mw"].astype(int).to_numpy()
    orig = df["orig_epoch_id"].to_numpy(dtype=int)
    p_eye = np.load(RESULTS / "p_eye_personal_cal.npy")[orig]
    _, test, fit, val = split_masks(df)
    idx_fit = torch.tensor(np.flatnonzero(fit), device=device, dtype=torch.long)
    idx_val = torch.tensor(np.flatnonzero(val), device=device, dtype=torch.long)
    idx_test = torch.tensor(np.flatnonzero(test), device=device, dtype=torch.long)
    z_eye = torch.tensor(logit(np.clip(p_eye, EPS, 1 - EPS)).astype("float32"), device=device)
    eye_ll = ll(y[test], p_eye[test])
    eye_auc = float(roc_auc_score(y[test], p_eye[test]))
    log(f"eye baseline ll={eye_ll:.6f} auc={eye_auc:.6f} n={len(df)} fit={fit.sum()} val={val.sum()} test={test.sum()}")

    x, _ = load_patchmean_to_vram(device)
    z = fit_pca_train_only(x, idx_fit, n_components=300)

    rows: list[dict[str, object]] = []
    wds = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 50.0]
    lr = 0.03
    epochs = 160
    for k in [100, 200, 300]:
        for combo in [False, True]:
            label = ("eye_reve10s_patchpca" if combo else "reve10s_patchpca") + str(k)
            xs = make_design(z, k, z_eye if combo else None, idx_fit)
            torch.cuda.synchronize()
            log(f"training {label} xs={tuple(xs.shape)} vram_gb={torch.cuda.memory_allocated() / 1e9:.2f}")
            pred, best = train_select(xs, y, idx_fit, idx_val, idx_test, wds, lr, epochs, label)
            test_ll = ll(y[test], pred)
            test_auc = float(roc_auc_score(y[test], pred))
            row = {
                "model": label,
                "k": k,
                "best_wd": best["wd"],
                "best_epoch": best["epoch"],
                "val_ll": best["val_ll"],
                "test_ll": test_ll,
                "test_auc": test_auc,
                "ll_gain_over_eye": eye_ll - test_ll,
                "auc_gain_over_eye": test_auc - eye_auc,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT_SUMMARY, index=False)
            np.save(RESULTS / f"p_{label}.npy", pred)
            log(f"{label}: test_ll={test_ll:.6f} auc={test_auc:.6f} ll_gain={eye_ll - test_ll:+.6f} auc_gain={test_auc - eye_auc:+.6f}")
            del xs
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    combo = out[out["model"].str.startswith("eye_reve10s_patchpca")].copy()
    # Select the dimensionality by validation log-loss only, before the test-set permutation assessment.
    selected = combo.sort_values(["val_ll", "k"]).iloc[0].to_dict()
    selected_k = int(selected["k"])
    selected_wd = float(selected["best_wd"])
    selected_epoch = int(selected["best_epoch"])
    observed_ll_gain = float(selected["ll_gain_over_eye"])
    observed_auc_gain = float(selected["auc_gain_over_eye"])
    log(f"selected for permutation by val_ll: k={selected_k} wd={selected_wd:g} epoch={selected_epoch} observed_ll_gain={observed_ll_gain:+.6f} observed_auc_gain={observed_auc_gain:+.6f}")

    perm_rows: list[dict[str, object]] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED + 1000)
    n_perm = 500
    y_test = y[test]
    for i in range(n_perm):
        perm = subject_permutation_indices(df, device, gen)
        zp = z.index_select(0, perm)
        xs = make_design(zp, selected_k, z_eye, idx_fit)
        pred = train_fixed(xs, y, idx_fit, idx_test, selected_wd, lr, selected_epoch)
        pll = ll(y_test, pred)
        pauc = float(roc_auc_score(y_test, pred))
        perm_rows.append({
            "perm": i + 1,
            "k": selected_k,
            "wd": selected_wd,
            "epoch": selected_epoch,
            "test_ll": pll,
            "test_auc": pauc,
            "ll_gain_over_eye": eye_ll - pll,
            "auc_gain_over_eye": pauc - eye_auc,
        })
        if (i + 1) % 25 == 0:
            pd.DataFrame(perm_rows).to_csv(OUT_PERMS, index=False)
            null_ll = np.array([r["ll_gain_over_eye"] for r in perm_rows], dtype=float)
            log(f"permutation {i + 1}/{n_perm} null_ll_gain_mean={null_ll.mean():+.6f} max={null_ll.max():+.6f}")
        del perm, zp, xs
        torch.cuda.empty_cache()

    null = pd.DataFrame(perm_rows)
    null.to_csv(OUT_PERMS, index=False)
    null_ll = null["ll_gain_over_eye"].to_numpy(dtype=float)
    null_auc = null["auc_gain_over_eye"].to_numpy(dtype=float)
    p_ll = (1.0 + float(np.sum(null_ll >= observed_ll_gain))) / (n_perm + 1.0)
    p_auc = (1.0 + float(np.sum(null_auc >= observed_auc_gain))) / (n_perm + 1.0)
    final_row = {
        "model": "permutation_test_selected_eye_reve10s_patchpca",
        "k": selected_k,
        "best_wd": selected_wd,
        "best_epoch": selected_epoch,
        "val_ll": selected["val_ll"],
        "test_ll": selected["test_ll"],
        "test_auc": selected["test_auc"],
        "ll_gain_over_eye": observed_ll_gain,
        "auc_gain_over_eye": observed_auc_gain,
        "perm_n": n_perm,
        "perm_p_ll_gain_ge_observed": p_ll,
        "perm_p_auc_gain_ge_observed": p_auc,
        "perm_ll_gain_mean": float(null_ll.mean()),
        "perm_ll_gain_p95": float(np.quantile(null_ll, 0.95)),
        "perm_auc_gain_mean": float(null_auc.mean()),
        "perm_auc_gain_p95": float(np.quantile(null_auc, 0.95)),
    }
    final = pd.concat([out, pd.DataFrame([final_row])], ignore_index=True)
    final.to_csv(OUT_SUMMARY, index=False)

    OUT_README.write_text(
        "# 10s Patch-PCA Permutation Probe\n\n"
        "This run uses the saved "
        "`(13822, 64, 11, 512)` 10s REVE tensor, averages over REVE patches, keeps the "
        "channel-resolved `64 x 512` representation, standardizes and fits PCA on the fit split only, "
        "then tests eye-calibrated predictions plus PCA scores. The selected eye+REVE PCA model is "
        "assessed against 500 within-subject shuffles of the REVE PCA block.\n\n"
        f"Eye baseline LL={eye_ll:.6f}, AUC={eye_auc:.6f}.\n\n"
        f"Selected k={selected_k}, wd={selected_wd:g}, epoch={selected_epoch}.\n\n"
        f"Observed LL gain={observed_ll_gain:+.6f}, permutation p={p_ll:.6f}.\n"
        f"Observed AUC gain={observed_auc_gain:+.6f}, permutation p={p_auc:.6f}.\n",
        encoding="utf-8",
    )
    log("DONE patch-PCA permutation probe")
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()


