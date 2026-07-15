from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann

SEED = 42
EPS = 1e-3
RIDGE = 1e-3


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def load_meta(root: Path) -> dict:
    return json.load(open(root / "roamm_balanced_eeg10s_uV_200hz_meta.json", encoding="utf-8"))


def open_eeg(root: Path, meta: dict) -> np.memmap:
    return np.memmap(
        root / "roamm_balanced_eeg10s_uV_200hz.dat",
        dtype=meta["dtype"],
        mode="r",
        shape=tuple(meta["shape"]),
    )


def sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + np.swapaxes(a, -1, -2))


def add_ridge(cov: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    cov = sym(cov)
    d = cov.shape[-1]
    tr = np.trace(cov, axis1=-2, axis2=-1) / d
    tr = np.maximum(tr, 1e-12).astype(cov.dtype, copy=False)
    ii = np.arange(d)
    cov[..., ii, ii] += ridge * tr[..., None]
    return cov


def invsqrtm_spd(c: np.ndarray) -> np.ndarray:
    c = sym(np.asarray(c, dtype=np.float64))
    vals, vecs = eigh(c)
    floor = max(1e-10, 1e-8 * float(np.mean(vals[vals > 0])) if np.any(vals > 0) else 1e-8)
    vals = np.maximum(vals, floor)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def condition_one(c: np.ndarray) -> np.ndarray:
    c = sym(np.asarray(c, dtype=np.float64))
    vals, vecs = eigh(c)
    floor = max(1e-9, 1e-7 * float(np.mean(vals[vals > 0])) if np.any(vals > 0) else 1e-7)
    vals = np.maximum(vals, floor)
    out = (vecs * vals) @ vecs.T
    return sym(out)


def ensure_covariances(root: Path, meta: dict, batch: int) -> Path:
    shape = tuple(meta["shape"])
    n, ch, times = shape
    out = root / "roamm_eeg10s_cov_ridge1e-3_f32.dat"
    expected = n * ch * ch * 4
    if out.exists() and out.stat().st_size == expected:
        log(f"covariance cache exists {out} size_gb={out.stat().st_size / 1e9:.2f}")
        return out
    eeg = open_eeg(root, meta)
    cov = np.memmap(out, dtype="float32", mode="w+", shape=(n, ch, ch))
    log(f"computing covariance cache n={n} ch={ch} times={times} batch={batch}")
    for start in range(0, n, batch):
        stop = min(n, start + batch)
        x = np.asarray(eeg[start:stop], dtype=np.float32)
        x = x - x.mean(axis=2, keepdims=True)
        cb = np.einsum("bct,bdt->bcd", x, x, optimize=True) / np.float32(times - 1)
        cb = add_ridge(cb, RIDGE)
        cov[start:stop] = cb.astype(np.float32)
        if stop == n or (start // batch) % 10 == 0:
            cov.flush()
            log(f"covariance {stop}/{n}")
    cov.flush()
    return out


def open_cov(root: Path, meta: dict, path: Path, mode: str = "r") -> np.memmap:
    n, ch, _ = tuple(meta["shape"])
    return np.memmap(path, dtype="float32", mode=mode, shape=(n, ch, ch))


def ensure_aligned_all(root: Path, meta: dict, idx: pd.DataFrame, cov_path: Path, batch: int) -> Path:
    n, ch, _ = tuple(meta["shape"])
    out = root / "roamm_eeg10s_cov_ea_all_f32.dat"
    expected = n * ch * ch * 4
    if out.exists() and out.stat().st_size == expected:
        log(f"EA all-run cache exists {out} size_gb={out.stat().st_size / 1e9:.2f}")
        return out
    cov = open_cov(root, meta, cov_path)
    aligned = np.memmap(out, dtype="float32", mode="w+", shape=(n, ch, ch))
    log("computing Euclidean Alignment from each subject's unlabeled 10s covariances")
    for s_i, (sub, g) in enumerate(idx.groupby("sub"), 1):
        rows = g.index.to_numpy()
        mean_c = condition_one(np.asarray(cov[rows], dtype=np.float64).mean(axis=0))
        w = invsqrtm_spd(mean_c)
        for start in range(0, len(rows), batch):
            r = rows[start:start + batch]
            cb = np.asarray(cov[r], dtype=np.float64)
            ab = np.einsum("ij,bjk,kl->bil", w, cb, w.T, optimize=True)
            aligned[r] = add_ridge(ab.astype(np.float32), RIDGE)
        aligned.flush()
        log(f"EA all-run subject {s_i}/{idx['sub'].nunique()} sub={sub} rows={len(rows)}")
    aligned.flush()
    return out


def aligned_subset(cov: np.memmap, rows: np.ndarray) -> np.ndarray:
    return np.asarray(cov[rows], dtype=np.float64)


def subject_class_means(aligned: np.memmap, idx: pd.DataFrame, rows: np.ndarray | None = None) -> dict[tuple[int, int], np.ndarray]:
    if rows is None:
        frame = idx
    else:
        frame = idx.loc[rows]
    means: dict[tuple[int, int], np.ndarray] = {}
    for (sub, y), g in frame.groupby(["sub", "mw"]):
        r = g.index.to_numpy()
        if len(r) < 2:
            continue
        means[(int(sub), int(y))] = mean_riemann(aligned_subset(aligned, r), tol=1e-4, maxiter=30)
    return means


def riemann_mean_of_subjects(means: dict[tuple[int, int], np.ndarray], subjects: list[int], y: int) -> np.ndarray:
    mats = [means[(int(s), int(y))] for s in subjects if (int(s), int(y)) in means]
    if len(mats) < 2:
        raise ValueError(f"Not enough subject means for class {y}")
    return mean_riemann(np.stack(mats, axis=0), tol=1e-5, maxiter=50)


def distance_to_ref(covs: np.ndarray, ref: np.ndarray, batch: int = 512) -> np.ndarray:
    w = invsqrtm_spd(ref)
    out = np.empty(covs.shape[0], dtype=np.float64)
    for start in range(0, covs.shape[0], batch):
        stop = min(covs.shape[0], start + batch)
        cb = np.asarray(covs[start:stop], dtype=np.float64)
        xb = np.einsum("ij,bjk,kl->bil", w, cb, w.T, optimize=True)
        xb = sym(xb)
        vals = np.linalg.eigvalsh(xb)
        vals = np.maximum(vals, 1e-12)
        out[start:stop] = np.sqrt(np.sum(np.log(vals) ** 2, axis=1))
    return out


def score_covs(covs: np.ndarray, ref_on: np.ndarray, ref_mw: np.ndarray) -> np.ndarray:
    d_on = distance_to_ref(covs, ref_on)
    d_mw = distance_to_ref(covs, ref_mw)
    return np.asarray(d_on - d_mw, dtype=float)


def fit_logistic(score: np.ndarray, y: np.ndarray) -> LogisticRegression:
    cal = LogisticRegression(max_iter=1000, random_state=SEED)
    cal.fit(np.asarray(score).reshape(-1, 1), np.asarray(y, dtype=int))
    return cal


def metrics(y: np.ndarray, p: np.ndarray, score: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    pred = (p >= 0.5).astype(int)
    if score is None:
        score = p
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    on_acc = tn / max(1, tn + fp)
    mw_acc = tp / max(1, tp + fn)
    return {
        "n": float(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "balanced_acc": float((on_acc + mw_acc) / 2.0),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "on_task_acc": float(on_acc),
        "mw_acc": float(mw_acc),
    }


def loso_reference(idx: pd.DataFrame, aligned: np.memmap, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    subjects = sorted(int(s) for s in idx["sub"].unique())
    scm = subject_class_means(aligned, idx)
    rows = []
    pred_parts = []
    for i, sub in enumerate(subjects, 1):
        train_subjects = [s for s in subjects if s != sub]
        ref_on = riemann_mean_of_subjects(scm, train_subjects, 0)
        ref_mw = riemann_mean_of_subjects(scm, train_subjects, 1)
        train_rows = idx.index[idx["sub"].astype(int) != sub].to_numpy()
        test_rows = idx.index[idx["sub"].astype(int) == sub].to_numpy()
        train_score = score_covs(aligned_subset(aligned, train_rows), ref_on, ref_mw)
        test_score = score_covs(aligned_subset(aligned, test_rows), ref_on, ref_mw)
        cal = fit_logistic(train_score, idx.loc[train_rows, "mw"].to_numpy())
        p_test = cal.predict_proba(test_score.reshape(-1, 1))[:, 1]
        y_test = idx.loc[test_rows, "mw"].to_numpy(dtype=int)
        m = metrics(y_test, p_test, test_score)
        rows.append({"subject": sub, **m})
        part = idx.loc[test_rows, ["epoch_id", "sub", "run", "mw"]].copy()
        part["riemann_score_loso"] = test_score
        part["p_riemann_loso"] = p_test
        pred_parts.append(part)
        log(f"LOSO {i:02d}/{len(subjects)} sub={sub} acc={m['acc']:.3f} auc={m['auc']:.3f}")
    by_subject = pd.DataFrame(rows)
    preds = pd.concat(pred_parts, ignore_index=True)
    overall = metrics(preds["mw"].to_numpy(), preds["p_riemann_loso"].to_numpy(), preds["riemann_score_loso"].to_numpy())
    by_subject.to_csv(out_dir / "riemann_shared_reference_loso_by_subject.csv", index=False)
    preds.to_csv(out_dir / "riemann_shared_reference_loso_predictions.csv", index=False)
    return by_subject, preds, overall


def calibration_cost(preds: pd.DataFrame, sizes: list[int], repeats: int, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for sub, g0 in preds.groupby("sub"):
        g = g0.reset_index(drop=True)
        y_all = g["mw"].to_numpy(dtype=int)
        score_all = g["riemann_score_loso"].to_numpy(dtype=float)
        p0 = g["p_riemann_loso"].to_numpy(dtype=float)
        pos_idx = np.flatnonzero(y_all == 1)
        neg_idx = np.flatnonzero(y_all == 0)
        for n in sizes:
            if n == 0:
                m = metrics(y_all, p0, score_all)
                rows.append({"subject": sub, "calibration_trials": 0, "repeat": 0, "usable": True, **m})
                continue
            n_pos = n // 2
            n_neg = n - n_pos
            if len(pos_idx) <= n_pos or len(neg_idx) <= n_neg:
                rows.append({"subject": sub, "calibration_trials": n, "repeat": -1, "usable": False})
                continue
            for r in range(repeats):
                cal_pos = rng.choice(pos_idx, size=n_pos, replace=False)
                cal_neg = rng.choice(neg_idx, size=n_neg, replace=False)
                cal_idx = np.concatenate([cal_pos, cal_neg])
                test_mask = np.ones(len(g), dtype=bool)
                test_mask[cal_idx] = False
                test_idx = np.flatnonzero(test_mask)
                cal = fit_logistic(score_all[cal_idx], y_all[cal_idx])
                p = cal.predict_proba(score_all[test_idx].reshape(-1, 1))[:, 1]
                m = metrics(y_all[test_idx], p, score_all[test_idx])
                rows.append({"subject": sub, "calibration_trials": n, "repeat": r, "usable": True, **m})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "riemann_calibration_cost_trials.csv", index=False)
    summary = out[out["usable"] == True].groupby("calibration_trials").agg(
        reps=("repeat", "count"),
        subjects=("subject", "nunique"),
        acc_mean=("acc", "mean"),
        acc_sd=("acc", "std"),
        auc_mean=("auc", "mean"),
        log_loss_mean=("log_loss", "mean"),
        balanced_acc_mean=("balanced_acc", "mean"),
        on_task_acc_mean=("on_task_acc", "mean"),
        mw_acc_mean=("mw_acc", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "riemann_calibration_cost_summary.csv", index=False)
    return summary


def aligned_for_rows(cov: np.memmap, rows: np.ndarray, subjects: np.ndarray, batch: int) -> tuple[np.ndarray, pd.Index]:
    out = np.empty((len(rows), cov.shape[1], cov.shape[2]), dtype=np.float32)
    pos_by_row = {int(r): i for i, r in enumerate(rows)}
    frame = pd.DataFrame({"row": rows, "sub": subjects})
    for sub, g in frame.groupby("sub"):
        rr = g["row"].to_numpy(dtype=int)
        pp = np.array([pos_by_row[int(r)] for r in rr], dtype=int)
        mean_c = condition_one(np.asarray(cov[rr], dtype=np.float64).mean(axis=0))
        w = invsqrtm_spd(mean_c)
        for start in range(0, len(rr), batch):
            r = rr[start:start + batch]
            p = pp[start:start + batch]
            cb = np.asarray(cov[r], dtype=np.float64)
            ab = np.einsum("ij,bjk,kl->bil", w, cb, w.T, optimize=True)
            out[p] = add_ridge(ab.astype(np.float32), RIDGE)
    return out, pd.Index(rows)


def run_stability(idx: pd.DataFrame, cov: np.memmap, batch: int, out_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    train_rows_all = idx.index[idx["run"].isin([1, 2])].to_numpy()
    test_rows_all = idx.index[idx["run"].isin([4, 5])].to_numpy()
    log(f"run stability: aligning train rows={len(train_rows_all)} and test rows={len(test_rows_all)} separately")
    a_train, train_index = aligned_for_rows(cov, train_rows_all, idx.loc[train_rows_all, "sub"].to_numpy(), batch)
    a_test, test_index = aligned_for_rows(cov, test_rows_all, idx.loc[test_rows_all, "sub"].to_numpy(), batch)
    train_df = idx.loc[train_rows_all].copy()
    train_df.index = np.arange(len(train_df))
    test_df = idx.loc[test_rows_all].copy()
    test_df.index = np.arange(len(test_df))
    train_subjects = sorted(int(s) for s in train_df["sub"].unique())
    test_subjects = sorted(int(s) for s in test_df["sub"].unique())
    train_mm = a_train
    test_mm = a_test
    scm = subject_class_means(train_mm, train_df)
    rows = []
    pred_parts = []
    for i, sub in enumerate(test_subjects, 1):
        if sub not in train_subjects:
            continue
        train_subs = [s for s in train_subjects if s != sub]
        if not all((s, 0) in scm and (s, 1) in scm for s in train_subs):
            train_subs = [s for s in train_subs if (s, 0) in scm and (s, 1) in scm]
        if len(train_subs) < 5:
            continue
        test_rows = test_df.index[test_df["sub"].astype(int) == sub].to_numpy()
        if test_df.loc[test_rows, "mw"].nunique() < 2:
            continue
        ref_on = riemann_mean_of_subjects(scm, train_subs, 0)
        ref_mw = riemann_mean_of_subjects(scm, train_subs, 1)
        train_rows = train_df.index[train_df["sub"].astype(int) != sub].to_numpy()
        train_score = score_covs(np.asarray(train_mm[train_rows], dtype=np.float64), ref_on, ref_mw)
        test_score = score_covs(np.asarray(test_mm[test_rows], dtype=np.float64), ref_on, ref_mw)
        cal = fit_logistic(train_score, train_df.loc[train_rows, "mw"].to_numpy())
        p_test = cal.predict_proba(test_score.reshape(-1, 1))[:, 1]
        y_test = test_df.loc[test_rows, "mw"].to_numpy(dtype=int)
        m = metrics(y_test, p_test, test_score)
        rows.append({"subject": sub, **m})
        part = test_df.loc[test_rows, ["epoch_id", "sub", "run", "mw"]].copy()
        part["riemann_score_cross_run"] = test_score
        part["p_riemann_cross_run"] = p_test
        pred_parts.append(part)
        log(f"RUN {i:02d}/{len(test_subjects)} sub={sub} acc={m['acc']:.3f} auc={m['auc']:.3f}")
    by_subject = pd.DataFrame(rows)
    by_subject.to_csv(out_dir / "riemann_cross_run_by_subject.csv", index=False)
    if pred_parts:
        preds = pd.concat(pred_parts, ignore_index=True)
        preds.to_csv(out_dir / "riemann_cross_run_predictions.csv", index=False)
        overall = metrics(preds["mw"].to_numpy(), preds["p_riemann_cross_run"].to_numpy(), preds["riemann_score_cross_run"].to_numpy())
    else:
        overall = {}
    return by_subject, overall


def main() -> None:
    ap = argparse.ArgumentParser(description="ROAMM 10s EEG Riemannian shared-reference tests")
    ap.add_argument("--root", default="artifacts/roamm_eeg10s")
    ap.add_argument("--out", default="artifacts/roamm_riemann_eeg10s")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--calibration-repeats", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(root)
    idx = pd.read_csv(root / "balanced_epoch_index_10s.csv").reset_index(drop=True)
    idx["sub"] = idx["sub"].astype(int)
    idx["mw"] = idx["mw"].astype(int)
    log(f"loaded index n={len(idx)} subjects={idx['sub'].nunique()} labels={idx['mw'].value_counts().to_dict()}")

    cov_path = ensure_covariances(root, meta, args.batch)
    aligned_path = ensure_aligned_all(root, meta, idx, cov_path, args.batch)
    aligned = open_cov(root, meta, aligned_path)
    by_subject, preds, shared = loso_reference(idx, aligned, out_dir)
    cost = calibration_cost(preds, [0, 5, 10, 20, 50], args.calibration_repeats, out_dir)
    cov = open_cov(root, meta, cov_path)
    cross_subject, cross = run_stability(idx, cov, args.batch, out_dir)
    summary = pd.DataFrame([
        {"test": "riemann_shared_reference_loso", **shared},
        {"test": "riemann_cross_run_train12_test45_loso", **cross},
    ])
    summary.to_csv(out_dir / "riemann_summary.csv", index=False)
    log("SUMMARY")
    print(summary.to_string(index=False), flush=True)
    log("CALIBRATION")
    print(cost.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

