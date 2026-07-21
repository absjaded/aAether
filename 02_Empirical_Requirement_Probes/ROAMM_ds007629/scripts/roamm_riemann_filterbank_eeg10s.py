from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression

from roamm_riemann_eeg10s_tests import (
    RIDGE,
    SEED,
    add_ridge,
    aligned_subset,
    condition_one,
    distance_to_ref,
    fit_logistic,
    invsqrtm_spd,
    load_meta,
    log,
    mean_riemann,
    metrics,
    open_cov,
    open_eeg,
    riemann_mean_of_subjects,
    score_covs,
)

BANDS = [
    ("theta4_8", 4.0, 8.0),
    ("alpha8_13", 8.0, 13.0),
    ("beta13_30", 13.0, 30.0),
]


def band_cov_path(root: Path, name: str) -> Path:
    return root / f"roamm_eeg10s_cov_{name}_butter4_ridge1e-3_f32.dat"


def band_ea_path(root: Path, name: str) -> Path:
    return root / f"roamm_eeg10s_cov_{name}_butter4_ea_all_f32.dat"


def ensure_band_covariances(root: Path, meta: dict, name: str, lo: float, hi: float, batch: int) -> Path:
    shape = tuple(meta["shape"])
    n, ch, times = shape
    out = band_cov_path(root, name)
    expected = n * ch * ch * 4
    if out.exists() and out.stat().st_size == expected:
        log(f"band covariance cache exists {name} {out} size_gb={out.stat().st_size / 1e9:.2f}")
        return out
    eeg = open_eeg(root, meta)
    cov = np.memmap(out, dtype="float32", mode="w+", shape=(n, ch, ch))
    sos = butter(4, [lo, hi], btype="bandpass", fs=float(meta["sfreq"]), output="sos")
    log(f"computing {name} band covariance lo={lo} hi={hi} n={n} ch={ch} times={times} batch={batch}")
    for start in range(0, n, batch):
        stop = min(n, start + batch)
        x = np.asarray(eeg[start:stop], dtype=np.float32)
        x = x - x.mean(axis=2, keepdims=True)
        xb = sosfiltfilt(sos, x, axis=2).astype(np.float32, copy=False)
        xb = xb - xb.mean(axis=2, keepdims=True)
        cb = np.einsum("bct,bdt->bcd", xb, xb, optimize=True) / np.float32(times - 1)
        cov[start:stop] = add_ridge(cb.astype(np.float32), RIDGE)
        if stop == n or (start // batch) % 10 == 0:
            cov.flush()
            log(f"{name} covariance {stop}/{n}")
    cov.flush()
    return out


def ensure_band_ea_all(root: Path, meta: dict, idx: pd.DataFrame, cov_path: Path, name: str, batch: int) -> Path:
    n, ch, _ = tuple(meta["shape"])
    out = band_ea_path(root, name)
    expected = n * ch * ch * 4
    if out.exists() and out.stat().st_size == expected:
        log(f"band EA cache exists {name} {out} size_gb={out.stat().st_size / 1e9:.2f}")
        return out
    cov = open_cov(root, meta, cov_path)
    aligned = np.memmap(out, dtype="float32", mode="w+", shape=(n, ch, ch))
    log(f"computing all-run Euclidean Alignment for {name}")
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
        log(f"{name} EA subject {s_i}/{idx['sub'].nunique()} sub={sub} rows={len(rows)}")
    aligned.flush()
    return out


def subject_class_means_for_band(aligned, idx: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    means: dict[tuple[int, int], np.ndarray] = {}
    for (sub, y), g in idx.groupby(["sub", "mw"]):
        rows = g.index.to_numpy()
        if len(rows) < 2:
            continue
        means[(int(sub), int(y))] = mean_riemann(np.asarray(aligned[rows], dtype=np.float64), tol=1e-4, maxiter=30)
    return means


def stacked_scores_for_refs(covs_by_band: list[np.ndarray], refs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    cols = []
    for covs, (ref_on, ref_mw) in zip(covs_by_band, refs):
        cols.append(score_covs(covs, ref_on, ref_mw))
    return np.column_stack(cols)


def fit_multiband(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    clf.fit(scores, y.astype(int))
    return clf


def loso_filterbank(idx: pd.DataFrame, aligned_bands: dict[str, np.memmap], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    subjects = sorted(int(s) for s in idx["sub"].unique())
    scm = {name: subject_class_means_for_band(mm, idx) for name, mm in aligned_bands.items()}
    band_names = list(aligned_bands.keys())
    rows = []
    pred_parts = []
    for i, sub in enumerate(subjects, 1):
        train_subjects = [s for s in subjects if s != sub]
        refs = []
        for name in band_names:
            refs.append((
                riemann_mean_of_subjects(scm[name], train_subjects, 0),
                riemann_mean_of_subjects(scm[name], train_subjects, 1),
            ))
        train_rows = idx.index[idx["sub"].astype(int) != sub].to_numpy()
        test_rows = idx.index[idx["sub"].astype(int) == sub].to_numpy()
        train_covs = [np.asarray(aligned_bands[name][train_rows], dtype=np.float64) for name in band_names]
        test_covs = [np.asarray(aligned_bands[name][test_rows], dtype=np.float64) for name in band_names]
        x_train = stacked_scores_for_refs(train_covs, refs)
        x_test = stacked_scores_for_refs(test_covs, refs)
        y_train = idx.loc[train_rows, "mw"].to_numpy(dtype=int)
        y_test = idx.loc[test_rows, "mw"].to_numpy(dtype=int)
        clf = fit_multiband(x_train, y_train)
        p = clf.predict_proba(x_test)[:, 1]
        m = metrics(y_test, p, p)
        rows.append({"subject": sub, **m})
        part = idx.loc[test_rows, ["epoch_id", "sub", "run", "mw"]].copy()
        for j, name in enumerate(band_names):
            part[f"score_{name}"] = x_test[:, j]
        part["p_filterbank_loso"] = p
        pred_parts.append(part)
        log(f"FB LOSO {i:02d}/{len(subjects)} sub={sub} acc={m['acc']:.3f} auc={m['auc']:.3f}")
    by_subject = pd.DataFrame(rows)
    preds = pd.concat(pred_parts, ignore_index=True)
    overall = metrics(preds["mw"].to_numpy(), preds["p_filterbank_loso"].to_numpy(), preds["p_filterbank_loso"].to_numpy())
    by_subject.to_csv(out_dir / "filterbank_loso_by_subject.csv", index=False)
    preds.to_csv(out_dir / "filterbank_loso_predictions.csv", index=False)
    return by_subject, preds, overall


def calibration_cost(preds: pd.DataFrame, sizes: list[int], repeats: int, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    score_cols = [c for c in preds.columns if c.startswith("score_")]
    rows = []
    for sub, g0 in preds.groupby("sub"):
        g = g0.reset_index(drop=True)
        y_all = g["mw"].to_numpy(dtype=int)
        x_all = g[score_cols].to_numpy(dtype=float)
        p0 = g["p_filterbank_loso"].to_numpy(dtype=float)
        pos_idx = np.flatnonzero(y_all == 1)
        neg_idx = np.flatnonzero(y_all == 0)
        for n in sizes:
            if n == 0:
                rows.append({"subject": sub, "calibration_trials": 0, "repeat": 0, "usable": True, **metrics(y_all, p0, p0)})
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
                clf = fit_multiband(x_all[cal_idx], y_all[cal_idx])
                p = clf.predict_proba(x_all[test_idx])[:, 1]
                rows.append({"subject": sub, "calibration_trials": n, "repeat": r, "usable": True, **metrics(y_all[test_idx], p, p)})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "filterbank_calibration_cost_trials.csv", index=False)
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
    summary.to_csv(out_dir / "filterbank_calibration_cost_summary.csv", index=False)
    return summary


def align_subject_train_test(cov, train_rows: np.ndarray, test_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_c = condition_one(np.asarray(cov[train_rows], dtype=np.float64).mean(axis=0))
    w = invsqrtm_spd(mean_c)
    train = np.einsum("ij,bjk,kl->bil", w, np.asarray(cov[train_rows], dtype=np.float64), w.T, optimize=True)
    test = np.einsum("ij,bjk,kl->bil", w, np.asarray(cov[test_rows], dtype=np.float64), w.T, optimize=True)
    return add_ridge(train.astype(np.float32)), add_ridge(test.astype(np.float32))


def within_subject_filterbank(idx: pd.DataFrame, band_covs: dict[str, np.memmap], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    band_names = list(band_covs.keys())
    rows = []
    pred_parts = []
    subjects = sorted(int(s) for s in idx["sub"].unique())
    for i, sub in enumerate(subjects, 1):
        sub_mask = idx["sub"].astype(int) == sub
        train_rows = idx.index[sub_mask & idx["run"].isin([1, 2, 3])].to_numpy()
        test_rows = idx.index[sub_mask & idx["run"].isin([4, 5])].to_numpy()
        if len(train_rows) < 8 or len(test_rows) < 8:
            continue
        y_train = idx.loc[train_rows, "mw"].to_numpy(dtype=int)
        y_test = idx.loc[test_rows, "mw"].to_numpy(dtype=int)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        x_train_cols = []
        x_test_cols = []
        for name in band_names:
            train_cov, test_cov = align_subject_train_test(band_covs[name], train_rows, test_rows)
            ref_on = mean_riemann(train_cov[y_train == 0].astype(np.float64), tol=1e-4, maxiter=30)
            ref_mw = mean_riemann(train_cov[y_train == 1].astype(np.float64), tol=1e-4, maxiter=30)
            x_train_cols.append(score_covs(train_cov.astype(np.float64), ref_on, ref_mw))
            x_test_cols.append(score_covs(test_cov.astype(np.float64), ref_on, ref_mw))
        x_train = np.column_stack(x_train_cols)
        x_test = np.column_stack(x_test_cols)
        clf = fit_multiband(x_train, y_train)
        p = clf.predict_proba(x_test)[:, 1]
        m = metrics(y_test, p, p)
        rows.append({"subject": sub, "n_train": len(train_rows), **m})
        part = idx.loc[test_rows, ["epoch_id", "sub", "run", "mw"]].copy()
        for j, name in enumerate(band_names):
            part[f"score_{name}"] = x_test[:, j]
        part["p_filterbank_within"] = p
        pred_parts.append(part)
        log(f"FB WITHIN {i:02d}/{len(subjects)} sub={sub} train={len(train_rows)} test={len(test_rows)} acc={m['acc']:.3f} auc={m['auc']:.3f}")
    by_subject = pd.DataFrame(rows)
    preds = pd.concat(pred_parts, ignore_index=True)
    overall = metrics(preds["mw"].to_numpy(), preds["p_filterbank_within"].to_numpy(), preds["p_filterbank_within"].to_numpy())
    by_subject.to_csv(out_dir / "filterbank_within_subject_by_subject.csv", index=False)
    preds.to_csv(out_dir / "filterbank_within_subject_predictions.csv", index=False)
    return by_subject, preds, overall


def main() -> None:
    ap = argparse.ArgumentParser(description="ROAMM filter-bank Riemannian tests")
    ap.add_argument("--root", default="data/roamm_eeg10s")
    ap.add_argument("--out", default="work/roamm_riemann_filterbank_eeg10s")
    ap.add_argument("--batch", type=int, default=16)
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

    band_cov_paths = {}
    band_covs = {}
    aligned_bands = {}
    for name, lo, hi in BANDS:
        cp = ensure_band_covariances(root, meta, name, lo, hi, args.batch)
        band_cov_paths[name] = cp
        band_covs[name] = open_cov(root, meta, cp)
        apath = ensure_band_ea_all(root, meta, idx, cp, name, args.batch)
        aligned_bands[name] = open_cov(root, meta, apath)

    _, loso_preds, loso = loso_filterbank(idx, aligned_bands, out_dir)
    cal = calibration_cost(loso_preds, [0, 5, 10, 20, 50], args.calibration_repeats, out_dir)
    _, within_preds, within = within_subject_filterbank(idx, band_covs, out_dir)
    summary = pd.DataFrame([
        {"test": "filterbank_riemann_loso_39to1", **loso},
        {"test": "filterbank_riemann_within_train123_test45", **within},
    ])
    summary.to_csv(out_dir / "filterbank_summary.csv", index=False)
    log("SUMMARY")
    print(summary.to_string(index=False), flush=True)
    log("CALIBRATION")
    print(cal.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
