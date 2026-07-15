from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from roamm_riemann_eeg10s_tests import (
    SEED,
    add_ridge,
    condition_one,
    ensure_covariances,
    fit_logistic,
    invsqrtm_spd,
    load_meta,
    log,
    mean_riemann,
    metrics,
    open_cov,
    score_covs,
)


def align_with_train(cov, train_rows: np.ndarray, test_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_c = condition_one(np.asarray(cov[train_rows], dtype=np.float64).mean(axis=0))
    w = invsqrtm_spd(mean_c)
    train = np.einsum("ij,bjk,kl->bil", w, np.asarray(cov[train_rows], dtype=np.float64), w.T, optimize=True)
    test = np.einsum("ij,bjk,kl->bil", w, np.asarray(cov[test_rows], dtype=np.float64), w.T, optimize=True)
    return add_ridge(train.astype(np.float32)), add_ridge(test.astype(np.float32))


def run_within_subject(root: Path, out_dir: Path, batch: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    meta = load_meta(root)
    idx = pd.read_csv(root / "balanced_epoch_index_10s.csv").reset_index(drop=True)
    idx["sub"] = idx["sub"].astype(int)
    idx["mw"] = idx["mw"].astype(int)
    cov_path = ensure_covariances(root, meta, batch)
    cov = open_cov(root, meta, cov_path)

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
        train_cov, test_cov = align_with_train(cov, train_rows, test_rows)
        ref_on = mean_riemann(train_cov[y_train == 0].astype(np.float64), tol=1e-4, maxiter=30)
        ref_mw = mean_riemann(train_cov[y_train == 1].astype(np.float64), tol=1e-4, maxiter=30)
        train_score = score_covs(train_cov.astype(np.float64), ref_on, ref_mw)
        test_score = score_covs(test_cov.astype(np.float64), ref_on, ref_mw)
        cal = fit_logistic(train_score, y_train)
        p_test = cal.predict_proba(test_score.reshape(-1, 1))[:, 1]
        m = metrics(y_test, p_test, test_score)
        rows.append({"subject": sub, "n_train": len(train_rows), **m})
        part = idx.loc[test_rows, ["epoch_id", "sub", "run", "mw"]].copy()
        part["riemann_score_within_broadband"] = test_score
        part["p_riemann_within_broadband"] = p_test
        pred_parts.append(part)
        log(f"WITHIN {i:02d}/{len(subjects)} sub={sub} train={len(train_rows)} test={len(test_rows)} acc={m['acc']:.3f} auc={m['auc']:.3f}")

    by_subject = pd.DataFrame(rows)
    by_subject.to_csv(out_dir / "riemann_within_subject_broadband_by_subject.csv", index=False)
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    preds.to_csv(out_dir / "riemann_within_subject_broadband_predictions.csv", index=False)
    overall = metrics(
        preds["mw"].to_numpy(dtype=int),
        preds["p_riemann_within_broadband"].to_numpy(dtype=float),
        preds["riemann_score_within_broadband"].to_numpy(dtype=float),
    ) if len(preds) else {}
    pd.DataFrame([{ "test": "riemann_within_subject_broadband_train123_test45", **overall }]).to_csv(
        out_dir / "riemann_within_subject_broadband_summary.csv", index=False
    )
    return by_subject, preds, overall


def main() -> None:
    ap = argparse.ArgumentParser(description="Within-subject broadband Riemannian control for ROAMM 10s EEG")
    ap.add_argument("--root", default="data/roamm_eeg10s")
    ap.add_argument("--out", default="artifacts/roamm_riemann_eeg10s")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, overall = run_within_subject(root, out_dir, args.batch)
    log("SUMMARY")
    print(pd.DataFrame([{ "test": "riemann_within_subject_broadband_train123_test45", **overall }]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
