#!/usr/bin/env python3
"""Stage 1d shrinkage: cohort backbone plus subject calibration.

This keeps the Stage 1c/1d feature pipeline and centroid-gap score, then changes
only the reference estimator:

  cohort_only  : leave-this-subject-out cohort intact/violated centroids
  subject_only : held subject calibration centroids
  shrinkage    : class-wise blend of cohort and subject centroids

The shrinkage weight is empirical-Bayes style:

  lambda_class = n_subject_calibration_class / (n_subject_calibration_class + K_class)

K_class is estimated from other subjects by comparing within-subject centroid
noise to between-subject centroid spread in the same feature space. It is not
hand-tuned against the answer.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from synthetic_radiology import GeneratorConfig, simulate


DEFAULT_CALIBRATION_VALUES = [0, 70, 140, 200, 300, 500, 700]
ESTIMATORS = ("cohort_only", "subject_only", "shrinkage")
EPS = 1e-12


def _inv_sqrtm(c: np.ndarray) -> np.ndarray:
    vals, vecs = eigh(0.5 * (c + c.T))
    vals = np.clip(vals, 1e-8, None)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def _logm_spd(c: np.ndarray) -> np.ndarray:
    vals, vecs = eigh(0.5 * (c + c.T))
    vals = np.clip(vals, 1e-8, None)
    return (vecs * np.log(vals)) @ vecs.T


def _vec_upper(mats: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(mats.shape[1])
    return mats[:, iu[0], iu[1]]


def riemannian_ea_features_fast(covs: np.ndarray, subjects: np.ndarray, n_components: int) -> np.ndarray:
    aligned = np.empty_like(covs, dtype=np.float64)
    for s in np.unique(subjects):
        idx = subjects == s
        whitener = _inv_sqrtm(covs[idx].mean(axis=0))
        aligned[idx] = whitener @ covs[idx] @ whitener.T

    logs = np.empty_like(aligned, dtype=np.float64)
    for i, c in enumerate(aligned):
        logs[i] = _logm_spd(c)
    x = _vec_upper(logs)
    d = min(n_components, x.shape[1], x.shape[0] - 2)
    return PCA(n_components=d, random_state=0).fit_transform(x)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    denom = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    if denom <= EPS:
        return float("nan")
    return float((a.mean() - b.mean()) / denom)


def signflip_p(values: list[float], n_perm: int = 4096, seed: int = 17) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return float("nan")
    observed = float(vals.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(vals)))
    null = (signs * vals).mean(axis=1)
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def score_to_centroids(x: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    d0 = np.linalg.norm(x - c0, axis=1)
    d1 = np.linalg.norm(x - c1, axis=1)
    return d0 - d1


def metric_row(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, scores)), cohen_d(scores[y == 1], scores[y == 0])


def estimate_k(x_train: np.ndarray, y_train: np.ndarray, subject_train: np.ndarray, seed: int) -> dict[int, float]:
    rng = np.random.default_rng(seed)
    out: dict[int, float] = {}
    for cls in (0, 1):
        means = []
        within_terms = []
        for s in np.unique(subject_train):
            idx = np.flatnonzero((subject_train == s) & (y_train == cls))
            if len(idx) < 4:
                continue
            rng.shuffle(idx)
            half = len(idx) // 2
            a = idx[:half]
            b = idx[half:]
            if len(a) == 0 or len(b) == 0:
                continue
            ma = x_train[a].mean(axis=0)
            mb = x_train[b].mean(axis=0)
            mall = x_train[idx].mean(axis=0)
            means.append(mall)
            denom = (1.0 / len(a)) + (1.0 / len(b))
            within_terms.append(float(np.sum((ma - mb) ** 2) / max(denom, EPS)))
        if len(means) < 3:
            out[cls] = 25.0
            continue
        means_arr = np.vstack(means)
        cohort = means_arr.mean(axis=0)
        between = float(np.mean(np.sum((means_arr - cohort) ** 2, axis=1)))
        within = float(np.mean(within_terms)) if within_terms else between
        out[cls] = float(np.clip(within / max(between, EPS), 0.5, 500.0))
    return out


def evaluate_split(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    cal_size: int,
    split: int,
    rep: int,
    mode: str,
    seed: int,
    min_test_class_count: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for held in np.unique(subjects):
        train = subjects != held
        held_idx = np.flatnonzero(subjects == held)
        y_held = y[held_idx]
        if cal_size >= len(held_idx):
            for estimator in ESTIMATORS:
                rows.append({
                    "mode": mode,
                    "rep": rep,
                    "split": split,
                    "subject": int(held),
                    "calibration_reads": int(cal_size),
                    "estimator": estimator,
                    "usable": False,
                    "reason": "calibration_exhausts_subject",
                    "cal_pos": int(np.sum(y_held == 1)),
                    "cal_neg": int(np.sum(y_held == 0)),
                    "test_pos": 0,
                    "test_neg": 0,
                    "lambda_pos": float("nan"),
                    "lambda_neg": float("nan"),
                    "k_pos": float("nan"),
                    "k_neg": float("nan"),
                    "auc": float("nan"),
                    "per_subject_d": float("nan"),
                })
            continue

        cal_local = rng.choice(np.arange(len(held_idx)), size=cal_size, replace=False) if cal_size else np.array([], dtype=int)
        cal_mask = np.zeros(len(held_idx), dtype=bool)
        cal_mask[cal_local] = True
        cal_idx = held_idx[cal_mask]
        test_idx = held_idx[~cal_mask]
        y_cal = y[cal_idx]
        y_test = y[test_idx]
        cal_pos = int(np.sum(y_cal == 1))
        cal_neg = int(np.sum(y_cal == 0))
        test_pos = int(np.sum(y_test == 1))
        test_neg = int(np.sum(y_test == 0))

        if test_pos < min_test_class_count or test_neg < min_test_class_count:
            for estimator in ESTIMATORS:
                rows.append({
                    "mode": mode,
                    "rep": rep,
                    "split": split,
                    "subject": int(held),
                    "calibration_reads": int(cal_size),
                    "estimator": estimator,
                    "usable": False,
                    "reason": "test_missing_class",
                    "cal_pos": cal_pos,
                    "cal_neg": cal_neg,
                    "test_pos": test_pos,
                    "test_neg": test_neg,
                    "lambda_pos": float("nan"),
                    "lambda_neg": float("nan"),
                    "k_pos": float("nan"),
                    "k_neg": float("nan"),
                    "auc": float("nan"),
                    "per_subject_d": float("nan"),
                })
            continue

        scaler = StandardScaler().fit(x[train])
        xs_train = scaler.transform(x[train])
        xs_cal = scaler.transform(x[cal_idx]) if len(cal_idx) else np.empty((0, xs_train.shape[1]))
        xs_test = scaler.transform(x[test_idx])
        y_train = y[train]
        subject_train = subjects[train]

        if len(np.unique(y_train)) < 2:
            continue
        cohort_c0 = xs_train[y_train == 0].mean(axis=0)
        cohort_c1 = xs_train[y_train == 1].mean(axis=0)
        k = estimate_k(xs_train, y_train, subject_train, seed=seed + int(held) * 7919)
        k_neg = k[0]
        k_pos = k[1]
        lambda_neg = cal_neg / (cal_neg + k_neg) if cal_neg > 0 else 0.0
        lambda_pos = cal_pos / (cal_pos + k_pos) if cal_pos > 0 else 0.0

        subject_has_neg = cal_neg > 0
        subject_has_pos = cal_pos > 0
        subject_c0 = xs_cal[y_cal == 0].mean(axis=0) if subject_has_neg else cohort_c0
        subject_c1 = xs_cal[y_cal == 1].mean(axis=0) if subject_has_pos else cohort_c1
        shrink_c0 = (1.0 - lambda_neg) * cohort_c0 + lambda_neg * subject_c0
        shrink_c1 = (1.0 - lambda_pos) * cohort_c1 + lambda_pos * subject_c1

        refs = {
            "cohort_only": (cohort_c0, cohort_c1, True, "ok"),
            "subject_only": (
                subject_c0,
                subject_c1,
                bool(cal_size == 0 or (subject_has_neg and subject_has_pos)),
                "ok" if cal_size == 0 or (subject_has_neg and subject_has_pos) else "calibration_missing_class",
            ),
            "shrinkage": (shrink_c0, shrink_c1, True, "ok"),
        }
        for estimator, (c0, c1, usable, reason) in refs.items():
            if not usable:
                auc = float("nan")
                d = float("nan")
            else:
                scores = score_to_centroids(xs_test, c0, c1)
                auc, d = metric_row(scores, y_test)
            rows.append({
                "mode": mode,
                "rep": rep,
                "split": split,
                "subject": int(held),
                "calibration_reads": int(cal_size),
                "estimator": estimator,
                "usable": bool(usable),
                "reason": reason,
                "cal_pos": cal_pos,
                "cal_neg": cal_neg,
                "test_pos": test_pos,
                "test_neg": test_neg,
                "lambda_pos": lambda_pos if estimator == "shrinkage" else (0.0 if estimator == "cohort_only" else (1.0 if subject_has_pos else float("nan"))),
                "lambda_neg": lambda_neg if estimator == "shrinkage" else (0.0 if estimator == "cohort_only" else (1.0 if subject_has_neg else float("nan"))),
                "k_pos": k_pos,
                "k_neg": k_neg,
                "auc": auc,
                "per_subject_d": d,
            })
    return rows


def run_job(
    mode: str,
    rep: int,
    cfg: GeneratorConfig,
    components: int,
    calibration_values: list[int],
    splits: int,
    min_test_class_count: int,
) -> list[dict]:
    cfg = replace(cfg, seed=cfg.seed + rep * 100003)
    ds = simulate(mode, cfg)
    labels = ds["labels"]
    subjects = labels["subject"].astype(int)
    y = labels["violation"].astype(int)
    x = riemannian_ea_features_fast(ds["covariances"], subjects, n_components=components)

    rows: list[dict] = []
    for cal_size in calibration_values:
        n_splits = 1 if cal_size == 0 else splits
        for split in range(n_splits):
            rows.extend(evaluate_split(
                x=x,
                y=y,
                subjects=subjects,
                cal_size=cal_size,
                split=split,
                rep=rep,
                mode=mode,
                seed=cfg.seed + 1009 * rep + 9176 * split + 37 * cal_size + (0 if mode == "signal" else 1),
                min_test_class_count=min_test_class_count,
            ))
    return rows


def aggregate(detail_rows: list[dict], mode: str, cal_size: int, estimator: str, seed: int) -> dict:
    subset = [
        r for r in detail_rows
        if r["mode"] == mode and int(r["calibration_reads"]) == cal_size and r["estimator"] == estimator
    ]
    usable = [r for r in subset if r["usable"] and np.isfinite(float(r["auc"])) and np.isfinite(float(r["per_subject_d"]))]

    def mean(name: str, source: list[dict]) -> float:
        vals = np.asarray([float(r[name]) for r in source if np.isfinite(float(r[name]))], dtype=float)
        return float(vals.mean()) if len(vals) else float("nan")

    by_subject: dict[int, list[float]] = {}
    for r in usable:
        by_subject.setdefault(int(r["subject"]), []).append(float(r["per_subject_d"]))
    subject_d = [float(np.mean(v)) for v in by_subject.values() if len(v)]
    return {
        "usable_fraction": float(len(usable) / len(subset)) if subset else float("nan"),
        "usable_subject_repeats": int(len(usable)),
        "total_subject_repeats": int(len(subset)),
        "subjects_valid": int(len(by_subject)),
        "subjects_in_direction": int(np.sum(np.asarray(subject_d) > 0)) if subject_d else 0,
        "auc": mean("auc", usable),
        "d": float(np.mean(subject_d)) if subject_d else float("nan"),
        "p": signflip_p(subject_d, seed=seed),
        "mean_cal_pos": mean("cal_pos", usable),
        "mean_cal_neg": mean("cal_neg", usable),
        "mean_test_pos": mean("test_pos", usable),
        "mean_test_neg": mean("test_neg", usable),
        "mean_lambda_pos": mean("lambda_pos", usable),
        "mean_lambda_neg": mean("lambda_neg", usable),
        "mean_k_pos": mean("k_pos", usable),
        "mean_k_neg": mean("k_neg", usable),
    }


def verdict(signal: dict, null: dict) -> str:
    if np.isfinite(null["auc"]) and abs(null["auc"] - 0.5) > 0.10:
        return "leak_check_failed"
    if signal["usable_fraction"] < 0.80:
        return "calibration_undercovered"
    auc = signal["auc"]
    p = signal["p"]
    if auc >= 0.75 and p <= 0.05:
        return "strong"
    if auc >= 0.65 and p <= 0.05:
        return "weak_real"
    if 0.45 <= auc <= 0.55:
        return "chance"
    return "inconclusive"


def build_summary(
    detail_rows: list[dict],
    calibration_values: list[int],
    cfg: GeneratorConfig,
    components: int,
    reps: int,
    splits: int,
    min_test_class_count: int,
) -> list[dict]:
    rows = []
    for cal_size in calibration_values:
        row = {
            "calibration_reads": int(cal_size),
            "subjects": cfg.n_subjects,
            "trials": cfg.n_trials,
            "channels": cfg.n_channels,
            "components": components,
            "gestalt_effect": cfg.gestalt_effect,
            "violation_rate": cfg.violation_rate,
            "response_effect": cfg.response_effect,
            "reps": reps,
            "splits": 1 if cal_size == 0 else splits,
            "min_test_class_count": min_test_class_count,
        }
        for estimator in ESTIMATORS:
            signal = aggregate(detail_rows, "signal", cal_size, estimator, seed=cfg.seed + cal_size)
            null = aggregate(detail_rows, "N1_null", cal_size, estimator, seed=cfg.seed + cal_size + 101)
            prefix = estimator
            row[f"{prefix}_usable_fraction"] = signal["usable_fraction"]
            row[f"{prefix}_usable_subject_repeats"] = signal["usable_subject_repeats"]
            row[f"{prefix}_subjects_valid"] = signal["subjects_valid"]
            row[f"{prefix}_subjects_in_direction"] = signal["subjects_in_direction"]
            row[f"{prefix}_mean_cal_pos"] = signal["mean_cal_pos"]
            row[f"{prefix}_mean_cal_neg"] = signal["mean_cal_neg"]
            row[f"{prefix}_mean_test_pos"] = signal["mean_test_pos"]
            row[f"{prefix}_mean_test_neg"] = signal["mean_test_neg"]
            row[f"{prefix}_mean_lambda_pos"] = signal["mean_lambda_pos"]
            row[f"{prefix}_mean_lambda_neg"] = signal["mean_lambda_neg"]
            row[f"{prefix}_mean_k_pos"] = signal["mean_k_pos"]
            row[f"{prefix}_mean_k_neg"] = signal["mean_k_neg"]
            row[f"{prefix}_d"] = signal["d"]
            row[f"{prefix}_auc"] = signal["auc"]
            row[f"{prefix}_p"] = signal["p"]
            row[f"{prefix}_null_usable_fraction"] = null["usable_fraction"]
            row[f"{prefix}_null_auc"] = null["auc"]
            row[f"{prefix}_null_d"] = null["d"]
            row[f"{prefix}_null_p"] = null["p"]
            row[f"{prefix}_verdict"] = verdict(signal, null)
        rows.append(row)
    return rows


def write_outputs(out_dir: Path, detail_rows: list[dict], summary_rows: list[dict], payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stage1d_shrinkage_trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    with (out_dir / "stage1d_shrinkage_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (out_dir / "stage1d_shrinkage.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="03_In_Silico_Instrument_And_Power/Radiology_Gestalt_Validation/outputs/stage1d_shrinkage")
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--channels", type=int, default=28)
    ap.add_argument("--components", type=int, default=24)
    ap.add_argument("--gestalt-effect", type=float, default=0.07)
    ap.add_argument("--violation-rate", type=float, default=0.10)
    ap.add_argument("--response-effect", type=float, default=0.24)
    ap.add_argument("--calibration-values", default=",".join(str(v) for v in DEFAULT_CALIBRATION_VALUES))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--splits", type=int, default=25)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-test-class-count", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    calibration_values = [0, 20, 70] if args.quick else parse_int_list(args.calibration_values)
    reps = 1 if args.quick else args.reps
    splits = min(args.splits, 4) if args.quick else args.splits
    cfg = GeneratorConfig(
        n_subjects=args.subjects,
        n_trials=args.trials,
        n_channels=args.channels,
        gestalt_effect=args.gestalt_effect,
        violation_rate=args.violation_rate,
        response_effect=args.response_effect,
    )
    jobs = [
        (mode, rep, cfg, args.components, calibration_values, splits, args.min_test_class_count)
        for mode in ("signal", "N1_null")
        for rep in range(reps)
    ]

    detail_rows: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_job, *job) for job in jobs]
            for fut in as_completed(futs):
                rows = fut.result()
                detail_rows.extend(rows)
                print(json.dumps({"job_rows": len(rows), "total_rows": len(detail_rows)}), flush=True)
    else:
        for job in jobs:
            rows = run_job(*job)
            detail_rows.extend(rows)
            print(json.dumps({"job": {"mode": job[0], "rep": job[1]}, "rows": len(rows)}), flush=True)

    summary_rows = build_summary(
        detail_rows=detail_rows,
        calibration_values=calibration_values,
        cfg=cfg,
        components=args.components,
        reps=reps,
        splits=splits,
        min_test_class_count=args.min_test_class_count,
    )
    payload = {
        "analysis": "Stage 1d shrinkage: cohort backbone plus subject calibration",
        "config": cfg.__dict__ | {
            "components": args.components,
            "reps": reps,
            "splits": splits,
            "calibration_values": calibration_values,
            "min_test_class_count": args.min_test_class_count,
        },
        "summary": summary_rows,
    }
    write_outputs(Path(args.out_dir), detail_rows, summary_rows, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
