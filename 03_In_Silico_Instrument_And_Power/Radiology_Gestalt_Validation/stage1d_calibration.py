#!/usr/bin/env python3
"""Stage 1d: calibration-constrained subject-reference sweep.

Stage 1c tests whether a cohort-shared geometry can recover the planted
gestalt contrast. This script asks the stricter operational question: how many
labelled reads from the same held subject are needed before that subject has a
usable intact/violated reference of their own?
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


CALIBRATION_VALUES = [0, 5, 10, 20, 40, 70, 100, 140]
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


def metric_row(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(y, scores))
    d = cohen_d(scores[y == 1], scores[y == 0])
    return auc, d


def group_loso_scores(x: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    subjects = np.asarray(subjects, dtype=int)
    scores = np.full(len(y), np.nan, dtype=float)
    for held in np.unique(subjects):
        train = subjects != held
        test = subjects == held
        if len(np.unique(y[train])) < 2:
            continue
        scaler = StandardScaler().fit(x[train])
        xt = scaler.transform(x[train])
        xv = scaler.transform(x[test])
        c0 = xt[y[train] == 0].mean(axis=0)
        c1 = xt[y[train] == 1].mean(axis=0)
        d0 = np.linalg.norm(xv - c0, axis=1)
        d1 = np.linalg.norm(xv - c1, axis=1)
        scores[test] = d0 - d1
    return scores


def baseline_rows(x: np.ndarray, y: np.ndarray, subjects: np.ndarray, mode: str, rep: int) -> list[dict]:
    scores = group_loso_scores(x, y, subjects)
    rows = []
    for s in np.unique(subjects):
        idx = (subjects == s) & np.isfinite(scores)
        y_s = y[idx]
        pos = int(np.sum(y_s == 1))
        neg = int(np.sum(y_s == 0))
        usable = pos >= 2 and neg >= 2
        auc, d = metric_row(scores[idx], y_s) if usable else (float("nan"), float("nan"))
        rows.append({
            "mode": mode,
            "rep": rep,
            "subject": int(s),
            "calibration_trials": 0,
            "split": 0,
            "selection": "group_loso",
            "usable": bool(usable),
            "reason": "ok" if usable else "test_missing_class",
            "cal_pos": 0,
            "cal_neg": 0,
            "test_pos": pos,
            "test_neg": neg,
            "n_test": int(idx.sum()),
            "auc": auc,
            "per_subject_d": d,
        })
    return rows


def subject_calibration_rows(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    mode: str,
    rep: int,
    calibration_trials: int,
    splits: int,
    min_class_count: int,
    min_test_class_count: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for s in np.unique(subjects):
        sub_idx = np.flatnonzero(subjects == s)
        y_sub = y[sub_idx]
        for split in range(splits):
            base = {
                "mode": mode,
                "rep": rep,
                "subject": int(s),
                "calibration_trials": int(calibration_trials),
                "split": int(split),
                "selection": "natural_subject_calibration",
            }
            if calibration_trials >= len(sub_idx):
                rows.append(base | {
                    "usable": False,
                    "reason": "calibration_exhausts_subject",
                    "cal_pos": int(np.sum(y_sub == 1)),
                    "cal_neg": int(np.sum(y_sub == 0)),
                    "test_pos": 0,
                    "test_neg": 0,
                    "n_test": 0,
                    "auc": float("nan"),
                    "per_subject_d": float("nan"),
                })
                continue

            cal_local = rng.choice(np.arange(len(sub_idx)), size=calibration_trials, replace=False)
            cal_mask = np.zeros(len(sub_idx), dtype=bool)
            cal_mask[cal_local] = True
            cal_idx = sub_idx[cal_mask]
            test_idx = sub_idx[~cal_mask]
            y_cal = y[cal_idx]
            y_test = y[test_idx]
            cal_pos = int(np.sum(y_cal == 1))
            cal_neg = int(np.sum(y_cal == 0))
            test_pos = int(np.sum(y_test == 1))
            test_neg = int(np.sum(y_test == 0))

            if cal_pos < min_class_count or cal_neg < min_class_count:
                reason = "calibration_missing_class"
            elif test_pos < min_test_class_count or test_neg < min_test_class_count:
                reason = "test_missing_class"
            else:
                reason = "ok"

            if reason != "ok":
                rows.append(base | {
                    "usable": False,
                    "reason": reason,
                    "cal_pos": cal_pos,
                    "cal_neg": cal_neg,
                    "test_pos": test_pos,
                    "test_neg": test_neg,
                    "n_test": int(len(test_idx)),
                    "auc": float("nan"),
                    "per_subject_d": float("nan"),
                })
                continue

            scaler = StandardScaler().fit(x[cal_idx])
            xc = scaler.transform(x[cal_idx])
            xt = scaler.transform(x[test_idx])
            c0 = xc[y_cal == 0].mean(axis=0)
            c1 = xc[y_cal == 1].mean(axis=0)
            d0 = np.linalg.norm(xt - c0, axis=1)
            d1 = np.linalg.norm(xt - c1, axis=1)
            scores = d0 - d1
            auc, d = metric_row(scores, y_test)
            rows.append(base | {
                "usable": True,
                "reason": "ok",
                "cal_pos": cal_pos,
                "cal_neg": cal_neg,
                "test_pos": test_pos,
                "test_neg": test_neg,
                "n_test": int(len(test_idx)),
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
    min_class_count: int,
    min_test_class_count: int,
) -> list[dict]:
    cfg = replace(cfg, seed=cfg.seed + rep * 100003)
    ds = simulate(mode, cfg)
    labels = ds["labels"]
    subjects = labels["subject"].astype(int)
    y = labels["violation"].astype(int)
    x = riemannian_ea_features_fast(ds["covariances"], subjects, n_components=components)

    rows: list[dict] = []
    for n in calibration_values:
        if n == 0:
            rows.extend(baseline_rows(x, y, subjects, mode, rep))
        else:
            seed = cfg.seed + 13007 * rep + 997 * n + 53 * (0 if mode == "signal" else 1)
            rows.extend(subject_calibration_rows(
                x=x,
                y=y,
                subjects=subjects,
                mode=mode,
                rep=rep,
                calibration_trials=n,
                splits=splits,
                min_class_count=min_class_count,
                min_test_class_count=min_test_class_count,
                seed=seed,
            ))
    return rows


def aggregate_mode(rows: list[dict], mode: str, calibration_trials: int, seed: int) -> dict:
    subset = [r for r in rows if r["mode"] == mode and int(r["calibration_trials"]) == calibration_trials]
    usable = [r for r in subset if r["usable"] and np.isfinite(r["auc"]) and np.isfinite(r["per_subject_d"])]
    total = len(subset)

    def mean(name: str, source: list[dict]) -> float:
        vals = np.asarray([float(r[name]) for r in source if np.isfinite(float(r[name]))], dtype=float)
        return float(vals.mean()) if len(vals) else float("nan")

    by_subject: dict[int, list[float]] = {}
    for r in usable:
        by_subject.setdefault(int(r["subject"]), []).append(float(r["per_subject_d"]))
    subject_d = [float(np.mean(v)) for v in by_subject.values() if len(v)]
    return {
        "usable_subject_repeats": int(len(usable)),
        "total_subject_repeats": int(total),
        "usable_fraction": float(len(usable) / total) if total else float("nan"),
        "subjects_valid": int(len(by_subject)),
        "subjects_in_direction": int(np.sum(np.asarray(subject_d) > 0)) if subject_d else 0,
        "gestalt_auc": mean("auc", usable),
        "per_subject_d": float(np.mean(subject_d)) if subject_d else float("nan"),
        "cohort_perm_p": signflip_p(subject_d, seed=seed),
        "mean_cal_pos": mean("cal_pos", usable),
        "mean_cal_neg": mean("cal_neg", usable),
        "mean_test_pos": mean("test_pos", usable),
        "mean_test_neg": mean("test_neg", usable),
    }


def verdict(signal: dict, null: dict) -> str:
    if np.isfinite(null["gestalt_auc"]) and abs(null["gestalt_auc"] - 0.5) > 0.10:
        return "leak_check_failed"
    if signal["usable_fraction"] < 0.80:
        return "calibration_undercovered"
    auc = signal["gestalt_auc"]
    p = signal["cohort_perm_p"]
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
    min_class_count: int,
    min_test_class_count: int,
) -> list[dict]:
    rows = []
    for n in calibration_values:
        signal = aggregate_mode(detail_rows, "signal", n, seed=cfg.seed + n)
        null = aggregate_mode(detail_rows, "N1_null", n, seed=cfg.seed + n + 177)
        row = {
            "calibration_trials": n,
            "selection": "group_loso" if n == 0 else "natural_subject_calibration",
            "subjects": cfg.n_subjects,
            "trials": cfg.n_trials,
            "channels": cfg.n_channels,
            "components": components,
            "gestalt_effect": cfg.gestalt_effect,
            "violation_rate": cfg.violation_rate,
            "response_effect": cfg.response_effect,
            "reps": reps,
            "splits": 1 if n == 0 else splits,
            "min_class_count": min_class_count,
            "min_test_class_count": min_test_class_count,
            "usable_fraction": signal["usable_fraction"],
            "usable_subject_repeats": signal["usable_subject_repeats"],
            "total_subject_repeats": signal["total_subject_repeats"],
            "subjects_valid": signal["subjects_valid"],
            "subjects_in_direction": signal["subjects_in_direction"],
            "mean_cal_pos": signal["mean_cal_pos"],
            "mean_cal_neg": signal["mean_cal_neg"],
            "mean_test_pos": signal["mean_test_pos"],
            "mean_test_neg": signal["mean_test_neg"],
            "per_subject_d": signal["per_subject_d"],
            "gestalt_auc": signal["gestalt_auc"],
            "cohort_perm_p": signal["cohort_perm_p"],
            "null_usable_fraction": null["usable_fraction"],
            "null_auc": null["gestalt_auc"],
            "null_d": null["per_subject_d"],
            "null_p": null["cohort_perm_p"],
        }
        row["verdict"] = verdict(signal, null)
        rows.append(row)
    return rows


def write_outputs(out_dir: Path, detail_rows: list[dict], summary_rows: list[dict], payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_fields = list(detail_rows[0]) if detail_rows else []
    with (out_dir / "stage1d_calibration_trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_fields = list(summary_rows[0]) if summary_rows else []
    with (out_dir / "stage1d_calibration_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    (out_dir / "stage1d_calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="03_In_Silico_Instrument_And_Power/Radiology_Gestalt_Validation/outputs/stage1d")
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--channels", type=int, default=28)
    ap.add_argument("--components", type=int, default=24)
    ap.add_argument("--gestalt-effect", type=float, default=0.07)
    ap.add_argument("--violation-rate", type=float, default=0.10)
    ap.add_argument("--response-effect", type=float, default=0.24)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-class-count", type=int, default=2)
    ap.add_argument("--min-test-class-count", type=int, default=2)
    ap.add_argument("--calibration-values", default=",".join(str(v) for v in CALIBRATION_VALUES))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    calibration_values = [0, 5, 10, 20, 40] if args.quick else parse_int_list(args.calibration_values)
    splits = min(args.splits, 10) if args.quick else args.splits
    reps = 1 if args.quick else args.reps
    cfg = GeneratorConfig(
        n_subjects=args.subjects,
        n_trials=args.trials,
        n_channels=args.channels,
        gestalt_effect=args.gestalt_effect,
        violation_rate=args.violation_rate,
        response_effect=args.response_effect,
    )

    jobs = [
        (mode, rep, cfg, args.components, calibration_values, splits, args.min_class_count, args.min_test_class_count)
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
                print(json.dumps({"job_rows": len(rows), "done_jobs": len(detail_rows)}), flush=True)
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
        min_class_count=args.min_class_count,
        min_test_class_count=args.min_test_class_count,
    )
    payload = {
        "analysis": "Stage 1d natural per-subject calibration-constrained reference sweep",
        "config": cfg.__dict__ | {
            "components": args.components,
            "reps": reps,
            "splits": splits,
            "calibration_values": calibration_values,
            "min_class_count": args.min_class_count,
            "min_test_class_count": args.min_test_class_count,
        },
        "summary": summary_rows,
    }
    write_outputs(Path(args.out_dir), detail_rows, summary_rows, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
