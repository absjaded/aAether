#!/usr/bin/env python3
"""Stage 1e: entangled response false-positive limits.

N4 plants no gestalt-violation geometry. It plants only the response axis, while
making the continuous response drive statistically correlated with the violation
label. Any cohort-backbone detection of violation in N4 is therefore a response
confound, not a gestalt result.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import argparse
import csv
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from synthetic_radiology import GeneratorConfig, simulate
from stage3_power_curve import (
    centroid_gap_scores,
    cohen_d,
    parse_float_list,
    parse_int_list,
    riemannian_ea_features_fast,
    signflip_p,
)


DEFAULT_RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8]
DEFAULT_SUBJECTS = [20, 30]
DEFAULT_TRIALS = [70, 140]
DEFAULT_VIOLATIONS = [0.08, 0.10]


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a = a[ok]
    b = b[ok]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze_once(cfg: GeneratorConfig, components: int, rep: int) -> dict:
    cfg = replace(cfg, seed=cfg.seed + rep * 100003 + int(round(cfg.response_gestalt_rho * 1000)) * 1009)
    ds = simulate("N4_entangled_response", cfg)
    labels = ds["labels"]
    subjects = labels["subject"].astype(int)
    y = labels["violation"].astype(int)
    response_drive = labels["response_drive"].astype(float)
    response_quality = labels["response_quality"].astype(float)

    x = riemannian_ea_features_fast(ds["covariances"], subjects, n_components=components)
    scores = centroid_gap_scores(x, y, subjects)
    ok = np.isfinite(scores)
    auc = float(roc_auc_score(y[ok], scores[ok])) if len(np.unique(y[ok])) == 2 else float("nan")

    per_subject = []
    valid_subjects = 0
    for subject in np.unique(subjects):
        idx = (subjects == subject) & ok
        y_s = y[idx]
        if idx.sum() == 0 or len(np.unique(y_s)) < 2:
            continue
        valid_subjects += 1
        d = cohen_d(scores[idx & (y == 1)], scores[idx & (y == 0)])
        if np.isfinite(d):
            per_subject.append(d)

    p = signflip_p(per_subject, seed=cfg.seed + 31)
    weak_fp = bool(np.isfinite(auc) and auc >= 0.65 and np.isfinite(p) and p <= 0.05)
    strong_fp = bool(np.isfinite(auc) and auc >= 0.75 and np.isfinite(p) and p <= 0.05)
    clean_null = bool(not weak_fp and np.isfinite(auc) and abs(auc - 0.5) <= 0.10)
    return {
        "rep": rep,
        "auc": auc,
        "per_subject_d": float(np.mean(per_subject)) if per_subject else float("nan"),
        "cohort_perm_p": p,
        "weak_false_positive": weak_fp,
        "strong_false_positive": strong_fp,
        "clean_null": clean_null,
        "subjects_valid": int(valid_subjects),
        "subjects_in_direction": int(np.sum(np.asarray(per_subject) > 0)) if per_subject else 0,
        "observed_violation_rate": float(y.mean()),
        "response_drive_violation_corr": finite_corr(response_drive, y),
        "response_quality_violation_corr": finite_corr(response_quality, y),
    }


def mean_metric(results: list[dict], name: str) -> float:
    vals = np.asarray([float(r[name]) for r in results if np.isfinite(float(r[name]))], dtype=float)
    return float(vals.mean()) if len(vals) else float("nan")


def summarize_cell(results: list[dict], cfg: GeneratorConfig, components: int, reps: int) -> dict:
    weak_fp_rate = float(np.mean([r["weak_false_positive"] for r in results]))
    strong_fp_rate = float(np.mean([r["strong_false_positive"] for r in results]))
    clean_rate = float(np.mean([r["clean_null"] for r in results]))
    mean_auc = mean_metric(results, "auc")

    if strong_fp_rate >= 0.80 or mean_auc >= 0.75:
        verdict = "severe_leak"
    elif weak_fp_rate >= 0.80 and mean_auc >= 0.65:
        verdict = "leaky"
    elif weak_fp_rate >= 0.20 or mean_auc >= 0.60:
        verdict = "borderline"
    else:
        verdict = "clean"

    return {
        "rho": cfg.response_gestalt_rho,
        "violation_rate": cfg.violation_rate,
        "subjects": cfg.n_subjects,
        "trials": cfg.n_trials,
        "total_reads": cfg.n_subjects * cfg.n_trials,
        "expected_violations_per_subject": cfg.n_trials * cfg.violation_rate,
        "channels": cfg.n_channels,
        "components": components,
        "response_effect": cfg.response_effect,
        "reps": reps,
        "n4_false_positive_weak": weak_fp_rate,
        "n4_false_positive_strong": strong_fp_rate,
        "n4_clean_rate": clean_rate,
        "mean_n4_auc": mean_auc,
        "mean_n4_d": mean_metric(results, "per_subject_d"),
        "mean_n4_p": mean_metric(results, "cohort_perm_p"),
        "mean_response_drive_violation_corr": mean_metric(results, "response_drive_violation_corr"),
        "mean_response_quality_violation_corr": mean_metric(results, "response_quality_violation_corr"),
        "mean_observed_violation_rate": mean_metric(results, "observed_violation_rate"),
        "mean_subjects_valid": mean_metric(results, "subjects_valid"),
        "mean_subjects_in_direction": mean_metric(results, "subjects_in_direction"),
        "verdict": verdict,
    }


def run_cell(cell: tuple[float, float, int, int], base_cfg: GeneratorConfig, components: int, reps: int) -> dict:
    rho, violation_rate, subjects, trials = cell
    cfg = replace(
        base_cfg,
        response_gestalt_rho=rho,
        violation_rate=violation_rate,
        n_subjects=subjects,
        n_trials=trials,
        gestalt_effect=0.0,
    )
    results = [analyze_once(cfg, components, rep) for rep in range(reps)]
    return summarize_cell(results, cfg, components, reps)


def limit_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, int, int], list[dict]] = {}
    for row in rows:
        key = (float(row["violation_rate"]), int(row["subjects"]), int(row["trials"]))
        grouped.setdefault(key, []).append(row)

    out = []
    for (violation_rate, subjects, trials), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda r: float(r["rho"]))
        clean = [r for r in ordered if r["verdict"] == "clean"]
        nonclean = [r for r in ordered if r["verdict"] != "clean"]
        leaky = [r for r in ordered if r["verdict"] in {"leaky", "severe_leak"}]
        severe = [r for r in ordered if r["verdict"] == "severe_leak"]
        out.append({
            "violation_rate": violation_rate,
            "subjects": subjects,
            "trials": trials,
            "total_reads": subjects * trials,
            "expected_violations_per_subject": trials * violation_rate,
            "max_clean_rho_tested": max((float(r["rho"]) for r in clean), default=float("nan")),
            "first_nonclean_rho_tested": float(nonclean[0]["rho"]) if nonclean else float("nan"),
            "first_leaky_rho_tested": float(leaky[0]["rho"]) if leaky else float("nan"),
            "first_severe_rho_tested": float(severe[0]["rho"]) if severe else float("nan"),
            "lowest_rho_verdict": ordered[0]["verdict"],
            "highest_rho_verdict": ordered[-1]["verdict"],
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="03_In_Silico_Instrument_And_Power/Radiology_Gestalt_Validation/outputs/stage1e_entanglement")
    ap.add_argument("--rho-values", default=",".join(str(v) for v in DEFAULT_RHOS))
    ap.add_argument("--violation-values", default=",".join(str(v) for v in DEFAULT_VIOLATIONS))
    ap.add_argument("--subjects-values", default=",".join(str(v) for v in DEFAULT_SUBJECTS))
    ap.add_argument("--trials-values", default=",".join(str(v) for v in DEFAULT_TRIALS))
    ap.add_argument("--channels", type=int, default=28)
    ap.add_argument("--components", type=int, default=24)
    ap.add_argument("--response-effect", type=float, default=0.24)
    ap.add_argument("--reps", type=int, default=36)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    rhos = [0.0, 0.3] if args.quick else parse_float_list(args.rho_values)
    violations = [0.10] if args.quick else parse_float_list(args.violation_values)
    subjects_values = [8] if args.quick else parse_int_list(args.subjects_values)
    trials_values = [70] if args.quick else parse_int_list(args.trials_values)
    reps = 2 if args.quick else args.reps

    base_cfg = GeneratorConfig(
        n_subjects=20,
        n_trials=140,
        n_channels=args.channels,
        response_effect=args.response_effect,
        gestalt_effect=0.0,
    )
    cells = [
        (rho, violation, subjects, trials)
        for rho in rhos
        for violation in violations
        for subjects in subjects_values
        for trials in trials_values
    ]

    rows: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_cell, cell, base_cfg, args.components, reps) for cell in cells]
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                print(json.dumps(row), flush=True)
    else:
        for cell in cells:
            row = run_cell(cell, base_cfg, args.components, reps)
            rows.append(row)
            print(json.dumps(row), flush=True)

    rows.sort(key=lambda r: (r["violation_rate"], r["subjects"], r["trials"], r["rho"]))
    limits = limit_rows(rows)
    payload = {
        "analysis": "Stage 1e N4 entangled-response false-positive limit sweep",
        "config": {
            "rho_values": rhos,
            "violation_rates": violations,
            "subjects_values": subjects_values,
            "trials_values": trials_values,
            "channels": args.channels,
            "components": args.components,
            "response_effect": args.response_effect,
            "reps": reps,
            "criterion": "N4 is a confound-only mode; weak false positive means AUC >= 0.65 and cohort permutation p <= 0.05.",
        },
        "limits": limits,
        "rows": rows,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "stage1e_entanglement_summary.csv", rows)
    write_csv(out_dir / "stage1e_entanglement_limits.csv", limits)
    (out_dir / "stage1e_entanglement.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"limits": limits}, indent=2), flush=True)


if __name__ == "__main__":
    main()