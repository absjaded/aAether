#!/usr/bin/env python3
"""Stage 1c sweep: run the in-silico extractor down to failure."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import argparse
import csv
import json
import math
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from synthetic_radiology import GeneratorConfig, simulate
from stage1b_extract import riemannian_ea_features


EFFECT_VALUES = [0.10, 0.07, 0.05, 0.035, 0.025, 0.018, 0.012, 0.008, 0.005, 0.0]
TRIAL_VALUES = [40, 70, 100, 140, 200, 300, 500]
SUBJECT_VALUES = [5, 8, 12, 20, 30, 50]
VIOLATION_VALUES = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
JOINT_TRIALS = [70, 140, 300]
JOINT_SUBJECTS = [8, 20, 50]


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    denom = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    if denom <= 1e-12:
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


def centroid_gap_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-subject-out centroid distance gap.

    Positive score means closer to the violated centroid than the intact centroid.
    """
    y = np.asarray(y, int)
    groups = np.asarray(groups, int)
    scores = np.full(len(y), np.nan, dtype=float)
    for held in np.unique(groups):
        train = groups != held
        test = groups == held
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


def analyze_once(cfg: GeneratorConfig, mode: str, components: int, rep: int) -> dict:
    cfg = replace(cfg, seed=cfg.seed + rep * 100003)
    ds = simulate(mode, cfg)
    covs = ds["covariances"]
    labels = ds["labels"]
    subjects = labels["subject"].astype(int)
    y = labels["violation"].astype(int)
    x = riemannian_ea_features(covs, subjects, n_components=components)
    scores = centroid_gap_scores(x, y, subjects)
    ok = np.isfinite(scores)
    auc = float(roc_auc_score(y[ok], scores[ok])) if len(np.unique(y[ok])) == 2 else float("nan")

    per_subject = []
    for s in np.unique(subjects):
        idx = (subjects == s) & ok
        if idx.sum() == 0 or len(np.unique(y[idx])) < 2:
            continue
        d = cohen_d(scores[idx & (y == 1)], scores[idx & (y == 0)])
        if np.isfinite(d):
            per_subject.append(d)

    return {
        "mode": mode,
        "rep": rep,
        "auc": auc,
        "per_subject_d": float(np.mean(per_subject)) if per_subject else float("nan"),
        "subjects_in_direction": int(np.sum(np.asarray(per_subject) > 0)) if per_subject else 0,
        "subjects_valid": int(len(per_subject)),
        "cohort_perm_p": signflip_p(per_subject, seed=cfg.seed + 31),
        "observed_violation_rate": float(y.mean()),
    }


def aggregate(results: list[dict]) -> dict:
    def mean(name: str) -> float:
        vals = np.asarray([r[name] for r in results if np.isfinite(r[name])], dtype=float)
        return float(vals.mean()) if len(vals) else float("nan")

    return {
        "gestalt_auc": mean("auc"),
        "per_subject_d": mean("per_subject_d"),
        "cohort_perm_p": mean("cohort_perm_p"),
        "subjects_in_direction": mean("subjects_in_direction"),
        "subjects_valid": mean("subjects_valid"),
        "observed_violation_rate": mean("observed_violation_rate"),
    }


def verdict(signal: dict, null: dict, zero_check: dict | None = None) -> str:
    if np.isfinite(null["gestalt_auc"]) and abs(null["gestalt_auc"] - 0.5) > 0.10:
        return "leak_check_failed"
    if zero_check and np.isfinite(zero_check["gestalt_auc"]) and abs(zero_check["gestalt_auc"] - 0.5) > 0.10:
        return "zero_effect_failed"
    auc = signal["gestalt_auc"]
    p = signal["cohort_perm_p"]
    if auc >= 0.75 and p <= 0.05:
        return "strong"
    if auc >= 0.65 and p <= 0.05:
        return "weak_real"
    if 0.45 <= auc <= 0.55:
        return "chance"
    return "inconclusive"


def run_point(axis: str, value: float, cfg: GeneratorConfig, components: int, reps: int) -> dict:
    signal_results = [analyze_once(cfg, "signal", components, rep) for rep in range(reps)]
    null_cfg = replace(cfg, gestalt_effect=0.0)
    null_results = [analyze_once(null_cfg, "N1_null", components, rep) for rep in range(reps)]
    signal = aggregate(signal_results)
    null = aggregate(null_results)
    row = {
        "axis": axis,
        "value": value,
        "subjects": cfg.n_subjects,
        "trials": cfg.n_trials,
        "channels": cfg.n_channels,
        "components": components,
        "gestalt_effect": cfg.gestalt_effect,
        "violation_rate": cfg.violation_rate,
        "response_effect": cfg.response_effect,
        "reps": reps,
        "per_subject_d": signal["per_subject_d"],
        "gestalt_auc": signal["gestalt_auc"],
        "cohort_perm_p": signal["cohort_perm_p"],
        "subjects_in_direction": signal["subjects_in_direction"],
        "subjects_valid": signal["subjects_valid"],
        "observed_violation_rate": signal["observed_violation_rate"],
        "null_auc": null["gestalt_auc"],
        "null_d": null["per_subject_d"],
        "null_p": null["cohort_perm_p"],
    }
    row["verdict"] = verdict(signal, null)
    return row


def nearest_effect_for_auc(rows: list[dict], target: float = 0.68) -> float:
    candidates = [r for r in rows if r["axis"] == "effect" and np.isfinite(r["gestalt_auc"])]
    if not candidates:
        return 0.025
    return float(min(candidates, key=lambda r: abs(r["gestalt_auc"] - target))["gestalt_effect"])


def build_jobs(anchor: GeneratorConfig, components: int, reps: int, quick: bool) -> list[tuple[str, object, GeneratorConfig, int, int]]:
    effect_values = EFFECT_VALUES[:5] + [0.0] if quick else EFFECT_VALUES
    trial_values = [70, 140] if quick else TRIAL_VALUES
    subject_values = [8, 20] if quick else SUBJECT_VALUES
    violation_values = [0.05, 0.10, 0.20] if quick else VIOLATION_VALUES
    jobs = []
    for effect in effect_values:
        cfg = replace(anchor, gestalt_effect=effect)
        jobs.append(("effect", effect, cfg, components, reps))
    # Placeholder effect for non-effect axes; full main replaces it after effect rows if run sequentially.
    for trials in trial_values:
        cfg = replace(anchor, n_trials=trials)
        jobs.append(("trials", trials, cfg, components, reps))
    for subjects in subject_values:
        cfg = replace(anchor, n_subjects=subjects)
        jobs.append(("subjects", subjects, cfg, components, reps))
    for vr in violation_values:
        cfg = replace(anchor, n_trials=200, violation_rate=vr)
        jobs.append(("violation_rate", vr, cfg, components, reps))
    joint_trials = [70, 140] if quick else JOINT_TRIALS
    joint_subjects = [8, 20] if quick else JOINT_SUBJECTS
    for trials in joint_trials:
        for subjects in joint_subjects:
            cfg = replace(anchor, n_trials=trials, n_subjects=subjects)
            jobs.append(("joint_trials_subjects", f"{trials}x{subjects}", cfg, components, reps))
    return jobs


def write_outputs(out_dir: Path, rows: list[dict], summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (out_dir / "stage1c_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "stage1c_sweep.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="03_In_Silico_Instrument_And_Power/Radiology_Gestalt_Validation/outputs/stage1c")
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--trials", type=int, default=140)
    ap.add_argument("--channels", type=int, default=28)
    ap.add_argument("--components", type=int, default=24)
    ap.add_argument("--violation-rate", type=float, default=0.10)
    ap.add_argument("--response-effect", type=float, default=0.24)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    anchor = GeneratorConfig(
        n_subjects=args.subjects,
        n_trials=args.trials,
        n_channels=args.channels,
        violation_rate=args.violation_rate,
        response_effect=args.response_effect,
        gestalt_effect=0.10,
    )

    # Run the effect axis first so the weak-real effect can drive later axes.
    effect_jobs = [("effect", effect, replace(anchor, gestalt_effect=effect), args.components, args.reps)
                   for effect in ((EFFECT_VALUES[:5] + [0.0]) if args.quick else EFFECT_VALUES)]
    rows: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_point, *job) for job in effect_jobs]
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                print(json.dumps(row), flush=True)
    else:
        for job in effect_jobs:
            row = run_point(*job)
            rows.append(row)
            print(json.dumps(row), flush=True)

    weak_effect = nearest_effect_for_auc(rows, target=0.68)
    axis_jobs = []
    for axis, value, cfg, components, reps in build_jobs(replace(anchor, gestalt_effect=weak_effect), args.components, args.reps, args.quick):
        if axis == "effect":
            continue
        axis_jobs.append((axis, value, cfg, components, reps))

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_point, *job) for job in axis_jobs]
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                print(json.dumps(row), flush=True)
    else:
        for job in axis_jobs:
            row = run_point(*job)
            rows.append(row)
            print(json.dumps(row), flush=True)

    rows.sort(key=lambda r: (r["axis"], r["value"]))
    summary = {
        "anchor": anchor.__dict__,
        "components": args.components,
        "reps": args.reps,
        "weak_effect_selected": weak_effect,
        "note": "Matched N1_null run is included in null_* columns for every row.",
    }
    write_outputs(Path(args.out_dir), rows, summary)


if __name__ == "__main__":
    main()