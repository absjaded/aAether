#!/usr/bin/env python3
"""Stage 3: simulation power curve for the cohort-backbone detector.

Stage 1c established the cohort-shared recovery curve. Stage 1d and the
shrinkage iteration showed that pure subject references do not rescue sparse
natural calibration at a 10% violation base rate. Stage 3 therefore sizes the
collection around the detector that actually worked: leave-one-subject-out
cohort backbone, with a matched no-signal null at every design cell.
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


DEFAULT_EFFECTS = [0.05, 0.07, 0.10]
DEFAULT_SUBJECTS = [8, 12, 20, 30, 50]
DEFAULT_TRIALS = [70, 140, 300]
DEFAULT_VIOLATIONS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
EPS = 1e-12


def parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


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


def centroid_gap_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-subject-out cohort centroid distance gap.

    Positive score means closer to the violated centroid than the intact centroid.
    """
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=int)
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
    cfg = replace(cfg, seed=cfg.seed + rep * 100003 + (0 if mode == "signal" else 1000))
    ds = simulate(mode, cfg)
    labels = ds["labels"]
    subjects = labels["subject"].astype(int)
    y = labels["violation"].astype(int)
    x = riemannian_ea_features_fast(ds["covariances"], subjects, n_components=components)
    scores = centroid_gap_scores(x, y, subjects)
    ok = np.isfinite(scores)
    auc = float(roc_auc_score(y[ok], scores[ok])) if len(np.unique(y[ok])) == 2 else float("nan")

    per_subject = []
    valid_subjects = 0
    for s in np.unique(subjects):
        idx = (subjects == s) & ok
        y_s = y[idx]
        if idx.sum() == 0 or len(np.unique(y_s)) < 2:
            continue
        valid_subjects += 1
        d = cohen_d(scores[idx & (y == 1)], scores[idx & (y == 0)])
        if np.isfinite(d):
            per_subject.append(d)
    p = signflip_p(per_subject, seed=cfg.seed + 31)
    weak_pass = bool(np.isfinite(auc) and auc >= 0.65 and np.isfinite(p) and p <= 0.05)
    strong_pass = bool(np.isfinite(auc) and auc >= 0.75 and np.isfinite(p) and p <= 0.05)
    clean_null = bool(not weak_pass and np.isfinite(auc) and abs(auc - 0.5) <= 0.10)
    return {
        "mode": mode,
        "rep": rep,
        "auc": auc,
        "per_subject_d": float(np.mean(per_subject)) if per_subject else float("nan"),
        "cohort_perm_p": p,
        "weak_pass": weak_pass,
        "strong_pass": strong_pass,
        "clean_null": clean_null,
        "subjects_valid": int(valid_subjects),
        "subjects_in_direction": int(np.sum(np.asarray(per_subject) > 0)) if per_subject else 0,
        "observed_violation_rate": float(y.mean()),
    }


def mean_metric(results: list[dict], name: str) -> float:
    vals = np.asarray([float(r[name]) for r in results if np.isfinite(float(r[name]))], dtype=float)
    return float(vals.mean()) if len(vals) else float("nan")


def summarize_cell(signal_results: list[dict], null_results: list[dict], cfg: GeneratorConfig, components: int, reps: int) -> dict:
    power_weak = float(np.mean([r["weak_pass"] for r in signal_results]))
    power_strong = float(np.mean([r["strong_pass"] for r in signal_results]))
    null_fp_weak = float(np.mean([r["weak_pass"] for r in null_results]))
    null_clean = float(np.mean([r["clean_null"] for r in null_results]))
    row = {
        "effect": cfg.gestalt_effect,
        "violation_rate": cfg.violation_rate,
        "subjects": cfg.n_subjects,
        "trials": cfg.n_trials,
        "total_reads": cfg.n_subjects * cfg.n_trials,
        "expected_violations_per_subject": cfg.n_trials * cfg.violation_rate,
        "channels": cfg.n_channels,
        "components": components,
        "reps": reps,
        "signal_power_weak": power_weak,
        "signal_power_strong": power_strong,
        "null_false_positive_weak": null_fp_weak,
        "null_clean_rate": null_clean,
        "mean_signal_auc": mean_metric(signal_results, "auc"),
        "mean_signal_d": mean_metric(signal_results, "per_subject_d"),
        "mean_signal_p": mean_metric(signal_results, "cohort_perm_p"),
        "mean_null_auc": mean_metric(null_results, "auc"),
        "mean_null_d": mean_metric(null_results, "per_subject_d"),
        "mean_subjects_valid": mean_metric(signal_results, "subjects_valid"),
        "mean_subjects_in_direction": mean_metric(signal_results, "subjects_in_direction"),
        "mean_observed_violation_rate": mean_metric(signal_results, "observed_violation_rate"),
    }
    if power_weak >= 0.80 and null_fp_weak <= 0.05 and null_clean >= 0.90:
        row["verdict"] = "stage3_powered"
    elif power_weak >= 0.60 and null_fp_weak <= 0.05:
        row["verdict"] = "borderline"
    elif power_weak < 0.20:
        row["verdict"] = "underpowered"
    else:
        row["verdict"] = "inconclusive"
    return row


def run_cell(cell: tuple[float, float, int, int], base_cfg: GeneratorConfig, components: int, reps: int) -> dict:
    effect, violation_rate, subjects, trials = cell
    cfg = replace(
        base_cfg,
        gestalt_effect=effect,
        violation_rate=violation_rate,
        n_subjects=subjects,
        n_trials=trials,
    )
    signal_results = [analyze_once(cfg, "signal", components, rep) for rep in range(reps)]
    null_cfg = replace(cfg, gestalt_effect=0.0)
    null_results = [analyze_once(null_cfg, "N1_null", components, rep) for rep in range(reps)]
    return summarize_cell(signal_results, null_results, cfg, components, reps)


def recommendations(rows: list[dict]) -> list[dict]:
    out = []
    grouped: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((float(row["effect"]), float(row["violation_rate"])), []).append(row)
    for (effect, violation_rate), group in sorted(grouped.items()):
        powered = [r for r in group if r["verdict"] == "stage3_powered"]
        if powered:
            best = sorted(powered, key=lambda r: (r["total_reads"], r["subjects"], r["trials"]))[0]
            out.append({
                "effect": effect,
                "violation_rate": violation_rate,
                "recommendation": "powered",
                "subjects": best["subjects"],
                "trials": best["trials"],
                "total_reads": best["total_reads"],
                "expected_violations_per_subject": best["expected_violations_per_subject"],
                "mean_signal_d": best["mean_signal_d"],
                "mean_signal_auc": best["mean_signal_auc"],
                "signal_power_weak": best["signal_power_weak"],
                "null_false_positive_weak": best["null_false_positive_weak"],
            })
        else:
            best = sorted(group, key=lambda r: (-r["signal_power_weak"], -r["mean_signal_auc"], r["total_reads"]))[0]
            out.append({
                "effect": effect,
                "violation_rate": violation_rate,
                "recommendation": "not_powered_in_grid",
                "subjects": best["subjects"],
                "trials": best["trials"],
                "total_reads": best["total_reads"],
                "expected_violations_per_subject": best["expected_violations_per_subject"],
                "mean_signal_d": best["mean_signal_d"],
                "mean_signal_auc": best["mean_signal_auc"],
                "signal_power_weak": best["signal_power_weak"],
                "null_false_positive_weak": best["null_false_positive_weak"],
            })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir: Path, rows: list[dict], recs: list[dict], payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "stage3_power_summary.csv", rows)
    write_csv(out_dir / "stage3_recommendations.csv", recs)
    (out_dir / "stage3_power_curve.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="03_In_Silico_Instrument_And_Power/Radiology_Gestalt_Validation/outputs/stage3_power_curve")
    ap.add_argument("--effect-values", default=",".join(str(v) for v in DEFAULT_EFFECTS))
    ap.add_argument("--violation-values", default=",".join(str(v) for v in DEFAULT_VIOLATIONS))
    ap.add_argument("--subjects-values", default=",".join(str(v) for v in DEFAULT_SUBJECTS))
    ap.add_argument("--trials-values", default=",".join(str(v) for v in DEFAULT_TRIALS))
    ap.add_argument("--channels", type=int, default=28)
    ap.add_argument("--components", type=int, default=24)
    ap.add_argument("--response-effect", type=float, default=0.24)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    effects = [0.07] if args.quick else parse_float_list(args.effect_values)
    violations = [0.08, 0.10] if args.quick else parse_float_list(args.violation_values)
    subjects_values = [8, 20] if args.quick else parse_int_list(args.subjects_values)
    trials_values = [70, 140] if args.quick else parse_int_list(args.trials_values)
    reps = 2 if args.quick else args.reps

    base_cfg = GeneratorConfig(
        n_subjects=20,
        n_trials=140,
        n_channels=args.channels,
        response_effect=args.response_effect,
    )
    cells = [
        (effect, violation, subjects, trials)
        for effect in effects
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

    rows.sort(key=lambda r: (r["effect"], r["violation_rate"], r["subjects"], r["trials"]))
    recs = recommendations(rows)
    payload = {
        "analysis": "Stage 3 simulation power curve for cohort-backbone detector",
        "config": {
            "effects": effects,
            "violation_rates": violations,
            "subjects_values": subjects_values,
            "trials_values": trials_values,
            "channels": args.channels,
            "components": args.components,
            "response_effect": args.response_effect,
            "reps": reps,
            "criterion": "stage3_powered if weak-real power >= 0.80, null false positive <= 0.05, null clean rate >= 0.90",
        },
        "recommendations": recs,
        "rows": rows,
    }
    write_outputs(Path(args.out_dir), rows, recs, payload)
    print(json.dumps({"recommendations": recs}, indent=2), flush=True)


if __name__ == "__main__":
    main()
