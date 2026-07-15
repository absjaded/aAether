#!/usr/bin/env python3
"""Blind Stage 1b extraction for the in-silico radiology generator."""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
from scipy.linalg import eigh, logm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from synthetic_radiology import GeneratorConfig, MODES, simulate


def _inv_sqrtm(c: np.ndarray) -> np.ndarray:
    w, v = eigh(c)
    w = np.clip(w, 1e-8, None)
    return (v * (1.0 / np.sqrt(w))) @ v.T


def _vec_upper(mats: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(mats.shape[1])
    return mats[:, iu[0], iu[1]]


def riemannian_ea_features(covs: np.ndarray, subjects: np.ndarray, n_components: int) -> np.ndarray:
    aligned = np.empty_like(covs, dtype=np.float64)
    for s in np.unique(subjects):
        idx = subjects == s
        mean_cov = covs[idx].mean(axis=0)
        whitener = _inv_sqrtm(mean_cov)
        aligned[idx] = whitener @ covs[idx] @ whitener.T

    logs = np.empty_like(aligned, dtype=np.float64)
    for i, c in enumerate(aligned):
        logs[i] = np.real(logm(c))
    x = _vec_upper(logs)
    d = min(n_components, x.shape[1], x.shape[0] - 2)
    return PCA(n_components=d, random_state=0).fit_transform(x)


def cv_binary_score(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "balanced_accuracy": float("nan")}

    prob = np.full(len(y), np.nan, dtype=float)
    pred = np.full(len(y), -1)
    cv = GroupKFold(n_splits=min(6, len(np.unique(groups))))
    for train, test in cv.split(x, y, groups):
        if len(np.unique(y[train])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
        )
        clf.fit(x[train], y[train])
        prob[test] = clf.predict_proba(x[test])[:, 1]
        pred[test] = clf.predict(x[test])

    ok = ~np.isnan(prob)
    return {
        "auc": float(roc_auc_score(y[ok], prob[ok])),
        "balanced_accuracy": float(balanced_accuracy_score(y[ok], pred[ok])),
    }


def response_score(x: np.ndarray, response: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    # Score response confound as low-vs-high image quality; the middle bin is excluded.
    keep = response != 1
    y = (response[keep] == 2).astype(int)
    return cv_binary_score(x[keep], y, groups[keep])


def analyze_mode(mode: str, cfg: GeneratorConfig, n_components: int) -> dict[str, float | str | int]:
    ds = simulate(mode, cfg)
    covs = ds["covariances"]
    labels = ds["labels"]
    subjects = labels["subject"].astype(int)
    x = riemannian_ea_features(covs, subjects, n_components=n_components)

    violation = labels["violation"].astype(int)
    response = labels["response_quality"].astype(int)
    response_drive = labels["response_drive"].astype(float)
    gestalt = cv_binary_score(x, violation, subjects)
    motor = response_score(x, response, subjects)
    response_drive_corr = float(np.corrcoef(violation, response_drive)[0, 1])
    response_quality_corr = float(np.corrcoef(violation, response)[0, 1])
    return {
        "mode": mode,
        "n_subjects": int(cfg.n_subjects),
        "n_trials_total": int(len(violation)),
        "violation_rate": float(violation.mean()),
        "gestalt_auc": gestalt["auc"],
        "gestalt_balanced_accuracy": gestalt["balanced_accuracy"],
        "response_auc": motor["auc"],
        "response_balanced_accuracy": motor["balanced_accuracy"],
        "response_drive_violation_corr": response_drive_corr,
        "response_quality_violation_corr": response_quality_corr,
    }


def verdict(row: dict[str, float | str | int]) -> str:
    mode = str(row["mode"])
    g = float(row["gestalt_auc"])
    r = float(row["response_auc"])
    if mode == "signal":
        return "pass" if g >= 0.62 else "fail"
    if mode == "N1_null":
        return "pass" if 0.40 <= g <= 0.60 and 0.40 <= r <= 0.60 else "fail"
    if mode == "N2_response_confound":
        return "pass" if 0.40 <= g <= 0.60 and r >= 0.62 else "fail"
    if mode == "N3_motor_null":
        return "pass" if g >= 0.62 and r >= 0.62 else "fail"
    if mode == "N4_entangled_response":
        return "pass" if 0.40 <= g <= 0.60 else "fail"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="03_In_Silico_Instrument_And_Power/Radiology_Gestalt_Validation/outputs/stage1b_results.json")
    ap.add_argument("--subjects", type=int, default=GeneratorConfig.n_subjects)
    ap.add_argument("--trials", type=int, default=GeneratorConfig.n_trials)
    ap.add_argument("--channels", type=int, default=GeneratorConfig.n_channels)
    ap.add_argument("--components", type=int, default=32)
    ap.add_argument("--gestalt-effect", type=float, default=GeneratorConfig.gestalt_effect)
    ap.add_argument("--response-effect", type=float, default=GeneratorConfig.response_effect)
    ap.add_argument("--violation-rate", type=float, default=GeneratorConfig.violation_rate)
    ap.add_argument("--response-gestalt-rho", type=float, default=GeneratorConfig.response_gestalt_rho)
    args = ap.parse_args()

    cfg = GeneratorConfig(
        n_subjects=args.subjects,
        n_trials=args.trials,
        n_channels=args.channels,
        gestalt_effect=args.gestalt_effect,
        response_effect=args.response_effect,
        violation_rate=args.violation_rate,
        response_gestalt_rho=args.response_gestalt_rho,
    )
    rows = []
    for mode in MODES:
        row = analyze_mode(mode, cfg, n_components=args.components)
        row["verdict"] = verdict(row)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "Stage 1b blind Riemannian-EA-log tangent extraction",
        "config": cfg.__dict__ | {"components": args.components},
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
