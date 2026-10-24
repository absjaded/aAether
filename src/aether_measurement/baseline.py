#!/usr/bin/env python3
"""Cheap-observable scaffold for validated Aether behavioral ledgers.

Adapted from public_aether_runtime/aether_cheap_baseline.py. Cross-validation
uses stratified trial folds; it does not hold out participants or sessions.
Permutation labels are shuffled globally rather than within participant/session.
The v0 CLI validates ledgers and includes matched rows only when usable_trial
and usable_covariates are true. This release boundary does not revise historical
scores. The cheap_auc helper scores every supplied row. Small samples report
apparent, in-sample AUC.
The utility target assumes equal-variance Gaussian score distributions. This is
behavioral software scaffolding, not evidence of a neural phenomenon.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Sequence, Tuple

from . import ledger

BOOLEAN_VALUES = {"true": True, "false": False, "1": True, "0": False}
CHEAP_FEATURES = [
    "confidence", "uncertainty", "response_latency_ms", "hold_duration_ms",
    "revision_pressure", "trial_time_fraction", "session_time_fraction",
    "fatigue_proxy", "difficulty_proxy",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="ascii", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return BOOLEAN_VALUES.get(str(value).strip().lower(), False)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except ValueError:
        return default


def join_ledgers(events: List[Dict[str, str]], covariates: List[Dict[str, str]], truth: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    cov_by_id = {row["trial_id"]: row for row in covariates}
    truth_by_id = {row["trial_id"]: row for row in truth}
    rows = []
    for event in events:
        trial = event["trial_id"]
        if trial not in cov_by_id or trial not in truth_by_id:
            continue
        joined: Dict[str, Any] = {}
        joined.update(event)
        joined.update(cov_by_id[trial])
        joined.update(truth_by_id[trial])
        rows.append(joined)
    return rows


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else None


def feature_matrix(rows: List[Dict[str, Any]]) -> List[List[float]]:
    return [[to_float(row.get(feature)) for feature in CHEAP_FEATURES] for row in rows]


def normalize(train_x: List[List[float]], test_x: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    if not train_x:
        return train_x, test_x
    means = []
    scales = []
    for index in range(len(train_x[0])):
        values = [row[index] for row in train_x]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        means.append(mean)
        scales.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
    norm_train = [[(row[i] - means[i]) / scales[i] for i in range(len(row))] for row in train_x]
    norm_test = [[(row[i] - means[i]) / scales[i] for i in range(len(row))] for row in test_x]
    return norm_train, norm_test


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1 / (1 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def fit_logistic(train_x: List[List[float]], train_y: List[int], steps: int = 300, lr: float = 0.05, l2: float = 0.1) -> List[float]:
    weights = [0.0] * ((len(train_x[0]) if train_x else 0) + 1)
    if not train_x:
        return weights
    for _ in range(steps):
        grads = [0.0] * len(weights)
        for x_row, y_value in zip(train_x, train_y):
            z_value = weights[0] + sum(weight * value for weight, value in zip(weights[1:], x_row))
            error = sigmoid(z_value) - y_value
            grads[0] += error
            for index, value in enumerate(x_row, start=1):
                grads[index] += error * value
        n_rows = len(train_x)
        weights[0] -= lr * grads[0] / n_rows
        for index in range(1, len(weights)):
            weights[index] -= lr * ((grads[index] / n_rows) + l2 * weights[index])
    return weights


def predict(weights: List[float], rows: List[List[float]]) -> List[float]:
    return [sigmoid(weights[0] + sum(weight * value for weight, value in zip(weights[1:], row))) for row in rows]


def stratified_folds(labels: List[int], k: int) -> List[List[int]]:
    positives = [index for index, label in enumerate(labels) if label == 1]
    negatives = [index for index, label in enumerate(labels) if label == 0]
    folds = [[] for _ in range(k)]
    for index, item in enumerate(positives):
        folds[index % k].append(item)
    for index, item in enumerate(negatives):
        folds[index % k].append(item)
    return [fold for fold in folds if fold]


def cheap_auc(rows: List[Dict[str, Any]], label_field: str) -> Tuple[float | None, str]:
    labels = [1 if to_bool(row.get(label_field)) else 0 for row in rows]
    matrix = feature_matrix(rows)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None, "undefined_single_class"
    if positives < 2 or negatives < 2 or len(labels) < 20:
        train_x, test_x = normalize(matrix, matrix)
        scores = predict(fit_logistic(train_x, labels), test_x)
        return auc_score(labels, scores), "apparent_small_sample"
    k = min(5, positives, negatives)
    scores = [0.0] * len(labels)
    for fold in stratified_folds(labels, k):
        test_indices = set(fold)
        train_x_raw = [row for index, row in enumerate(matrix) if index not in test_indices]
        train_y = [label for index, label in enumerate(labels) if index not in test_indices]
        # Preserve the same order when constructing and placing fold predictions.
        test_x_raw = [matrix[index] for index in fold]
        train_x, test_x = normalize(train_x_raw, test_x_raw)
        predictions = predict(fit_logistic(train_x, train_y), test_x)
        for index, score in zip(fold, predictions):
            scores[index] = score
    return auc_score(labels, scores), f"{k}_fold_cv"


def permutation_null(rows: List[Dict[str, Any]], label_field: str, n_perm: int, seed: int) -> Dict[str, Any]:
    observed, mode = cheap_auc(rows, label_field)
    labels = [1 if to_bool(row.get(label_field)) else 0 for row in rows]
    if observed is None or len(set(labels)) < 2:
        return {"n": 0, "mean_auc": None, "p_ge_observed": None, "mode": mode}
    rng = random.Random(seed)
    aucs = []
    copied = [dict(row) for row in rows]
    for _ in range(n_perm):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        for row, label in zip(copied, shuffled):
            row[label_field] = str(bool(label)).lower()
        auc, _ = cheap_auc(copied, label_field)
        if auc is not None:
            aucs.append(auc)
    p_value = (sum(1 for auc in aucs if auc >= observed) + 1) / (len(aucs) + 1) if aucs else None
    return {"n": len(aucs), "mean_auc": sum(aucs) / len(aucs) if aucs else None, "p_ge_observed": p_value, "mode": mode}


def utility_target(base_rate: float, flag_rate: float, target_lift: float) -> Dict[str, Any]:
    target_precision = target_lift * base_rate
    recall = target_precision * flag_rate / base_rate if base_rate else None
    if recall is None or recall >= 1 or recall <= 0 or base_rate <= 0 or base_rate >= 1 or flag_rate <= 0 or flag_rate >= 1:
        return {"target_auc": None, "target_precision": target_precision, "target_recall": recall, "false_positive_rate": None}
    false_positive_rate = (flag_rate - base_rate * recall) / (1 - base_rate)
    if false_positive_rate <= 0 or false_positive_rate >= 1:
        return {"target_auc": None, "target_precision": target_precision, "target_recall": recall, "false_positive_rate": false_positive_rate}
    normal = NormalDist()
    threshold = normal.inv_cdf(1 - false_positive_rate)
    signal_quantile = normal.inv_cdf(1 - recall)
    auc = normal.cdf((threshold - signal_quantile) / math.sqrt(2))
    return {"target_auc": auc, "target_precision": target_precision, "target_recall": recall, "false_positive_rate": false_positive_rate}


def nearest_rho(rho_map: Path | None, observed_auc: float | None) -> Dict[str, Any]:
    if rho_map is None or observed_auc is None:
        return {"cheap_state_rho": None, "status": "not_requested"}
    candidates = []
    for row in read_csv(rho_map):
        if "cheap_state_rho" in row and "cheap_auc" in row:
            candidates.append((to_float(row["cheap_state_rho"]), to_float(row["cheap_auc"])))
    if not candidates:
        return {"cheap_state_rho": None, "status": "rho_map_missing_columns"}
    best = min(candidates, key=lambda item: abs(item[1] - observed_auc))
    return {"cheap_state_rho": best[0], "matched_cheap_auc": best[1], "status": "nearest_match"}


def write_report(report: Dict[str, Any], args: argparse.Namespace) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="ascii")
    if args.markdown:
        lines = [
            "# Aether Cheap-Observable Baseline Report", "",
            "Behavioral scaffold only. No neural data and no claim about the phenomenon.", "",
            "Evaluation uses trial folds, without subject/session holdout.", "",
            f"- Decision: `{report['decision']}`",
        ]
        if "cheap_stack" in report:
            lines.extend([
                f"- Joined rows: `{report['joined_rows']}`",
                f"- Included usable rows: `{report['included_rows']}`",
                f"- Excluded unusable rows: `{report['excluded_rows']}`",
                f"- Primary residue base rate: `{report['primary_residue_base_rate']:.3f}`",
                f"- Cheap-only primary AUC: `{report['cheap_stack']['primary_auc']}`",
                f"- Cheap-only primary mode: `{report['cheap_stack']['primary_mode']}`",
                f"- Secondary skill-channel AUC: `{report['cheap_stack']['secondary_auc']}`",
                f"- Utility target AUC: `{report['utility_target'].get('target_auc')}`",
            ])
        errors = report.get("validation", {}).get("errors", [])
        if errors:
            lines.extend(["", "## Validation errors"])
            lines.extend(f"- {error}" for error in errors)
        if report.get("warnings"):
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in report["warnings"])
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(rendered)


def analyze(args: argparse.Namespace) -> int:
    try:
        raw_events = ledger.read_csv(args.event)
        raw_covariates = ledger.read_csv(args.covariate)
        raw_truth = ledger.read_csv(args.ground_truth)
        validation = ledger.validate_rows(raw_events, raw_covariates, raw_truth)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        validation = {"summary": None, "errors": [str(exc)], "warnings": [], "passed": False}
    if not validation["passed"]:
        write_report({
            "decision": "fail_invalid_ledgers",
            "validation": validation,
            "warnings": validation["warnings"],
            "claim_boundary": "Behavioral scaffold only. No neural data and no claim about the phenomenon.",
        }, args)
        return 1
    # Use the same normalized identifiers/values that passed validation.
    joined_rows = join_ledgers(
        [ledger.typed_row(row) for row in raw_events],
        [ledger.typed_row(row) for row in raw_covariates],
        [ledger.typed_row(row) for row in raw_truth],
    )
    rows = [row for row in joined_rows if to_bool(row.get("usable_trial")) and to_bool(row.get("usable_covariates"))]
    primary_count = sum(1 for row in rows if to_bool(row.get("primary_residue_label")))
    secondary_count = sum(1 for row in rows if to_bool(row.get("secondary_skill_label")))
    divergence_count = sum(1 for row in rows if to_bool(row.get("divergence_label")))
    base_rate = primary_count / len(rows) if rows else 0.0
    primary_auc, primary_mode = cheap_auc(rows, "primary_residue_label")
    secondary_auc, secondary_mode = cheap_auc(rows, "secondary_skill_label")
    decision = "ready_for_review"
    warnings = list(validation["warnings"])
    if not joined_rows:
        decision = "fail_no_joined_rows"
    elif not rows:
        decision = "fail_no_usable_rows"
    elif primary_auc is None:
        decision = "rerun_primary_label_single_class"
    elif primary_mode == "apparent_small_sample":
        decision = "insufficient_sample_for_decision"
        warnings.append("apparent small-sample AUC; do not use for proceed/send-back decisions")
    elif primary_auc > args.cheap_auc_limit:
        decision = "send_back_cheap_observables_explain_label"
    if rows and not 0.05 <= base_rate <= 0.15:
        warnings.append("primary residue base rate outside 0.05-0.15 target band")
    report = {
        "validation": validation,
        "evaluation_scope": "stratified trial folds; no subject/session holdout; global label permutations",
        "joined_rows": len(joined_rows),
        "included_rows": len(rows),
        "excluded_rows": len(joined_rows) - len(rows),
        "exclusion_policy": "usable_trial and usable_covariates must both be true",
        "primary_residue_count": primary_count,
        "secondary_skill_count": secondary_count,
        "divergence_count": divergence_count,
        "primary_residue_base_rate": base_rate,
        "cheap_features": CHEAP_FEATURES,
        "cheap_stack": {
            "primary_auc": primary_auc,
            "primary_mode": primary_mode,
            "primary_permutation_null": permutation_null(rows, "primary_residue_label", args.permutations, args.seed),
            "secondary_auc": secondary_auc,
            "secondary_mode": secondary_mode,
            "secondary_permutation_null": permutation_null(rows, "secondary_skill_label", args.permutations, args.seed + 1),
        },
        "utility_target": utility_target(base_rate, args.flag_rate, args.target_lift),
        "cheap_state_rho_inversion": nearest_rho(args.rho_map, primary_auc),
        "decision": decision,
        "warnings": warnings,
        "claim_boundary": "Behavioral scaffold only. No neural data and no claim about the phenomenon.",
    }
    write_report(report, args)
    return 0 if rows else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate cheap-observable baseline for Aether ledgers")
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--covariate", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--flag-rate", type=float, default=0.20)
    parser.add_argument("--target-lift", type=float, default=2.0)
    parser.add_argument("--cheap-auc-limit", type=float, default=0.75)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--rho-map", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    if args.permutations < 0:
        parser.error("--permutations must be nonnegative")
    if not 0 < args.flag_rate < 1:
        parser.error("--flag-rate must be strictly between 0 and 1")
    if not math.isfinite(args.target_lift) or args.target_lift <= 0:
        parser.error("--target-lift must be finite and positive")
    if not 0 <= args.cheap_auc_limit <= 1:
        parser.error("--cheap-auc-limit must be between 0 and 1")
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
