#!/usr/bin/env python3
"""Public Aether ledger validator.

The script validates three CSV ledgers: event timing, cheap covariates, and
trial-level ground truth. It is intentionally self-contained and uses only the
Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

EVENT_FIELDS = [
    "participant_id", "session_id", "block_id", "trial_id", "trial_index",
    "stimulus_id", "stimulus_family", "state_label", "report_label",
    "stimulus_onset_ms", "hold_start_ms", "hold_end_ms", "report_prompt_onset_ms",
    "response_onset_ms", "response_commit_ms", "confidence_prompt_onset_ms",
    "trial_end_ms", "run_order_index", "condition_order_index", "usable_trial",
    "quality_flag",
]
COVARIATE_FIELDS = [
    "participant_id", "session_id", "trial_id", "confidence", "uncertainty",
    "response_latency_ms", "hold_duration_ms", "revision_pressure",
    "gaze_available", "gaze_missing_fraction", "fixation_count", "saccade_count",
    "dwell_entropy", "trial_time_fraction", "session_time_fraction",
    "fatigue_proxy", "device_quality", "usable_covariates",
]
GROUND_TRUTH_FIELDS = [
    "trial_id", "stimulus_id", "participant_id", "session_id",
    "ground_truth_source", "reference_answer", "reported_answer",
    "reference_score_before", "reference_score_after_report", "loss_score",
    "divergence_label", "repeat_consistency_label", "primary_residue_label",
    "secondary_skill_label", "curation_source", "difficulty_proxy",
]

TIMING_CHAIN = [
    "stimulus_onset_ms", "hold_start_ms", "hold_end_ms", "report_prompt_onset_ms",
    "response_onset_ms", "response_commit_ms", "confidence_prompt_onset_ms",
    "trial_end_ms",
]
STATE_LABELS = {"stable_judgment", "unstable_judgment", "withheld_or_delayed_report", "control_neutral"}
QUALITY_FLAGS = {"ok", "missed_response", "timing_error", "interrupted", "invalid_covariates", "exclude"}
DEVICE_QUALITY = {"good", "usable", "poor", "exclude"}
REPEAT_LABELS = {
    "not_repeated", "diverged_inconsistent", "diverged_consistent",
    "agreement_inconsistent", "agreement_consistent",
}
BOOLEAN_VALUES = {"true": True, "false": False, "1": True, "0": False}

NUMERIC_FIELDS = {
    "trial_index", "stimulus_onset_ms", "hold_start_ms", "hold_end_ms",
    "report_prompt_onset_ms", "response_onset_ms", "response_commit_ms",
    "confidence_prompt_onset_ms", "trial_end_ms", "run_order_index",
    "condition_order_index", "confidence", "uncertainty", "response_latency_ms",
    "hold_duration_ms", "revision_pressure", "gaze_missing_fraction",
    "fixation_count", "saccade_count", "dwell_entropy", "trial_time_fraction",
    "session_time_fraction", "fatigue_proxy", "reference_score_before",
    "reference_score_after_report", "loss_score", "difficulty_proxy",
}
BOOLEAN_FIELDS = {"usable_trial", "gaze_available", "usable_covariates", "divergence_label", "primary_residue_label", "secondary_skill_label"}
RANGE_0_1 = {"confidence", "uncertainty", "revision_pressure", "gaze_missing_fraction", "trial_time_fraction", "session_time_fraction", "fatigue_proxy"}
NONNEGATIVE_FIELDS = NUMERIC_FIELDS - {"reference_score_before", "reference_score_after_report"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="ascii", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_template(path: Path, fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()


def parse_bool(value: Any) -> Tuple[bool | Any, bool]:
    lowered = str(value).strip().lower()
    if lowered in BOOLEAN_VALUES:
        return BOOLEAN_VALUES[lowered], True
    return value, False


def parse_float(value: Any) -> Tuple[float | Any, bool]:
    try:
        if value is None or str(value).strip() == "":
            return value, False
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return value, False
        return number, True
    except ValueError:
        return value, False


def typed_row(row: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if key in BOOLEAN_FIELDS:
            out[key], _ = parse_bool(value)
        elif key in NUMERIC_FIELDS:
            out[key], _ = parse_float(value)
        else:
            out[key] = value.strip() if value is not None else value
    return out


def duplicate_ids(rows: Iterable[Dict[str, Any]], field: str) -> List[str]:
    seen = set()
    dupes = set()
    for row in rows:
        value = row.get(field)
        if value in seen:
            dupes.add(str(value))
        seen.add(value)
    return sorted(dupes)


def validate_fields(rows: List[Dict[str, str]], required: List[str], name: str) -> List[str]:
    errors: List[str] = []
    for line, row in enumerate(rows, start=2):
        keys = set(row)
        missing = [field for field in required if field not in keys or row.get(field, "") == ""]
        extra = sorted(keys - set(required))
        for field in missing:
            errors.append(f"{name}:csv_line_{line}: missing {field}")
        for field in extra:
            errors.append(f"{name}:csv_line_{line}: unexpected {field}")
        for field in NUMERIC_FIELDS & keys:
            parsed, ok = parse_float(row[field])
            if not ok:
                errors.append(f"{name}:csv_line_{line}: {field} is not numeric")
                continue
            if field in RANGE_0_1 and not 0 <= parsed <= 1:
                errors.append(f"{name}:csv_line_{line}: {field} outside 0..1")
            if field in NONNEGATIVE_FIELDS and parsed < 0:
                errors.append(f"{name}:csv_line_{line}: {field} below zero")
        for field in BOOLEAN_FIELDS & keys:
            _, ok = parse_bool(row[field])
            if not ok:
                errors.append(f"{name}:csv_line_{line}: {field} is not boolean")
    return errors


def validate_enums(events: List[Dict[str, Any]], covariates: List[Dict[str, Any]], truth: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for row in events:
        trial = row.get("trial_id", "<missing>")
        if row.get("state_label") not in STATE_LABELS:
            errors.append(f"trial {trial}: state_label outside allowed set")
        if row.get("quality_flag") not in QUALITY_FLAGS:
            errors.append(f"trial {trial}: quality_flag outside allowed set")
    for row in covariates:
        trial = row.get("trial_id", "<missing>")
        if row.get("device_quality") not in DEVICE_QUALITY:
            errors.append(f"trial {trial}: device_quality outside allowed set")
    for row in truth:
        trial = row.get("trial_id", "<missing>")
        if row.get("repeat_consistency_label") not in REPEAT_LABELS:
            errors.append(f"trial {trial}: repeat_consistency_label outside allowed set")
    return errors


def validate_timing(events: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for row in events:
        trial = row.get("trial_id", "<missing>")
        for left, right in zip(TIMING_CHAIN, TIMING_CHAIN[1:]):
            if left in row and right in row and row[left] > row[right]:
                errors.append(f"trial {trial}: timing order violation {left} > {right}")
    return errors


def validate_cross_ledger(events: List[Dict[str, Any]], covariates: List[Dict[str, Any]], truth: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    event_by_id = {row.get("trial_id"): row for row in events}
    cov_by_id = {row.get("trial_id"): row for row in covariates}
    truth_by_id = {row.get("trial_id"): row for row in truth}
    event_ids = set(event_by_id)
    cov_ids = set(cov_by_id)
    truth_ids = set(truth_by_id)

    for trial in sorted(cov_ids - event_ids):
        errors.append(f"covariate trial {trial} has no event row")
    for trial in sorted(truth_ids - event_ids):
        errors.append(f"ground_truth trial {trial} has no event row")
    for trial in sorted(event_ids - cov_ids):
        warnings.append(f"event trial {trial} has no covariate row")
    for trial in sorted(event_ids - truth_ids):
        errors.append(f"event trial {trial} has no ground_truth row")

    for trial in sorted(event_ids & cov_ids):
        event = event_by_id[trial]
        cov = cov_by_id[trial]
        expected_hold = event["hold_end_ms"] - event["hold_start_ms"]
        if abs(cov["hold_duration_ms"] - expected_hold) > 2:
            warnings.append(f"trial {trial}: hold_duration_ms differs from event timing by >2 ms")
        expected_latency = event["response_onset_ms"] - event["report_prompt_onset_ms"]
        if abs(cov["response_latency_ms"] - expected_latency) > 2:
            warnings.append(f"trial {trial}: response_latency_ms differs from event timing by >2 ms")

    for trial in sorted(event_ids & truth_ids):
        event = event_by_id[trial]
        row = truth_by_id[trial]
        for field in ("participant_id", "session_id", "stimulus_id"):
            if event.get(field) != row.get(field):
                errors.append(f"trial {trial}: ground_truth {field} does not match event ledger")
        if row.get("primary_residue_label") and not row.get("divergence_label"):
            errors.append(f"trial {trial}: primary_residue_label requires divergence_label")
        if row.get("secondary_skill_label") and not row.get("divergence_label"):
            errors.append(f"trial {trial}: secondary_skill_label requires divergence_label")
        if row.get("primary_residue_label") and row.get("secondary_skill_label"):
            errors.append(f"trial {trial}: primary and secondary labels cannot both be true")

    if events:
        matched = sum(1 for row in events if row.get("state_label") == row.get("report_label")) / len(events)
        if matched >= 0.95:
            errors.append("state_label and report_label are mechanically identical in >=95% of trials")
        elif matched >= 0.50:
            warnings.append("state_label and report_label match in >=50% of trials; inspect separability")
    if truth:
        rate = sum(1 for row in truth if row.get("primary_residue_label") is True) / len(truth)
        if rate < 0.05 or rate > 0.15:
            warnings.append(f"primary residue rate {rate:.3f} outside target band 0.05-0.15")
    return errors, warnings


def summary(events: List[Dict[str, Any]], covariates: List[Dict[str, Any]], truth: List[Dict[str, Any]]) -> Dict[str, Any]:
    primary = sum(1 for row in truth if row.get("primary_residue_label") is True)
    secondary = sum(1 for row in truth if row.get("secondary_skill_label") is True)
    return {
        "event_rows": len(events),
        "covariate_rows": len(covariates),
        "ground_truth_rows": len(truth),
        "usable_event_rows": sum(1 for row in events if row.get("usable_trial") is True),
        "usable_covariate_rows": sum(1 for row in covariates if row.get("usable_covariates") is True),
        "primary_residue_rows": primary,
        "secondary_skill_rows": secondary,
        "primary_residue_rate": primary / len(truth) if truth else None,
    }


def validate(args: argparse.Namespace) -> int:
    raw_events = read_csv(args.event)
    raw_covariates = read_csv(args.covariate)
    raw_truth = read_csv(args.ground_truth)
    errors = []
    warnings: List[str] = []
    errors.extend(validate_fields(raw_events, EVENT_FIELDS, "event"))
    errors.extend(validate_fields(raw_covariates, COVARIATE_FIELDS, "covariate"))
    errors.extend(validate_fields(raw_truth, GROUND_TRUTH_FIELDS, "ground_truth"))
    if not raw_events:
        errors.append("event ledger has zero rows")
    if not raw_covariates:
        errors.append("covariate ledger has zero rows")
    if not raw_truth:
        errors.append("ground_truth ledger has zero rows")

    events = [typed_row(row) for row in raw_events]
    covariates = [typed_row(row) for row in raw_covariates]
    truth = [typed_row(row) for row in raw_truth]
    for name, rows in (("event", events), ("covariate", covariates), ("ground_truth", truth)):
        for duplicate in duplicate_ids(rows, "trial_id"):
            errors.append(f"duplicate {name} trial_id {duplicate}")
    errors.extend(validate_enums(events, covariates, truth))
    errors.extend(validate_timing(events))
    cross_errors, cross_warnings = validate_cross_ledger(events, covariates, truth)
    errors.extend(cross_errors)
    warnings.extend(cross_warnings)

    report = {
        "summary": summary(events, covariates, truth),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="ascii")
    print(rendered)
    return 0 if not errors else 1


def init(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    write_template(args.out / "event_ledger.csv", EVENT_FIELDS)
    write_template(args.out / "covariate_ledger.csv", COVARIATE_FIELDS)
    write_template(args.out / "ground_truth_ledger.csv", GROUND_TRUTH_FIELDS)
    print(f"templates written to {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Aether behavioral ledgers")
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init", help="write blank ledger CSV templates")
    init_parser.add_argument("--out", type=Path, required=True)
    validate_parser = sub.add_parser("validate", help="validate completed ledger CSVs")
    validate_parser.add_argument("--event", type=Path, required=True)
    validate_parser.add_argument("--covariate", type=Path, required=True)
    validate_parser.add_argument("--ground-truth", type=Path, required=True)
    validate_parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        return init(args)
    if args.command == "validate":
        return validate(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
