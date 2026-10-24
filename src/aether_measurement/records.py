"""Verify compact saved records and replay their aggregation, without inference.

Temporal/behavior summaries follow the original stage1f/stage1g aggregators.
ROAMM tables retain saved empirical aggregates and documented split boundaries;
their AUCs are not recomputed from private epoch or prediction records.
The bundled Phase 4 surfaces describe planted simulations through an observed
reliability envelope. Integrity and aggregation checks do not validate a human
phenomenon or regenerate the original research runs.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
from importlib import resources
import io
import json
import math
import statistics
from typing import Any


class RecordIntegrityError(ValueError):
    """A saved asset, declared count, or recomputed summary is inconsistent."""


EXPECTED_ASSETS = frozenset({
    "synthetic_temporal", "synthetic_behavior", "synthetic_power", "phase4_manifest",
    "phase4_auc_readout_surface", "phase4_gate_surface", "phase4_incremental_effect_surface",
    "phase4_observation_cost_surface", "phase4_utility_surface", "phase4_observation_contract",
    "phase4_utility_target",
    "empirical_eye_summary", "empirical_eye_calibration", "empirical_eeg_summary",
    "empirical_eeg_calibration", "empirical_eeg_within_summary", "empirical_filterbank_summary",
    "empirical_filterbank_calibration",
})
EMPIRICAL_TESTS = {
    "empirical_eye_summary": ("shared_reference_loso", "cross_run_train12_test45_loso"),
    "empirical_eeg_summary": ("riemann_shared_reference_loso", "riemann_cross_run_train12_test45_loso"),
    "empirical_eeg_within_summary": ("riemann_within_subject_broadband_train123_test45",),
    "empirical_filterbank_summary": ("filterbank_riemann_loso_39to1", "filterbank_riemann_within_train123_test45"),
}
SUMMARY_COLUMNS = ("test", "n", "acc", "balanced_acc", "auc", "log_loss", "on_task_acc", "mw_acc")
CALIBRATION_COLUMNS = ("calibration_trials", "reps", "subjects", "acc_mean", "acc_sd", "auc_mean",
                       "log_loss_mean", "balanced_acc_mean", "on_task_acc_mean", "mw_acc_mean")


def _read_asset_bytes(filename: str) -> bytes:
    if (not isinstance(filename, str) or not filename.endswith((".json", ".csv"))
            or "/" in filename or "\\" in filename or filename.startswith(".")):
        raise RecordIntegrityError(f"Unsafe evidence asset name: {filename!r}")
    try:
        return resources.files("aether_measurement").joinpath("data", filename).read_bytes()
    except (FileNotFoundError, OSError) as error:
        raise RecordIntegrityError(f"Missing evidence asset: {filename}") from error


def _decode(payload: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise RecordIntegrityError(f"Non-finite JSON constant in {label}: {value}")

    try:
        return json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RecordIntegrityError(f"Malformed evidence JSON: {label}") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordIntegrityError(message)


def _compare(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"Field mismatch: {label}")
        for key, value in expected.items():
            _compare(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), f"Row-count mismatch: {label}")
        for index, value in enumerate(expected):
            _compare(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        _require(type(actual) is type(expected) and actual == expected, f"Value mismatch: {label}")
    elif isinstance(expected, (int, float)):
        _require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                 and math.isfinite(actual) and math.isfinite(expected)
                 and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                 f"Numeric mismatch: {label}: {actual!r} != {expected!r}")
    else:
        raise RecordIntegrityError(f"Unsupported record value: {label}")


def _decode_csv(payload: bytes, label: str, columns: tuple[str, ...]) -> list[dict]:
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
        _require(tuple(reader.fieldnames or ()) == columns, f"Aggregate CSV columns mismatch: {label}")
        rows = []
        for raw in reader:
            _require(raw.keys() == set(columns) and all(value is not None and value != "" for value in raw.values()),
                     f"Missing or extra aggregate CSV field: {label}")
            row = {key: value if key == "test" else float(value) for key, value in raw.items()}
            _require(all(not isinstance(value, float) or math.isfinite(value) for value in row.values()),
                     f"Non-finite aggregate CSV value: {label}")
            rows.append(row)
        return rows
    except (UnicodeError, csv.Error, ValueError) as error:
        if isinstance(error, RecordIntegrityError):
            raise
        raise RecordIntegrityError(f"Malformed aggregate CSV: {label}") from error


def load_records() -> tuple[dict, dict[str, Any]]:
    """Load package resources only after verifying every catalogued byte stream."""
    catalog = _decode(_read_asset_bytes("catalog.json"), "catalog.json")
    _require(isinstance(catalog, dict) and catalog.get("schema_version") == 1, "Unsupported evidence catalog")
    assets = catalog.get("assets", [])
    _require(isinstance(assets, list) and len(assets) == len(EXPECTED_ASSETS), "Evidence catalog asset count mismatch")
    _require({entry.get("id") for entry in assets} == EXPECTED_ASSETS, "Evidence catalog identifiers mismatch")
    records = {}
    for entry in assets:
        payload = _read_asset_bytes(entry["file"])
        _require(len(payload) == entry["bytes"], f"Size mismatch: {entry['file']}")
        _require(hashlib.sha256(payload).hexdigest() == entry["sha256"], f"SHA256 mismatch: {entry['file']}")
        if entry.get("format") == "csv":
            columns = CALIBRATION_COLUMNS if entry["id"].endswith("_calibration") else SUMMARY_COLUMNS
            records[entry["id"]] = _decode_csv(payload, entry["file"], columns)
            _require(len(records[entry["id"]]) == entry["rows"], f"Aggregate row-count mismatch: {entry['file']}")
        else:
            records[entry["id"]] = _decode(payload, entry["file"])
    return catalog, records


def _mean(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    _require(bool(values), f"No finite replicate values: {field}")
    return statistics.fmean(values)


def _temporal_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], float(row["drift_effect"]))].append(row)
    summaries = []
    for _, group in sorted(grouped.items()):
        first = group[0]
        out = {key: first[key] for key in (
            "scenario", "drift_effect", "subjects", "trials", "total_reads", "violation_rate_target",
            "channels", "components", "gestalt_effect", "drift_phi", "drift_rank", "ramp_weight", "label_time_strength")}
        out["reps"] = len(group)
        for target, source in (("mean_violation_rate_observed", "violation_rate_observed"),
                               ("mean_label_lag1", "label_lag1"), ("mean_feature_pc1_lag1", "feature_pc1_lag1")):
            out[target] = _mean(group, source)
        for name in ("observed", "time_only", "detrended"):
            for metric in ("auc", "d", "p"):
                out[f"{name}_{metric}"] = _mean(group, f"{name}_{metric}")
            out[f"{name}_weak_rate"] = statistics.fmean(bool(row[f"{name}_weak"]) for row in group)
        for name in ("naive_shuffle", "circular_shift", "block_shuffle"):
            for metric in ("auc", "weak_rate"):
                out[f"{name}_{metric}"] = _mean(group, f"{name}_{metric}")
        if first["scenario"] == "clean_signal":
            if out["observed_weak_rate"] >= .80 and out["time_only_auc"] < .60 and out["detrended_auc"] >= .62:
                verdict = "survives_temporal_controls"
            elif out["time_only_auc"] >= .60:
                verdict = "time_structure_suspect"
            else:
                verdict = "weakened_by_temporal_controls"
        elif first["scenario"] == "independent_null":
            verdict = "clean_temporal_null" if out["observed_weak_rate"] <= .05 and .45 <= out["observed_auc"] <= .55 else "temporal_false_positive"
        elif out["observed_weak_rate"] >= .50 and out["time_only_auc"] >= .60 and out["detrended_auc"] < .62:
            verdict = "temporal_trap_detected"
        elif out["observed_weak_rate"] < .20:
            verdict = "trap_not_strong_enough"
        else:
            verdict = "temporal_trap_partial"
        out["verdict"] = verdict
        summaries.append(out)
    return summaries


def _behavior_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(row)
    summaries = []
    for _, group in sorted(grouped.items()):
        first = group[0]
        out = {key: first[key] for key in (
            "scenario", "mode", "rho", "subjects", "trials", "total_reads", "channels", "components", "gestalt_effect", "response_effect")}
        out["reps"] = len(group)
        out["mean_violation_rate"] = _mean(group, "violation_rate")
        for field in ("response_drive_violation_corr", "response_quality_violation_corr", "confidence_violation_corr",
                      "incremental_auc_vs_cheap", "neural_top_decile_enrichment", "cheap_top_decile_enrichment"):
            out[field] = _mean(group, field)
        for name in ("neural", "cheap", "combined"):
            for metric in ("auc", "d", "p"):
                out[f"{name}_{metric}"] = _mean(group, f"{name}_{metric}")
            out[f"{name}_weak_rate"] = statistics.fmean(row[f"{name}_weak"] for row in group)
        if first["scenario"] == "signal":
            if out["neural_auc"] >= .65 and out["cheap_auc"] < .60 and out["combined_auc"] >= out["neural_auc"] - .02:
                verdict = "neural_adds_beyond_cheap"
            elif out["cheap_auc"] >= .65:
                verdict = "cheap_behavior_competes"
            else:
                verdict = "weak_or_mixed"
        elif first["scenario"].startswith("N4"):
            verdict = "cheap_baseline_exposes_response_leak" if out["cheap_auc"] >= .65 or abs(out["response_drive_violation_corr"]) >= .15 else "response_leak_not_detected_by_cheap_baseline"
        else:
            verdict = "clean" if out["neural_auc"] < .60 and out["cheap_auc"] < .60 else "suspect"
        out["verdict"] = verdict
        summaries.append(out)
    return summaries


def _verify_replicates(record: dict, label: str, expected_rows: int, expected_groups: int, summarize) -> list[dict]:
    rows = record["rows"]
    _require(len(rows) == expected_rows and len(record["summary"]) == expected_groups, f"Saved row-count mismatch: {label}")
    config = record["config"]
    for row in rows:
        _require(row["subjects"] == config["subjects"] and row["trials"] == config["trials"], f"Configuration mismatch: {label}")
        _require(row["total_reads"] == row["subjects"] * row["trials"], f"Read-count mismatch: {label}")
    recomputed = summarize(rows)
    _compare(recomputed, record["summary"], f"{label}.summary")
    for group in recomputed:
        members = [r for r in rows if r["scenario"] == group["scenario"]
                   and ("drift_effect" not in group or r["drift_effect"] == group["drift_effect"])]
        _require(len(members) == config["reps"] and {r["rep"] for r in members} == set(range(config["reps"])),
                 f"Replicate identity mismatch: {label}.{group['scenario']}")
    return recomputed


def _surface_key(row: dict) -> tuple[str, float]:
    return row["reliability_channel"], float(row["state_effect"])


def _empirical_report(data: dict) -> dict:
    """Validate saved aggregate schemas/metric ranges, not underlying predictions."""
    tables = {}
    counts = {}
    for identifier, expected_tests in EMPIRICAL_TESTS.items():
        rows = data[identifier]
        _require(tuple(row["test"] for row in rows) == expected_tests, f"Empirical test labels mismatch: {identifier}")
        for row in rows:
            _require(row["n"] > 0 and row["n"].is_integer(), f"Empirical epoch count invalid: {identifier}")
            for field in ("acc", "balanced_acc", "auc", "on_task_acc", "mw_acc"):
                _require(0 <= row[field] <= 1, f"Empirical metric out of range: {identifier}.{field}")
            _require(row["log_loss"] >= 0, f"Negative empirical log loss: {identifier}")
            _compare(row["balanced_acc"], (row["on_task_acc"] + row["mw_acc"]) / 2, f"{identifier}.balanced_acc")
        tables[identifier] = rows
        counts[identifier] = len(rows)
    calibration = {}
    for name in ("eye", "eeg", "filterbank"):
        identifier = f"empirical_{name}_calibration"
        rows = data[identifier]
        _require([row["calibration_trials"] for row in rows] == [0, 5, 10, 20, 50], f"Calibration grid mismatch: {identifier}")
        for row in rows:
            _require(row["subjects"] == 40 and row["reps"] == (40 if row["calibration_trials"] == 0 else 8000),
                     f"Calibration aggregate count mismatch: {identifier}")
            for field in ("acc_mean", "auc_mean", "balanced_acc_mean", "on_task_acc_mean", "mw_acc_mean"):
                _require(0 <= row[field] <= 1, f"Calibration metric out of range: {identifier}.{field}")
            _require(row["acc_sd"] >= 0 and row["log_loss_mean"] >= 0, f"Invalid calibration deviation/loss: {identifier}")
            _compare(row["balanced_acc_mean"], (row["on_task_acc_mean"] + row["mw_acc_mean"]) / 2,
                     f"{identifier}.balanced_acc_mean")
        calibration[identifier] = rows
        counts[identifier] = len(rows)
    return dict(
        dataset="ROAMM ds007629", status="saved_aggregate_not_raw_reanalysis", eligible_subjects=40,
        row_counts=counts, pooled_prediction_metrics=tables, mean_subject_repeat_calibration=calibration,
        split_definitions=dict(
            eye="LOSO over held-out subjects; no run/time features. Cross-run trains on other subjects' runs 1–2 and tests held-out subject runs 4–5.",
            eeg_shared="Broadband/filterbank LOSO uses held-out-subject unlabeled all-run covariances for Euclidean alignment (transductive preprocessing).",
            eeg_cross_run="Broadband cross-run trains other subjects' runs 1–2, tests held-out subject runs 4–5; train/test blocks are aligned separately using their unlabeled covariances.",
            eeg_within="Within-subject EEG trains runs 1–3, tests runs 4–5, with train-only alignment.",
            calibration="Balanced labeled calibration subsets are sampled from each held-out subject; evaluated on remaining epochs. Means span subject/repeat metrics, not pooled AUC.",
        ),
        comparison_boundary="Eye/reading tables and 10-second EEG probes have different usable epoch counts and preprocessing. These are parallel probes, not an exact matched-row comparison or a human pre-report-state result.",
        replay_boundary="Integrity, table schemas, counts and balanced-accuracy identities are checked. AUCs are saved aggregates; subject predictions/raw epochs are not bundled or reanalyzed.",
    )


def replay_records() -> dict:
    """Verify byte integrity and saved aggregations; never simulate or fit models."""
    try:
        catalog, data = load_records()
        temporal = _verify_replicates(data["synthetic_temporal"], "temporal", 108, 9, _temporal_summary)
        behavior = _verify_replicates(data["synthetic_behavior"], "behavior", 144, 6, _behavior_summary)
        signal = next(row for row in behavior if row["scenario"] == "signal")
        power = data["synthetic_power"]["rows"]
        _require(len(power) == 1, "Power summary row-count mismatch")
        _compare(power[0]["mean_signal_auc"], signal["neural_auc"], "power.mean_signal_auc")
        _compare(power[0]["signal_power_weak"], signal["neural_weak_rate"], "power.signal_power_weak")

        manifest = data["phase4_manifest"]
        surface_counts = {}
        expected_counts = {"utility_surface": 17, "incremental_effect_surface": 17,
                           "observation_cost_surface": 4, "gate_surface": 6, "auc_readout_surface": 28}
        _require(len(manifest["surfaces"]) == 5, "Phase 4 manifest surface-count mismatch")
        for declaration in manifest["surfaces"]:
            name = declaration["file"].rsplit("/", 1)[-1].removesuffix(".json")
            _require(name in expected_counts and name not in surface_counts, f"Unexpected Phase 4 surface: {name}")
            surface = data[f"phase4_{name}"]
            _require(surface["surface_id"] == declaration["surface_id"], f"Surface identifier mismatch: {name}")
            grid = surface["grid"]
            count = sum(len(rows) for rows in grid.values()) if isinstance(grid, dict) else len(grid)
            _require(count == declaration["rows"] == expected_counts[name], f"Surface row-count mismatch: {name}")
            surface_counts[name] = count
        _require(surface_counts.keys() == expected_counts.keys(), "Phase 4 surface identifiers incomplete")
        effects = {_surface_key(row): row for row in data["phase4_incremental_effect_surface"]["grid"]}
        utility = {_surface_key(row): row for row in data["phase4_utility_surface"]["grid"]}
        _require(len(effects) == len(utility) == 17 and effects.keys() == utility.keys(), "Phase 4 effect-grid keys mismatch")
        for key, row in utility.items():
            for field in ("combined_auc", "cheap_auc", "power_30x70", "power_30x140"):
                _compare(row[field], effects[key][field], f"phase4.{key}.{field}")
        _compare(data["phase4_utility_target"], manifest["utility_target"], "phase4.utility_target")
        summary = manifest["summary_numbers"]
        for channel, prefix in (("primary_uncorrected", "primary_uncorrected"), ("spearman_brown", "spearman_brown")):
            best = max((row for (kind, _), row in utility.items() if kind == channel), key=lambda row: row["combined_auc"])
            for suffix, field in (("auc", "combined_auc"), ("effect", "state_effect"), ("lift", "lift")):
                _compare(best[field], summary[f"best_{prefix}_{suffix}"], f"phase4.best_{prefix}_{suffix}")
            first_pass = min(row["state_effect"] for (kind, _), row in effects.items() if kind == channel and row["power_30x70_pass"])
            _compare(first_pass, summary[f"{prefix}_first_power_pass_effect"], f"phase4.{prefix}_first_power_pass_effect")
        cost = data["phase4_observation_cost_surface"]["grid"]
        _compare(cost, data["phase4_observation_contract"]["auc_observation_cost"], "phase4.observation_cost_contract")
        return dict(
            status="verified_saved_records", scope="Hash verification and aggregation replay; no raw-neural inference or model fitting.",
            assets=dict(count=len(catalog["assets"]), bytes=sum(entry["bytes"] for entry in catalog["assets"]), sha256_verified=True),
            empirical_ROAMM=_empirical_report(data),
            synthetic_instrument=dict(temporal_replicates=108, temporal_summary=temporal, behavior_replicates=144,
                                      behavior_summary=behavior, power_summary=power),
            observation_calibration=dict(surface_counts=surface_counts, auc_parity_rows=17,
                                         summary_numbers=summary, utility_target=manifest["utility_target"],
                                         utility_assumption="Equal-variance binormal scores; modeled precision/lift, not observed deployment utility.",
                                         claim_boundary=manifest["framing"]),
            reported_unreproduced=catalog["reported_unreproduced"],
        )
    except RecordIntegrityError:
        raise
    except (KeyError, TypeError, ValueError, StopIteration, OverflowError) as error:
        raise RecordIntegrityError(f"Malformed or inconsistent evidence record: {error}") from error
