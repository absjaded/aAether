"""Small contract/regression checks; no neural data or research runs."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aether_measurement import baseline, ledger


def make_ledgers(n: int = 30):
    """Create artificial schema fixtures, not a proposed collection design."""
    events, covariates, truth = [], [], []
    for index in range(n):
        trial = f"trial-{index}"
        participant = f"participant-{index // 10}"
        shared = {"participant_id": participant, "session_id": "session-0", "trial_id": trial}
        event = dict.fromkeys(ledger.EVENT_FIELDS, "0")
        event.update(shared)
        event.update({
            "block_id": "block-0", "trial_index": str(index),
            "stimulus_id": f"stimulus-{index}", "stimulus_family": "fixture",
            "state_label": "stable_judgment", "report_label": "reported",
            "stimulus_onset_ms": "0", "hold_start_ms": "10", "hold_end_ms": "110",
            "report_prompt_onset_ms": "120", "response_onset_ms": "150",
            "response_commit_ms": "160", "confidence_prompt_onset_ms": "170",
            "trial_end_ms": "180", "usable_trial": "true", "quality_flag": "ok",
        })
        cov = dict.fromkeys(ledger.COVARIATE_FIELDS, "0")
        cov.update(shared)
        cov.update({
            "confidence": str(index / max(1, n - 1)), "uncertainty": "0.5",
            "response_latency_ms": "30", "hold_duration_ms": "100",
            "gaze_available": "false", "device_quality": "good", "usable_covariates": "true",
        })
        target = dict.fromkeys(ledger.GROUND_TRUTH_FIELDS, "0")
        target.update(shared)
        target.update({
            "stimulus_id": event["stimulus_id"], "ground_truth_source": "fixture",
            "reference_answer": "reference", "reported_answer": "report",
            "divergence_label": str(index % 10 in (0, 1)).lower(),
            "repeat_consistency_label": "not_repeated",
            "primary_residue_label": str(index % 10 == 0).lower(),
            "secondary_skill_label": str(index % 10 == 1).lower(),
            "curation_source": "artificial_fixture",
        })
        events.append(event)
        covariates.append(cov)
        truth.append(target)
    return events, covariates, truth


def write_ledgers(root: Path, rows):
    paths = []
    for name, fields, contents in zip(
        ("event", "covariate", "ground_truth"),
        (ledger.EVENT_FIELDS, ledger.COVARIATE_FIELDS, ledger.GROUND_TRUTH_FIELDS),
        rows,
    ):
        path = root / f"{name}.csv"
        with path.open("w", encoding="ascii", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(contents)
        paths.append(path)
    return paths


def baseline_args(paths):
    return ["--event", str(paths[0]), "--covariate", str(paths[1]),
            "--ground-truth", str(paths[2]), "--permutations", "0"]


def run_baseline(paths):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = baseline.main(baseline_args(paths))
    report = json.loads(output.getvalue())
    # Reject nonstandard NaN/Infinity output as well as accidental serialization.
    json.dumps(report, allow_nan=False)
    return status, report


class LedgerTests(unittest.TestCase):
    def test_valid_ledgers(self):
        report = ledger.validate_rows(*make_ledgers())
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["warnings"], [])
        self.assertEqual(report["summary"]["primary_residue_rows"], 3)

    def test_malformed_values_do_not_reach_timing_arithmetic(self):
        rows = make_ledgers()
        rows[0][0].pop("hold_end_ms")
        rows[1][0]["confidence"] = "invalid"
        report = ledger.validate_rows(*rows)
        self.assertFalse(report["passed"])
        self.assertTrue(any("missing hold_end_ms" in error for error in report["errors"]))
        self.assertTrue(any("confidence is not numeric" in error for error in report["errors"]))

    def test_duplicate_trial_ids_are_rejected(self):
        rows = make_ledgers()
        for contents in rows:
            contents.append(dict(contents[0]))
        report = ledger.validate_rows(*rows)
        self.assertFalse(report["passed"])
        self.assertEqual(sum("duplicate" in error for error in report["errors"]), 3)

    def test_cross_ledger_identity_and_timing_errors(self):
        rows = make_ledgers()
        rows[1][0]["participant_id"] = "different-participant"
        rows[0][1]["response_onset_ms"] = "119"
        rows[2][2]["stimulus_id"] = "different-stimulus"
        report = ledger.validate_rows(*rows)
        self.assertFalse(report["passed"])
        self.assertTrue(any("covariate participant_id" in error for error in report["errors"]))
        self.assertTrue(any("timing order violation" in error for error in report["errors"]))
        self.assertTrue(any("ground_truth stimulus_id" in error for error in report["errors"]))

    def test_invalid_enum_and_incompatible_labels(self):
        rows = make_ledgers()
        rows[0][0]["quality_flag"] = "unknown"
        rows[2][0]["secondary_skill_label"] = "true"
        report = ledger.validate_rows(*rows)
        self.assertFalse(report["passed"])
        self.assertTrue(any("quality_flag" in error for error in report["errors"]))
        self.assertTrue(any("cannot both be true" in error for error in report["errors"]))

    def test_empty_ledgers_and_missing_truth_are_rejected(self):
        self.assertFalse(ledger.validate_rows([], [], [])["passed"])
        rows = make_ledgers()
        rows[2].pop()
        report = ledger.validate_rows(*rows)
        self.assertTrue(any("has no ground_truth row" in error for error in report["errors"]))

    def test_duplicate_csv_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            path.write_text("trial_id,trial_id\nfirst,second\n", encoding="ascii")
            report = ledger.validate_ledgers(path, path, path)
        self.assertFalse(report["passed"])
        self.assertIn("duplicate CSV header", report["errors"][0])

    def test_init_and_validate_cli_accept_explicit_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ledger.main(["init", "--out", str(root)]), 0)
            self.assertEqual(ledger.read_csv(root / "event_ledger.csv"), [])
            paths = write_ledgers(root, make_ledgers())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = ledger.main(["validate", *baseline_args(paths)[:-2]])
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(output.getvalue())["passed"])


class BaselineTests(unittest.TestCase):
    def test_interleaved_labels_preserve_out_of_fold_score_placement(self):
        rows = []
        for index in range(20):
            row = dict.fromkeys(baseline.CHEAP_FEATURES, 0.0)
            row.update({"confidence": index / 20, "label": bool(index % 2)})
            rows.append(row)
        with (
            mock.patch.object(baseline, "normalize", side_effect=lambda train, test: (train, test)),
            mock.patch.object(baseline, "fit_logistic", return_value=[]),
            mock.patch.object(baseline, "predict", side_effect=lambda weights, data: [row[0] for row in data]),
            mock.patch.object(baseline, "auc_score", wraps=baseline.auc_score) as scored,
        ):
            _, mode = baseline.cheap_auc(rows, "label")
        self.assertEqual(mode, "5_fold_cv")
        labels, scores = scored.call_args.args
        self.assertEqual(labels, [index % 2 for index in range(20)])
        self.assertEqual(scores, [index / 20 for index in range(20)])

    def test_tiny_separable_trial_fixture_has_expected_auc(self):
        rows = []
        for index in range(24):
            label = bool(index % 2)
            row = dict.fromkeys(baseline.CHEAP_FEATURES, 0.0)
            row.update({"confidence": float(label), "label": label})
            rows.append(row)
        auc, mode = baseline.cheap_auc(rows, "label")
        self.assertEqual(mode, "5_fold_cv")
        self.assertEqual(auc, 1.0)

    def test_small_sample_and_undefined_helpers_are_explicit(self):
        rows = baseline.join_ledgers(*make_ledgers(10))
        auc, mode = baseline.cheap_auc(rows, "primary_residue_label")
        self.assertIsNotNone(auc)
        self.assertEqual(mode, "apparent_small_sample")
        for row in rows:
            row["primary_residue_label"] = "false"
        self.assertEqual(baseline.cheap_auc(rows, "primary_residue_label"), (None, "undefined_single_class"))
        self.assertEqual(baseline.permutation_null(rows, "primary_residue_label", 3, 1)["n"], 0)
        self.assertEqual(baseline.cheap_auc([], "label"), (None, "undefined_single_class"))

    def test_cli_small_sample_and_single_class_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = make_ledgers(10)
            paths = write_ledgers(root, rows)
            status, report = run_baseline(paths)
            self.assertEqual(status, 0)
            self.assertEqual(report["decision"], "insufficient_sample_for_decision")
            self.assertEqual(report["cheap_stack"]["primary_mode"], "apparent_small_sample")
            for row in rows[2]:
                row["primary_residue_label"] = "false"
            paths = write_ledgers(root, rows)
            status, report = run_baseline(paths)
            self.assertEqual(status, 0)
            self.assertEqual(report["decision"], "rerun_primary_label_single_class")
            self.assertIsNone(report["cheap_stack"]["primary_auc"])
            self.assertIsNone(report["utility_target"]["target_recall"])

    def test_cli_refuses_invalid_ledgers_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = make_ledgers()
            rows[1][0]["confidence"] = "invalid"
            paths = write_ledgers(Path(directory), rows)
            with mock.patch.object(baseline, "cheap_auc") as score:
                status, report = run_baseline(paths)
            score.assert_not_called()
            self.assertEqual(status, 1)
            self.assertEqual(report["decision"], "fail_invalid_ledgers")
            self.assertFalse(report["validation"]["passed"])

    def test_cli_excludes_existing_unusable_flags_and_reports_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = make_ledgers()
            rows[0][0]["usable_trial"] = "false"
            rows[1][1]["usable_covariates"] = "false"
            rows[0][2]["usable_trial"] = "false"
            rows[1][2]["usable_covariates"] = "false"
            paths = write_ledgers(Path(directory), rows)
            status, report = run_baseline(paths)
        self.assertEqual(status, 0)
        self.assertEqual(report["joined_rows"], 30)
        self.assertEqual(report["included_rows"], 27)
        self.assertEqual(report["excluded_rows"], 3)
        self.assertEqual(report["primary_residue_count"], 2)
        self.assertIn("no subject/session holdout", report["evaluation_scope"])

    def test_cli_no_usable_rows_returns_failure_without_model_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = make_ledgers()
            for event in rows[0]:
                event["usable_trial"] = "false"
            paths = write_ledgers(Path(directory), rows)
            with mock.patch.object(baseline, "fit_logistic") as fit:
                status, report = run_baseline(paths)
            fit.assert_not_called()
        self.assertEqual(status, 1)
        self.assertEqual(report["decision"], "fail_no_usable_rows")
        self.assertEqual(report["included_rows"], 0)
        self.assertEqual(report["excluded_rows"], 30)
        self.assertIsNone(report["cheap_stack"]["primary_auc"])

    def test_cli_joins_the_same_normalized_identifiers_as_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = make_ledgers(10)
            for event in rows[0]:
                event["trial_id"] = " " + event["trial_id"] + " "
            paths = write_ledgers(Path(directory), rows)
            status, report = run_baseline(paths)
        self.assertEqual(status, 0)
        self.assertTrue(report["validation"]["passed"])
        self.assertEqual(report["joined_rows"], 10)
        self.assertEqual(report["included_rows"], 10)

    def test_missing_input_is_a_structured_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            status, report = run_baseline([missing] * 3)
        self.assertEqual(status, 1)
        self.assertEqual(report["decision"], "fail_invalid_ledgers")

    def test_undefined_utility_values_are_json_null(self):
        target = baseline.utility_target(0.0, 0.2, 2.0)
        self.assertIsNone(target["target_auc"])
        self.assertIsNone(target["target_recall"])
        json.dumps(target, allow_nan=False)

    def test_negative_permutation_count_is_rejected(self):
        arguments = ["--event", "unused", "--covariate", "unused", "--ground-truth", "unused",
                     "--permutations", "-1"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            baseline.main(arguments)
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
