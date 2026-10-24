"""Saved-record replay and corruption diagnostics, without research reruns."""
import hashlib
import json
from unittest.mock import patch

import pytest

from aether_measurement import records


def test_saved_records_replay():
    report = records.replay_records()
    assert report["status"] == "verified_saved_records"
    assert report["assets"] == {"count": 18, "bytes": 458122, "sha256_verified": True}
    assert report["synthetic_instrument"]["temporal_replicates"] == 108
    assert report["synthetic_instrument"]["behavior_replicates"] == 144
    assert report["observation_calibration"]["auc_parity_rows"] == 17
    assert report["reported_unreproduced"][0]["status"] == "reported_unreproduced"
    json.dumps(report, allow_nan=False)


def test_saved_empirical_metrics_and_explicit_scope():
    empirical = records.replay_records()["empirical_ROAMM"]
    assert empirical["status"] == "saved_aggregate_not_raw_reanalysis"
    assert sum(empirical["row_counts"].values()) == 22
    eye = empirical["pooled_prediction_metrics"]["empirical_eye_summary"]
    eeg = empirical["pooled_prediction_metrics"]["empirical_eeg_summary"]
    assert eye[0]["auc"] == 0.7937596782034959
    assert eye[1]["auc"] == 0.7805612018072601
    assert eeg[0]["auc"] == 0.5175821661744285
    assert "transductive" in empirical["split_definitions"]["eeg_shared"]
    assert "not an exact matched-row" in empirical["comparison_boundary"]


def test_empirical_csv_hash_mismatch_is_clear():
    original = records._read_asset_bytes

    def corrupted(filename):
        payload = original(filename)
        return payload.replace(b"0.7937596782034959", b"0.9937596782034959") if filename == "empirical_eye_summary.csv" else payload

    with patch.object(records, "_read_asset_bytes", side_effect=corrupted):
        with pytest.raises(records.RecordIntegrityError, match="SHA256 mismatch: empirical_eye_summary"):
            records.replay_records()


def test_empirical_csv_requires_aggregate_only_schema():
    with pytest.raises(records.RecordIntegrityError, match="CSV columns mismatch"):
        records._decode_csv(b"subject,auc\nperson,0.8\n", "not_aggregate.csv", records.SUMMARY_COLUMNS)


def test_empirical_csv_rejects_nonfinite_numbers():
    payload = b"test,n,acc,balanced_acc,auc,log_loss,on_task_acc,mw_acc\nexample,10,0.5,0.5,NaN,0.7,0.5,0.5\n"
    with pytest.raises(records.RecordIntegrityError, match="Non-finite aggregate CSV"):
        records._decode_csv(payload, "bad.csv", records.SUMMARY_COLUMNS)


def test_empirical_balanced_accuracy_identity():
    _, data = records.load_records()
    data["empirical_eye_summary"][0]["balanced_acc"] = .2
    with pytest.raises(records.RecordIntegrityError, match="balanced_acc"):
        records._empirical_report(data)


def test_all_summary_fields_are_recomputed():
    _, data = records.load_records()
    records._compare(records._temporal_summary(data["synthetic_temporal"]["rows"]),
                     data["synthetic_temporal"]["summary"], "temporal")
    records._compare(records._behavior_summary(data["synthetic_behavior"]["rows"]),
                     data["synthetic_behavior"]["summary"], "behavior")


def test_hash_mismatch_is_clear():
    original = records._read_asset_bytes

    def corrupted(filename):
        payload = original(filename)
        return payload.replace(b'"subjects": 30', b'"subjects": 31', 1) if filename == "synthetic_behavior.json" else payload

    with patch.object(records, "_read_asset_bytes", side_effect=corrupted):
        with pytest.raises(records.RecordIntegrityError, match="SHA256 mismatch: synthetic_behavior"):
            records.replay_records()


def test_missing_asset_is_clear():
    with pytest.raises(records.RecordIntegrityError, match="Missing evidence asset"):
        records._read_asset_bytes("not_bundled.json")


@pytest.mark.parametrize("filename", ["../outside.json", "a/b.json", "a\\b.json", ".hidden.json"])
def test_path_traversal_is_rejected(filename):
    with pytest.raises(records.RecordIntegrityError, match="Unsafe evidence asset name"):
        records._read_asset_bytes(filename)


def _tamper_and_recatalog(filename, mutate):
    """Exercise aggregation checks independently of the first-line hash check."""
    original = records._read_asset_bytes
    record = json.loads(original(filename))
    mutate(record)
    payload = json.dumps(record).encode("utf-8")
    catalog = json.loads(original("catalog.json"))
    entry = next(entry for entry in catalog["assets"] if entry["file"] == filename)
    entry.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    replacement_catalog = json.dumps(catalog).encode("utf-8")

    def replacement(name):
        return replacement_catalog if name == "catalog.json" else payload if name == filename else original(name)

    return patch.object(records, "_read_asset_bytes", side_effect=replacement)


def test_temporal_row_change_fails_summary_parity():
    with _tamper_and_recatalog("synthetic_temporal.json", lambda data: data["rows"][0].update(observed_auc=.1)):
        with pytest.raises(records.RecordIntegrityError, match="Numeric mismatch: temporal.summary"):
            records.replay_records()


def test_behavior_verdict_change_fails_summary_parity():
    with _tamper_and_recatalog("synthetic_behavior.json", lambda data: data["summary"][0].update(verdict="invented")):
        with pytest.raises(records.RecordIntegrityError, match="Value mismatch: behavior.summary"):
            records.replay_records()


def test_phase4_auc_mismatch_is_detected():
    with _tamper_and_recatalog("phase4_utility_surface.json", lambda data: data["grid"][0].update(combined_auc=.99)):
        with pytest.raises(records.RecordIntegrityError, match="combined_auc"):
            records.replay_records()


def test_phase4_row_count_mismatch_is_detected():
    with _tamper_and_recatalog("phase4_gate_surface.json", lambda data: data["grid"]["leakage"].pop()):
        with pytest.raises(records.RecordIntegrityError, match="Surface row-count mismatch: gate_surface"):
            records.replay_records()


def test_nonfinite_json_is_rejected():
    with pytest.raises(records.RecordIntegrityError, match="Non-finite JSON constant"):
        records._decode(b'{"invalid": NaN}', "bad.json")
