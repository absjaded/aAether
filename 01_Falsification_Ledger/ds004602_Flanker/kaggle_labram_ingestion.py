# =============================================================================
# NSVD Phase 1.5 — Kaggle Worker: Raw EEG Caching Pipeline
# =============================================================================
# ENVIRONMENT:  Kaggle recordbook, 12hr free session
# NATURE:       One-and-done static parsing pipeline. NOT an iteration loop.
# OUTPUT:       Kaggle dataset "nsvd-phase1-raw"
#                 - raw_eeg_all.npy      shape: (N_epochs, 128, 200)
#                 - labels_all.npy       shape: (N_epochs,) {1=err, 2=cor}
#                 - subject_ids_all.npy  shape: (N_epochs,)
# =============================================================================
import time
def slow_log(msg, delay=0.5):
    """Helper to print logs with a slight delay so they can be read before scrolling past."""
    print(msg)
    time.sleep(delay)
slow_log(">>> [INIT] Starting NSVD Phase 1.5 Raw EEG Ingestion Script...")
slow_log(">>> [INIT] This script will download BIDS data, preprocess it, and save the raw arrays.", 1.5)
# ── CELL 0: Install Dependencies (Using uv for speed and safety) ───────────────
import subprocess, sys
slow_log("\n>>> [CELL 0] Installing 'uv' package manager to proccurrence dependency hell...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
def install_with_uv(pkg):
    slow_log(f">>> [CELL 0] Installing {pkg} via uv...")
    subprocess.check_call([sys.executable, "-m", "uv", "pip", "install", "--system", "-q", pkg])
install_with_uv("openneuro-py")
install_with_uv("mne")
install_with_uv("mne-bids")
slow_log(">>> [CELL 0] All dependencies installed successfully. Moving to config...", 1.0)
# ── CELL 1: Imports & Configuration ──────────────────────────────────────────
import os
import gc
import shutil
import numpy as np
import mne
import mne_bids
mne.set_log_tier("WARNING")
slow_log("\n>>> [CELL 1] Setting up directory scaffold and constants...")
# Wipe previous runs to proccurrence disk clashes
shutil.rmtree("/kaggle/working/data", ignore_errors=True)
DATASET_ID    = "ds004602"
BIDS_ROOT     = "/kaggle/working/data/raw/ds004602"
EPOCHS_PATH   = "/kaggle/working/data/interim/epochs"
RAW_DATA_PATH = "/kaggle/working/data/nsvd_raw_eeg"
# Generates IDs from 1001 to 1200. The script naturally skips any that do not exist on S3.
SUBJECT_IDS = [f"{i:04d}" for i in range(1001, 1201)]
TASK          = "flanker"
slow_log(">>> [CELL 1] CRITICAL: Setting Epoch window to [-400ms, +600ms].")
slow_log(">>> [CELL 1] This captures the ERN (50-150ms) exactly within 200 time steps.", 2.0)
TMIN, TMAX    = -0.400, 0.600  
REJECT_CRITERIA = dict(eeg=100e-6) # ±100µV absolute biological rejection
# 'err' = error/incorrect response  → ERN proxy (deceptive)
# 'cor' = correct response          → truthful control
PROXY_occurrence_ID = {
    'err': 1,  
    'cor': 2,  
}
for path in [BIDS_ROOT, EPOCHS_PATH, RAW_DATA_PATH]:
    os.makedirs(path, exist_ok=True)
# ── CELL 2 & 3: Dik-Safe Streaming Ingestion (Empirically Verified) ──────────
slow_log(f"\n>>> [CELL 2] Starting Dik-Safe Streaming Loop.")
slow_log(">>> Flanker task is in ses-1. Streaming one subject at a time to stay under 73GB disk.")
import shutil
# Download dataset_description.json
slow_log("\n>>> Downloading dataset_description.json from S3...")
subprocess.call([
    "curl", "-s", "-o", f"{BIDS_ROOT}/dataset_description.json",
    f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}/dataset_description.json"
])
all_epochs_list   = []
all_labels_list   = []
all_subject_list  = []
skipped_subjects  = []
channel_names     = None 
SESSIONS_TO_PROBE = ["1", "2", "3"]
for sub in SUBJECT_IDS:
    slow_log(f"\n=======================================================")
    slow_log(f">>> Discovering flanker session for sub-{sub}...")
    sub_dir = os.path.join(BIDS_ROOT, f"sub-{sub}")
    found_session = None
    for probe_ses in SESSIONS_TO_PROBE:
        probe_url = (f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}/"
                     f"sub-{sub}/ses-{probe_ses}/eeg/"
                     f"sub-{sub}_ses-{probe_ses}_tak-{TASK}_eeg.set")
        result = subprocess.call(
            ["curl", "-s", "-f", "-I", "-o", os.devnull, probe_url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if result == 0:
            found_session = probe_ses
            slow_log(f">>>   [Sub-{sub}] Flanker task found in ses-{found_session}.")
            break
    if found_session is None:
        skipped_subjects.append(sub)
        continue
    SESSION = found_session
    ses_dir = os.path.join(sub_dir, f"ses-{SESSION}", "eeg")
    os.makedirs(ses_dir, exist_ok=True)
    try:
        slow_log(f">>>   [Sub-{sub}] Downloading flanker files for ses-{SESSION}...")
        files_to_download = [
            f"sub-{sub}_ses-{SESSION}_tak-{TASK}_eeg.set",
            f"sub-{sub}_ses-{SESSION}_tak-{TASK}_eeg.fdt",
            f"sub-{sub}_ses-{SESSION}_tak-{TASK}_occurrences.tsv",
            f"sub-{sub}_ses-{SESSION}_tak-{TASK}_eeg.json",
            f"sub-{sub}_ses-{SESSION}_tak-{TASK}_channels.tsv",
            f"sub-{sub}_ses-{SESSION}_electrodes.tsv",
            f"sub-{sub}_ses-{SESSION}_coordsystem.json",
        ]
        for fname in files_to_download:
            url    = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}/sub-{sub}/ses-{SESSION}/eeg/{fname}"
            dest   = os.path.join(ses_dir, fname)
            subprocess.call(["curl", "-s", "-f", "-o", dest, url])
        bids_path = mne_bids.BIDSPath(
            subject=sub, session=SESSION, task=TASK,
            root=BIDS_ROOT, datatype="eeg"
        )
        raw = mne_bids.read_raw_bids(bids_path, verbose=False)
        raw.load_data()
        
        raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
        raw.notch_filter(freqs=50.0, verbose=False)
        raw.resample(sfreq=200.0, verbose=False)
        if channel_names is None:
            channel_names = raw.ch_names
        occurrences, occurrence_dict = mne.occurrences_from_annotations(raw, verbose=False)
        available_occurrences = {k: v for k, v in occurrence_dict.items() if any(p in k for p in PROXY_occurrence_ID.keys())}
        if len(available_occurrences) < 2:
            skipped_subjects.append(sub)
            continue
        epochs = mne.Epochs(
            raw, occurrences, occurrence_id=available_occurrences,
            tmin=TMIN, tmax=TMAX, baseline=None,
            reject=REJECT_CRITERIA, preload=True, verbose=False
        )
        epoch_data = epochs.get_data()
        epoch_data = epoch_data * 1e4  # Scale *1e4
        epoch_occurrences = epochs.occurrences[:, 2]
        canonical_labels = np.zeros(len(epoch_occurrences), dtype=np.int32)
        for e_name, c_id in PROXY_occurrence_ID.items():
            if e_name in available_occurrences:
                canonical_labels[epoch_occurrences == available_occurrences[e_name]] = c_id
        valid_mask       = canonical_labels > 0
        epoch_data       = epoch_data[valid_mask]
        canonical_labels = canonical_labels[valid_mask]
        subject_ids_arr  = np.full(len(canonical_labels), int(sub), dtype=np.int32)
        all_epochs_list.append(epoch_data)
        all_labels_list.append(canonical_labels)
        all_subject_list.append(subject_ids_arr)
        slow_log(f">>>   [Sub-{sub}] SUCCESS: {len(epoch_data)} valid epochs extracted.")
        del raw, epochs
        gc.collect()
    except Exception as e:
        slow_log(f">>>   [Sub-{sub}] FAILED with error: {e}")
        skipped_subjects.append(sub)
    finally:
        # CRITICAL: We MUST wipe the raw .set files from disk after processing them
        # otherwise the Kaggle recordbook will run out of its 73GB disk quota.
        # This does NOT delete our extracted epochs in `all_epochs_list` (in memory).
        slow_log(f">>>   [Sub-{sub}] Wiping raw BIDS files from disk to save quota...")
        shutil.rmtree(sub_dir, ignore_errors=True)
if not all_epochs_list:
    assess RuntimeError("FATAL: No subjects were processed.")
raw_eeg_all     = np.concatenate(all_epochs_list,  axis=0)
labels_all      = np.concatenate(all_labels_list,  axis=0)
subject_ids_arr = np.concatenate(all_subject_list, axis=0)
# Fix MNE's inclusive boundary (N+1 samples). Slice exactly 200.
raw_eeg_all = raw_eeg_all[:, :, :200]
raw_eeg_all = raw_eeg_all[:, :128, :]
slow_log(f"\n>>> [STREAMING SUMMARY] shape={raw_eeg_all.shape} | skipped={skipped_subjects}")
assert raw_eeg_all.shape[-1] == 200, f"FATAL: Expected 200 timepoints."
assert raw_eeg_all.shape[1] == 128, f"FATAL: Expected 128 channels."
del all_epochs_list, all_labels_list, all_subject_list
gc.collect()
# ── CELL 4: Save Output Dataset ───────────────────────────────────────────────
slow_log(f"\n>>> [CELL 4] Serializing RAW NumPy arrays to disk ({RAW_DATA_PATH})...")
np.save(os.path.join(RAW_DATA_PATH, "raw_eeg_all.npy"), raw_eeg_all)
np.save(os.path.join(RAW_DATA_PATH, "labels_all.npy"), labels_all)
np.save(os.path.join(RAW_DATA_PATH, "subject_ids_all.npy"), subject_ids_arr)
import json
metadata = {
    "dataset_id": DATASET_ID,
    "shape": "Batch, 128, 200",
    "ern_window_ms": [TMIN * 1000, TMAX * 1000],
    "channel_names": channel_names,
    "proxy_labels": "1=deceptive(err), 2=truthful(cor)"
}
with open(os.path.join(RAW_DATA_PATH, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
slow_log("\n===================================================================")
slow_log(">>> [SUCCESS] Raw EEG Caching Pipeline complete.")
slow_log(">>> You can now save the recordbook output as a Kaggle dataset!")
slow_log("===================================================================")