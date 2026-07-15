# ===========================================================================
# OMP and Threading Settings must be at the very top (R4)
# ===========================================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import json
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
# import tracker (redacted)

# Reproducibility rng_state
rng_state = 42
np.random.rng_state(rng_state)
random.rng_state(rng_state)

DATA_DIR = os.environ.get("AETHER_DATA_DIR", "/workspace/data/nsvd_fusion")
OUTPUT_JSON = "/workspace/results/ml_results_v2.json"
LOG_FILE = "/workspace/experiments_meg_wm.log"
CHECKPOINT_FILE = "/workspace/results/loso_checkpoint.jsonl"  # Append-only fold-tier checkpoint

EXCLUDED_SUBJECTS = set()

TIER1_ROI_INDICES = [1, 2, 6, 25, 26, 29, 35, 36, 40, 59, 60, 63]
MOTOR_SENSORY_INDICES = [27, 28, 24, 61, 62, 58]

ROI_NAMES_MAP = {
    1: 'L_caudalanteriorcingulate',
    2: 'L_caudalmiddlefrontal',
    6: 'L_inferiorparietal',
    25: 'L_rostralmiddlefrontal',
    26: 'L_superiorfrontal',
    29: 'L_supramarginal',
    35: 'R_caudalanteriorcingulate',
    36: 'R_caudalmiddlefrontal',
    40: 'R_inferiorparietal',
    59: 'R_rostralmiddlefrontal',
    60: 'R_superiorfrontal',
    63: 'R_supramarginal',
    24: 'L_paracentral',
    27: 'L_precentral',
    28: 'L_postcentral',
    58: 'R_paracentral',
    61: 'R_precentral',
    62: 'R_postcentral'
}

LCMV_DISCLAIMER = (
    "DISCLAIMER/[Detail]: ON BILATERAL HOMOLOGOUS SUPPRESSION INVARIANT (I-6): "
    "LCMV beamforming is known to underestimate source activity for homologous bilateral pairs "
    "due to signal cancellation/suppression when source activities are highly correlated. "
    "Specifically, homologous pairs like left caudalmiddlefrontal (index 2) and right "
    "caudalmiddlefrontal (index 36) may have their active source contributions underestimated by LCMV."
)

def get_roi_name(idx):
    return ROI_NAMES_MAP.get(idx, f"ROI_{idx}")

# ===========================================================================
# Helper Functions
# ===========================================================================

def condition_spd(matrix, eps=1e-6):
    """Enforce Symmetric Positive Definite constraints safely."""
    if matrix.ndim == 2:
        matrix = (matrix + matrix.T) / 2
        trace = np.trace(matrix)
        matrix += np.eye(matrix.shape[-1]) * (trace * eps if trace > 0 else eps)
    elif matrix.ndim == 3:
        matrix = (matrix + matrix.transpose(0, 2, 1)) / 2
        for i in range(matrix.shape[0]):
            trace = np.trace(matrix[i])
            matrix[i] += np.eye(matrix.shape[-1]) * (trace * eps if trace > 0 else eps)
    return matrix

def custom_invsqrtm(matrix):
    """Robust custom inverse square root of a symmetric matrix."""
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, 1e-15, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

def unvech(vector, n):
    """Reconstruct a symmetric matrix from tangent space vector."""
    matrix = np.zeros((n, n))
    idx = 0
    for i in range(n):
        matrix[i, i] = vector[idx]
        idx += 1
        for j in range(i + 1, n):
            matrix[i, j] = vector[idx] / np.sqrt(2)
            matrix[j, i] = matrix[i, j]
            idx += 1
    return matrix

def load_subject(subject_id):
    """Load subject, filter out Fixation trials (imgType == 0) and check validity."""
    if subject_id in EXCLUDED_SUBJECTS:
        return None
    try:
        X = np.load(f"{DATA_DIR}/{subject_id}_X.npy")
        y_sem = np.load(f"{DATA_DIR}/{subject_id}_y_semantic.npy")
    except Exception as e:
        print(f"  [WARN] Cannot load {subject_id}: {e}")
        return None

    # Filter to active trials (Face=1, Tool=2) and exclude Fixation (0)
    img_type = y_sem[:, 0]
    valid_mask = img_type != 0
    X_valid = X[valid_mask]
    y_valid = img_type[valid_mask]

    if len(X_valid) < 2:
        print(f"  [WARN] {subject_id} has <2 valid trials — skipping.")
        return None
    if len(np.unique(y_valid)) < 2:
        print(f"  [WARN] {subject_id} has only one class present — skipping.")
        return None

    return subject_id, X_valid, y_valid

# ===========================================================================
# Main offline LOSO classifier pipeline (modular run)
# ===========================================================================

def run_offline_loso_pipeline(subjects_aligned_data, shuffle_labels=False, mask_motor=False):
    """Runs the main offline 68x68 LOSO classifier pipeline on subjects_aligned_data.
    Returns:
        loso_results: dict of test_sid -> test_acc, best_c, coef
        average_accuracy: float
        average_std: float
    """
    loso_results = {}
    all_subject_ids = sorted(subjects_aligned_data.keys())
    
    for test_sid in all_subject_ids:
        # Prepare splits
        train_sids = sorted([sid for sid in all_subject_ids if sid != test_sid])
        
        # Deterministic split using rng_state=42
        rng = np.random.default_rng(rng_state)
        shuffled_train_sids = list(train_sids)
        rng.shuffle(shuffled_train_sids)
        
        n_val = int(round(0.15 * len(train_sids)))
        val_subjects = shuffled_train_sids[:n_val]
        true_train = shuffled_train_sids[n_val:]
        
        # Helper to get covariances and labels
        def get_data(sids, shuffle=False, mask=False):
            covs = []
            labels = []
            for sid in sids:
                c = subjects_aligned_data[sid]["cov_aligned"]
                if mask:
                    c = c.copy()
                    c[:, MOTOR_SENSORY_INDICES, :] = 0
                    c[:, :, MOTOR_SENSORY_INDICES] = 0
                    c = condition_spd(c)
                y = subjects_aligned_data[sid]["y"].copy()
                if shuffle:
                    # Shuffle individually to preserve balance
                    np.random.shuffle(y)
                covs.append(c)
                labels.append(y)
            return np.concatenate(covs, axis=0), np.concatenate(labels, axis=0)
            
        cov_true_train, y_true_train = get_data(true_train, shuffle=shuffle_labels, mask=mask_motor)
        
        # Tangent Space Projection
        ts = TangentSpace(metric='riemann')
        X_true_train_ts = ts.fit_transform(cov_true_train)
        
        if len(val_subjects) > 0:
            cov_val, y_val = get_data(val_subjects, shuffle=shuffle_labels, mask=mask_motor)
            X_val_ts = ts.transform(cov_val)
        else:
            X_val_ts, y_val = None, None
            
        # C parameter sweep
        best_c = 1.0
        best_val_acc = -1.0
        c_candidates = [0.01, 0.1, 1.0, 10.0]
        
        if X_val_ts is not None and len(y_val) > 0:
            for C in c_candidates:
                clf = LogisticRegression(penalty='l2', C=C, max_iter=1000, random_state=rng_state)
                clf.fit(X_true_train_ts, y_true_train)
                val_acc = accuracy_score(y_val, clf.predict(X_val_ts))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_c = C
        
        # assess on test subject
        cov_test, y_test = get_data([test_sid], shuffle=shuffle_labels, mask=mask_motor)
        X_test_ts = ts.transform(cov_test)
        
        clf_final = LogisticRegression(penalty='l2', C=best_c, max_iter=1000, random_state=rng_state)
        clf_final.fit(X_true_train_ts, y_true_train)
        test_acc = accuracy_score(y_test, clf_final.predict(X_test_ts))
        
        loso_results[test_sid] = {
            "best_acc": test_acc,
            "best_c": best_c,
            "coef": clf_final.coef_[0].tolist()
        }
        
    accuracies = [loso_results[sid]["best_acc"] for sid in all_subject_ids]
    return loso_results, float(np.mean(accuracies)), float(np.std(accuracies))

# ===========================================================================
# R5: RGD Demo Soundness Metrics Profiling
# ===========================================================================

def profile_rgd_metrics(subjects_aligned_data):
    """Profile baseline variance, threshold, and latencies for each subject."""
    rgd_profiles = {}
    
    # Pre-calculate sliding windows to map index to ms latency
    WINDOW_SIZE = 12
    WINDOW_STEP = 3
    MAX_END_INDEX = 175
    windows = []
    for start in range(0, MAX_END_INDEX - WINDOW_SIZE + 1, WINDOW_STEP):
        windows.append((start, start + WINDOW_SIZE))

    cov_extractor = Covariances(estimator='lwf')

    for sid in sorted(subjects_aligned_data.keys()):
        # Retrieve raw time series, aligned 68x68 covariances, and mean covariance
        X_raw = subjects_aligned_data[sid]["X_raw"] # (N, 68, 255)
        cov_aligned = subjects_aligned_data[sid]["cov_aligned"] # (N, 68, 68)
        mean_cov = subjects_aligned_data[sid]["mean_cov"] # (68, 68)
        
        # 1. Slice aligned 68x68 covariances to 12x12
        cov_aligned_12 = cov_aligned[:, TIER1_ROI_INDICES, :][:, :, TIER1_ROI_INDICES]
        
        # Compute 12x12 baseline Fréchet mean of aligned active trials
        baseline_frechet_12 = mean_riemann(cov_aligned_12, tol=1e-4, maxiter=50)

        # Riemannian distances of each trial covariance to the baseline mean
        dists = [distance_riemann(c, baseline_frechet_12) for c in cov_aligned_12]
        
        baseline_variance = float(np.var(dists))
        threshold_sd = float(np.std(dists))
        rgd_threshold = float(np.mean(dists) + 2.5 * threshold_sd)

        # 2. Profile rgd_spike_latency_ms
        # We need the whitening matrix R_invsqrt of the subject (68x68)
        R_invsqrt = custom_invsqrtm(mean_cov)
        
        latencies = []
        for trial_idx in range(X_raw.shape[0]):
            first_crossing_time = None
            for w_start, w_end in windows:
                # 68x68 window covariance
                X_window = X_raw[trial_idx, :, w_start:w_end] # (68, 12)
                cov_window = cov_extractor.fit_transform(X_window[np.newaxis, ...])[0]
                cov_window = condition_spd(cov_window)
                cov_window /= np.trace(cov_window)
                
                # Apply Euclidean Alignment using the subject's 68x68 whitening matrix
                cov_window_aligned = R_invsqrt @ cov_window @ R_invsqrt
                cov_window_aligned = condition_spd(cov_window_aligned)
                
                # Slice to 12x12
                cov_window_12 = cov_window_aligned[TIER1_ROI_INDICES, :][:, TIER1_ROI_INDICES]
                cov_window_12 = condition_spd(cov_window_12)

                # RGD distance to the 12x12 baseline Fréchet mean
                rgd_score = distance_riemann(cov_window_12, baseline_frechet_12)
                
                if rgd_score > rgd_threshold:
                    first_crossing_time = float((w_start + w_end) / 2 * 4)
                    break
            
            if first_crossing_time is not None:
                latencies.append(first_crossing_time)

        avg_spike_latency = float(np.mean(latencies)) if latencies else 0.0

        # 3. Profile veto_latency_ms
        # Measure duration for a single window processing
        n_profiling_runs = min(100, X_raw.shape[0])
        times_ms = []
        for i in range(n_profiling_runs):
            X_window = X_raw[i, :, 0:12]
            
            t0 = time.perf_counter()
            cov_win = cov_extractor.fit_transform(X_window[np.newaxis, ...])[0]
            cov_win = condition_spd(cov_win)
            cov_win /= np.trace(cov_win)
            
            cov_win_aligned = R_invsqrt @ cov_win @ R_invsqrt
            cov_win_aligned = condition_spd(cov_win_aligned)
            
            cov_win_12 = cov_win_aligned[TIER1_ROI_INDICES, :][:, TIER1_ROI_INDICES]
            cov_win_12 = condition_spd(cov_win_12)
            
            rgd = distance_riemann(cov_win_12, baseline_frechet_12)
            _ = rgd > rgd_threshold
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

        avg_veto_latency = float(np.mean(times_ms))

        rgd_profiles[sid] = {
            "baseline_variance": baseline_variance,
            "threshold_sd": threshold_sd,
            "rgd_threshold": rgd_threshold,
            "rgd_spike_latency_ms": avg_spike_latency,
            "veto_latency_ms": avg_veto_latency
        }

    return rgd_profiles

# ===========================================================================
# Main Execution Entrypoint
# ===========================================================================

def main():
    print("=" * 60)
    print("KAGGLE PIPELINE V2 — Compliance Remediation & Gate 3 Audits")
    print("=" * 60)
    print(LCMV_DISCLAIMER)
    print()

    # 1. Discover and load all valid subjects
    all_files = os.listdir(DATA_DIR)
    subject_files = [f for f in all_files if f.endswith("_X.npy")]
    all_subject_ids = sorted([f.split("_")[0] for f in subject_files])
    all_subject_ids = [sid for sid in all_subject_ids if sid not in EXCLUDED_SUBJECTS]

    print(f"Discovered {len(all_subject_ids)} valid subjects")
    
    subjects_aligned_data = {}
    for sid in all_subject_ids:
        res = load_subject(sid)
        if res is not None:
            subject_id, X_valid, y_valid = res
            
            # Compute 68x68 covariances
            cov_estimator = Covariances(estimator='lwf')
            cov = cov_estimator.fit_transform(X_valid)
            cov = condition_spd(cov)
            
            # Trace normalization
            traces = np.trace(cov, axis1=1, axis2=2)[:, None, None]
            traces = np.where(traces == 0, 1e-15, traces)
            cov /= traces
            cov = condition_spd(cov)
            
            # Within-subject Euclidean Alignment
            mean_cov = cov.mean(axis=0)
            R_invsqrt = custom_invsqrtm(mean_cov)
            cov_aligned = np.einsum('ij,njk,kl->nil', R_invsqrt, cov, R_invsqrt)
            cov_aligned = condition_spd(cov_aligned)
            
            subjects_aligned_data[sid] = {
                "X_raw": X_valid,
                "cov_aligned": cov_aligned,
                "y": y_valid,
                "mean_cov": mean_cov
            }
    
    print(f"Loaded and pre-aligned data for {len(subjects_aligned_data)} subjects.")

    # 2. RGD Soundness Metrics Profiling
    print("\nProfiling RGD Soundness Metrics...")
    rgd_profiles = profile_rgd_metrics(subjects_aligned_data)
    
    avg_spike_latency = float(np.mean([v["rgd_spike_latency_ms"] for v in rgd_profiles.values()]))
    avg_veto_latency = float(np.mean([v["veto_latency_ms"] for v in rgd_profiles.values()]))
    print(f"Average RGD Spike Latency: {avg_spike_latency:.2f} ms")
    print(f"Average Veto Latency: {avg_veto_latency:.2f} ms")

    # 3. Main offline LOSO classifier execution
    print("\n[R2] Running Main 68-ROI LOSO Pipeline...")
    t_start = time.time()

    # --- CHECKPOINT RESUME LOGIC ---
    # Load any already-completed folds from disk so an OOM crash at no point loses work.
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    loso_results = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as _ck:
            for _line in _ck:
                _line = _line.strip()
                if _line:
                    _fold = json.loads(_line)
                    loso_results[_fold["subject_id"]] = _fold["result"]
        print(f"[CHECKPOINT] Resumed {len(loso_results)} completed folds from {CHECKPOINT_FILE}")

    for fold_idx, test_sid in enumerate(all_subject_ids, 1):
        # Skip already-completed folds (crash recovery)
        if test_sid in loso_results:
            print(f"Fold {fold_idx}/{len(all_subject_ids)}: Subject {test_sid} already done — skipping.")
            continue
        print(f"Fold {fold_idx}/{len(all_subject_ids)}: Testing subject {test_sid}...")
        # Prepare splits
        train_sids = sorted([sid for sid in all_subject_ids if sid != test_sid])
        
        # Deterministic split using rng_state=42
        rng = np.random.default_rng(rng_state)
        shuffled_train_sids = list(train_sids)
        rng.shuffle(shuffled_train_sids)
        
        n_val = int(round(0.15 * len(train_sids)))
        val_subjects = shuffled_train_sids[:n_val]
        true_train = shuffled_train_sids[n_val:]
        
        # Concatenate train data
        cov_true_train = np.concatenate([subjects_aligned_data[sid]["cov_aligned"] for sid in true_train], axis=0)
        y_true_train = np.concatenate([subjects_aligned_data[sid]["y"] for sid in true_train], axis=0)
        
        # Concatenate val data
        if len(val_subjects) > 0:
            cov_val = np.concatenate([subjects_aligned_data[sid]["cov_aligned"] for sid in val_subjects], axis=0)
            y_val = np.concatenate([subjects_aligned_data[sid]["y"] for sid in val_subjects], axis=0)
        else:
            cov_val, y_val = None, None
            
        # Test data
        cov_test = subjects_aligned_data[test_sid]["cov_aligned"]
        y_test = subjects_aligned_data[test_sid]["y"]
        
        # Tangent Space Projection
        ts = TangentSpace(metric='riemann')
        X_true_train_ts = ts.fit_transform(cov_true_train)
        
        if cov_val is not None:
            X_val_ts = ts.transform(cov_val)
        else:
            X_val_ts = None
            
        X_test_ts = ts.transform(cov_test)
        
        # C parameter sweep
        best_c = 1.0
        best_val_acc = -1.0
        c_candidates = [0.01, 0.1, 1.0, 10.0]
        
        if X_val_ts is not None and len(y_val) > 0:
            for C in c_candidates:
                clf = LogisticRegression(penalty='l2', C=C, max_iter=1000, random_state=rng_state)
                clf.fit(X_true_train_ts, y_true_train)
                val_acc = accuracy_score(y_val, clf.predict(X_val_ts))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_c = C
                    
        # Train final classifier with best_c
        clf_final = LogisticRegression(penalty='l2', C=best_c, max_iter=1000, random_state=rng_state)
        clf_final.fit(X_true_train_ts, y_true_train)
        test_acc = accuracy_score(y_test, clf_final.predict(X_test_ts))
        
        coef = clf_final.coef_[0] # (2346,)
        
        # Reconstruct 68x68 coefficient matrix
        fold_tangent_matrix = unvech(coef, 68)
        fold_saliency_68 = np.abs(fold_tangent_matrix).sum(axis=1)
        fold_ranked_indices = np.argsort(fold_saliency_68)[::-1]
        
        # Get rank of L_caudalmiddlefrontal (index 2)
        fold_rank_2 = int(np.where(fold_ranked_indices == 2)[0][0] + 1)
        fold_top5_rois = [get_roi_name(idx) for idx in fold_ranked_indices[:5]]
        
        # Verify DMN suppression direction (posteriorcingulate indices 21/55, precuneus 23/57)
        dmn_indices = [21, 23, 55, 57]
        fold_dmn_weights = [fold_tangent_matrix[idx].mean() for idx in dmn_indices]
        fold_dmn_avg_weight = np.mean(fold_dmn_weights)
        fold_dmn_saliency_direction = "negative" if fold_dmn_avg_weight < 0 else "positive"
        
        # Get rank of motor ROIs
        fold_motor_ranks = [int(np.where(fold_ranked_indices == idx)[0][0] + 1) for idx in MOTOR_SENSORY_INDICES]
        fold_motor_roi_max_saliency_rank = int(np.min(fold_motor_ranks))
        
        # Global Fréchet mean of aligned training subjects
        train_covs_list = [subjects_aligned_data[sid]["cov_aligned"] for sid in train_sids]
        all_train_covs = np.concatenate(train_covs_list, axis=0)
        global_frechet_mean = mean_riemann(all_train_covs, tol=1e-4, maxiter=50)
        
        # Riemannian distance of test subject's mean 68x68 covariance to the global Fréchet mean
        frechet_distance = distance_riemann(subjects_aligned_data[test_sid]["mean_cov"], global_frechet_mean)
        
        # RGD profiling metrics for this subject
        resting_variance = rgd_profiles[test_sid]["baseline_variance"]
        rgd_spike_latency_ms = rgd_profiles[test_sid]["rgd_spike_latency_ms"]
        
        # Subject info
        n_face_trials = int(np.sum(y_test == 1))
        n_tool_trials = int(np.sum(y_test == 2))
        class_ratio = float(n_face_trials / n_tool_trials) if n_tool_trials > 0 else 0.0
        
        fold_record = {
            "best_acc": test_acc,
            "best_c": best_c,
            "coef": coef.tolist(),
            "l_caudalmiddlefrontal_rank": fold_rank_2,
            "top5_rois": fold_top5_rois,
            "dmn_saliency_direction": fold_dmn_saliency_direction,
            "motor_roi_max_saliency_rank": fold_motor_roi_max_saliency_rank,
            "frechet_distance_from_mean": frechet_distance,
            "resting_variance": resting_variance,
            "n_face_trials": n_face_trials,
            "n_tool_trials": n_tool_trials,
            "class_ratio": class_ratio,
            "rgd_spike_latency_ms": rgd_spike_latency_ms
        }
        loso_results[test_sid] = fold_record

        # --- WRITE TO CHECKPOINT IMMEDIATELY (crash-safe) ---
        with open(CHECKPOINT_FILE, "a") as _ck:
            _ck.write(json.dumps({"subject_id": test_sid, "result": fold_record}) + "\n")
            _ck.flush()
            os.fsync(_ck.fileno())  # Force OS to flush to disk now
        print(f"  [CHECKPOINT] Fold {fold_idx} saved to {CHECKPOINT_FILE}")
        
        # Weights & Biases Telemetry
        try:
            tracker_mode = os.environ.get("TRACKER_MODE", "offline")
            # tracker.init(
                project="aether-semantic",
                group="bet_a_v2",
                name=f"loso_fold_{test_sid}",
                config={
                    "rng_state": rng_state,
                    "test_subject": test_sid,
                    "classifier": "LogisticRegression",
                    "penalty": "l2",
                    "best_c": best_c,
                },
                mode=tracker_mode
            )
            
            # tracker.log({
                "subject_id": test_sid,
                "loso_accuracy": test_acc,
                "loso_chance": 0.5,
                "delta_vs_phase0": test_acc - 0.598,
                "l_caudalmiddlefrontal_rank": fold_rank_2,
                "top5_rois": fold_top5_rois,
                "dmn_saliency_direction": fold_dmn_saliency_direction,
                "motor_roi_max_saliency_rank": fold_motor_roi_max_saliency_rank,
                "frechet_distance_from_mean": frechet_distance,
                "resting_variance": resting_variance,
                "n_face_trials": n_face_trials,
                "n_tool_trials": n_tool_trials,
                "class_ratio": class_ratio,
                "rgd_spike_latency_ms": rgd_spike_latency_ms
            })
        finally:
            # tracker.finish()

    t_end = time.time()
    print(f"Main LOSO pipeline complete in {t_end - t_start:.2f} seconds.")

    # Compute average peak accuracy
    peak_accs = [v["best_acc"] for v in loso_results.values()]
    loso_mean = float(np.mean(peak_accs))
    loso_std = float(np.std(peak_accs))
    print(f"Average Peak LOSO Accuracy: {loso_mean*100:.2f}% ± {loso_std*100:.2f}%")

    # [R3] Gate 3 Anti-Leakage Audits
    print("\n[R3] Running Control 1: Random Label Shuffle...")
    _, c1_mean, _ = run_offline_loso_pipeline(subjects_aligned_data, shuffle_labels=True)
    print(f"Control 1 Mean Accuracy: {c1_mean*100:.2f}% (Expected ~50%)")

    print("\n[R3] Running Control 2: Covariance Time-Shuffle...")
    subjects_time_shuffled = {}
    for sid in all_subject_ids:
        res = load_subject(sid)
        if res is not None:
            subject_id, X_valid, y_valid = res
            X_shuffled = X_valid.copy()
            for i in range(X_valid.shape[0]):
                # Shuffling timepoints (axis 2) before covariance extraction
                perm = np.random.permutation(X_valid.shape[2])
                X_shuffled[i] = X_valid[i, :, perm]
                
            cov_estimator = Covariances(estimator='lwf')
            cov = cov_estimator.fit_transform(X_shuffled)
            cov = condition_spd(cov)
            
            # Trace normalization
            traces = np.trace(cov, axis1=1, axis2=2)[:, None, None]
            traces = np.where(traces == 0, 1e-15, traces)
            cov /= traces
            cov = condition_spd(cov)
            
            # Within-subject Euclidean Alignment
            mean_cov = cov.mean(axis=0)
            R_invsqrt = custom_invsqrtm(mean_cov)
            cov_aligned = np.einsum('ij,njk,kl->nil', R_invsqrt, cov, R_invsqrt)
            cov_aligned = condition_spd(cov_aligned)
            
            subjects_time_shuffled[sid] = {
                "cov_aligned": cov_aligned,
                "y": y_valid
            }
            
    _, c2_mean, _ = run_offline_loso_pipeline(subjects_time_shuffled, shuffle_labels=False)
    c2_delta = loso_mean - c2_mean
    print(f"Control 2 Mean Accuracy: {c2_mean*100:.2f}%  (Delta = {c2_delta*100:+.2f}pp)")

    print("\n[R3] Running Control 3: Motor ROI Masking check...")
    _, c3_masked_mean, _ = run_offline_loso_pipeline(subjects_aligned_data, shuffle_labels=False, mask_motor=True)
    c3_delta = loso_mean - c3_masked_mean
    print(f"Control 3 Masked: {c3_masked_mean*100:.2f}%  (Delta = {c3_delta*100:+.2f}pp)")

    # Complete Logging & ROI Saliency Extraction
    # 2. Extract best_coef vector and average across folds
    mean_coef = np.mean([loso_results[sid]["coef"] for sid in all_subject_ids], axis=0)
    
    # Symmetrize and reconstruct into a 68x68 matrix
    tangent_matrix = unvech(mean_coef, 68)
    
    # Rank ROIs by absolute weight
    saliency_68 = np.abs(tangent_matrix).sum(axis=1) # (68,)
    
    # Rank 68 ROIs descending
    ranked_indices = np.argsort(saliency_68)[::-1]
    
    # Verify Left DLPFC prominence (index 2) in top 3 by absolute weight (verbal stimuli)
    rank_2 = int(np.where(ranked_indices == 2)[0][0] + 1)
    is_top_3 = rank_2 <= 3
    print(f"L_caudalmiddlefrontal (index 2) Rank: {rank_2} (Is in top 3: {is_top_3})")

    # Clean the coefficients before saving to json to avoid large file size
    for sid in all_subject_ids:
        if "coef" in loso_results[sid]:
            del loso_results[sid]["coef"]

    # Save results to results/ml_results_v2.json
    results_json = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rng_state": rng_state,
        "average_peak_accuracy": loso_mean,
        "average_peak_accuracy_std": loso_std,
        "top_performing_window": [0, 255],
        "top_performing_window_ms": [0, 1020],
        "top_performing_window_accuracy": loso_mean,
        "l_caudalmiddlefrontal_rank": rank_2,
        "is_l_caudalmiddlefrontal_top_3": is_top_3,
        "rgd_average_spike_latency_ms": avg_spike_latency,
        "rgd_average_veto_latency_ms": avg_veto_latency,
        "lcmv_bilateral_homologous_suppression_disclaimer": LCMV_DISCLAIMER,
        "audits": {
            "control_1_random_label_shuffle": {
                "average_peak_accuracy": c1_mean
            },
            "control_2_covariance_time_shuffle": {
                "average_peak_accuracy": c2_mean,
                "delta_accuracy": c2_delta
            },
            "control_3_motor_roi_masking": {
                "average_peak_accuracy_unmasked_68": loso_mean,
                "average_peak_accuracy_masked_68": c3_masked_mean,
                "delta_accuracy": c3_delta
            }
        },
        "loso_folds": loso_results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_json, f, indent=4)
    print(f"Saved results to {OUTPUT_JSON}")

    # Format experiments_meg_wm.log block
    top5_roi_names = [get_roi_name(idx) for idx in ranked_indices[:5]]
    
    gate1_status = "PASS" if (loso_mean - 0.598) >= 0.03 else "FAIL"
    gate2_status = "PASS" if is_top_3 else "FAIL"
    gate3_random_status = "PASS" if c1_mean <= 0.55 else "FAIL"
    
    log_block = f"""
--- RUN ---
timestamp:           {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
rng_state:                {rng_state}
phase / bet:         Bet A v2
hypothesis:          Compliance Remediation: Within-subject Euclidean Alignment, 68x68 main classifier, proper val fold splits, and telemetry.
lever_changed:       compliance_remediation
lever_value:         68x68_main_and_validation_splits
lever_state:         estimator=lwf, unvech=68, within_subject_ea=true, validation_split=0.15
eda_pending_flag:    NO

loso_mean:           {loso_mean*100:.2f}%
loso_std:            ±{loso_std*100:.2f}%
within_subj:         N/A
gate1:               {gate1_status} (delta={loso_mean - 0.598:+.4f})
gate2:               {gate2_status} (l_caudalmiddlefrontal_rank={rank_2})
gate3_random:        {gate3_random_status} (shuffled={c1_mean*100:.2f}%)
gate3_shuffle:       Δ={c2_delta*100:.2f}pp
gate3_motor:         Δ={c3_delta*100:.2f}pp

top5_rois:           {top5_roi_names}
l_caudalmiddlefrontal_rank: {rank_2}
saliency_method:     unvech_absolute_sum

rgd_spike_latency_ms: {avg_spike_latency:.2f} ms
veto_latency_ms:      {avg_veto_latency:.2f} ms

disclaimer:          {LCMV_DISCLAIMER}

diagnosis:           None
next_action:         Document compliance remediation transition report.
--- END RUN ---
"""
    with open(LOG_FILE, "a") as f:
        f.write(log_block)
    
    print("\nAppended run to experiments_meg_wm.log:")
    print(log_block)
    print("=" * 60)

if __name__ == "__main__":
    main()
