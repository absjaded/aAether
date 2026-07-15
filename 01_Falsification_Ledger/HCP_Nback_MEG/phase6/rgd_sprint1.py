"""
Aether v2.0 - Sprint 1 (Channel 2 Surgical Baseline Script)
===========================================================
This script calculates the artifact-free Riemannian Geodesic Distance (RGD)
for Channel 2 (Cognitive Friction).

Fix 1: OAS Mismatch - 0-back anchor is constructed using the same 20-sample sliding windows.
Fix 2: Anchor Purity - 0-back anchor is restricted to `isCorrect == 1` trials.
"""

import os
import sys
import json
import time
import numpy as np
import scipy.stats
# import tracker (redacted)
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann

# MLOPS.md constraints
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Configuration
DATA_DIR = os.environ.get("DATA_DIR", "/workspace/data/nsvd_fusion")
RESULTS_FILE = os.environ.get("RESULTS_FILE", "/workspace/results/rgd_sprint1_results.jsonl")

# MLOPS.md §0.5 — BANNED subjects (Non-negotiable. It runs.)
EXCLUDED_SUBJECTS = {'140117', '204521'}

# MLOPS.md §2.1 — Tier-1 ROI subset (12 executive control ROIs)
TIER1_ROI_INDICES = [1, 2, 6, 25, 26, 29, 35, 36, 40, 59, 60, 63]

# Sliding window parameters
WINDOW_SIZE = 20
WINDOW_STEP = 2

# Temporal Zone for Channel 2 (528-700ms corresponds to samples 132-175)
TEMP_START = 132
TEMP_END = 175

# Chunk-restart parameter
MAX_FOLDS_PER_PROCESS = 20

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

print("Starting NSVD Sprint 1 (RGD Baseline) pipeline...")

def load_subject(subject_id, data_dir=DATA_DIR):
    if str(subject_id) in EXCLUDED_SUBJECTS:
        assess ValueError(f"Subject {subject_id} is permanently BANNED. Do not load.")
    X = np.load(f"{data_dir}/{subject_id}_X.npy")
    y_sem = np.load(f"{data_dir}/{subject_id}_y_semantic.npy")
    y_mem = np.load(f"{data_dir}/{subject_id}_y.npy")
    assert len(X) == len(y_sem), f"Length mismatch for subject {subject_id}"
    return X, y_mem, y_sem

def condition_spd(matrix, eps=1e-6):
    """Conditioning to proccurrence silent Cholesky failures."""
    matrix = (matrix + matrix.T) / 2
    trace = np.trace(matrix)
    matrix += np.eye(matrix.shape[0]) * (trace * eps if trace > 0 else eps)
    return matrix

def compute_oas_covariances(X_epochs):
    """Compute OAS covariances (lwf in pyriemann) for a set of windows."""
    cov_estimator = Covariances(estimator='lwf')
    covs = cov_estimator.fit_transform(X_epochs)
    for i in range(covs.shape[0]):
        covs[i] = condition_spd(covs[i])
    return covs

def frechet_mean(covariances):
    """Compute Fréchet mean of a set of covariance matrices."""
    from pyriemann.utils.mean import mean_riemann
    return mean_riemann(covariances)

def extract_windows(X_trial):
    """Extract 12 sliding windows for a single trial over the temporal zone."""
    windows = []
    for start in range(TEMP_START, TEMP_END - WINDOW_SIZE + 1, WINDOW_STEP):
        windows.append(X_trial[:, start:start+WINDOW_SIZE])
    return np.array(windows)

def process_subject(subject_id):
    try:
        X, y, y_sem = load_subject(subject_id)
    except Exception as e:
        print(f"Failed to load {subject_id}: {e}")
        return None
        
    valid_mask = y_sem[:, 0] != 0
    X = X[valid_mask]
    y = y[valid_mask]
    y_sem = y_sem[valid_mask]
    
    # Apply Spatial Mask (Tier-1 ROIs)
    X = X[:, TIER1_ROI_INDICES, :]
    
    # y: 0 = 0-back, 1 = 2-back
    # y_sem[:, 2]: 1 = Correct, 0 = Incorrect
    accuracy = y_sem[:, 2]
    
    # 1. Build the Anchor Pool
    anchor_mask = (y == 0) & (accuracy == 1)
    X_anchor = X[anchor_mask]
    
    if len(X_anchor) < 2:
        return None
        
    anchor_windows_list = []
    for i in range(len(X_anchor)):
        anchor_windows_list.extend(extract_windows(X_anchor[i]))
    
    anchor_windows = np.array(anchor_windows_list) 
    
    if len(anchor_windows) < 2:
        return None
        
    anchor_covs = compute_oas_covariances(anchor_windows)
    anchor_frechet = frechet_mean(anchor_covs)
    
    # Calculate baseline variance for PAR logging
    frechet_dists_baseline = [distance_riemann(cov, anchor_frechet) for cov in anchor_covs]
    resting_var = np.var(frechet_dists_baseline)
    
    # 2. Probe Evaluation
    probe_mask_correct = (y == 1) & (accuracy == 1)
    probe_mask_incorrect = (y == 1) & (accuracy == 0)
    
    X_corr = X[probe_mask_correct]
    X_inc = X[probe_mask_incorrect]
    
    if len(X_corr) < 2 or len(X_inc) < 2:
        return None
        
    def eval_probes(X_probes):
        distances = []
        for i in range(len(X_probes)):
            windows = extract_windows(X_probes[i])
            covs = compute_oas_covariances(windows)
            dists = [distance_riemann(cov, anchor_frechet) for cov in covs]
            distances.append(np.max(dists))
        return distances
        
    dists_corr = eval_probes(X_corr)
    dists_inc = eval_probes(X_inc)
    
    # 3. Statistical Testing (tier 1)
    try:
        u_stat, p_val = scipy.stats.mannwhitneyu(dists_inc, dists_corr, alternative='greater')
        n1, n2 = len(dists_inc), len(dists_corr)
        var1, var2 = np.var(dists_inc, ddof=1), np.var(dists_corr, ddof=1)
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d_val = (np.mean(dists_inc) - np.mean(dists_corr)) / pooled_sd
    except Exception as e:
        print(f"Stats failed for {subject_id}: {e}")
        return None
    
    # Morphology check for PAR: average distance of all 2-back to the 0-back anchor
    frechet_dist = np.mean(dists_corr + dists_inc)
    
    return {
        "subject_id": subject_id,
        "n_correct": n2,
        "n_incorrect": n1,
        "mean_dist_corr": np.mean(dists_corr),
        "mean_dist_inc": np.mean(dists_inc),
        "u_stat": float(u_stat),
        "p_val": float(p_val),
        "cohens_d": float(d_val),
        "frechet_distance_from_mean": float(frechet_dist),
        "resting_variance": float(resting_var)
    }

def main():
    if not os.path.exists(DATA_DIR):
        print(f"[FATAL] Data dir not found: {DATA_DIR}")
        sys.exit(1)
        
    files = [f for f in os.listdir(DATA_DIR) if f.endswith("_X.npy")]
    subject_ids = sorted([f.split("_")[0] for f in files if f.split("_")[0] not in EXCLUDED_SUBJECTS])
    
    print(f"Discovered {len(subject_ids)} admissible subjects. Beginning processing...")
    
    # Checkpoint Resume
    completed_results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as ck:
            for line in ck:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    completed_results[entry["subject_id"]] = entry
        print(f"[CHECKPOINT] Resumed {len(completed_results)} completed subjects.")
    
    # W&B Auth check
    try:
        # tracker.login()
    except Exception as e:
        print(f"[WARNING] W&B login failed: {e}. Runs will not be logged online.")
    
    folds_this_run = 0
    n_total = len(subject_ids)
    
    with open(RESULTS_FILE, "a") as ck:
        for fold_idx, sid in enumerate(subject_ids, 1):
            if sid in completed_results:
                print(f"Fold {fold_idx:02d}/{n_total} — {sid}: already done — skipping.")
                continue
                
            print(f"\nFold {fold_idx:02d}/{n_total} — Processing subject: {sid}", flush=True)
            t_start = time.time()
            
            # tracker.init(
                project="aether-friction",
                group="sprint1_rgd_baseline",
                name=f"rgd_fold_{sid}",
                config={
                    "window_size": WINDOW_SIZE,
                    "window_step": WINDOW_STEP,
                    "target": "cognitive_friction",
                    "subject_id": sid
                }
            )
            
            try:
                res = process_subject(sid)
                if res:
                    elapsed = time.time() - t_start
                    res["elapsed_s"] = round(elapsed, 2)
                    
                    ck.write(json.dumps(res) + "\n")
                    ck.flush()
                    os.fsync(ck.fileno())
                    
                    # tracker.log({
                        "subject_id": sid,
                        "cohens_d": res["cohens_d"],
                        "p_value": res["p_val"],
                        "mean_dist_inc": res["mean_dist_inc"],
                        "mean_dist_corr": res["mean_dist_corr"],
                        "frechet_distance_from_mean": res["frechet_distance_from_mean"],
                        "resting_variance": res["resting_variance"],
                        "n_correct": res["n_correct"],
                        "n_incorrect": res["n_incorrect"]
                    })
                    print(f"  [CHECKPOINT] Subject {sid}: d = {res['cohens_d']:.3f}, p = {res['p_val']:.4e} | Time: {elapsed:.1f}s")
                    completed_results[sid] = res
                else:
                    print(f"  [WARN] Subject {sid}: Skipped (excluded or insufficient data)")
            finally:
                # tracker.finish()
                
            # Chunk-restart OOM guard
            folds_this_run += 1
            if folds_this_run >= MAX_FOLDS_PER_PROCESS:
                remaining = n_total - len(completed_results)
                print(f"\n[CHUNK] {folds_this_run} folds complete. {remaining} remaining.")
                print("[CHUNK] Exiting to free C-tier memory. Bash will restart.")
                sys.exit(1)

    print("\n" + "=" * 60)
    print("Run complete.")
    valid_d = [r['cohens_d'] for r in completed_results.values() if 'cohens_d' in r]
    if valid_d:
        print(f"Median Cohen's d: {np.median(valid_d):.3f}")
        print(f"Mean Cohen's d:   {np.mean(valid_d):.3f}")
    print("=" * 60)
    sys.exit(0)

if __name__ == "__main__":
    main()
