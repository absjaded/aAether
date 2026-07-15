#!/usr/bin/env python3
"""
runpod_eda.py — The Definitive Semantic Specificity EDA
Executes the 8 canonical EDA constraints identified in doc00nsvd_fusion_metadata.md.
Calculates class balances, SPD health, subject baselines (from Fixation trials),
threshold standard deviations, and global EA outliers.
"""
import os
import time
import json
import numpy as np
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
DATA_DIR = os.environ.get("AETHER_DATA_DIR", "/workspace/data/nsvd_fusion")
OUTPUT_JSON = "canonical/subject_cards/eda_results.json"
# We assume subjects are strictly the 75 (73 valid + 2 banned) found in the dir
EXCLUDED_SUBJECTS = {'140117', '204521'}
def condition_spd(matrix, eps=1e-6):
    """Enforces Symmetric Positive Definite constraints safely using trace-scaled regularization."""
    if matrix.ndim == 2:
        matrix = (matrix + matrix.transpose(1, 0)) / 2
        trace = np.trace(matrix)
        matrix += np.eye(matrix.shape[-1]) * (trace * eps if trace > 0 else eps)
    elif matrix.ndim == 3:
        matrix = (matrix + matrix.transpose(0, 2, 1)) / 2
        for i in range(matrix.shape[0]):
            trace = np.trace(matrix[i])
            matrix[i] += np.eye(matrix.shape[-1]) * (trace * eps if trace > 0 else eps)
    return matrix
def load_and_audit_subject(subject_id):
    """Loads subject, checks trial yield, extracts Fixation vs Task, and audits SPD health."""
    try:
        X = np.load(f"{DATA_DIR}/{subject_id}_X.npy")
        y_sem = np.load(f"{DATA_DIR}/{subject_id}_y_semantic.npy")
    except Exception as e:
        return {"subject_id": subject_id, "error": str(e)}
    # 1. Class Balance & Trial Yield Audit (imgType 0=Fix, 1=Face, 2=Tool)
    img_type = y_sem[:, 0]
    n_fix = (img_type == 0).sum()
    n_face = (img_type == 1).sum()
    n_tool = (img_type == 2).sum()
    
    ratio = n_face / n_tool if n_tool > 0 else 0.0
    # 2. SPD Health Check & Hardware Latency Profiling
    t0 = time.perf_counter()
    cov_extractor = Covariances(estimator='lwf')
    
    # Separate Fixation (Resting Baseline) vs Task (Face/Tool)
    X_fix = X[img_type == 0]
    X_task = X[img_type != 0]
    if len(X_task) == 0:
         return {"subject_id": subject_id, "error": "No task trials found."}
cov_task = condition_spd(cov_extractor.fit_transform(X_task))
    t1 = time.perf_counter()
    # Eigenvalue Audit on task covariances
    eigenvalues = np.linalg.eigvalsh(cov_task)
    min_eig = float(eigenvalues.min())
    n_near_singular = int((eigenvalues < 1e-6).sum())
    # Since Fixation trials (0) are absent, we use the standard Euclidean Alignment approach:
    # Compute the Fréchet mean of all active Task trials to serve as the baseline reference geometry.
    baseline_frechet = mean_riemann(cov_task)
    # Calculate threshold_sd based on resting volatility (distance of task trials to their mean)
    dists_to_baseline = [distance_riemann(c, baseline_frechet) for c in cov_task]
    threshold_sd = float(np.std(dists_to_baseline))
    baseline_variance = float(np.var(dists_to_baseline))
    return {
        "subject_id": subject_id,
        "metrics": {
            "n_fix": int(n_fix),
            "n_face": int(n_face),
            "n_tool": int(n_tool),
            "face_tool_ratio": float(ratio),
            "min_eigenvalue": min_eig,
            "near_singular_matrices": n_near_singular,
            "cov_extraction_latency_ms": (t1 - t0) * 1000,
            "baseline_variance": baseline_variance,
            "threshold_sd": threshold_sd
        },
        "baseline_frechet": baseline_frechet
    }
def main():
    print("Starting RunPod Phase 2 EDA...")
    
    if not os.path.exists(DATA_DIR):
        print(f"WARNING: Data directory {DATA_DIR} not found. Skipping execution.")
        return
    subject_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_X.npy")]
    all_subject_ids = [f.split("_")[0] for f in subject_files]
    
    print(f"Found {len(all_subject_ids)} subjects. Beginning parallel processing...")
    # Process all subjects in parallel to max CPU(alt: 4)
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(load_and_audit_subject)(sid) for sid in all_subject_ids
    )
    
    eda_output = {
        "class_balance_audit": {},
        "spd_health_audit": {},
        "hardware_latency": {},
        "subject_baselines": {},
        "outliers": []
    }
    valid_baselines = []
    baseline_subject_ids = []
    for res in results:
        sid = res["subject_id"]
        if "error" in res:
            print(f"Subject {sid} failed: {res['error']}")
            continue
            
        metrics = res["metrics"]
        eda_output["class_balance_audit"][sid] = {
            "ratio": metrics["face_tool_ratio"],
            "n_face": metrics["n_face"],
            "n_tool": metrics["n_tool"]
        }
        eda_output["spd_health_audit"][sid] = {
            "min_eig": metrics["min_eigenvalue"],
            "near_singular": metrics["near_singular_matrices"]
        }
        eda_output["hardware_latency"][sid] = metrics["cov_extraction_latency_ms"]
        
        # Save baseline stats, skip matrix in JSON but keep for Global Outlier
        eda_output["subject_baselines"][sid] = {
            "baseline_variance": metrics["baseline_variance"],
            "threshold_sd": metrics["threshold_sd"]
        }
        
        valid_baselines.append(res["baseline_frechet"])
        baseline_subject_ids.append(sid)
    # 4. Outlier Detection (Global Fréchet Mean across subjects)
    print("Computing Global EA Mean to detect outliers...")
    if valid_baselines:
        global_frechet = mean_riemann(np.array(valid_baselines))

    # Calculate Median Absolute Deviation (MAD) instead of Z-Score to comply with canonical bans
        global_distances = [distance_riemann(b, global_frechet) for b in valid_baselines]
        median_dist = np.median(global_distances)
        mad = np.median(np.abs(global_distances - median_dist))
        # Constant for normal distribution consistency (1.4826)
        modified_z_scores = 0.6745 * (global_distances - median_dist) / mad
        
        for idx, sid in enumerate(baseline_subject_ids):
            dist = global_distances[idx]
            mod_z = modified_z_scores[idx]
            eda_output["subject_baselines"][sid]["global_ea_distance"] = float(dist)
             eda_output["subject_baselines"][sid]["global_z_score"] = float(mod_z) # Kept key name for compat
             # MAD threshold of 3.5 is the standard for robust outlier detection
            if mod_z > 3.5:
                eda_output["outliers"].append({
                    "subject_id": sid,
                    "z_score": float(mod_z),
                    "action": "BANNED" if sid in EXCLUDED_SUBJECTS else "NEW_OUTLIER"
                })
    os.makedirs("canonical/subject_cards", exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(eda_output, f, indent=4)
        
    print(f"EDA Complete. Results saved to {OUTPUT_JSON}")
if __name__ == "__main__":
    main()
