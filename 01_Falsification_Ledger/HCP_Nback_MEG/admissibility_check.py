"""
Within-Subject Admissibility Check (ENGINEERING_RGD.md §6)
==========================================================
assesss the internal 0-back geometric dispersion of each subject.
Flags subjects as INADMISSIBLE if their internal variance is an outlier.
Criteria: sigma_s > median(cohort_sigma) + 3 * MAD(cohort_sigma)
Required minimum 0-back trials: 20
"""

import os
import sys
import json
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann

# MLOPS constraints
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

DATA_DIR = os.environ.get("DATA_DIR", "/workspace/data/nsvd_fusion")
RESULTS_FILE = os.environ.get("RESULTS_FILE", "/workspace/results/admissibility_results.json")

TIER1_ROI_INDICES = [1, 2, 6, 25, 26, 29, 35, 36, 40, 59, 60, 63]
TEMP_START = 132
TEMP_END = 175 # Full 43-sample zone for admissibility

def condition_spd(matrix, eps=1e-6):
    matrix = (matrix + matrix.T) / 2
    trace = np.trace(matrix)
    matrix += np.eye(matrix.shape[0]) * (trace * eps if trace > 0 else eps)
    return matrix

def compute_oas_covariances(X_epochs):
    cov_estimator = Covariances(estimator='lwf')
    covs = cov_estimator.fit_transform(X_epochs)
    for i in range(covs.shape[0]):
        covs[i] = condition_spd(covs[i])
    return covs

def frechet_mean(covariances):
    from pyriemann.utils.mean import mean_riemann
    return mean_riemann(covariances)

def assess_subject(subject_id):
    try:
        X = np.load(f"{DATA_DIR}/{subject_id}_X.npy")
        y = np.load(f"{DATA_DIR}/{subject_id}_y.npy")
        y_sem = np.load(f"{DATA_DIR}/{subject_id}_y_semantic.npy")
    except Exception:
        return None
        
    valid_mask = y_sem[:, 0] != 0
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Admissible 0-back trials
    X_0back = X[y == 0]
    
    if len(X_0back) < 20:
        return {"subject_id": subject_id, "status": "INSUFFICIENT_DATA", "n_trials": len(X_0back)}
        
    X_0back = X_0back[:, TIER1_ROI_INDICES, TEMP_START:TEMP_END]
    
    try:
        covs = compute_oas_covariances(X_0back)
        M_s_full = frechet_mean(covs)
        
        distances = [distance_riemann(c, M_s_full) for c in covs]
        sigma_s = np.mean(distances)
        
        return {
            "subject_id": subject_id, 
            "status": "assessD", 
            "n_trials": len(X_0back), 
            "sigma_s": float(sigma_s)
        }
    except Exception as e:
        print(f"Error evaluating {subject_id}: {e}")
        return None

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Data dir not found: {DATA_DIR}")
        sys.exit(1)
        
    files = [f for f in os.listdir(DATA_DIR) if f.endswith("_X.npy")]
    subject_ids = sorted([f.split("_")[0] for f in files])
    
    print(f"Evaluating admissibility for {len(subject_ids)} subjects...")
    
    results = []
    for sid in subject_ids:
        res = assess_subject(sid)
        if res:
            results.append(res)
            
    eval_results = [r for r in results if r["status"] == "assessD"]
    sigmas = [r["sigma_s"] for r in eval_results]
    
    if not sigmas:
        print("No valid subjects to assess.")
        sys.exit(1)
        
    median_sigma = np.median(sigmas)
    mad_sigma = np.median(np.abs(sigmas - median_sigma))
    threshold = median_sigma + 3 * mad_sigma
    
    print(f"\nCohort Median Sigma: {median_sigma:.4f}")
    print(f"Cohort MAD Sigma: {mad_sigma:.4f}")
    print(f"Rejection Threshold: {threshold:.4f}\n")
    
    final_output = {
        "metadata": {
            "median_sigma": median_sigma,
            "mad_sigma": mad_sigma,
            "threshold": threshold
        },
        "subjects": {}
    }
    
    rejected_count = 0
    insufficient_count = sum(1 for r in results if r["status"] == "INSUFFICIENT_DATA")
    
    for r in results:
        sid = r["subject_id"]
        if r["status"] == "INSUFFICIENT_DATA":
            final_output["subjects"][sid] = {"admissible": False, "reason": "INSUFFICIENT_DATA"}
            print(f"Subject {sid}: REJECTED (Insufficient Data: {r['n_trials']} trials)")
        else:
            if r["sigma_s"] > threshold:
                final_output["subjects"][sid] = {"admissible": False, "reason": "OUTLIER_DISPERSION", "sigma_s": r["sigma_s"]}
                rejected_count += 1
                print(f"Subject {sid}: REJECTED (Outlier: {r['sigma_s']:.4f} > {threshold:.4f})")
            else:
                final_output["subjects"][sid] = {"admissible": True, "sigma_s": r["sigma_s"]}
                
    print(f"\nSummary:")
    print(f"Total Admissible: {len(eval_results) - rejected_count}")
    print(f"Rejected (Outliers): {rejected_count}")
    print(f"Rejected (Insufficient Data): {insufficient_count}")
    
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
