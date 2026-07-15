import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.covariance import OAS

# ==============================================================================
# --- LOGGER SETUP ---
# ==============================================================================
logger = logging.getLogger("c02_Baseline")
logger.settier(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s | %(tiername)-8s | %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# ==============================================================================
# --- LOCKED CONSTANTS (ENGINEERING_RGD.md) ---
# ==============================================================================
TIER_1_ROI_INDICES = [1, 2, 6, 25, 26, 29, 35, 36, 40, 59, 60, 63]
ZONE_START_IDX = 132  # 528ms at 250Hz
ZONE_END_IDX = 175    # 700ms at 250Hz
WINDOW_SIZE = 20      # 80ms
WINDOW_STEP = 2       # 8ms

def condition_spd(matrix, eps=1e-6):
    """Ensures a matrix is strictly Symmetric Positive Definite."""
    matrix = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def compute_oas_conditioned(X_trials):
    """Computes OAS covariance for a 3D array of trials (N, p, samples)."""
    N, p, n_samples = X_trials.shape
    covs = np.zeros((N, p, p))
    oas = OAS(assume_centered=False)
    for i in range(N):
        C = oas.fit(X_trials[i].T).covariance_
        covs[i] = condition_spd(C)
    return covs

def distance_riemann(A, B):
    """Affine-Invariant Riemannian Metric."""
    inv_sqrt_A = np.linalg.inv(np.linalg.cholesky(A))
    mid = inv_sqrt_A @ B @ inv_sqrt_A.T
    mid = (mid + mid.T) / 2.0
    eigvals = np.linalg.eigvalsh(mid)
    eigvals = np.maximum(eigvals, 1e-12)
    return np.sqrt(np.sum(np.log(eigvals)**2))

def mean_covariance(covmats, metric='riemann', maxiter=50, tol=1e-6):
    """Fréchet mean of SPD matrices."""
    N, p, _ = covmats.shape
    M = np.mean(covmats, axis=0)
    for _ in range(maxiter):
        sqrt_M = np.linalg.cholesky(M)
        inv_sqrt_M = np.linalg.inv(sqrt_M)
        
        tangent_sum = np.zeros((p, p))
        for i in range(N):
            C = covmats[i]
            mid = inv_sqrt_M @ C @ inv_sqrt_M.T
            mid = (mid + mid.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(mid)
            eigvals = np.maximum(eigvals, 1e-12)
            log_mid = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T
            tangent_sum += log_mid
            
        tangent_mean = tangent_sum / N
        norm = np.linalg.norm(tangent_mean, ord='fro')
        if norm < tol:
            break
            
        eigvals, eigvecs = np.linalg.eigh(tangent_mean)
        exp_tangent = eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T
        M = sqrt_M @ exp_tangent @ sqrt_M.T
        M = (M + M.T) / 2.0
    return condition_spd(M)

def cohen_d(x, y):
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2: return np.nan
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0: return 0.0
    return (np.mean(x) - np.mean(y)) / s_pooled

def process_subject(data_dir, subject_id):
    X_path = os.path.join(data_dir, f"{subject_id}_X.npy")
    y_path = os.path.join(data_dir, f"{subject_id}_y.npy")
    y_sem_path = os.path.join(data_dir, f"{subject_id}_y_semantic.npy")
    
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        return None
        
    X = np.load(X_path)
    X = X / (np.std(X) + 1e-12)
    y = np.load(y_path)
    y_sem = np.load(y_sem_path)
    
    # Step 1: Trial Filtering (Drop Fixation)
    fixation_mask = (y_sem[:, 0] != 0)
    
    # Step 2: Spatial Masking
    X_filt = X[fixation_mask][:, TIER_1_ROI_INDICES, :]
    y_filt = y[fixation_mask]
    y_sem_filt = y_sem[fixation_mask]
    
    # Step 3: Trial Pool Assignment
    z_idx = np.where(y_filt == 0)[0]  # All 0-back trials
    tb_mask = (y_filt == 1)
    tbc_idx = np.where(tb_mask & (y_sem_filt[:, 2] == 1))[0]
    tbi_idx = np.where(tb_mask & (y_sem_filt[:, 2] == 0))[0]
    
    if len(z_idx) < 20:
        return {'subject_id': subject_id, 'status': 'INADMISSIBLE_DATA'}
        
    # Step 5: Anchor/Probe Split
    np.random.rng_state(42 + int(subject_id))
    shuffled_idx = np.random.permutation(z_idx)
    half = len(shuffled_idx) // 2
    anchor_idx = shuffled_idx[:half]
    probe_idx = shuffled_idx[half:]
    
    # Step 4 & 6: Admissibility Check using full 43-sample OAS
    all_0back_zone_trials = X_filt[z_idx][:, :, ZONE_START_IDX:ZONE_END_IDX]
    all_0back_covs = compute_oas_conditioned(all_0back_zone_trials)
    M_s_full = mean_covariance(all_0back_covs, metric='riemann')
    disp_dists = [distance_riemann(C, M_s_full) for C in all_0back_covs]
    sigma_s = np.mean(disp_dists)
    
    # Step 7: Fréchet Mean Anchor Construction (FLAW 1: Using 43-sample full zone covariances)
    # FLAW 2: The anchor_idx includes ALL 0-back trials (no isCorrect filter)
    anchor_zone_trials = X_filt[anchor_idx][:, :, ZONE_START_IDX:ZONE_END_IDX]
    anchor_covs = compute_oas_conditioned(anchor_zone_trials)
    M_s_anchor = mean_covariance(anchor_covs, metric='riemann')
    
    # Step 8: Sliding-Window RGD Extraction
    def extract_scalars(indices):
        scalars = []
        for idx in indices:
            trial_zone = X_filt[idx, :, ZONE_START_IDX:ZONE_END_IDX]
            cov_windows = []
            n_windows = (trial_zone.shape[1] - WINDOW_SIZE) // WINDOW_STEP + 1
            for w in range(n_windows):
                start = w * WINDOW_STEP
                end = start + WINDOW_SIZE
                window_data = trial_zone[:, start:end]
                C = OAS(assume_centered=False).fit(window_data.T).covariance_
                cov_windows.append(condition_spd(C))
            max_d = np.max([distance_riemann(Cw, M_s_anchor) for Cw in cov_windows])
            scalars.append(max_d)
        return np.array(scalars)

    d_0b = extract_scalars(probe_idx)
    d_c2b = extract_scalars(tbc_idx)
    d_i2b = extract_scalars(tbi_idx)
    
    return {
        'subject_id': subject_id,
        'status': 'ADMISSIBLE',
        'sigma_s': sigma_s,
        'd_0b': d_0b,
        'd_c2b': d_c2b,
        'd_i2b': d_i2b
    }

def main():
    data_dir = sys.argv[1]
    files = glob.glob(os.path.join(data_dir, "*_X.npy"))
    subject_ids = [os.path.basename(f).split('_')[0] for f in files]
    
    logger.info(f"Found {len(subject_ids)} subjects. Processing Strict c02 Baseline...")
    
    # Process sequentially for easier debugging
    results = []
    for sid in subject_ids:
        res = process_subject(data_dir, sid)
        if res is not None:
            results.append(res)
    
    # 1. Subject Admissibility Filter
    valid_results = [r for r in results if r['status'] == 'ADMISSIBLE']
    if len(valid_results) == 0:
        logger.error("No admissible subjects found!")
        return
        
    sigmas = [r['sigma_s'] for r in valid_results]
    cohort_median = np.median(sigmas)
    cohort_mad = np.median(np.abs(sigmas - cohort_median))
    excl_thresh = cohort_median + 3 * cohort_mad
    
    logger.info(f"Cohort Median Sigma: {cohort_median:.4f}, MAD: {cohort_mad:.4f}, Exclusion > {excl_thresh:.4f}")
    
    admissible_subjects = []
    for r in valid_results:
        if r['sigma_s'] > excl_thresh:
            logger.warning(f"Subject {r['subject_id']} EXCLUDED: Dispersion {r['sigma_s']:.4f} > Threshold")
        else:
            admissible_subjects.append(r)
            
    logger.info(f"Admissible cohort size: {len(admissible_subjects)}")
    
    # 4. Gate 2: Friction Effect
    g2_d = []
    m_null, m_c2b, m_i2b = [], [], []
    for r in admissible_subjects:
        if len(r['d_i2b']) >= 2 and len(r['d_c2b']) >= 2:
            g2_d.append(cohen_d(r['d_i2b'], r['d_c2b']))
            if len(r['d_0b']) >= 2:
                m_null.append(np.median(r['d_0b']))
                m_c2b.append(np.median(r['d_c2b']))
                m_i2b.append(np.median(r['d_i2b']))
                
    if len(g2_d) > 0:
        logger.info(f"GATE 2 (Friction): N={len(g2_d)}, Median d={np.median(g2_d):.4f}")
        logger.info("GATE 2 Monotonicity Check:")
        logger.info(f"  Null: {np.median(m_null):.4f}")
        logger.info(f"  Corr: {np.median(m_c2b):.4f}")
        logger.info(f"  Inco: {np.median(m_i2b):.4f}")
    else:
        logger.warning(f"GATE 2: No valid comparisons found.")
        
    # --- SAVE FINAL CSV ---
    rows = []
    for r in admissible_subjects:
        sid = r['subject_id']
        for val in r['d_0b']: rows.append({'subject_id': sid, 'condition': '0-back', 'is_correct': 1, 'rgd_scalar': val})
        for val in r['d_c2b']: rows.append({'subject_id': sid, 'condition': '2-back', 'is_correct': 1, 'rgd_scalar': val})
        for val in r['d_i2b']: rows.append({'subject_id': sid, 'condition': '2-back', 'is_correct': 0, 'rgd_scalar': val})
    
    df = pd.DataFrame(rows)
    out_path = os.path.join(data_dir, "c02_baseline_scalars.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"SUCCESS: Saved {len(df)} trials to {out_path}")
        
if __name__ == "__main__":
    main()
