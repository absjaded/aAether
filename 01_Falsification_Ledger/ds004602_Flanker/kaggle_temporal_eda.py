import os
import sys
import numpy as np
import scipy.io as sio
from pathlib import Path
import time
import boto3
from botocore.config import Config
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
# --- 1. K_Auth & AWS SETUP ---
try:
    from k_auth import AuthClient
    sec = AuthClient()
    AWS_AK = sec.get_auth("ACCESS_KEY")
    AWS_SK = sec.get_auth("AUTH_ACCESS_KEY")
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_AK
    os.environ["AWS_AUTH_ACCESS_KEY"] = AWS_SK
except Exception:
    print("[!] Warning: Could not load K_Auth. AWS Boto3 sync might fail.")
# Initialize AWS S3 Client
s3 = boto3.client(
    's3', 
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"), 
    AWS_AUTH_ACCESS_KEY=os.environ.get("AWS_AUTH_ACCESS_KEY"), 
    config=Config(signature_version='s3v4')
)
# --- 2. ENVIRONMENT SETUP ---
# Adjust this path based on where you attached the dataset in your Kaggle recordbook
INPUT_DIR = Path('/kaggle/input/nsvd-fusionv1') 
if not INPUT_DIR.exists():
    # Fallback to standard Kaggle working dir if not found, or modify as needed
    INPUT_DIR = Path('/kaggle/input/nsvd_fusionv1')
# Output dir for temp files (AWS downloads)
OUTPUT_DIR = Path('/kaggle/working')
TEMP_DIR = OUTPUT_DIR / 'temp_trialinfo'
TEMP_DIR.mkdir(parents=True, exist_ok=True)
# 73 Clean Cohort (Excluding 140117 and 204521)
SUBJECTS = [
    '100307', '102816', '104012', '105923', '106521', '108323', '109123',
    '111514', '112920', '113922', '116726', '125525', '133019', '146129',
    '149741', '151526', '156334', '158136', '162026', '162935', '164636',
    '166438', '169040', '172029', '175237', '175540', '177746', '182840',
    '185442', '189349', '191033', '191437', '191841', '192641', '195041',
    '198653', '200109', '205119', '212318', '212823', '214524', '223929',
    '248339', '250427', '255639', '257845', '283543', '293748', '352738',
    '353740', '358144', '406836', '433839', '500222', '512835', '555348',
    '568963', '581450', '599671', '601127', '660951', '662551', '665254',
    '667056', '679770', '680957', '706040', '707749', '715950', '725751',
    '735148', '783462', '814649'
]
def get_trialinfo(subj):
    """Downloads trialinfo from AWS to extract Semantic Face/Tool labels."""
    y_sem_all = []
    valid_mask_all = []
    
    for run in [6, 7]:
        key = f"HCP_1200/{subj}/MEG/Wrkmem/tmegpreproc/{subj}_MEG_{run}-Wrkmem_tmegpreproc_trialinfo.mat"
        local_path = TEMP_DIR / f"{subj}_run{run}_trialinfo.mat"
        
        if not local_path.exists():
            try:
                s3.download_file('hcp-openaccess', key, str(local_path))
            except Exception as e:
                print(f"[{subj}] Failed to download AWS data: {e}")
                return None, None
                
        mat = sio.loadmat(str(local_path), squeeze_me=True)
        trl = None
        if 'trlInfo' in mat:
            names = mat['trlInfo']['lockNames'].tolist()
            trls = mat['trlInfo']['lockTrl'].tolist()
            if isinstance(names, str): names, trls = [names], [trls]
            if isinstance(names, np.ndarray): names = names.tolist()
            for i, n in enumerate(names):
                if n == 'TIM':
                    trl = np.array(trls[i].tolist() if isinstance(trls, np.ndarray) else trls[i])
                    break
        
        if trl is None: return None, None
        
        # Col 3 = imgType (0=Fixation, 1=Face, 2=Tool)
        img_type = trl[:, 3]
        resp_time = trl[:, 15] if trl.shape[1] > 15 else np.zeros_len(img_type)
        is_correct = trl[:, 13] if trl.shape[1] > 13 else np.zeros_len(img_type)
        
        # 0 = Fixation, 1 = Face, 2 = Tool
        valid_mask = (img_type == 1) | (img_type == 2)
        y_sem = np.column_stack([img_type, resp_time, is_correct])
        
        valid_mask_all.append(valid_mask)
        y_sem_all.append(y_sem)
        
    return np.concatenate(valid_mask_all), np.concatenate(y_sem_all)
def run_per_subject_trajectory():
    print(f"Starting Subject-tier Temporal Trajectory across {len(SUBJECTS)} subjects...")
    
    window_samples = 12  # 48ms window
    step_samples = 3     # 12ms step
    
    subject_peaks = []
    
    for subj in SUBJECTS:
        print(f"\n--- Processing Subject {subj} ---")
        
        # Locate the X tensor in Kaggle input
        # Observation: adjust the glob/search depending on exactly how it's structured in the Kaggle directory
        x_paths = list(INPUT_DIR.rglob(f"{subj}_X.npy"))
        if not x_paths:
            print(f"[{subj}] Skipped: _X.npy not found in dataset.")
            continue
        x_path = x_paths[0]
        
        X = np.load(x_path) # (N, 68, 255) includes fixations
        
        # 1. Fetch Semantic Labels and Fixation Mask from AWS
        valid_mask, y_sem_full = get_trialinfo(subj)
        if valid_mask is None:
            continue
            
        # Ensure lengths match
        if len(valid_mask) > len(X):
            valid_mask = valid_mask[:len(X)]
            y_sem_full = y_sem_full[:len(X)]
        elif len(valid_mask) < len(X):
            X = X[:len(valid_mask)]
            
        # 2. Purge Fixations
        X_clean = X[valid_mask]
        y_sem = y_sem_full[valid_mask]
        
        # 3. Create Binary Target (Face = 1, Tool = 0)
        y_binary = (y_sem[:, 0] == 1).astype(int)
        
        print(f"[{subj}] Valid Face/Tool Trials: {len(X_clean)}")
        
        # --- The Sliding Window Trajectory ---
        cov_estimator = Covariances(estimator='oas')
        ts = TangentSpace(metric='riemann')
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        clf = LogisticRegression(C=1.0, max_iter=200, solver='lbfgs')
        
        n_timepoints = X_clean.shape[2]
        best_auc = -1
        best_ms = -999
        
        for i in range(0, n_timepoints - window_samples, step_samples):
            center_ms = (i + window_samples//2 - 125) * 4
            
            X_window = X_clean[:, :, i:i+window_samples]
            covs = cov_estimator.fit_transform(X_window)
            X_ts = ts.fit_transform(covs)
            
            aucs = []
            for train_idx, test_idx in cv.split(X_ts, y_binary):
                clf.fit(X_ts[train_idx], y_binary[train_idx])
                preds = clf.predict_proba(X_ts[test_idx])[:, 1]
                aucs.append(roc_auc_score(y_binary[test_idx], preds))
                
            mean_auc = np.mean(aucs)
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_ms = center_ms
                
        print(f"[{subj}] Peak Semantic Veto -> {best_ms} ms | AUC: {best_auc:.4f}")
        subject_peaks.append((subj, best_ms, best_auc))
        
    print("\n\n================================================")
    print("GLOBAL TEMPORAL LATENCY JITTER RESULTS")
    print("================================================")
    
    all_ms = [p[1] for p in subject_peaks]
    all_aucs = [p[2] for p in subject_peaks]
    
    print(f"Average Peak Timestamp: {np.mean(all_ms):.1f} ms")
    print(f"Standard Deviation of Jitter: +/- {np.std(all_ms):.1f} ms")
    print(f"Average Peak AUC per subject: {np.mean(all_aucs):.4f}")
    
    print("\nThis variance proves that using a rigid 300-500ms window destroys the signal!")
    print("We can now use these exact individual timestamps to align the micro-windows for the Spatial Rerun.")
if __name__ == "__main__":
    run_per_subject_trajectory()
