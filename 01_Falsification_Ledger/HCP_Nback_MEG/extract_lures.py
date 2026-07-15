import os
import boto3
import scipy.io
from io import BytesIO
import numpy as np
from dotenv import load_dotenv

load_dotenv('.env')

access_key = os.getenv('ACCESS_KEY')
auth_key = os.getenv('AUTH_ACCESS_KEY')

s3 = boto3.client(
    's3',
    aws_access_key_id=access_key,
    AWS_AUTH_ACCESS_KEY=auth_key,
    region_name='us-east-1'
)

bucket = 'hcp-openaccess'
data_dir = './.data/nsvd_fusion'

# Find subjects from the _y_semantic.npy files
subjects = []
for f in os.listdir(data_dir):
    if f.endswith('_y_semantic.npy'):
        subjects.append(f.split('_')[0])

print(f"Found {len(subjects)} subjects. Starting extraction...")

for subj in sorted(subjects):
    try:
        # We need run 1 and run 2
        all_trials = []
        for run in [6, 7]:
            key = f'HCP_1200/{subj}/MEG/Wrkmem/tmegpreproc/{subj}_MEG_{run}-Wrkmem_tmegpreproc_trialinfo.mat'
            obj = s3.get_object(Bucket=bucket, Key=key)
            mat_data = obj['Body'].read()
            mat = scipy.io.loadmat(BytesIO(mat_data))
            
            # The TIM array
            trl_tim = mat['trlInfo']['lockTrl'][0][0][0][0] # Shape (N, 40)
            all_trials.append(trl_tim)
            
        trl_arr = np.vstack(all_trials)
        
        imgType = trl_arr[:, 3]
        memoryType = trl_arr[:, 4]
        targetType = trl_arr[:, 5]
        isCorrect = trl_arr[:, 13]
        respTime = trl_arr[:, 15]
        
        # Filter fixations
        valid_mask = (imgType == 1) | (imgType == 2)
        
        imgType_clean = imgType[valid_mask]
        memoryType_clean = memoryType[valid_mask]
        targetType_clean = targetType[valid_mask]
        respTime_clean = respTime[valid_mask]
        isCorrect_clean = isCorrect[valid_mask]
        
        # Fill NaN targetType with 0 just to keep the array pure numeric if any exist
        targetType_clean = np.nan_to_num(targetType_clean, nan=0.0)
        
        y_semantic = np.column_stack((imgType_clean, memoryType_clean, targetType_clean, respTime_clean, isCorrect_clean))
        
        out_path = os.path.join(data_dir, f'{subj}_y_semantic.npy')
        
        # Verify length matches existing X array
        X_path = os.path.join(data_dir, f'{subj}_X.npy')
        if os.path.exists(X_path):
            X = np.load(X_path)
            if len(y_semantic) != len(X):
                print(f"[{subj}] Length mismatch: {len(y_semantic)} semantics vs {len(X)} epochs in X.npy")
            else:
                np.save(out_path, y_semantic)
                print(f"[{subj}] SUCCESS! Saved shape: {y_semantic.shape}")
        else:
             print(f"[{subj}] Skipped: _X.npy not found")
            
    except Exception as e:
        print(f"[{subj}] Failed: {e}")

print("Extraction complete!")
