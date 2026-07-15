import os
import json
import numpy as np

def generate_smart_synthetic():
    census_file = "results/c2_trial_census.json"
    out_dir = "scratch/synthetic_data"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(census_file, 'r') as f:
        census = json.load(f)
        
    print(f"Loaded census with {len(census['subjects'])} subjects.")
    
    # We will generate 10 subjects to keep local compute fast
    subjects = census["subjects"][:10]
    
    for sub in subjects:
        sid = sub["subject_id"]
        # +10 is to account for some fixation trials that get filtered out
        n_trials = sub["n_task"] + 10 
        
        # Base data: Random noise (simulated sensor data)
        X = np.random.randn(n_trials, 68, 255).astype(np.float32)
        
        y = np.zeros(n_trials, dtype=int)
        y_sem = np.zeros((n_trials, 3), dtype=int)
        
        # 1. First 10 trials are Fixation (y_sem[:, 0] == 0)
        idx = 0
        y_sem[idx:idx+10, 0] = 0 # Fixation
        idx += 10
        
        # 2. 0-back trials
        n_0back = sub["n_0back"]
        y[idx:idx+n_0back] = 0
        y_sem[idx:idx+n_0back, 0] = 1 # Task
        y_sem[idx:idx+sub["n_0back_correct"], 2] = 1 # Correct
        y_sem[idx+sub["n_0back_correct"]:idx+n_0back, 2] = 0 # Incorrect
        idx += n_0back
        
        # 3. 2-back correct trials
        n_2back_c = sub["n_2back_correct"]
        y[idx:idx+n_2back_c] = 1
        y_sem[idx:idx+n_2back_c, 0] = 1 # Task
        y_sem[idx:idx+n_2back_c, 2] = 1 # Correct
        
        # Inject subtle base signal to all 2-back correct so they differ slightly from 0-back
        X[idx:idx+n_2back_c, :, 132:175] *= 1.2 
        idx += n_2back_c
        
        # 4. 2-back incorrect trials (THE CRITICAL LOW-N GROUP)
        n_2back_i = sub["n_2back_incorrect"]
        if n_2back_i > 0:
            y[idx:idx+n_2back_i] = 1
            y_sem[idx:idx+n_2back_i, 0] = 1 # Task
            y_sem[idx:idx+n_2back_i, 2] = 0 # Incorrect
            
            # Inject a STRONG friction signal (power collapse / ERD) into the 12 Tier-1 ROIs late window
            tier_1 = [1, 2, 6, 25, 26, 29, 35, 36, 40, 59, 60, 63]
            X[idx:idx+n_2back_i, tier_1, 132:175] *= 0.5 # 50% power drop (ERD)
            idx += n_2back_i
            
        np.save(f"{out_dir}/{sid}_X.npy", X[:idx])
        np.save(f"{out_dir}/{sid}_y.npy", y[:idx])
        np.save(f"{out_dir}/{sid}_y_semantic.npy", y_sem[:idx])
        
        print(f"Generated {sid}: {n_0back} 0B, {n_2back_c} 2B-C, {n_2back_i} 2B-I")

if __name__ == "__main__":
    generate_smart_synthetic()
