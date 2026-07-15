import os
import sys
import pandas as pd
import numpy as np

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:
    import subprocess
    print("Installing statsmodels...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

def compute_hedges_g(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / s_pooled
    
    # Hedges' g correction factor for small samples
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return d * correction

def main():
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "scratch/synthetic_data"
    csv_path = os.path.join(data_dir, "c02_baseline_scalars.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run c02_baseline.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter to only 2-back trials to compare Correct vs Incorrect (Friction)
    df_2b = df[df['condition'] == '2-back'].copy()
    
    # is_correct == 1 (Correct), is_correct == 0 (Incorrect)
    # We want to see if Incorrect differs from Correct
    
    print("\n--- ABLATION 1: GLMM + Hedges' g ---")
    print(f"Total 2-back Trials: {len(df_2b)}")
    print(f"Correct: {len(df_2b[df_2b['is_correct'] == 1])}")
    print(f"Incorrect: {len(df_2b[df_2b['is_correct'] == 0])}")
    print("-" * 40)
    
    # Fit Generalized Linear Mixed Model
    # rgd_scalar ~ is_correct with random intercept for subject_id
    md = smf.mixedlm("rgd_scalar ~ is_correct", df_2b, groups=df_2b["subject_id"])
    try:
        mdf = md.fit()
        print(mdf.summary())
        
        # Extract p-value for is_correct
        p_val = mdf.pvalues['is_correct']
        coef = mdf.params['is_correct']
        print(f"\nGLMM Coefficient for is_correct: {coef:.4f} (p = {p_val:.4e})")
    except Exception as e:
        print(f"GLMM fitting failed: {e}")
        
    # Calculate Cohort-tier Hedges' g (Pooled across all subjects instead of per-subject averaging)
    g_correct = df_2b[df_2b['is_correct'] == 1]['rgd_scalar'].values
    g_incorrect = df_2b[df_2b['is_correct'] == 0]['rgd_scalar'].values
    
    h_g = compute_hedges_g(g_incorrect, g_correct)
    print(f"\nCohort-tier Hedges' g (Incorrect vs Correct): {h_g:.4f}")
    
    # Compare to naive baseline (mean of per-subject Cohen's d)
    subject_ds = []
    for sid in df_2b['subject_id'].unique():
        s_df = df_2b[df_2b['subject_id'] == sid]
        s_c = s_df[s_df['is_correct'] == 1]['rgd_scalar'].values
        s_i = s_df[s_df['is_correct'] == 0]['rgd_scalar'].values
        if len(s_i) >= 2 and len(s_c) >= 2:
            s_pooled = np.sqrt(((len(s_i)-1)*np.var(s_i, ddof=1) + (len(s_c)-1)*np.var(s_c, ddof=1)) / (len(s_i)+len(s_c)-2))
            if s_pooled > 0:
                subject_ds.append((np.mean(s_i) - np.mean(s_c)) / s_pooled)
                
    if len(subject_ds) > 0:
        print(f"Flawed Baseline (Median of per-subject d): {np.median(subject_ds):.4f}")
        print(f"Flawed Baseline (Mean of per-subject d):   {np.mean(subject_ds):.4f}")
    else:
        print("Flawed Baseline: N/A (No subjects had >=2 error trials)")
        
    # VERDICT
    if h_g > 0.3 and p_val < 0.05:
        print("\nVERDICT: Priority 1 RESCUED the signal!")
    else:
        print("\nVERDICT: Priority 1 FAILED to rescue the signal. Proceed to Priority 2/3.")

if __name__ == "__main__":
    main()
