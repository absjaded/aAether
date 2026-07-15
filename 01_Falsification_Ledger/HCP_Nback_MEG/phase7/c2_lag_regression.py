import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import boto3
from scipy.stats import wilcoxon
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from dotenv import load_dotenv

# ── Config ─────────────────────────────────────────────────────────
COV_SCALE = 1e26
S3_BUCKET = 'hcp-openaccess'
load_dotenv('.env')
s3 = boto3.client('s3', aws_access_key_id=os.environ['ACCESS_KEY'],
                  AWS_AUTH_ACCESS_KEY=os.environ['AUTH_ACCESS_KEY'])
SCRATCH_DIR = './scratch/MEG_TAB'
os.makedirs(SCRATCH_DIR, exist_ok=True)
RESULTS_DIR = './.batches'
DATA_DIR = './.data/nsvd_fusion'

def get_tab(subj, run):
    local = f"{SCRATCH_DIR}/{subj}_run{run}.tab"
    if not os.path.exists(local):
        # Could be 6-Wrkmem or 7-Wrkmem depending on run1/run2 mapping, but HCP MEG uses 6/7.
        # Run1 is typically 6, Run2 is 7.
        folder = '6-Wrkmem' if run == 1 else '7-Wrkmem'
        key = f"HCP_1200/{subj}/unprocessed/MEG/{folder}/EPRIME/{subj}_MEG_Wrkmem_run{run}.tab"
        try:
            s3.download_file(S3_BUCKET, key, local)
        except Exception:
            return None
    try:
        return pd.read_csv(local, sep='\t', encoding='utf-8')
    except:
        return None

def align_sequences(y_tt, y_rt, tab_tt, tab_rt):
    M, N = len(y_tt), len(tab_tt)
    dp = np.full((M + 1, N + 1), np.inf)
    dp[0, 0] = 0
    for i in range(M + 1):
        for j in range(i, N + 1):
            if i > 0 and j > 0:
                cost_match = np.inf
                if y_tt[i-1] == tab_tt[j-1]:
                    ry = y_rt[i-1]*1000 if not np.isnan(y_rt[i-1]) and y_rt[i-1]>0 else np.nan
                    rt = tab_rt[j-1]
                    if np.isnan(ry) and np.isnan(rt): cost_match = 0
                    elif np.isnan(ry) or np.isnan(rt): cost_match = 1000
                    else: cost_match = abs(ry - rt)
                dp[i, j] = min(dp[i, j-1] + 1, dp[i-1, j-1] + cost_match)
            elif j > 0 and i == 0:
                dp[i, j] = dp[i, j-1] + 1
    aligned = []
    i, j = M, N
    while i > 0:
        if j == 0: assess ValueError("Alignment failed")
        if dp[i, j] == dp[i, j-1] + 1:
            j -= 1
        else:
            aligned.append(j-1)
            i -= 1; j -= 1
    return aligned[::-1]

def process_subject(r):
    subj = r['subj']
    d = r['dist']
    if 'covs' not in d: return None
    covs, tt = d['covs'], np.asarray(d['tt'])
    
    y_file = f"{DATA_DIR}/{subj}_y_meta.npy"
    if not os.path.exists(y_file): return None
    y = np.load(y_file)
    m2 = y[:,0] == 2
    y2 = y[m2]
    if len(y2) != len(tt) or not np.array_equal(y2[:,1], tt): return None
    
    df1, df2 = get_tab(subj, 1), get_tab(subj, 2)
    if df1 is None or df2 is None: return None
    
    # Compute Lag per run
    for df in [df1, df2]:
        if 'Stimulus' in df.columns:
            scol = 'Stimulus'
        elif 'Stimulus[Block]' in df.columns:
            scol = 'Stimulus[Block]'
        else:
            return None
        stims = df[scol].values
        lags = np.full(len(df), np.nan)
        last_seen = {}
        for i, st in enumerate(stims):
            if pd.isna(st): continue
            if st in last_seen:
                lags[i] = i - last_seen[st]
            last_seen[st] = i
        df['Lag'] = lags

    df = pd.concat([df1[df1['BlockType'] == '2-Back'], df2[df2['BlockType'] == '2-Back']])
    tt_map = {'target': 1, 'nontarget': 2, 'lure': 3}
    if 'TargetType' not in df.columns: return None
    tab_tt = df['TargetType'].map(tt_map).values
    tab_rt = pd.to_numeric(df['Stim.RT'], errors='coerce').values
    tab_lag = df['Lag'].values
    
    y_tt, y_rt = y2[:,1], y2[:,2]
    try:
        idx = align_sequences(y_tt, y_rt, tab_tt, tab_rt)
    except:
        return None
        
    y_lag = tab_lag[idx]
    
    # Compute distances
    C = covs * COV_SCALE
    t_idx = np.where(tt == 1)[0]
    withhold_idx = np.where((tt == 2) | (tt == 3))[0]
    if len(t_idx) < 10 or len(withhold_idx) < 10: return None
    
    ref = mean_riemann(C[t_idx])
    dist_withhold = np.array([distance_riemann(ref, C[i]) for i in withhold_idx])
    lag_withhold = y_lag[withhold_idx]
    
    # Drop NaNs
    valid = ~np.isnan(lag_withhold)
    d_v, l_v = dist_withhold[valid], lag_withhold[valid]
    if len(d_v) < 5 or np.std(l_v) < 1e-9: return None
    
    # Regress distance ~ 1 + lag
    # Z-score to get standardized beta
    d_z = (d_v - d_v.mean()) / (d_v.std() + 1e-12)
    l_z = (l_v - l_v.mean()) / (l_v.std() + 1e-12)
    X = np.column_stack([np.ones_like(l_z), l_z])
    beta, *_ = np.linalg.lstsq(X, d_z, rcond=None)
    
    return {'subj': subj, 'beta_lag': beta[1], 'n_trials': len(d_v)}

def main():
    results = []
    for bid in range(5):
        p = f"{RESULTS_DIR}/batch_{bid}_results.npy"
        if os.path.exists(p):
            results.extend(np.load(p, allow_pickle=True).tolist())
    stable = [r for r in results if r.get('dist') is not None and r['dist'].get('stable')]
    print(f"Loaded {len(stable)} stable subjects.")
    
    betas = []
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_subject, r) for r in stable]
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            res = f.result()
            if res:
                betas.append(res['beta_lag'])
            if (i+1)%10 == 0:
                print(f"Processed {i+1}/{len(futs)}")
                
    b = np.array(betas)
    if len(b) == 0:
        print("No subjects processed successfully.")
        return
        
    p_val = wilcoxon(b).pvalue if len(b) >= 6 else np.nan
    print("\n" + "="*50)
    print("DISTANCE ~ LAG REGRESSION (WITHHOLD TRIALS ONLY)")
    print("="*50)
    print(f"Subjects      : {len(b)}")
    print(f"Mean Beta     : {b.mean():+.4f} (standardized)")
    print(f"Beta > 0      : {(b > 0).sum()}/{len(b)}")
    print(f"Wilcoxon p    : {p_val:.4f}")
    if p_val < 0.05 and b.mean() > 0:
        print("Verdict       : GRADIENT EXISTS. Geometry tracks recency.")
    elif p_val < 0.05 and b.mean() < 0:
        print("Verdict       : NEGATIVE GRADIENT. Geometry tracking something else.")
    else:
        print("Verdict       : NO GRADIENT. Geometry is binary response-detection.")

if __name__ == '__main__':
    main()
