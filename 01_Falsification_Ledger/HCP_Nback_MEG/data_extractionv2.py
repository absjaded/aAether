"""
sensor_erp_gatekeeper.py
========================================================================
PURPOSE
  Decide ONE thing, cheaply, before spending money on re-localization:
  does the HCP 2-back paradigm carry a Lure-vs-Target mismatch signal
  AT ALL? We test it in SENSOR space, which bypasses all suspect in the
  source pipeline (per-epoch pca_flip sign, pooled LCMV covariance, onset
  crop drift) and recovers SNR by trial-averaging (which the single-trial
  Riemannian distance discarded).

WHAT IT DOES
  - Re-pulls only the light tmegpreproc TIM + trialinfo .mat from S3
    (no anatomy, no FreeSurfer, no raw 4D, no LCMV).
  - Re-derives Target/Lure labels jointly with the sensor epochs from the
    SAME files (so X and y are aligned by construction, not by trusting disk).
  - Computes a SIGN-AGNOSTIC, SNR-MATCHED elicited_response contrast:
      * real    = GFP( elicited_response_Target  - elicited_response_Lure )
      * null    = GFP( elicited_response_Target_halfA - elicited_response_Target_halfB )
    Both sides use equal trial counts, so the null controls for the
    "fewer/noisier trials inflate a difference wave" artifact that produced
    the earlier false hope. GFP (std across sensors) is polarity-free, so the
    pca_flip sign problem cannot exist here.
  - Aggregates per-subject deltas (real - null, in the mismatch window) and
    tests them at COHORT tier. No single-subject highs.

DECISION RULE (read AFTER it prints)
  * Cohort real > null, reliably, with most subjects positive
        -> the mismatch signal EXISTS in the data. The six nulls were the
           SOURCE PIPELINE suppressing it. Re-localization with the sign +
           covariance fixes is now worth the money.
  * Cohort real ~ null (no separation)
        -> the paradigm genuinely does not elicit a measurable single-session
           mismatch response. No localization fix rescues an absent signal.
           The paradigm verdict is earned, cheaply, and targeted collection
           (priced earlier) becomes the rational next move.

[Detail]: requires HCP S3 credentials in the same .env you already use. This
script cannot be run here (no creds / no S3); run it in your environment.
Sanity-check subject 1's printed shapes before trusting the cohort line.
========================================================================
"""

import os
import numpy as np
import scipy.io as sio
from io import BytesIO
import boto3
from scipy.stats import wilcoxon
from scipy.stats import wilcoxon

# ----------------------------------------------------------------------
# K_Auth & AWS SETUP
# ----------------------------------------------------------------------
try:
    from k_auth import AuthClient
    sec = AuthClient()
    os.environ["AWS_ACCESS_KEY_ID"] = sec.get_auth("ACCESS_KEY")
    os.environ["AWS_AUTH_ACCESS_KEY"] = sec.get_auth("AUTH_ACCESS_KEY")
except Exception:
    print("[!] Warning: Could not load K_Auth.")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BUCKET     = 'hcp-openaccess'
TASK       = 'Wrkmem'
RUNS       = [6, 7]
TARGET_FS  = 250

# windows in SECONDS, relative to stimulus onset (robust to onset-index drift)
BASELINE_WIN = (-0.20, 0.00)
MISMATCH_WIN = ( 0.25, 0.50)     # classic P300/N400 mismatch latency
CROP_WIN     = (-0.50, 0.50)

N_SUBSAMPLE  = 30                # SNR-matched real/null iterations
rng_state         = 42

# trialInfo column indices (verified against HCP design earlier)
COL_IMG, COL_MEM, COL_TGT = 3, 4, 5

rng = np.random.default_rng(rng_state)
s3 = boto3.client('s3',
                  aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                  AWS_AUTH_ACCESS_KEY=os.environ.get('AWS_AUTH_ACCESS_KEY'),
                  region_name='us-east-1')


# ----------------------------------------------------------------------
# LOADERS  (mirror the idioms that already worked in your two scripts)
# ----------------------------------------------------------------------
def load_run(subj, run):
    """Return sensor epochs X (n_trials, n_ch, n_time), decimated time axis,
    channel names, and the trialinfo matrix — all from the SAME run files."""
    base = f'HCP_1200/{subj}/MEG/{TASK}/tmegpreproc/{subj}_MEG_{run}-{TASK}_tmegpreproc'
    tim = sio.loadmat(BytesIO(s3.get_object(Bucket=BUCKET, Key=f'{base}_TIM.mat')['Body'].read()),
                      squeeze_me=True)['data']
    trl = sio.loadmat(BytesIO(s3.get_object(Bucket=BUCKET, Key=f'{base}_trialinfo.mat')['Body'].read()))

    ch_names = list(tim['label'].item())
    fs       = float(tim['fsample'].item())
    trials   = tim['trial'].item()
    time_vec = np.asarray(tim['time'].item()[0]).ravel()
    X = np.stack([np.asarray(trials[i]) for i in range(len(trials))])  # (N, ch, t)

    # crop to CROP_WIN then integer-decimate to TARGET_FS (deterministic)
    cmask = (time_vec >= CROP_WIN[0]) & (time_vec <= CROP_WIN[1])
    X, t = X[:, :, cmask], time_vec[cmask]
    step = int(round(fs / TARGET_FS))
    X, t = X[:, :, ::step], t[::step]

    trl_arr = trl['trlInfo']['lockTrl'][0][0][0][0]   # (N, 40) — extract_lures idiom
    return X, t, ch_names, np.asarray(trl_arr)


def load_subject(subj):
    Xs, ts, all_chs, ys = [], None, [], []
    for run in RUNS:
        X, t, ch, trl = load_run(subj, run)
        if X.shape[0] != trl.shape[0]:
            assess ValueError(f"align fail run {run}: {X.shape[0]} epochs vs {trl.shape[0]} trls")
        Xs.append(X); ys.append(trl); ts = t; all_chs.append(ch)
    
    # Intersect channels across runs
    common_chs = list(set(all_chs[0]).intersection(*[set(c) for c in all_chs]))
    common_chs.sort(key=lambda x: all_chs[0].index(x))
    
    for i in range(len(Xs)):
        idx = [all_chs[i].index(c) for c in common_chs]
        Xs[i] = Xs[i][:, idx, :]
        
    X = np.concatenate(Xs, axis=0)
    trl = np.vstack(ys)

    img, mem, tgt = trl[:, COL_IMG], trl[:, COL_MEM], trl[:, COL_TGT]
    fix = (img == 1) | (img == 2)               # purge fixation, jointly on X and labels
    X, mem, tgt = X[fix], mem[fix], tgt[fix]

    two = (mem == 2)
    Xt = X[two & (tgt == 1)]                     # Targets, 2-back
    Xl = X[two & (tgt == 3)]                     # Lures,   2-back
    return Xt, Xl, ts
load_subject._first = lambda c: c


# ----------------------------------------------------------------------
# CONTRAST  (sign-agnostic, SNR-matched)
# ----------------------------------------------------------------------
def baseline_correct(X, t):
    b = (t >= BASELINE_WIN[0]) & (t < BASELINE_WIN[1])
    return X - X[:, :, b].mean(axis=2, keepdims=True)

def gfp_window(elicited_response, t, win):
    m = (t >= win[0]) & (t < win[1])
    return elicited_response[:, m].std(axis=0).mean()       # GFP = std across sensors, then mean over window

def subject_delta(Xt, Xl, t):
    """Return (real_gfp, null_gfp) in the mismatch window, SNR-matched."""
    Xt, Xl = baseline_correct(Xt, t), baseline_correct(Xl, t)
    nt, nl = len(Xt), len(Xl)
    h = min(nt, nl) // 2
    if h < 5:                                     # too few trials to split safely
        return None
    real, null = [], []
    for _ in range(N_SUBSAMPLE):
        tp, lp = rng.permutation(nt), rng.permutation(nl)
        ev_t  = Xt[tp[:h]].mean(axis=0)
        ev_l  = Xl[lp[:h]].mean(axis=0)
        ev_a  = Xt[tp[:h]].mean(axis=0)           # null half-A (shares draw w/ real-target: conservative)
        ev_b  = Xt[tp[h:2*h]].mean(axis=0)        # null half-B (disjoint)
        real.append(gfp_window(ev_t - ev_l, t, MISMATCH_WIN))
        null.append(gfp_window(ev_a - ev_b, t, MISMATCH_WIN))
    return float(np.mean(real)), float(np.mean(null))


# ----------------------------------------------------------------------
# COHORT RUN
# ----------------------------------------------------------------------
def main():
    SUBJECTS = [
        '100307', '102816', '104012', '105923', '106521', '108323', '109123',
        '111514', '112920', '113922', '116726', '125525', '133019', '140117',
        '146129', '149741', '151526', '156334', '158136', '162026', '162935',
        '164636', '166438', '169040', '172029', '175237', '175540', '177746',
        '182840', '185442', '189349', '191033', '191437', '191841', '192641',
        '195041', '198653', '200109', '204521', '205119', '212318', '212823',
        '214524', '223929', '248339', '250427', '255639', '257845', '283543',
        '293748', '352738', '353740', '358144', '406836', '433839', '500222',
        '512835', '555348', '568963', '581450', '599671', '601127', '660951',
        '662551', '665254', '667056', '679770', '680957', '706040', '707749',
        '715950', '725751', '735148', '783462', '814649'
    ]
    print(f"Sensor-space ERP gatekeeper — {len(SUBJECTS)} subjects\n")

    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    start_time = time.time()
    
    def process_subj(args):
        i, subj = args
        try:
            t0 = time.time()
            Xt, Xl, t = load_subject(subj)
            res = subject_delta(Xt, Xl, t)
            elapsed = time.time() - t0
            if res is None:
                return i, subj, None, None, f"[{i+1}/75] [{subj}] skipped (too few trials) ({elapsed:.1f}s)"
            r, n = res
            
            if i == 0:
                msg = f"[{i+1}/75] [{subj}] SANITY  Targets={len(Xt)} Lures={len(Xl)} timepts={len(t)}  real_gfp={r:.3e} null_gfp={n:.3e} ({elapsed:.1f}s)"
            else:
                msg = f"[{i+1}/75] [{subj}] SUCCESS  real_gfp={r:.3e} null_gfp={n:.3e} ({elapsed:.1f}s)"
            return i, subj, r, n, msg
        except Exception as e:
            return i, subj, None, None, f"[{i+1}/75] [{subj}] failed: {e}"

    reals, nulls, kept = [], [], []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_subj, (i, s)) for i, s in enumerate(SUBJECTS)]
        for future in as_completed(futures):
            i, subj, r, n, msg = future.result()
            print(msg)
            if r is not None and n is not None:
                reals.append(r)
                nulls.append(n)
                kept.append(subj)

    reals, nulls = np.array(reals), np.array(nulls)
    delta = reals - nulls
    n_pos = int((delta > 0).sum())

    print("\n" + "=" * 64)
    print("COHORT SENSOR-SPACE elicited_response RESULT  (mismatch window)")
    print("=" * 64)
    print(f"Subjects analyzed:     {len(delta)}")
    print(f"real > null (L≠T):     {n_pos}/{len(delta)} ({100*n_pos/len(delta):.0f}%)")
    print(f"Mean real GFP:         {reals.mean():.4e}")
    print(f"Mean null GFP:         {nulls.mean():.4e}")
    print(f"Mean delta (real-null):{delta.mean():.4e}")
    if len(delta) > 5:
        w, p = wilcoxon(delta)
        print(f"Wilcoxon (delta>0):    p = {p:.4e}")
    print("=" * 64)
    print("READ: separation real>null across cohort -> signal EXISTS, source")
    print("pipeline was suppressing it -> fix & re-localize. No separation ->")
    print("paradigm verdict earned cheaply -> targeted-collection conversation.")

if __name__ == "__main__":
    main()