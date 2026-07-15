"""
dual_contrast.py
========================================================================
Runs on the SIGN-CORRECTED source arrays (_Xsrc.npy) produced by the
patched extractor. Two contrasts, in parallel, answering two questions:

  CONTRAST A — elicited_response-source GFP  (the validation arm)
    Does the cohort-tier mismatch signal — proven in sensor space —
    SURVIVE the corrected source localization? Sign-agnostic GFP, SNR-
    matched null, same design as the sensor gatekeeper but on ROIs.
    PASS = real > null across cohort. Confirms the localization fixes
    preserved the signal (i.e., mean_flip un-cancelled the elicited_response).

  CONTRAST B — Covariance-source RGD  (the revival arm)
    Does the RIEMANNIAN framework — six-times null on contaminated data —
    come back to life once the sign is fixed? ERPCovariances on Tier-1+2
    ROIs, LOO distance-to-Target-mean, Lure-vs-Target gap, cohort tier.
    This decides whether the geometry-based structural advantage was WRONG or just STARVED
    of a clean signal. A revived gap here means the framework was starved.

DECISION TABLE
    A pass, B pass : best case. Signal localizes AND the geometry revives.
                     The Riemannian structural advantage is alive; proceed to formalize B.
    A pass, B null : signal is real and localizes, but it is an elicited_response-
                     AMPLITUDE phenomenon the covariance geometry can't see.
                     Pivot the representation to elicited_response/ERP-based, not pure
                     covariance. Thesis intact, method changes.
    A null         : the localization fixes did NOT recover the signal in
                     source space. Stay in sensor space for the existence
                     proof; debug the source pipeline separately. (Do not
                     read B if A is null — B inherits the same source data.)

Reads:  {DATA_DIR}/{subj}_Xsrc.npy   (n_trials, 68, n_time), sign-corrected
        {DATA_DIR}/{subj}_y_meta.npy (n_trials, 4) [memoryType,targetType,respTime,isCorrect]
        {DATA_DIR}/roi_names.npy     (68,) aparc label names
========================================================================
"""

import os
import numpy as np
from scipy.stats import wilcoxon

from pyriemann.estimation import ERPCovariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

# ----------------------------------------------------------------------
DATA_DIR     = './.data/nsvd_fusion'
FS           = 250
ONSET_SAMPLE = 127                      # confirmed: t=0 at index 127
BASELINE_WIN = (-0.20, 0.00)
MISMATCH_WIN = ( 0.25, 0.50)
N_SUBSAMPLE  = 30
rng_state         = 42
rng = np.random.default_rng(rng_state)

# Tier-1 (WM maintenance) + Tier-2 (conflict / mismatch detection), both hemispheres
TIER12 = {
    'caudalmiddlefrontal', 'rostralmiddlefrontal', 'superiorfrontal',
    'inferiorparietal', 'supramarginal', 'superiorparietal',          # Tier 1
    'caudalanteriorcingulate', 'rostralanteriorcingulate',
    'parsopercularis', 'parstriangularis', 'insula',                  # Tier 2
}


def time_axis(n):
    return (np.arange(n) - ONSET_SAMPLE) / FS

def win_mask(t, win):
    return (t >= win[0]) & (t < win[1])

def baseline_correct(X, t):
    b = win_mask(t, BASELINE_WIN)
    return X - X[:, :, b].mean(axis=2, keepdims=True)

def condition_spd(C, eps=1e-6):
    C = (C + C.transpose(0, 2, 1)) / 2
    for i in range(C.shape[0]):
        tr = np.trace(C[i])
        C[i] += np.eye(C.shape[-1]) * (tr * eps if tr > 0 else eps)
    return C


def load_subject(subj):
    X = np.load(os.path.join(DATA_DIR, f"{subj}_Xsrc.npy"))      # (N,68,T)
    y = np.load(os.path.join(DATA_DIR, f"{subj}_y_meta.npy"))    # (N,4)
    mem, tt = y[:, 0], y[:, 1]
    two = (mem == 2)
    return X[two], tt[two]                                        # 2-back only


# ----------------------------------------------------------------------
# CONTRAST A — elicited_response-source GFP, SNR-matched (sign-agnostic)
# ----------------------------------------------------------------------
def contrast_a(X, tt, t):
    Xt = baseline_correct(X[tt == 1], t)
    Xl = baseline_correct(X[tt == 3], t)
    nt, nl = len(Xt), len(Xl)
    h = min(nt, nl) // 2
    if h < 5:
        return None
    m = win_mask(t, MISMATCH_WIN)
    real, null = [], []
    for _ in range(N_SUBSAMPLE):
        tp, lp = rng.permutation(nt), rng.permutation(nl)
        ev_t = Xt[tp[:h]].mean(0); ev_l = Xl[lp[:h]].mean(0)
        ev_a = Xt[tp[:h]].mean(0); ev_b = Xt[tp[h:2 * h]].mean(0)
        real.append((ev_t - ev_l)[:, m].std(0).mean())
        null.append((ev_a - ev_b)[:, m].std(0).mean())
    return float(np.mean(real)), float(np.mean(null))


# ----------------------------------------------------------------------
# CONTRAST B — covariance-source RGD, ERPCovariances + LOO (the revival test)
# ----------------------------------------------------------------------
def contrast_b(X, tt, t, roi_idx):
    m = win_mask(t, MISMATCH_WIN)
    Xw = X[:, roi_idx][:, :, m]                          # (N, n_roi, n_win)
    y = tt.astype(int)
    t_idx = np.where(tt == 1)[0]
    l_idx = np.where(tt == 3)[0]
    if len(t_idx) < 6 or len(l_idx) < 6:
        return None

    # ERPCovariances: fold the Target elicited_response prototype into each trial's covariance,
    # which is what rescued the signal on contaminated data. Prototype = Target class.
    erp = ERPCovariances(classes=[1], estimator='lwf')
    covs = condition_spd(erp.fit_transform(Xw, y))

    # Target distances: leave-one-out (else biased small => fake gap)
    d_t = np.empty(len(t_idx))
    for k, i in enumerate(t_idx):
        ref = mean_riemann(covs[np.delete(t_idx, k)])
        d_t[k] = distance_riemann(ref, covs[i])
    ref_all = mean_riemann(covs[t_idx])
    d_l = np.array([distance_riemann(ref_all, covs[i]) for i in l_idx])

    pooled = np.sqrt((d_t.var(ddof=1) + d_l.var(ddof=1)) / 2)
    return float((d_l.mean() - d_t.mean()) / pooled) if pooled > 0 else 0.0


# ----------------------------------------------------------------------
def main():
    subjects = sorted({f.split('_')[0] for f in os.listdir(DATA_DIR)
                       if f.endswith('_Xsrc.npy')})
    roi_names = np.load(os.path.join(DATA_DIR, 'roi_names.npy'))
    roi_idx = [i for i, n in enumerate(roi_names)
               if str(n).rsplit('-', 1)[0] in TIER12]
    print(f"Dual contrast — {len(subjects)} subjects | Tier-1+2 ROIs: {len(roi_idx)}\n")

    A_real, A_null, B_d, kept = [], [], [], []
    for i, subj in enumerate(subjects):
        try:
            X, tt = load_subject(subj)
            t = time_axis(X.shape[2])
            a = contrast_a(X, tt, t)
            b = contrast_b(X, tt, t, roi_idx)
            if a is None or b is None:
                print(f"[{subj}] skipped (too few trials)"); continue
            A_real.append(a[0]); A_null.append(a[1]); B_d.append(b); kept.append(subj)
            if i == 0:
                print(f"[{subj}] SANITY  A_real={a[0]:.3e} A_null={a[1]:.3e}  B_d={b:+.3f}")
        except Exception as e:
            print(f"[{subj}] failed: {e}")

    A_real, A_null, B_d = map(np.array, (A_real, A_null, B_d))
    A_delta = A_real - A_null

    print("\n" + "=" * 64)
    print("CONTRAST A — elicited_response-SOURCE GFP  (does the signal survive localization?)")
    print("=" * 64)
    na = int((A_delta > 0).sum())
    print(f"  subjects:            {len(A_delta)}")
    print(f"  real > null:         {na}/{len(A_delta)} ({100*na/len(A_delta):.0f}%)")
    print(f"  mean delta:          {A_delta.mean():.4e}")
    if len(A_delta) > 5:
        print(f"  Wilcoxon p:          {wilcoxon(A_delta).pvalue:.4e}")

    print("\n" + "=" * 64)
    print("CONTRAST B — COVARIANCE-SOURCE RGD  (does the geometry revive?)")
    print("=" * 64)
    nb = int((B_d > 0).sum())
    print(f"  subjects:            {len(B_d)}")
    print(f"  Lure>Target gap >0:  {nb}/{len(B_d)} ({100*nb/len(B_d):.0f}%)")
    print(f"  mean cohort Cohen d: {B_d.mean():+.4f}")
    if len(B_d) > 5:
        print(f"  Wilcoxon p:          {wilcoxon(B_d).pvalue:.4e}")
    print("=" * 64)
    print("READ: A is the existence proof in source space. B decides whether the")
    print("Riemannian structural advantage was WRONG (B null) or merely STARVED (B revived).")


if __name__ == "__main__":
    main()