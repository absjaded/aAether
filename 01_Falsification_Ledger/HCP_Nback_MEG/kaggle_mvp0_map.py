"""
kaggle_mvp0_map.py  —  MAP PHASE
========================================================================
Run this in 5 separate Kaggle CPU recordbooks. Each processes 15 subjects
and saves per-subject results (including covariance matrices needed for
the permutation) to /kaggle/working/.

SETUP:
  1. Upload your nsvd_fusion/ folder as a Kaggle dataset called
     "nsvd-fusion" (or change DATA_DIR below to match your dataset name).
  2. In each of the 5 recordbooks, change BATCH_ID to 0, 1, 2, 3, or 4.
  3. Run all. Download batch_N_results.npy from each recordbook's output.
  4. Run kaggle_mvp0_reduce.py locally on all 5 batch files.

MLOps records:
  - One subject loaded at a time; X deleted immediately after covs extracted.
  - Per-subject checkpoint: skips already-completed subjects on rerun.
  - No multiprocessing / n_jobs (Kaggle CPU = shared; don't peg all cores).
  - Covariance matrices saved per subject for reduce-phase permutation.
========================================================================
"""

BATCH_ID = 0          # ← CHANGE THIS: 0, 1, 2, 3, or 4

import os, gc, warnings, sys
warnings.filterwarnings('ignore')

import numpy as np
from pyriemann.estimation import ERPCovariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

# ── Config (frozen — do not touch between batches) ─────────────────────────
DATA_DIR = '/kaggle/input/datasets/hansmax/nsvd-fusion'     # your Kaggle dataset path
OUT_DIR  = '/kaggle/working'
os.makedirs(OUT_DIR, exist_ok=True)

FS           = 250
ONSET_SAMPLE = 127
MISMATCH_WIN = (0.25, 0.50)
BASELINE_WIN = (-0.20, 0.00)
rng_state         = 42

# Negative control draw counts — balanced for Kaggle CPU time budget
K_SUBSETS = 40   # ARM 1: WM subsampled to motor-ROI count
N_NULL22  = 30   # ARM 2: random non-WM 22-ROI null sets

TIER12 = {
    'caudalmiddlefrontal', 'rostralmiddlefrontal', 'superiorfrontal',
    'inferiorparietal', 'supramarginal', 'superiorparietal',
    'caudalanteriorcingulate', 'rostralanteriorcingulate',
    'parsopercularis', 'parstriangularis', 'insula',
}
MOTOR = {'precentral', 'postcentral', 'paracentral'}

ALL_SUBJECTS = [
    '100307','102816','104012','105923','106521','108323','109123',
    '111514','112920','113922','116726','125525','133019','140117',
    '146129','149741','151526','156334','158136','162026','162935',
    '164636','166438','169040','172029','175237','175540','177746',
    '182840','185442','189349','191033','191437','191841','192641',
    '195041','198653','200109','204521','205119','212318','212823',
    '214524','223929','248339','250427','255639','257845','283543',
    '293748','352738','353740','358144','406836','433839','500222',
    '512835','555348','568963','581450','599671','601127','660951',
    '662551','665254','667056','679770','680957','706040','707749',
    '715950','725751','735148','783462','814649',
]

BATCH_SIZE = 15
SUBJECTS   = ALL_SUBJECTS[BATCH_ID * BATCH_SIZE : (BATCH_ID + 1) * BATCH_SIZE]

# ── Signal utilities ────────────────────────────────────────────────────────

def time_axis(n):
    return (np.arange(n) - ONSET_SAMPLE) / FS

def win_mask(t, w):
    return (t >= w[0]) & (t < w[1])

def baseline_correct(X, t):
    b = win_mask(t, BASELINE_WIN)
    return X - X[:, :, b].mean(axis=2, keepdims=True)

def condition_spd(C, eps=1e-6):
    C = (C + C.transpose(0, 2, 1)) / 2
    for i in range(C.shape[0]):
        tr = np.trace(C[i])
        C[i] += np.eye(C.shape[-1]) * (tr * eps if tr > 0 else eps)
    return C

def cohens_d(a, b):
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) + 1e-12
    return float((np.mean(a) - np.mean(b)) / pooled)

def residualize_rt(rgd, rt):
    valid = ~np.isnan(rt)
    resid = rgd.copy()
    if valid.sum() >= 3:
        coef = np.polyfit(rt[valid], rgd[valid], 1)
        resid[valid] = rgd[valid] - np.polyval(coef, rt[valid])
    return resid

def erpcov(X2, tt, roi_idx):
    """Compute ERPCov matrices for one subject, one ROI set."""
    t  = time_axis(X2.shape[2])
    Xw = baseline_correct(X2, t)[:, roi_idx][:, :, win_mask(t, MISMATCH_WIN)]
    Xw = Xw * 1e13                          # ← Anomaly-8 scale fix (canon)
    return condition_spd(
        ERPCovariances(classes=[1], estimator='lwf').fit_transform(Xw, tt)
    )

def loo_gap(covs, tt):
    """
    Full LOO distance dict for primary analysis.
    Returns None if too few trials. Includes split-half stability check.
    """
    t_idx  = np.where(tt == 1)[0]
    nt_idx = np.where(tt == 2)[0]
    l_idx  = np.where(tt == 3)[0]
    if len(t_idx) < 6 or len(l_idx) < 4:
        return None

    # Split-half stability (§4 Guard 2)
    rng  = np.random.default_rng(rng_state)
    perm = rng.permutation(len(t_idx))
    half = len(t_idx) // 2
    ref_h, score_h = t_idx[perm[:half]], t_idx[perm[half:]]
    C_sh  = mean_riemann(covs[ref_h])
    d_sh  = np.array([distance_riemann(C_sh, covs[i]) for i in score_h])
    all_t = np.arange(len(t_idx))
    d_loo_sh = np.array([
        distance_riemann(
            mean_riemann(covs[t_idx[np.setdiff1d(all_t, perm[half + k])]]),
            covs[si]
        ) for k, si in enumerate(score_h)
    ])
    pooled_sd  = np.std(np.concatenate([d_sh, d_loo_sh])) + 1e-12
    delta_frac = abs(d_sh.mean() - d_loo_sh.mean()) / pooled_sd
    stable     = bool(delta_frac < 0.50)

    # LOO distances for Targets
    d_target = np.array([
        distance_riemann(mean_riemann(covs[np.delete(t_idx, k)]), covs[i])
        for k, i in enumerate(t_idx)
    ])
    ref_all = mean_riemann(covs[t_idx])
    d_nt    = np.array([distance_riemann(ref_all, covs[i]) for i in nt_idx])
    d_lure  = np.array([distance_riemann(ref_all, covs[i]) for i in l_idx])

    return {
        'stable': stable, 'delta_frac': float(delta_frac),
        'd_target': d_target, 'd_nt': d_nt, 'd_lure': d_lure,
        'gap': cohens_d(d_lure, d_target),
        't_idx': t_idx, 'l_idx': l_idx,
    }

def quick_gap(covs, tt):
    """Minimal gap for negative-control draws. No full dict needed."""
    t_idx = np.where(tt == 1)[0]
    l_idx = np.where(tt == 3)[0]
    if len(t_idx) < 6 or len(l_idx) < 4:
        return None
    d_t = np.array([
        distance_riemann(mean_riemann(covs[np.delete(t_idx, k)]), covs[i])
        for k, i in enumerate(t_idx)
    ])
    ref = mean_riemann(covs[t_idx])
    d_l = np.array([distance_riemann(ref, covs[i]) for i in l_idx])
    return cohens_d(d_l, d_t)

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # Load ROI names once
    roi_names  = np.load(os.path.join(DATA_DIR, 'roi_names.npy'))
    base_name  = lambda n: str(n).rsplit('-', 1)[0]
    tier12_idx = [i for i, n in enumerate(roi_names) if base_name(n) in TIER12]
    motor_idx  = [i for i, n in enumerate(roi_names) if base_name(n) in MOTOR]
    nonwm_idx  = [i for i, n in enumerate(roi_names) if base_name(n) not in TIER12]
    k          = len(motor_idx)

    # Pre-draw negative-control ROI sets ONCE — deterministic, identical across batches
    rng_draw  = np.random.default_rng(rng_state)
    wm_k_sets = [list(rng_draw.choice(tier12_idx, size=k, replace=False))
                 for _ in range(K_SUBSETS)]
    null22_sets = [list(rng_draw.choice(nonwm_idx, size=len(tier12_idx), replace=False))
                   for _ in range(N_NULL22)]

    print('=' * 62)
    print(f'MVP-0 MAP PHASE  —  Batch {BATCH_ID}  ({len(SUBJECTS)} subjects)')
    print('=' * 62)
    print(f'Tier-1+2: {len(tier12_idx)} ROIs | Motor: {k} | non-WM pool: {len(nonwm_idx)}')
    print(f'ARM 1 draws: {K_SUBSETS} | ARM 2 draws: {N_NULL22}')
    print()

    batch_results = []

    for si, subj in enumerate(SUBJECTS):
        ckpt = os.path.join(OUT_DIR, f'subj_{subj}_result.npy')

        # Checkpoint: skip if already done
        if os.path.exists(ckpt):
            print(f'[{si+1}/{len(SUBJECTS)}] {subj}: checkpoint found — skipping')
            batch_results.append(np.load(ckpt, allow_pickle=True).item())
            continue

        print(f'\n[{si+1}/{len(SUBJECTS)}] {subj}')

        try:
            X = np.load(os.path.join(DATA_DIR, f'{subj}_Xsrc.npy'))   # (N,68,T)
            y = np.load(os.path.join(DATA_DIR, f'{subj}_y_meta.npy')) # (N,4)
        except FileNotFoundError:
            print(f'  [SKIP] file not found')
            continue

        # Filter to 2-back only, then free raw array
        m2  = y[:, 0] == 2
        X2  = X[m2].copy()
        tt  = y[m2, 1].astype(int)
        rt  = y[m2, 2]
        del X; gc.collect()

        # ── PRIMARY: Tier-1+2 ERPCov ──────────────────────────────────
        print(f'  [1/3] Primary ERPCov (Tier-1+2)...', end=' ', flush=True)
        covs_main = erpcov(X2, tt, tier12_idx)
        dist = loo_gap(covs_main, tt)
        if dist is not None:
            l_idx_rt = dist['l_idx']
            t_idx_rt = dist['t_idx']
            dist['gap_rt'] = cohens_d(
                residualize_rt(dist['d_lure'],   rt[l_idx_rt]),
                residualize_rt(dist['d_target'], rt[t_idx_rt]),
            )
            dist['covs'] = covs_main   # kept for reduce-phase permutation
            dist['tt']   = tt
        print(f"d={dist['gap']:+.3f}" if dist else 'too few trials')

        # ── MOTOR CONTROL ─────────────────────────────────────────────
        print(f'  [2/3] Motor control...', end=' ', flush=True)
        covs_m  = erpcov(X2, tt, motor_idx)
        motor_g = quick_gap(covs_m, tt)
        # covs_m KEPT (not deleted) — needed for per-trial motor null in reduce
        print(f'd={motor_g:+.3f}' if motor_g is not None else 'skip')

        # ── NEGATIVE CONTROL: ARM 1 (WM → k ROIs) ─────────────────────
        print(f'  [3/3] Negative control ARM 1 ({K_SUBSETS} draws) + ARM 2 ({N_NULL22} draws)...',
              flush=True)
        wm_k_gaps, null22_gaps = [], []
        for j, rset in enumerate(wm_k_sets):
            covs_j = erpcov(X2, tt, rset)
            wm_k_gaps.append(quick_gap(covs_j, tt))
            del covs_j
        for j, rset in enumerate(null22_sets):
            covs_j = erpcov(X2, tt, rset)
            null22_gaps.append(quick_gap(covs_j, tt))
            del covs_j
        gc.collect()

        del X2; gc.collect()

        result = {
            'subj'       : subj,
            'dist'       : dist,        # full primary result (covs inside for permutation)
            'motor_gap'  : motor_g,
            'covs_m'     : covs_m,      # motor covariances — for per-trial motor null
            'wm_k_gaps'  : wm_k_gaps,
            'null22_gaps': null22_gaps,
            'rt'         : rt,          # 2-back RT column (NaN = no response)
            'ic'         : y[m2, 3],    # 2-back isCorrect column
            'tt'         : tt,          # 2-back condition labels
        }
        np.save(ckpt, result)
        batch_results.append(result)
        print(f'  Saved → {ckpt}')

    # Save full batch file
    out = os.path.join(OUT_DIR, f'batch_{BATCH_ID}_results.npy')
    np.save(out, batch_results)
    print(f'\nBatch {BATCH_ID} complete — {len(batch_results)} subjects saved → {out}')
    print('Download this file, then run kaggle_mvp0_reduce.py locally.')


if __name__ == '__main__':
    main()
