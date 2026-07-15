"""
negative_control_matched.py
========================================================================
Dimensionality-matched negative control for the MVP-0 closeout.

THE PROBLEM IT FIXES
  The primary analysis runs on Tier-1+2 = 22 ROIs => ERPCovariances builds
  a ~44x44 matrix. The motor control runs on 6 ROIs => a ~12x12 matrix.
  Different size => different estimation stability => a clean motor null is
  CONFOUNDED with dimensionality. "Motor is null" might just mean "6 ROIs
  can't support the geometry," which proves nothing about spatial specificity.

THE FIX — match dimensionality, two complementary ways:

  ARM 1 (core) — DOWNWARD MATCH.
    Subsample Tier-1+2 down to the motor ROI count and compare WM-at-k vs
    motor-at-k at the SAME matrix size. If WM-k stays positive while motor-k
    sits at ~0, the motor null is BIOLOGICAL, not a dimensionality artifact.
    This is the direct refutation of the confound.

  ARM 2 (bonus) — SPATIAL-SPECIFICITY NULL.
    Draw random 22-ROI sets from non-WM/non-conflict cortex, build a null
    distribution of cohort d at FULL matched dimensionality, and locate the
    real Tier-1+2 d within it. Shows Tier-1+2 is specifically elevated, not
    just "any 22 ROIs give 0.33."

Reads the same _Xsrc.npy / _y_meta.npy / roi_names.npy. Self-contained:
the gap math is identical to mvp0_closeout.compute_erpcov_distances so the
numbers are directly comparable.
========================================================================
"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pyriemann.estimation import ERPCovariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

# ── Frozen config (identical to the closeout harness) ─────────────────────────
DATA_DIR     = './.data/nsvd_fusion'
FS           = 250
ONSET_SAMPLE = 127
MISMATCH_WIN = (0.25, 0.50)
BASELINE_WIN = (-0.20, 0.00)

K_SUBSETS    = 40       # random draws per matched arm (assess for tighter nulls)
N_NULL22     = 30       # random 22-ROI non-WM sets for Arm 2 (slower; lower to speed up)
rng_state         = 42

TIER12 = {
    'caudalmiddlefrontal', 'rostralmiddlefrontal', 'superiorfrontal',
    'inferiorparietal', 'supramarginal', 'superiorparietal',
    'caudalanteriorcingulate', 'rostralanteriorcingulate',
    'parsopercularis', 'parstriangularis', 'insula',
}
MOTOR = {'precentral', 'postcentral', 'paracentral'}

SUBJECTS = [
    '100307','102816','104012','105923','106521','108323','109123','111514','112920','113922',
    '116726','125525','133019','140117','146129','149741','151526','156334','158136','162026',
    '162935','164636','166438','169040','172029','175237','175540','177746','182840','185442',
    '189349','191033','191437','191841','192641','195041','198653','200109','204521','205119',
    '212318','212823','214524','223929','248339','250427','255639','257845','283543','293748',
    '352738','353740','358144','406836','433839','500222','512835','555348','568963','581450',
    '599671','601127','660951','662551','665254','667056','679770','680957','706040','707749',
    '715950','725751','735148','783462','814649',
]

# ── Gap math — identical to the harness (so numbers are comparable) ───────────

def time_axis(n): return (np.arange(n) - ONSET_SAMPLE) / FS
def win_mask(t, w): return (t >= w[0]) & (t < w[1])

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

def subject_gap(X, y, roi_idx):
    """Lure-vs-Target Cohen's d for one subject, one ROI set. Identical
    estimator + LOO discipline as the primary harness gap."""
    m2 = y[:, 0] == 2
    X2, tt = X[m2], y[m2, 1].astype(int)
    t_idx, l_idx = np.where(tt == 1)[0], np.where(tt == 3)[0]
    if len(t_idx) < 6 or len(l_idx) < 4:
        return None
    t = time_axis(X2.shape[2])
    Xw = baseline_correct(X2, t)[:, roi_idx][:, :, win_mask(t, MISMATCH_WIN)]
    covs = condition_spd(ERPCovariances(classes=[1], estimator='lwf').fit_transform(Xw, tt))
    d_t = np.empty(len(t_idx))
    for k, i in enumerate(t_idx):
        d_t[k] = distance_riemann(mean_riemann(covs[np.delete(t_idx, k)]), covs[i])
    ref = mean_riemann(covs[t_idx])
    d_l = np.array([distance_riemann(ref, covs[i]) for i in l_idx])
    return cohens_d(d_l, d_t)

def cohort_d(gaps):
    g = np.array([x for x in gaps if x is not None])
    return g.mean(), int((g > 0).sum()), len(g)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(rng_state)
    roi_names = np.load(os.path.join(DATA_DIR, 'roi_names.npy'))
    base = lambda n: str(n).rsplit('-', 1)[0]
    tier12_idx = [i for i, n in enumerate(roi_names) if base(n) in TIER12]
    motor_idx  = [i for i, n in enumerate(roi_names) if base(n) in MOTOR]
    nonwm_idx  = [i for i, n in enumerate(roi_names) if base(n) not in TIER12]
    k = len(motor_idx)

    print('=' * 66)
    print('DIMENSIONALITY-MATCHED NEGATIVE CONTROL')
    print('=' * 66)
    print(f'Tier-1+2: {len(tier12_idx)} ROIs | Motor: {k} ROIs | non-WM pool: {len(nonwm_idx)}')

    # Pre-draw ROI sets ONCE (same draws applied to all subject) ─────────────
    wm_k_sets    = [list(rng.choice(tier12_idx, size=k, replace=False)) for _ in range(K_SUBSETS)]
    null22_sets  = [list(rng.choice(nonwm_idx, size=len(tier12_idx), replace=False))
                    for _ in range(N_NULL22)]

    # Accumulators: name -> list-of-per-subject-gaps
    acc = {'tier12': [], 'motor': []}
    for j in range(K_SUBSETS):   acc[f'wm{k}_{j}'] = []
    for j in range(N_NULL22):    acc[f'null22_{j}'] = []

    print('\nScanning subjects (each loaded once)...')
    for si, subj in enumerate(SUBJECTS):
        try:
            X = np.load(os.path.join(DATA_DIR, f'{subj}_Xsrc.npy'))
            y = np.load(os.path.join(DATA_DIR, f'{subj}_y_meta.npy'))
        except FileNotFoundError:
            continue
        acc['tier12'].append(subject_gap(X, y, tier12_idx))
        acc['motor'].append(subject_gap(X, y, motor_idx))
        for j, s in enumerate(wm_k_sets):
            acc[f'wm{k}_{j}'].append(subject_gap(X, y, s))
        for j, s in enumerate(null22_sets):
            acc[f'null22_{j}'].append(subject_gap(X, y, s))
        if (si + 1) % 10 == 0:
            print(f'  {si+1}/{len(SUBJECTS)}', end='\r')
    print()

    # Aggregate ────────────────────────────────────────────────────────────────
    d_tier12, pos_t, n_t = cohort_d(acc['tier12'])
    d_motor,  pos_m, n_m = cohort_d(acc['motor'])
    wm_k_ds  = np.array([cohort_d(acc[f'wm{k}_{j}'])[0] for j in range(K_SUBSETS)])
    null_ds  = np.array([cohort_d(acc[f'null22_{j}'])[0] for j in range(N_NULL22)])

    print('\n' + '=' * 66)
    print('ARM 1 — DOWNWARD MATCH  (the direct fix)')
    print('=' * 66)
    print(f'  Tier-1+2 (full, {len(tier12_idx)} ROIs)   d = {d_tier12:+.4f}   '
          f'({pos_t}/{n_t} positive)   [context]')
    print(f'  WM subsampled to {k} ROIs        d = {wm_k_ds.mean():+.4f} '
          f'± {wm_k_ds.std():.4f}  (mean over {K_SUBSETS} draws)')
    print(f'  Motor ({k} ROIs)                 d = {d_motor:+.4f}   '
          f'({pos_m}/{n_m} positive)')
    wm_k_positive = wm_k_ds.mean() > 0.10
    motor_null    = abs(d_motor) < 0.12
    matched_ok    = wm_k_positive and motor_null
    print(f'\n  WM retains signal at {k} ROIs : {"YES" if wm_k_positive else "NO"}')
    print(f'  Motor null at {k} ROIs        : {"YES" if motor_null else "NO"}')
    if matched_ok:
        print('  VERDICT: motor null is BIOLOGICAL, not a dimensionality artifact. ✓')
    elif not wm_k_positive:
        print('  VERDICT: WM also collapses at low ROI count => the downward match is')
        print('           UNINFORMATIVE (6 ROIs too few). Rely on Arm 2 for specificity.')
    else:
        print('  VERDICT: motor is NOT null at matched dimensionality => the primary')
        print('           effect may carry a motor/response component. INVESTIGATE.')

    print('\n' + '=' * 66)
    print('ARM 2 — SPATIAL-SPECIFICITY NULL  (random 22-ROI non-WM sets)')
    print('=' * 66)
    emp_p = float(np.mean(np.abs(null_ds) >= abs(d_tier12)))
    print(f'  Real Tier-1+2 d              : {d_tier12:+.4f}')
    print(f'  Non-WM null d (mean ± SD)    : {null_ds.mean():+.4f} ± {null_ds.std():.4f}')
    print(f'  Null range                   : [{null_ds.min():+.4f}, {null_ds.max():+.4f}]')
    print(f'  Empirical p (|null| ≥ |real|): {emp_p:.4f}  over {N_NULL22} sets')
    spatial_ok = emp_p < 0.05
    print(f'  Tier-1+2 specifically elevated: {"YES ✓" if spatial_ok else "NO"}')

    print('\n' + '=' * 66)
    print('CONTROL SUMMARY')
    print('=' * 66)
    print(f'  Dimensionality-matched motor null : {"PASS ✓" if matched_ok else "CHECK"}')
    print(f'  Spatial specificity               : {"PASS ✓" if spatial_ok else "CHECK"}')
    print('  A negative control now controls for ROI count, not just region.')


if __name__ == '__main__':
    main()
