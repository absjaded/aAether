"""
mvp0_closeout.py
========================================================================
Formal MVP-0 closeout harness.

Uses the ERPCovariances backbone (aether_fusion.py, confirmed d=0.33 at
n=75) as the core representation, and adds the five Document 2 guards
that convert "we found a gap" into "the gap is real, specific, and
survives all alternative explanation":

  Guard 1 — Split-half reference stability check (§4 Guard 2)
  Guard 2 — Three-way ordinal: Lure > Non-Target ≈ Target (§6b)
  Guard 3 — RT-regression survival (§6a)
  Guard 4 — Motor-ROI negative control (§6d)
  Guard 5 — 1000-shuffle cohort permutation null (§7) ~0.8667
  Bonus  — Leave-one-subject-out influence (Cook's-distance analog)

Pre-registered, frozen design (do not tune between guards):
  Window  : MISMATCH_WIN = (0.25, 0.50) s post-stimulus
  ROIs    : TIER12 set (22 bilateral WM+conflict ROIs)
  Control : MOTOR set  (6 bilateral motor ROIs)
  Estimator: ERPCovariances(classes=[1], estimator='lwf')
  Distances: LOO for Targets, full-ref for NT and Lure

Pass condition (pre-declared, §8):
  All five guards must hold. If all five hold: MVP-0 STANDS.
========================================================================
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import wilcoxon

from pyriemann.estimation import ERPCovariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

# ── Config (frozen) ───────────────────────────────────────────────────────────
# Kaggle paths — dataset must be uploaded as 'nsvd-fusion'
DATA_DIR     = '/kaggle/input/nsvd-fusion'
RESULTS_DIR  = '/kaggle/working'
os.makedirs(RESULTS_DIR, exist_ok=True)

FS           = 250
ONSET_SAMPLE = 127          # t=0 at index 127 (confirmed)
MISMATCH_WIN = (0.25, 0.50) # seconds post-stimulus
BASELINE_WIN = (-0.20, 0.00)

N_PERMS      = 1000
MAX_WORKERS  = 2        # leaves 2 cores free for RAM / OS headroom
rng_state         = 42
rng_global   = np.random.default_rng(rng_state)

# Tier-1+2: WM maintenance + conflict/mismatch detection, bilateral (22 ROIs)
TIER12 = {
    'caudalmiddlefrontal', 'rostralmiddlefrontal', 'superiorfrontal',
    'inferiorparietal', 'supramarginal', 'superiorparietal',
    'caudalanteriorcingulate', 'rostralanteriorcingulate',
    'parsopercularis', 'parstriangularis', 'insula',
}

# Motor negative control: Tier-5, bilateral (6 ROIs)
MOTOR = {'precentral', 'postcentral', 'paracentral'}

SUBJECTS = [
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

# ── Utilities ─────────────────────────────────────────────────────────────────

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

def cohens_d(a, b):
    """Pooled-SD Cohen's d: positive = a > b."""
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt((var_a + var_b) / 2) + 1e-12
    return float((np.mean(a) - np.mean(b)) / pooled)

def residualize_rt(rgd, rt):
    """Remove linear RT trend from RGD. Returns residuals."""
    valid = ~np.isnan(rt)
    resid = rgd.copy()
    if valid.sum() >= 3:
        coef = np.polyfit(rt[valid], rgd[valid], 1)
        resid[valid] = rgd[valid] - np.polyval(coef, rt[valid])
    return resid, int(valid.sum())

# ── Core per-subject ERPCov geometry ──────────────────────────────────────────

def compute_erpcov_distances(X_full, y_meta, roi_idx):
    """
    Full per-subject pipeline using ERPCovariances backbone.

    Parameters
    ----------
    X_full  : (N, 68, T) — full sign-corrected source epochs
    y_meta  : (N, 4)     — [memType, tgtType, rt, acc]
    roi_idx : list[int]  — pre-registered ROI indices

    Returns
    -------
    dict with per-trial distances and metadata, or None if too few trials.
    """
    mask2 = y_meta[:, 0] == 2
    X2, y2 = X_full[mask2], y_meta[mask2]
    tt = y2[:, 1].astype(int)
    rt = y2[:, 2]

    t_idx  = np.where(tt == 1)[0]
    nt_idx = np.where(tt == 2)[0]
    l_idx  = np.where(tt == 3)[0]

    if len(t_idx) < 6 or len(l_idx) < 4:
        return None

    t_axis = time_axis(X2.shape[2])
    X2_bc  = baseline_correct(X2, t_axis)
    m      = win_mask(t_axis, MISMATCH_WIN)
    Xw     = X2_bc[:, roi_idx][:, :, m]   # (N, n_roi, n_win)

    # ERPCovariances: folds Target elicited_response prototype into each trial's matrix
    erp  = ERPCovariances(classes=[1], estimator='lwf')
    covs = condition_spd(erp.fit_transform(Xw, tt))

    # Split-half stability on Target reference (§4 Guard 2)
    rng = np.random.default_rng(rng_state)
    perm = rng.permutation(len(t_idx))
    half = len(t_idx) // 2
    ref_half   = t_idx[perm[:half]]
    score_half = t_idx[perm[half:]]
    C_sh  = mean_riemann(covs[ref_half])
    d_sh  = np.array([distance_riemann(C_sh, covs[i]) for i in score_half])
    # LOO on same score half
    d_loo_sh = np.empty(len(score_half))
    all_t = np.arange(len(t_idx))
    for k, si in enumerate(score_half):
        oth = t_idx[np.setdiff1d(all_t, perm[half + k])]
        C_loo = mean_riemann(covs[oth]) if len(oth) >= 2 else C_sh
        d_loo_sh[k] = distance_riemann(C_loo, covs[si])
    pooled_sd  = np.std(np.concatenate([d_sh, d_loo_sh])) + 1e-12
    delta_frac = abs(d_sh.mean() - d_loo_sh.mean()) / pooled_sd
    stable     = bool(delta_frac < 0.50)

    # LOO distances for Targets
    d_target = np.empty(len(t_idx))
    for k, i in enumerate(t_idx):
        ref = mean_riemann(covs[np.delete(t_idx, k)])
        d_target[k] = distance_riemann(ref, covs[i])

    # Full-ref distances for NT and Lure (unbiased — at no point in ref)
    ref_all   = mean_riemann(covs[t_idx])
    d_nt      = np.array([distance_riemann(ref_all, covs[i]) for i in nt_idx])
    d_lure    = np.array([distance_riemann(ref_all, covs[i]) for i in l_idx])

    # RT-residualized gap
    resid_l, n_rt_l = residualize_rt(d_lure, rt[l_idx])
    resid_t, n_rt_t = residualize_rt(d_target, rt[t_idx])
    gap_rt = cohens_d(resid_l, resid_t)

    return {
        'n_t': len(t_idx), 'n_nt': len(nt_idx), 'n_l': len(l_idx),
        'stable': stable, 'delta_frac': delta_frac,
        'd_target': d_target, 'd_nt': d_nt, 'd_lure': d_lure,
        'gap': cohens_d(d_lure, d_target),
        'gap_rt': gap_rt,
        'n_rt_l': n_rt_l, 'n_rt_t': n_rt_t,
        'rt_t': rt[t_idx], 'rt_l': rt[l_idx],
        'covs': covs, 'tt_pooled': tt,
    }


# ── Cohort permutation null (§7) — parallelized across 2 workers ──────────────

def _run_perm_batch(args):
    """Top-tier worker: runs a batch of permutations. Module-tier for pickling."""
    loaded_simple, n_batch, batch_rng_state = args
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.distance import distance_riemann
    import numpy as np
    rng = np.random.default_rng(batch_rng_state)
    null_batch = np.zeros(n_batch)
    for p in range(n_batch):
        perm_gaps = []
        for r in loaded_simple:
            covs  = r['covs']
            tt    = r['tt_pooled']
            t_idx = np.where(tt == 1)[0]
            l_idx = np.where(tt == 3)[0]
            pool  = np.concatenate([t_idx, l_idx])
            n_t   = len(t_idx)
            shuffled = rng.permutation(pool)
            p_t, p_l = shuffled[:n_t], shuffled[n_t:]
            if len(p_t) < 3 or len(p_l) < 1:
                continue
            C_ref_p = mean_riemann(covs[p_t])
            d_lp = np.array([distance_riemann(C_ref_p, covs[i]) for i in p_l])
            d_tp = np.empty(len(p_t))
            for k, i in enumerate(p_t):
                oth = np.delete(p_t, k)
                C_loo_p = mean_riemann(covs[oth]) if len(oth) >= 2 else C_ref_p
                d_tp[k] = distance_riemann(C_loo_p, covs[i])
            pooled = np.sqrt((np.var(d_lp, ddof=1) + np.var(d_tp, ddof=1)) / 2) + 1e-12
            perm_gaps.append(float((d_lp.mean() - d_tp.mean()) / pooled))
        null_batch[p] = float(np.mean(perm_gaps)) if perm_gaps else 0.0
    return null_batch


def cohort_permutation(loaded_results, n_perms=N_PERMS):
    """
    1000-shuffle cohort-tier permutation (§7).
    Parallelized: splits into MAX_WORKERS batches of n_perms//MAX_WORKERS each.
    Logic identical to serial version — rng_states differ per batch to avoid repetition.
    """
    from concurrent.futures import ProcessPoolExecutor
    print(f'  Running {n_perms} cohort permutations (max_workers={MAX_WORKERS})...')
    observed = float(np.mean([r['gap'] for r in loaded_results]))

    # Strip non-picklable keys — workers only need covs and tt
    loaded_simple = [{'covs': r['covs'], 'tt_pooled': r['tt_pooled']} for r in loaded_results]

    batch_size = n_perms // MAX_WORKERS
    batch_args = [
        (loaded_simple, batch_size, rng_state + 1 + i)
        for i in range(MAX_WORKERS)
    ]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        batches = list(executor.map(_run_perm_batch, batch_args))

    null  = np.concatenate(batches)
    p_val = float(np.mean(np.abs(null) >= np.abs(observed)))
    print(f'    {len(null)} permutations complete')
    return observed, null, p_val


# ── Leave-one-subject-out influence ───────────────────────────────────────────

def loso_influence(stable_results):
    """Cook's-distance analog: how much does dropping one subject move d?"""
    all_gaps  = np.array([r['gap'] for r in stable_results])
    full_mean = all_gaps.mean()
    influences = []
    for i in range(len(stable_results)):
        loo = np.delete(all_gaps, i).mean()
        influences.append(abs(full_mean - loo))
    return np.array(influences)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 68)
    print('AETHER MVP-0 CLOSEOUT — ERPCovariances + Document 2 Guards')
    print('=' * 68)

    # Load ROI index from saved roi_names
    roi_names = np.load(os.path.join(DATA_DIR, 'roi_names.npy'))
    tier12_idx = [i for i, n in enumerate(roi_names)
                  if str(n).rsplit('-', 1)[0] in TIER12]
    motor_idx  = [i for i, n in enumerate(roi_names)
                  if str(n).rsplit('-', 1)[0] in MOTOR]
    print(f'Tier-1+2 ROIs : {len(tier12_idx)}  |  Motor ROIs: {len(motor_idx)}')
    print(f'Subjects       : {len(SUBJECTS)}')
    print()

    # ── Phase 1: Per-subject primary analysis ─────────────────────────────────
    print('── Phase 1: Per-subject ERPCov distances (Tier-1+2) ──')
    results, unstable_list, skipped = [], [], []

    for subj in SUBJECTS:
        try:
            X = np.load(os.path.join(DATA_DIR, f'{subj}_Xsrc.npy'))
            y = np.load(os.path.join(DATA_DIR, f'{subj}_y_meta.npy'))
        except FileNotFoundError:
            skipped.append(subj); continue

        r = compute_erpcov_distances(X, y, tier12_idx)
        if r is None:
            print(f'  [{subj}] SKIP — too few trials'); skipped.append(subj); continue

        r['subj'] = subj
        results.append(r)
        stab = '✓' if r['stable'] else '✗ UNSTABLE'
        print(f"  [{subj}]  n=({r['n_t']}T/{r['n_nt']}NT/{r['n_l']}L)  "
              f"ref={stab}(Δ={r['delta_frac']:.3f})  "
              f"T={r['d_target'].mean():.4f}  "
              f"NT={r['d_nt'].mean():.4f}  "
              f"L={r['d_lure'].mean():.4f}  "
              f"d={r['gap']:+.3f}")
        if not r['stable']:
            unstable_list.append(subj)

    stable_results = [r for r in results if r['stable']]
    n_exc = len(results) - len(stable_results)
    n     = len(stable_results)
    print(f'\n  Total: {len(results)}/{len(SUBJECTS)}  |  '
          f'Excluded (unstable ref): {n_exc}  |  Cohort: {n}')
    if unstable_list:
        print(f'  Unstable: {unstable_list}')

    if n < 20:
        print('[HALT] Too few stable subjects for cohort analysis.'); return

    gaps   = np.array([r['gap'] for r in stable_results])
    n_pos  = int((gaps > 0).sum())
    mean_T  = float(np.mean([r['d_target'].mean() for r in stable_results]))
    mean_NT = float(np.mean([r['d_nt'].mean()     for r in stable_results]))
    mean_L  = float(np.mean([r['d_lure'].mean()   for r in stable_results]))
    ordinal_ok = (mean_L > mean_NT) and (mean_NT >= mean_T * 0.9)

    print(f'\n  Mean cohort d            : {gaps.mean():+.4f}')
    print(f'  Gap > 0                  : {n_pos}/{n} ({100*n_pos/n:.0f}%)')
    print(f'  Wilcoxon p (parametric)  : {wilcoxon(gaps).pvalue:.4e}')
    print(f'  Ordinal T/NT/L means     : {mean_T:.4f} / {mean_NT:.4f} / {mean_L:.4f}')
    print(f'  Ordinal L>NT≈T           : {"✓" if ordinal_ok else "✗"}')

    # ── Phase 2: RT-regression guard ──────────────────────────────────────────
    print('\n── Phase 2: RT-regression guard ──')
    gaps_rt  = np.array([r['gap_rt'] for r in stable_results])
    n_pos_rt = int((gaps_rt > 0).sum())
    print(f'  RT-residualized d        : {gaps_rt.mean():+.4f}')
    print(f'  Gap > 0 (RT-resid)       : {n_pos_rt}/{n} ({100*n_pos_rt/n:.0f}%)')
    rt_guard_ok = gaps_rt.mean() > 0 and n_pos_rt > n * 0.5

    # ── Phase 3: Motor-ROI negative control ───────────────────────────────────
    print('\n── Phase 3: Motor-ROI negative control ──')
    motor_results = []
    for subj in SUBJECTS:
        try:
            X = np.load(os.path.join(DATA_DIR, f'{subj}_Xsrc.npy'))
            y = np.load(os.path.join(DATA_DIR, f'{subj}_y_meta.npy'))
        except FileNotFoundError:
            continue
        r = compute_erpcov_distances(X, y, motor_idx)
        if r:
            motor_results.append(r)

    motor_null_ok = False
    if motor_results:
        m_gaps  = np.array([r['gap'] for r in motor_results])
        m_n_pos = int((m_gaps > 0).sum())
        print(f'  Motor subjects           : {len(motor_results)}')
        print(f'  Mean motor d             : {m_gaps.mean():+.4f}  (expected ≈ 0)')
        print(f'  Motor gap > 0            : {m_n_pos}/{len(motor_results)} '
              f'({100*m_n_pos/len(motor_results):.0f}%)')
        motor_null_ok = (abs(m_gaps.mean()) < 0.15 and
                         m_n_pos < len(motor_results) * 0.65)
        print(f'  Motor null               : {"✓ CLEAN" if motor_null_ok else "✗ CONTAMINATED"}')

    # ── Phase 4: 1000-shuffle cohort permutation ───────────────────────────────
    print(f'\n── Phase 4: Cohort permutation ({N_PERMS} shuffles) ──')
    observed, null_dist, perm_p = cohort_permutation(stable_results)
    print(f'  Observed cohort d        : {observed:+.4f}')
    print(f'  Null mean ± SD           : {null_dist.mean():+.4f} ± {null_dist.std():.4f}')
    print(f'  Two-sided perm p         : {perm_p:.4f}')
    perm_ok = perm_p < 0.05

    # ── Phase 5: Leave-one-subject-out influence ───────────────────────────────
    print('\n── Phase 5: Leave-one-subject-out influence ──')
    influences = loso_influence(stable_results)
    worst_idx  = int(np.argmax(influences))
    worst_subj = stable_results[worst_idx]['subj']
    print(f'  Max influence subject    : {worst_subj} (Δd = {influences[worst_idx]:.4f})')
    print(f'  Mean influence           : {influences.mean():.4f}')
    print(f'  d without most-inf. subj: {np.delete(gaps, worst_idx).mean():+.4f}')
    robust = bool(np.delete(gaps, worst_idx).mean() > 0)
    print(f'  Robust to removal        : {"✓" if robust else "✗"}')

    # Save null distribution
    np.save(os.path.join(RESULTS_DIR, 'mvp0_perm_null.npy'), null_dist)
    np.save(os.path.join(RESULTS_DIR, 'mvp0_gaps.npy'), gaps)

    # ── Final pass/fail report ─────────────────────────────────────────────────
    cond1 = n_exc < len(SUBJECTS) * 0.50
    cond2 = gaps.mean() > 0 and n_pos > n * 0.5 and ordinal_ok
    cond3 = rt_guard_ok
    cond4 = perm_ok
    cond5 = motor_null_ok

    def chk(c): return '✓ PASS' if c else '✗ FAIL'

    print('\n' + '=' * 68)
    print('MVP-0 PASS CONDITIONS  (pre-declared, §8 of 02_PIPELINE_LOGIC.md)')
    print('=' * 68)
    print(f'  1. Reference stable (< 50% excluded) : {chk(cond1)}')
    print(f'     ({n_exc} excluded, {n} stable of {len(results)} total)')
    print(f'  2. Ordinal Lure > NT ≈ Target        : {chk(cond2)}')
    print(f'     (T={mean_T:.4f} NT={mean_NT:.4f} L={mean_L:.4f}, {n_pos}/{n} positive)')
    print(f'  3. Gap survives RT regression        : {chk(cond3)}')
    print(f'     (RT-resid d={gaps_rt.mean():+.4f}, {n_pos_rt}/{n} positive)')
    print(f'  4. Cohort permutation p < 0.05       : {chk(cond4)}')
    print(f'     (p={perm_p:.4f})')
    print(f'  5. Motor-ROI negative control null   : {chk(cond5)}')
    print()
    all_pass = all([cond1, cond2, cond3, cond4, cond5])
    verdict  = '✓  MVP-0 EXISTENCE PROOF STANDS' if all_pass else '✗  NOT ALL CONDITIONS MET'
    print(f'  RESULT: {verdict}')
    print('=' * 68)
    print(f'\nResults saved → {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
