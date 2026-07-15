"""
kaggle_mvp0_reduce.py  —  REDUCE PHASE
========================================================================
Run this LOCALLY after downloading all 5 batch_N_results.npy files
from your 5 Kaggle recordbooks into a single folder.

Does NO heavy compute — just aggregation + the 1000-shuffle permutation
(fast because covariances are pre-loaded from the map phase).

Usage:
  python kaggle_mvp0_reduce.py --batch_dir /path/to/folder/with/batch_files

Outputs:
  Full MVP-0 pass/fail verdict (Document 2 §8)
  + Dimensionality-matched negative control verdict (ARM 1 + ARM 2)
========================================================================
"""

import os, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import wilcoxon

from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

# ── Config (must match map phase exactly) ──────────────────────────────────
N_PERMS = 1000
rng_state    = 42
K_SUBSETS = 40
N_NULL22  = 30

def cohens_d(a, b):
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) + 1e-12
    return float((np.mean(a) - np.mean(b)) / pooled)

def cohort_d(gaps):
    g = np.array([x for x in gaps if x is not None], dtype=float)
    return g.mean(), int((g > 0).sum()), len(g)

# ── Cohort-tier permutation (§7) ─────────────────────────────────────────

def run_permutation(stable_results, n_perms=N_PERMS):
    print(f'  Running {n_perms} cohort permutations...')
    rng = np.random.default_rng(rng_state + 1)
    observed = float(np.mean([r['dist']['gap'] for r in stable_results]))
    null = np.zeros(n_perms)

    for p in range(n_perms):
        perm_gaps = []
        for r in stable_results:
            covs = r['dist']['covs']
            tt   = r['dist']['tt']
            t_idx = np.where(tt == 1)[0]
            l_idx = np.where(tt == 3)[0]
            pool  = np.concatenate([t_idx, l_idx])
            n_t   = len(t_idx)
            shuf  = rng.permutation(pool)
            p_t, p_l = shuf[:n_t], shuf[n_t:]
            if len(p_t) < 3 or len(p_l) < 1:
                continue
            C_ref = mean_riemann(covs[p_t])
            d_lp  = np.array([distance_riemann(C_ref, covs[i]) for i in p_l])
            d_tp  = np.array([
                distance_riemann(
                    mean_riemann(covs[np.delete(p_t, k)]) if len(p_t) > 2 else C_ref,
                    covs[i]
                ) for k, i in enumerate(p_t)
            ])
            perm_gaps.append(cohens_d(d_lp, d_tp))
        null[p] = float(np.mean(perm_gaps)) if perm_gaps else 0.0
        if (p + 1) % 100 == 0:
            print(f'    {p+1}/{n_perms}', end='\r')
    print()
    p_val = float(np.mean(np.abs(null) >= np.abs(observed)))
    return observed, null, p_val

# ── Leave-one-subject-out influence ───────────────────────────────────────

def loso_influence(gaps):
    full = gaps.mean()
    return np.array([abs(np.delete(gaps, i).mean() - full) for i in range(len(gaps))])

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_dir', default='.',
                        help='Folder containing batch_0..4_results.npy')
    args = parser.parse_args()

    # Load all 5 batches
    all_results = []
    for bid in range(5):
        path = os.path.join(args.batch_dir, f'batch_{bid}_results.npy')
        if not os.path.exists(path):
            print(f'[WARN] {path} not found — skipping batch {bid}')
            continue
        batch = np.load(path, allow_pickle=True).tolist()
        all_results.extend(batch)
        print(f'Batch {bid}: loaded {len(batch)} subjects')

    print(f'\nTotal subjects loaded: {len(all_results)}')

    # Separate stable / unstable
    valid   = [r for r in all_results if r['dist'] is not None]
    stable  = [r for r in valid       if r['dist']['stable']]
    unstable = [r['dist'] for r in valid if not r['dist']['stable']]
    n = len(stable)
    print(f'Valid: {len(valid)}  |  Stable ref: {n}  |  Unstable: {len(valid)-n}')

    if n < 20:
        print('[HALT] Too few stable subjects.'); return

    gaps    = np.array([r['dist']['gap']    for r in stable])
    gaps_rt = np.array([r['dist']['gap_rt'] for r in stable])
    mean_T  = np.mean([r['dist']['d_target'].mean() for r in stable])
    mean_NT = np.mean([r['dist']['d_nt'].mean()     for r in stable])
    mean_L  = np.mean([r['dist']['d_lure'].mean()   for r in stable])
    n_pos   = int((gaps > 0).sum())
    n_pos_rt= int((gaps_rt > 0).sum())
    ordinal = mean_L > mean_NT and mean_NT >= mean_T * 0.9

    # ── MVP-0 Guards 1-3 ──────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('MVP-0 CLOSEOUT — ERPCovariances + Document 2 Guards')
    print('=' * 65)
    print(f'  Mean cohort d           : {gaps.mean():+.4f}')
    print(f'  Gap > 0                 : {n_pos}/{n} ({100*n_pos/n:.0f}%)')
    print(f'  Wilcoxon p              : {wilcoxon(gaps).pvalue:.4e}')
    print(f'  Ordinal T/NT/L          : {mean_T:.4f} / {mean_NT:.4f} / {mean_L:.4f}')
    print(f'  Ordinal L>NT>=T         : {"YES" if ordinal else "NO"}')
    print(f'  RT-residualized d       : {gaps_rt.mean():+.4f}')
    print(f'  RT gap > 0              : {n_pos_rt}/{n} ({100*n_pos_rt/n:.0f}%)')

    # ── Guard 4: cohort permutation ───────────────────────────────────────
    print(f'\n── Cohort permutation ({N_PERMS} shuffles) ──')
    observed, null_dist, perm_p = run_permutation(stable)
    print(f'  Observed cohort d       : {observed:+.4f}')
    print(f'  Null mean ± SD          : {null_dist.mean():+.4f} ± {null_dist.std():.4f}')
    print(f'  Two-sided perm p        : {perm_p:.4f}')

    # ── LOSO influence ────────────────────────────────────────────────────
    infl = loso_influence(gaps)
    wi   = int(np.argmax(infl))
    print(f'\n── LOSO influence ──')
    print(f'  Most influential        : {stable[wi]["dist"].get("subj", "?")} '
          f'(Δd={infl[wi]:.4f})')
    print(f'  d without it            : {np.delete(gaps, wi).mean():+.4f}')

    # ── Guard 5: Motor null ───────────────────────────────────────────────
    motor_gaps = np.array([r['motor_gap'] for r in stable
                           if r['motor_gap'] is not None], dtype=float)
    m_pos = int((motor_gaps > 0).sum())
    motor_null_ok = abs(motor_gaps.mean()) < 0.15 and m_pos < len(motor_gaps) * 0.65
    print(f'\n── Motor-ROI negative control ──')
    print(f'  Mean motor d            : {motor_gaps.mean():+.4f}  (expected ≈ 0)')
    print(f'  Motor gap > 0           : {m_pos}/{len(motor_gaps)}')
    print(f'  Motor null              : {"CLEAN ✓" if motor_null_ok else "CONTAMINATED"}')

    # ── Negative control ARM 1 (downward match) ───────────────────────────
    print(f'\n── Negative Control ARM 1 — Downward match ──')
    wm_k_cohort = np.array([
        cohort_d([r['wm_k_gaps'][j] for r in stable])[0]
        for j in range(K_SUBSETS)
    ])
    motor_d_all, _, _ = cohort_d([r['motor_gap'] for r in stable])
    wm_k_positive = wm_k_cohort.mean() > 0.10
    motor_null_arm1 = abs(motor_d_all) < 0.12
    matched_ok = wm_k_positive and motor_null_arm1
    print(f'  WM subsampled (mean ± SD): {wm_k_cohort.mean():+.4f} ± {wm_k_cohort.std():.4f}')
    print(f'  Motor at matched size   : {motor_d_all:+.4f}')
    print(f'  Motor null BIOLOGICAL   : {"YES ✓" if matched_ok else "NO — check"}')

    # ── Negative control ARM 2 (spatial specificity) ──────────────────────
    print(f'\n── Negative Control ARM 2 — Spatial specificity ──')
    null22_cohort = np.array([
        cohort_d([r['null22_gaps'][j] for r in stable])[0]
        for j in range(N_NULL22)
    ])
    d_tier12 = gaps.mean()
    emp_p = float(np.mean(np.abs(null22_cohort) >= abs(d_tier12)))
    spatial_ok = emp_p < 0.05
    print(f'  Real Tier-1+2 d         : {d_tier12:+.4f}')
    print(f'  Non-WM null (mean ± SD) : {null22_cohort.mean():+.4f} ± {null22_cohort.std():.4f}')
    print(f'  Empirical p             : {emp_p:.4f}  ({N_NULL22} null sets)')
    print(f'  Specifically elevated   : {"YES ✓" if spatial_ok else "NO"}')

    # ── Final verdict ─────────────────────────────────────────────────────
    cond1 = (len(valid) - n) < len(all_results) * 0.50
    cond2 = gaps.mean() > 0 and n_pos > n * 0.5 and ordinal
    cond3 = gaps_rt.mean() > 0 and n_pos_rt > n * 0.5
    cond4 = perm_p < 0.05
    cond5 = motor_null_ok
    cond6 = matched_ok       # stronger motor null (dimensionality matched)
    cond7 = spatial_ok       # signal is spatially specific

    def chk(c): return '✓ PASS' if c else '✗ FAIL'

    print('\n' + '=' * 65)
    print('MVP-0 PASS CONDITIONS  (pre-declared, §8 of 02_PIPELINE_LOGIC.md)')
    print('=' * 65)
    print(f'  1. Reference stable (<50% excluded)     : {chk(cond1)}')
    print(f'  2. Ordinal L > NT ≈ T, gap > 0          : {chk(cond2)}')
    print(f'  3. Gap survives RT regression            : {chk(cond3)}')
    print(f'  4. Cohort permutation p < 0.05           : {chk(cond4)}  (p={perm_p:.4f})')
    print(f'  5. Motor null (simple)                   : {chk(cond5)}')
    print(f'  6. Motor null (dimensionality-matched)   : {chk(cond6)}')
    print(f'  7. Spatially specific (non-WM null)      : {chk(cond7)}')
    print()
    core_pass = all([cond1, cond2, cond3, cond4, cond5])
    full_pass = all([cond1, cond2, cond3, cond4, cond5, cond6, cond7])
    if full_pass:
        print('  RESULT: ✓  MVP-0 EXISTENCE PROOF STANDS  (all 7 conditions)')
    elif core_pass:
        print('  RESULT: ✓  MVP-0 CORE PASSES  (5/5) — negative control pending')
    else:
        print('  RESULT: ✗  NOT ALL CONDITIONS MET — inspect output above')
    print('=' * 65)

    # Save
    np.save('mvp0_perm_null.npy', null_dist)
    np.save('mvp0_gaps.npy', gaps)
    print('\nSaved: mvp0_perm_null.npy, mvp0_gaps.npy')


if __name__ == '__main__':
    main()
