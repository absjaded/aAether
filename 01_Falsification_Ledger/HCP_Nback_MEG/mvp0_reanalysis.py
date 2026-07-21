"""
mvp0_reanalysis.py  -  REDUCE-SIDE REANALYSIS (no map re-run)
========================================================================
Adjudicates the three live questions on the EXISTING batch files, using
the Tier-1+2 covariances already saved in the map phase:

  A. SCALE FIX (the 1e13 bug).  ERPCovariances was run on unscaled signal.
     AIRM distance is scale-invariant in EXACT arithmetic, and lwf shrinkage
     + condition_spd are scale-equivariant - so multiplying the SAVED
     covariances by 1e26 EXACTLY reproduces what the map would have saved
     had it scaled the signal by 1e13. Any difference between the as-saved
     distances and the x1e26 distances is therefore PURELY the numerical
     artifact the scale was meant to proccurrence. This settles "did the bug
     matter?" without re-running the expensive map.

  B. ORDINAL SPACING (the real intent-specificity test).  Replaces the weak
     boolean `L > NT and NT >= 0.9*T` with the canon test: is NT clustered
     WITH T (g_NT_vs_T ~ 0) while L separates from NT (g_L_vs_NT > 0)?

  C. RT INCREMENTAL VALIDITY (the correct RT test).  The map's residualize_rt
     mean-centers each class separately, which mechanically forces the gap to
     ~0 regardless of whether RT explains anything. The correct test asks
     whether `is_lure` predicts distance OVER AND ABOVE RT, jointly across
     both classes: per-subject OLS  rgd ~ 1 + is_lure + rt,  then Wilcoxon on
     the per-subject is_lure coefficients.

  D. CORRECT-ONLY.  Re-run the gap and ordinal on isCorrect==1 trials only,
     reference rebuilt from correct Targets. Watch the Lure n collapse.

  E. COARSE MOTOR-INCREMENTAL.  Per-subject (exec_gap - motor_gap), Wilcoxon.
     This is a WEAK subject-tier proxy. The proper per-trial motor control
     and the matched-dimensionality motor null REQUIRE a map re-run that saves
     motor covariances (see the patch shipped alongside this file).

Needs, per subject: the saved covs (in the batch files) + y_meta reloaded
locally for RT and isCorrect. Point DATA_DIR at your local nsvd_fusion folder.
========================================================================
"""

import os, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import wilcoxon

from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

# - Config -
COV_SCALE = 1e13 ** 2          # = 1e26. Signal x1e13 => covariance x1e26.
rng_state      = 42

def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) + 1e-12
    return float((np.mean(a) - np.mean(b)) / s)

# - Per-subject recompute at a chosen covariance scale -

def recompute(covs, tt, scale):
    """All distances to the within-subject Target geometry. Target is LOO."""
    C = covs * scale
    t_idx  = np.where(tt == 1)[0]
    nt_idx = np.where(tt == 2)[0]
    l_idx  = np.where(tt == 3)[0]
    d_t = np.array([distance_riemann(mean_riemann(C[np.delete(t_idx, k)]), C[i])
                    for k, i in enumerate(t_idx)])
    ref = mean_riemann(C[t_idx])
    d_nt = np.array([distance_riemann(ref, C[i]) for i in nt_idx])
    d_l  = np.array([distance_riemann(ref, C[i]) for i in l_idx])
    return d_t, d_nt, d_l, t_idx, nt_idx, l_idx

def correct_only_gap(covs, tt, ic2, scale):
    """Gap + ordinal restricted to isCorrect==1, reference = correct Targets."""
    C = covs * scale
    t_idx  = np.where(tt == 1)[0]; nt_idx = np.where(tt == 2)[0]; l_idx = np.where(tt == 3)[0]
    tC = t_idx[ic2[t_idx] == 1]; nC = nt_idx[ic2[nt_idx] == 1]; lC = l_idx[ic2[l_idx] == 1]
    if len(tC) < 6 or len(lC) < 4:
        return None
    d_t = np.array([distance_riemann(mean_riemann(C[np.delete(tC, k)]), C[i])
                    for k, i in enumerate(tC)])
    ref = mean_riemann(C[tC])
    d_l  = np.array([distance_riemann(ref, C[i]) for i in lC])
    d_nt = np.array([distance_riemann(ref, C[i]) for i in nC]) if len(nC) else np.array([])
    return {'gap': cohens_d(d_l, d_t), 'n_lure': len(lC), 'n_lure_all': len(l_idx),
            'mT': d_t.mean(), 'mNT': d_nt.mean() if len(nC) else np.nan, 'mL': d_l.mean()}

def rt_incremental_beta(d_t, d_l, rt_t, rt_l):
    """Per-subject OLS: rgd_z ~ 1 + is_lure + rt_z. Return is_lure coefficient
    (in within-subject SD units), or None if underpowered / rank-deficient."""
    rgd = np.concatenate([d_t, d_l]).astype(float)
    isl = np.concatenate([np.zeros(len(d_t)), np.ones(len(d_l))])
    rt  = np.concatenate([rt_t, rt_l]).astype(float)
    keep = ~np.isnan(rt)
    rgd, isl, rt = rgd[keep], isl[keep], rt[keep]
    # need both classes present and RT variation
    if (isl == 0).sum() < 3 or (isl == 1).sum() < 3 or np.std(rt) < 1e-9:
        return None, int((isl == 1).sum())
    rgd = (rgd - rgd.mean()) / (rgd.std() + 1e-12)
    rt  = (rt  - rt.mean())  / (rt.std()  + 1e-12)
    Xd  = np.column_stack([np.ones_like(rgd), isl, rt])
    beta, *_ = np.linalg.lstsq(Xd, rgd, rcond=None)
    return float(beta[1]), int((isl == 1).sum())   # coefficient on is_lure

# - Main -

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_dir', default='.', help='folder with batch_0..4_results.npy')
    ap.add_argument('--data_dir',  default='', help='local nsvd_fusion (for RT/isCorrect)')
    args = ap.parse_args()

    results = []
    for bid in range(5):
        p = os.path.join(args.batch_dir, f'batch_{bid}_results.npy')
        if os.path.exists(p):
            results.extend(np.load(p, allow_pickle=True).tolist())
    stable = [r for r in results
              if r.get('dist') is not None and r['dist'].get('stable')
              and 'covs' in r['dist']]
    print(f'Loaded {len(results)} subjects | stable w/ covs: {len(stable)}')
    if len(stable) < 20:
        print('[HALT] too few stable subjects with saved covariances.'); return

    have_meta = bool(args.data_dir) and os.path.isdir(args.data_dir)
    if not have_meta:
        print('[WARN] --data_dir not set/found: RT and correct-only will be SKIPPED.')

    # accumulators
    g_LT_saved, g_LT_corr = [], []
    g_NT_T, g_L_NT        = [], []
    ord_T, ord_NT, ord_L  = [], [], []
    rt_betas, rt_drop     = [], 0
    co_gaps, co_nlure     = [], []
    exec_minus_motor      = []
    scale_delta_gap       = []

    for r in stable:
        d = r['dist']
        covs, tt = d['covs'], np.asarray(d['tt'])

        # A. corrected (x1e26) distances
        d_t, d_nt, d_l, t_idx, nt_idx, l_idx = recompute(covs, tt, COV_SCALE)

        g_corr  = cohens_d(d_l, d_t)
        g_LT_corr.append(g_corr)
        g_LT_saved.append(d['gap'])
        scale_delta_gap.append(abs(g_corr - d['gap']))

        # B. ordinal spacings (standardized), corrected scale
        g_NT_T.append(cohens_d(d_nt, d_t))
        g_L_NT.append(cohens_d(d_l, d_nt))
        ord_T.append(d_t.mean()); ord_NT.append(d_nt.mean()); ord_L.append(d_l.mean())

        # E. coarse motor-incremental
        if r.get('motor_gap') is not None:
            exec_minus_motor.append(g_corr - r['motor_gap'])

        if not have_meta:
            continue

        # reload y_meta for this subject -> align RT / isCorrect to 2-back trials
        try:
            y = np.load(os.path.join(args.data_dir, f"{r['subj']}_y_meta.npy"))
        except FileNotFoundError:
            continue
        m2 = y[:, 0] == 2
        tt_chk = y[m2, 1].astype(int)
        if len(tt_chk) != len(tt) or not np.array_equal(tt_chk, tt):
            print(f"  [skip RT] {r['subj']}: y_meta trial order mismatch")
            continue
        rt2, ic2 = y[m2, 2], y[m2, 3]

        # C. RT incremental validity
        beta, n_l = rt_incremental_beta(d_t, d_l, rt2[t_idx], rt2[l_idx])
        if beta is not None:
            rt_betas.append(beta)
        rt_drop += int(np.isnan(np.concatenate([rt2[t_idx], rt2[l_idx]])).sum())

        # D. correct-only
        co = correct_only_gap(covs, tt, ic2.astype(float), COV_SCALE)
        if co is not None:
            co_gaps.append(co['gap']); co_nlure.append((co['n_lure_all'], co['n_lure']))

    n = len(g_LT_corr)
    def wp(x):
        x = np.asarray(x, float)
        return wilcoxon(x).pvalue if len(x) >= 6 and np.any(x != 0) else np.nan

    print('\n' + '=' * 66)
    print('A - SCALE FIX  (did the missing 1e13 matter numerically?)')
    print('=' * 66)
    print(f'  Gap as-saved (no scale)   : {np.mean(g_LT_saved):+.4f}')
    print(f'  Gap corrected (x1e26)     : {np.mean(g_LT_corr):+.4f}')
    print(f'  Max per-subject |Deltagap|    : {np.max(scale_delta_gap):.4f}')
    print(f'  Mean per-subject |Deltagap|   : {np.mean(scale_delta_gap):.4f}')
    settled = np.max(scale_delta_gap) < 0.02
    print(f'  -> Scale bug verdict       : '
          + ('NUMERICALLY HARMLESS - failures are real, not a scale artifact.'
             if settled else
             'MATERIAL - corrected values differ; trust the x1e26 column above.'))

    print('\n' + '=' * 66)
    print('B - ORDINAL SPACING  (is Non-Target clustered with Target?)')
    print('=' * 66)
    print(f'  Ordinal T / NT / L (means): {np.mean(ord_T):.4f} / {np.mean(ord_NT):.4f} / {np.mean(ord_L):.4f}')
    print(f'  g(NT vs T)  [want ~0]     : {np.mean(g_NT_T):+.4f}   (Wilcoxon p={wp(g_NT_T):.4f})')
    print(f'  g(L  vs NT) [want > 0]    : {np.mean(g_L_NT):+.4f}   (Wilcoxon p={wp(g_L_NT):.4f})')
    nt_clusters = abs(np.mean(g_NT_T)) < 0.10 and not (wp(g_NT_T) < 0.05 and np.mean(g_NT_T) > 0)
    l_separates = np.mean(g_L_NT) > 0.15 and wp(g_L_NT) < 0.05
    ordinal_ok  = nt_clusters and l_separates
    print(f'  NT clusters with T        : {"YES" if nt_clusters else "NO - NT separates from T"}')
    print(f'  L separates from NT       : {"YES" if l_separates else "NO"}')
    print(f'  -> Intent-specific ordinal : {"PASS" if ordinal_ok else "FAIL (response/novelty axis)"}')

    if have_meta and rt_betas:
        print('\n' + '=' * 66)
        print('C - RT INCREMENTAL VALIDITY  (does is_lure predict beyond RT?)')
        print('=' * 66)
        rb = np.asarray(rt_betas)
        print(f'  Subjects with usable RT   : {len(rb)}   (NaN-RT trials dropped: {rt_drop})')
        print(f'  Mean is_lure coef (SD u.) : {rb.mean():+.4f}')
        print(f'  Coef > 0                  : {int((rb>0).sum())}/{len(rb)}   (Wilcoxon p={wp(rb):.4f})')
        rt_ok = rb.mean() > 0 and wp(rb) < 0.05
        print(f'  -> Survives RT (correctly) : {"PASS" if rt_ok else "FAIL"}')
        print(f'  CAVEAT: a correctly-handled Lure is a WITHHELD response -> RT is NaN.')
        print(f'          The Lures that HAVE an RT are mostly misfires. Read C with D.')
    else:
        rt_ok = None
        print('\nC - RT INCREMENTAL VALIDITY: skipped (no --data_dir).')

    if have_meta and co_gaps:
        print('\n' + '=' * 66)
        print('D - CORRECT-ONLY  (does the gap survive on correctly-handled trials?)')
        print('=' * 66)
        cg = np.asarray(co_gaps)
        print(f'  Subjects (>= trial floor) : {len(cg)}')
        print(f'  Correct-only mean gap     : {cg.mean():+.4f}   (Wilcoxon p={wp(cg):.4f})')
        meanall = np.mean([a for a, _ in co_nlure]); meankept = np.mean([b for _, b in co_nlure])
        print(f'  Lure n  all -> correct     : {meanall:.1f} -> {meankept:.1f} per subject')
        co_ok = cg.mean() > 0 and wp(cg) < 0.05
        print(f'  -> Gap holds among correct : {"YES" if co_ok else "NO / underpowered"}')
    else:
        co_ok = None
        print('\nD - CORRECT-ONLY: skipped (no --data_dir).')

    print('\n' + '=' * 66)
    print('E - COARSE MOTOR-INCREMENTAL  (subject-tier proxy ONLY)')
    print('=' * 66)
    if exec_minus_motor:
        em = np.asarray(exec_minus_motor)
        print(f'  exec_gap - motor_gap      : {em.mean():+.4f}   (Wilcoxon p={wp(em):.4f})')
        print(f'  exec > motor              : {int((em>0).sum())}/{len(em)}')
        print(f'  -> WEAK proxy. The decisive motor null (matched-dimensionality,')
        print(f'    per-trial) needs a map re-run that saves motor covariances.')

    print('\n' + '=' * 66)
    print('HONEST VERDICT (frozen gates)')
    print('=' * 66)
    print(f'  Scale bug              : {"settled (harmless)" if settled else "MATERIAL - recheck"}')
    print(f'  Ordinal (NTapproxT, L>NT)   : {"PASS" if ordinal_ok else "FAIL"}')
    print(f'  RT incremental         : {"PASS" if rt_ok else ("FAIL" if rt_ok is False else "n/a")}')
    print(f'  Correct-only           : {"PASS" if co_ok else ("WEAK/FAIL" if co_ok is False else "n/a")}')
    print(f'  Motor null             : UNRESOLVED reduce-side - re-run map w/ scale fix + motor covs')
    print('=' * 66)
    print('  This add-on can confirm intent-SPECIFICITY (ordinal + RT-done-right +')
    print('  correct-only). It CANNOT clear the motor confound. If ordinal still')
    print('  shows NTapproxL>T and correct-only does not rescue a clean L>NTapproxT, the')
    print('  honest read is: this public task couples rule-violation with the')
    print('  response decision - and isolating intent needs a purpose-built')
    print('  paradigm. That is the paradigm-design structural advantage arriving as a finding.')


if __name__ == '__main__':
    main()
