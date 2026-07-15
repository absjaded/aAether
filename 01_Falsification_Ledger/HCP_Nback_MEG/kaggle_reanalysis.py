"""
kaggle_reanalysis.py
========================================================================
PHASE 1 REDUCE — runs off existing batch files. No map re-run needed.
Adjudicates the four live oracle questions using the Tier-1+2 covariances
already saved in the map phase.

Upload to a Kaggle recordbook with:
  - mvp0-batches  dataset  (your 5 batch_N_results.npy files)
  - nsvd-fusion   dataset  (original subject files, for RT + isCorrect)

Then: Run All. Takes ~15-20 min.
========================================================================
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy.stats import wilcoxon
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_DIR = '/kaggle/input/datasets/hansmax/nsvd-batches'
DATA_DIR  = '/kaggle/input/datasets/hansmax/nsvd-fusion'
OUT_DIR   = '/kaggle/working'
os.makedirs(OUT_DIR, exist_ok=True)

COV_SCALE = 1e26   # signal * 1e13 => covariance * 1e26 (AIRM scale-equivariant)
rng_state      = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) + 1e-12
    return float((np.mean(a) - np.mean(b)) / s)

def wp(x):
    x = np.asarray(x, float)
    if len(x) < 6 or not np.any(x != 0):
        return np.nan
    return float(wilcoxon(x).pvalue)

def recompute(covs, tt, scale):
    """Recompute all LOO distances at a given covariance scale."""
    C = covs * scale
    t_idx  = np.where(tt == 1)[0]
    nt_idx = np.where(tt == 2)[0]
    l_idx  = np.where(tt == 3)[0]
    d_t = np.array([
        distance_riemann(mean_riemann(C[np.delete(t_idx, k)]), C[i])
        for k, i in enumerate(t_idx)
    ])
    ref  = mean_riemann(C[t_idx])
    d_nt = np.array([distance_riemann(ref, C[i]) for i in nt_idx])
    d_l  = np.array([distance_riemann(ref, C[i]) for i in l_idx])
    return d_t, d_nt, d_l, t_idx, nt_idx, l_idx

def rt_incremental_beta(d_t, d_l, rt_t, rt_l):
    """
    Per-subject OLS: rgd_z ~ 1 + is_lure + rt_z
    Returns is_lure coefficient (SD units), or None if underpowered.
    [Detail]: Lures with NaN RT (correct withholds) are dropped — those trials
    are structurally the cleanest intent signal but have no RT to regress.
    Read Section C together with Section D (correct-only).
    """
    rgd = np.concatenate([d_t, d_l]).astype(float)
    isl = np.concatenate([np.zeros(len(d_t)), np.ones(len(d_l))])
    rt  = np.concatenate([rt_t, rt_l]).astype(float)
    keep = ~np.isnan(rt)
    rgd, isl, rt = rgd[keep], isl[keep], rt[keep]
    if (isl == 0).sum() < 3 or (isl == 1).sum() < 3 or np.std(rt) < 1e-9:
        return None, int((isl == 1).sum())
    rgd = (rgd - rgd.mean()) / (rgd.std() + 1e-12)
    rt  = (rt  - rt.mean())  / (rt.std()  + 1e-12)
    X   = np.column_stack([np.ones_like(rgd), isl, rt])
    beta, *_ = np.linalg.lstsq(X, rgd, rcond=None)
    return float(beta[1]), int((isl == 1).sum())

def correct_only_gap(covs, tt, ic, scale):
    """Gap + ORDINAL SPACING restricted to isCorrect==1 trials."""
    C = covs * scale
    t_idx  = np.where(tt == 1)[0]
    nt_idx = np.where(tt == 2)[0]
    l_idx  = np.where(tt == 3)[0]
    tC = t_idx[ic[t_idx] == 1]
    nC = nt_idx[ic[nt_idx] == 1]
    lC = l_idx[ic[l_idx] == 1]
    if len(tC) < 6 or len(lC) < 4:
        return None
    d_t = np.array([
        distance_riemann(mean_riemann(C[np.delete(tC, k)]), C[i])
        for k, i in enumerate(tC)
    ])
    ref  = mean_riemann(C[tC])
    d_l  = np.array([distance_riemann(ref, C[i]) for i in lC])
    d_nt = np.array([distance_riemann(ref, C[i]) for i in nC]) if len(nC) else np.array([])
    g_NT_T = cohens_d(d_nt, d_t) if len(d_nt) else np.nan
    g_L_NT = cohens_d(d_l, d_nt) if len(d_nt) else np.nan
    return {
        'gap'           : cohens_d(d_l, d_t),
        'g_NT_T'        : g_NT_T,   # want ~0 for intent-specific signal
        'g_L_NT'        : g_L_NT,   # want >0 for intent-specific signal
        'mT'            : float(d_t.mean()),
        'mNT'           : float(d_nt.mean()) if len(d_nt) else np.nan,
        'mL'            : float(d_l.mean()),
        'n_lure_all'    : len(l_idx),
        'n_lure_correct': len(lC),
    }


# ── Load batches ──────────────────────────────────────────────────────────────

def main():
    results = []
    for bid in range(5):
        p = os.path.join(BATCH_DIR, f'batch_{bid}_results.npy')
        if os.path.exists(p):
            results.extend(np.load(p, allow_pickle=True).tolist())
            print(f'Batch {bid}: loaded')

    stable = [r for r in results
              if r.get('dist') is not None
              and r['dist'].get('stable')
              and 'covs' in r['dist']]
    print(f'\nTotal: {len(results)} | Stable w/ covs: {len(stable)}')
    if len(stable) < 20:
        print('[HALT] too few stable subjects.'); return

    have_meta = os.path.isdir(DATA_DIR)
    if not have_meta:
        print('[WARN] nsvd-fusion not found: Sections C + D will be skipped.')

    # Accumulators
    g_saved, g_corr, scale_delta = [], [], []
    g_NT_T, g_L_NT = [], []
    ord_T, ord_NT, ord_L = [], [], []
    rt_betas, rt_nan_total = [], 0
    co_gaps, co_nlure, co_raw = [], [], []
    exec_minus_motor = []

    for r in stable:
        d = r['dist']
        covs = d['covs']
        tt   = np.asarray(d['tt'])

        # A. Scale fix
        d_t, d_nt, d_l, t_idx, nt_idx, l_idx = recompute(covs, tt, COV_SCALE)
        g_c = cohens_d(d_l, d_t)
        g_corr.append(g_c)
        g_saved.append(d['gap'])
        scale_delta.append(abs(g_c - d['gap']))

        # B. Ordinal spacings
        g_NT_T.append(cohens_d(d_nt, d_t))
        g_L_NT.append(cohens_d(d_l, d_nt))
        ord_T.append(d_t.mean()); ord_NT.append(d_nt.mean()); ord_L.append(d_l.mean())

        # E. Coarse motor incremental (scalar proxy)
        if r.get('motor_gap') is not None:
            exec_minus_motor.append(g_c - float(r['motor_gap']))

        if not have_meta:
            continue

        # Reload y_meta for RT + isCorrect
        try:
            y = np.load(os.path.join(DATA_DIR, f"{r['subj']}_y_meta.npy"))
        except FileNotFoundError:
            continue
        m2 = y[:, 0] == 2
        tt_chk = y[m2, 1].astype(int)
        if len(tt_chk) != len(tt) or not np.array_equal(tt_chk, tt):
            continue
        rt2 = y[m2, 2]
        ic2 = y[m2, 3].astype(float)

        # C. RT incremental validity
        beta, n_l = rt_incremental_beta(d_t, d_l, rt2[t_idx], rt2[l_idx])
        if beta is not None:
            rt_betas.append(beta)
        rt_nan_total += int(np.isnan(np.concatenate([rt2[t_idx], rt2[l_idx]])).sum())

        # D. Correct-only
        co = correct_only_gap(covs, tt, ic2, COV_SCALE)
        if co is not None:
            co_gaps.append(co['gap'])
            co_nlure.append((co['n_lure_all'], co['n_lure_correct']))
            co_raw.append(co)

    n = len(g_corr)

    # ── A. Scale fix ──────────────────────────────────────────────────────────
    print('\n' + '=' * 64)
    print('A  SCALE FIX  (did the missing 1e13 matter?)')
    print('=' * 64)
    print(f'  Gap as-saved              : {np.mean(g_saved):+.4f}')
    print(f'  Gap corrected (x1e26)     : {np.mean(g_corr):+.4f}')
    print(f'  Max per-subj |delta gap|  : {np.max(scale_delta):.4f}')
    print(f'  Mean per-subj |delta gap| : {np.mean(scale_delta):.4f}')
    settled = np.max(scale_delta) < 0.02
    print(f'  Verdict: {"NUMERICALLY HARMLESS — failures are real." if settled else "MATERIAL — trust x1e26 column."}')
    np.save(os.path.join(OUT_DIR, 'reanalysis_g_corr.npy'), np.array(g_corr))

    # ── B. Ordinal spacing ────────────────────────────────────────────────────
    print('\n' + '=' * 64)
    print('B  ORDINAL SPACING  (is NT clustered WITH T?)')
    print('=' * 64)
    print(f'  T / NT / L  (mean dist)   : {np.mean(ord_T):.4f} / {np.mean(ord_NT):.4f} / {np.mean(ord_L):.4f}')
    print(f'  g(NT vs T)  [want ~0]     : {np.mean(g_NT_T):+.4f}   (Wilcoxon p={wp(g_NT_T):.4f})')
    print(f'  g(L  vs NT) [want >0]     : {np.mean(g_L_NT):+.4f}   (Wilcoxon p={wp(g_L_NT):.4f})')
    nt_clusters  = abs(np.mean(g_NT_T)) < 0.10 and not (wp(g_NT_T) < 0.05 and np.mean(g_NT_T) > 0)
    l_separates  = np.mean(g_L_NT) > 0.15 and wp(g_L_NT) < 0.05
    ordinal_ok   = nt_clusters and l_separates
    print(f'  NT clusters with T        : {"YES" if nt_clusters else "NO -- NT has lifted toward L"}')
    print(f'  L separates from NT       : {"YES" if l_separates else "NO"}')
    print(f'  Intent-specific ordinal   : {"PASS" if ordinal_ok else "FAIL (response/novelty axis)"}')

    # ── C. RT incremental validity ────────────────────────────────────────────
    print('\n' + '=' * 64)
    print('C  RT INCREMENTAL VALIDITY  (is_lure coef beyond RT)')
    print('=' * 64)
    if rt_betas:
        rb = np.asarray(rt_betas)
        print(f'  Subjects w/ usable RT     : {len(rb)}   (NaN-RT trials dropped: {rt_nan_total})')
        print(f'  Mean is_lure coef (SD u.) : {rb.mean():+.4f}')
        print(f'  Coef > 0                  : {int((rb>0).sum())}/{len(rb)}   (Wilcoxon p={wp(rb):.4f})')
        rt_ok = rb.mean() > 0 and wp(rb) < 0.05
        print(f'  Survives RT (correct test): {"PASS" if rt_ok else "FAIL"}')
        print(f'  CAVEAT: correct Lure withholds have RT=NaN and are dropped here.')
        print(f'          Lures WITH RT are mostly misfires. Read with Section D.')
    else:
        rt_ok = None
        print('  Skipped (nsvd-fusion not mounted).')

    # ── D. Correct-only + ORDINAL (the decisive fork) ────────────────────────
    print('\n' + '=' * 64)
    print('D  CORRECT-ONLY ORDINAL  (adjudicates B vs D)')
    print('   The fork: does NT drop back to T on correct trials?')
    print('=' * 64)
    if co_gaps:
        cg     = np.asarray(co_gaps)
        g_NT_T_co = np.array([c['g_NT_T'] for c in co_raw if c is not None and not np.isnan(c['g_NT_T'])])
        g_L_NT_co = np.array([c['g_L_NT'] for c in co_raw if c is not None and not np.isnan(c['g_L_NT'])])
        mT_co  = np.mean([c['mT']  for c in co_raw if c is not None])
        mNT_co = np.nanmean([c['mNT'] for c in co_raw if c is not None])
        mL_co  = np.mean([c['mL']  for c in co_raw if c is not None])
        avg_all  = np.mean([a for a, _ in co_nlure])
        avg_kept = np.mean([b for _, b in co_nlure])
        print(f'  Subjects (>= trial floor) : {len(cg)}')
        print(f'  Lure n: all -> correct    : {avg_all:.1f} -> {avg_kept:.1f} per subject')
        print(f'  Correct-only mean gap     : {cg.mean():+.4f}   (Wilcoxon p={wp(cg):.4f})')
        print()
        print(f'  T / NT / L (correct only) : {mT_co:.4f} / {mNT_co:.4f} / {mL_co:.4f}')
        print(f'  g(NT vs T) [want ~0]      : {np.mean(g_NT_T_co):+.4f}   (Wilcoxon p={wp(g_NT_T_co):.4f})')
        print(f'  g(L  vs NT)[want >0]      : {np.mean(g_L_NT_co):+.4f}   (Wilcoxon p={wp(g_L_NT_co):.4f})')
        nt_drops  = abs(np.mean(g_NT_T_co)) < 0.10 and not (wp(g_NT_T_co) < 0.05 and np.mean(g_NT_T_co) > 0)
        l_pulls   = np.mean(g_L_NT_co) > 0.15 and wp(g_L_NT_co) < 0.05
        co_ok = cg.mean() > 0 and wp(cg) < 0.05
        print()
        if nt_drops and l_pulls:
            print('  VERDICT: ORDINAL RECOVERS on correct trials.')
            print('           Intent signal exists under correct performance.')
            print('           Confound lives in error trials. Proceed to Phase 2.')
        elif not nt_drops:
            print('  VERDICT: NT stays up with L even on correct trials.')
            print('           Confound is total. Task cannot separate intent from response.')
            print('           This is the paradigm-design structural advantage arriving as a finding.')
        else:
            print('  VERDICT: Partial recovery. L pulls from NT but NT still elevated.')
            print('           Mixed signal. Read with oracle before Phase 2.')
    else:
        co_ok = None
        print('  Skipped (nsvd-fusion not mounted).')


    # ── E. Coarse motor incremental ───────────────────────────────────────────
    print('\n' + '=' * 64)
    print('E  COARSE MOTOR-INCREMENTAL  (scalar proxy, weak)')
    print('=' * 64)
    if exec_minus_motor:
        em = np.asarray(exec_minus_motor)
        print(f'  exec_gap - motor_gap      : {em.mean():+.4f}   (Wilcoxon p={wp(em):.4f})')
        print(f'  exec > motor              : {int((em>0).sum())}/{len(em)}')
    print(f'  [Detail]: per-trial motor null needs Phase 2 map (covs_m now saved).')

    # ── Honest verdict ────────────────────────────────────────────────────────
    print('\n' + '=' * 64)
    print('HONEST VERDICT')
    print('=' * 64)
    print(f'  Scale bug         : {"settled (harmless)" if settled else "MATERIAL"}')
    print(f'  Ordinal (NT~T)    : {"PASS" if ordinal_ok else "FAIL"}')
    print(f'  RT incremental    : {"PASS" if rt_ok else ("FAIL" if rt_ok is False else "n/a")}')
    print(f'  Correct-only      : {"PASS" if co_ok else ("FAIL/weak" if co_ok is False else "n/a")}')
    print(f'  Motor (per-trial) : PENDING -- Phase 2 map re-run required')
    print('=' * 64)


if __name__ == '__main__':
    main()
