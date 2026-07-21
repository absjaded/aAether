"""
synth_validation_scaffold.py  —  PHASE A: Synthetic Instrument-Validation
========================================================================
Purpose (AGENTS_v8 §9, ENGINEERING_v8 §7, RESEARCH_v8 §XI):
  Prove the Riemannian INSTRUMENT recovers a RESPONSE-DECOUPLED goal-violation
  signal under realistic noise, stays NULL on the negative controls, and produce
  a RECOVERY CURVE that SIZES the real EEG collection.

  This proves the instrument. It does NOT prove the phenomenon — no brain is
  involved. at no point report a synthetic recovery as proof intent geometry exists.

What is LIFTED, not reinvented (from the validated public-data harness — Tier 2):
  condition_spd, cohens_d, Euclidean Alignment, the LOO reference + split-half
  gate, the cohort-permutation structure. Signatures match ENGINEERING_v8.

THE FIREWALL (non-negotiable — RESEARCH_v8 §XI):
  The generator injects a GENERIC separable SPD shift on violating-labelled trials.
  It must NOT encode the paradigm's assumed structure. If the injected signal is
  shaped like what the paradigm predicts, the validation flatters the design and
  proves nothing. Keep the generator dumb and generic on purpose.

This is a SCAFFOLD: the data-generating model is deliberately simple and explicit.
Extend the realism (calibrate nuisance from real covariances) where marked [Pending];
do not restructure the gates or the firewall.
========================================================================
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pyriemann.estimation import ERPCovariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.base import invsqrtm
from scipy.stats import wilcoxon

rng_state = 42
rng_global = np.random.default_rng(rng_state)

# ── Schema constants (ENGINEERING_v8 §0; values are illustrative, fixed at PILOT) ──
GOAL_CONGRUENT, GOAL_ODDBALL, GOAL_VIOLATING = 1, 2, 3
SCALE = 1e6          # modality-specific underflow guard; EEG/synthetic units, NOT the MEG 1e13
N_PERMS = 1000       # RESEARCH_v8 §V floor — at no point below 1000

# ── Calibration pool (nuisance realism) ───────────────────────────────────────
# Three calibration states, in increasing trustworthiness for SIZING the real run:
#   (0) None        → generic Gaussian nuisance. Validates instrument BEHAVIOUR only;
#                     recovery-curve NUMBERS are NOT trustworthy for collection sizing.
#   (1) public MEG  → real SPD nuisance STRUCTURE from saved public covariances. Better
#                     realism, but WRONG modality/dimensionality (MEG, not EEG montage).
#   (2) EEG pilot   → real EEG nuisance. The ONLY state whose recovery-curve numbers may
#                     be quoted in an ethics_assessment power analysis / proposal. Does not exist pre-pilot.
# The FIREWALL is unchanged: calibration shapes only the NUISANCE (base_cov). The injected
# violation effect stays generic — at no point calibrated to the paradigm.
CALIB_POOL = None    # list of real SPD covariances, or None

def load_calibration_pool(batch_dir, max_covs=3000):
    """State (1): pool real SPD covariances from saved public-data batches.
    Each batch file is a list of subject dicts with r['dist']['covs'] (N,d,d).
    Sets the module global CALIB_POOL. Returns the pool size (0 if none found)."""
    import os
    global CALIB_POOL
    pool = []
    for bid in range(5):
        p = os.path.join(batch_dir, f'batch_{bid}_results.npy')
        if not os.path.exists(p):
            continue
        for r in np.load(p, allow_pickle=True).tolist():
            d = r.get('dist')
            if d is not None and 'covs' in d:
                for C in d['covs']:
                    pool.append(C)
                    if len(pool) >= max_covs:
                        break
    CALIB_POOL = pool if pool else None
    print(f"[calibration] pooled {len(pool)} real covariances "
          f"({'STATE 1: MEG-structure realism' if pool else 'STATE 0: none → generic Gaussian'})")
    return len(pool)

def draw_base_cov(n_chan, rng):
    """Per-subject nuisance covariance. Uses the real pool if loaded (a principal
    submatrix of a real SPD cov is still SPD and carries real correlation structure),
    else a generic random SPD. n_chan stays the knob in both cases."""
    if CALIB_POOL is not None:
        C = CALIB_POOL[rng.integers(len(CALIB_POOL))]
        d = C.shape[0]
        if d >= n_chan:
            sub = np.asarray(C)[:n_chan, :n_chan]
        else:                                   # pad if pool dim < requested
            sub = np.eye(n_chan); sub[:d, :d] = C
        return (sub + sub.T) / 2 + np.eye(n_chan) * 1e-6
    A = rng.standard_normal((n_chan, n_chan))
    return A @ A.T / n_chan + np.eye(n_chan) * 0.5

# ════════════════════════════════════════════════════════════════════════
# LIFTED BACKBONE  (verbatim from the validated harness — do not rewrite)
# ════════════════════════════════════════════════════════════════════════

def condition_spd(C, eps=1e-6):
    C = (C + C.transpose(0, 2, 1)) / 2
    for i in range(C.shape[0]):
        tr = np.trace(C[i])
        C[i] += np.eye(C.shape[-1]) * (tr * eps if tr > 0 else eps)
    return C

def cohens_d(a, b):
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) + 1e-12
    return float((np.mean(a) - np.mean(b)) / pooled)

def per_subject_euclidean_alignment(cov):
    """Within-subject whitening (ENGINEERING_v8 §1). cov: (N,d,d) SPD for ONE subject."""
    R = cov.mean(axis=0)
    P = invsqrtm(R)
    return P @ cov @ P

def subject_gap(covs, gc):
    """
    Per-subject standardized gap + ordinal triplet, on conditioned+aligned covs.
    Reference = goal-intact (congruent) geometry. Congruent scored LOO; violating
    and oddball scored against the full congruent mean. (ENGINEERING_v8 §4)
    Returns None if underpowered.
    """
    cong = np.where(gc == GOAL_CONGRUENT)[0]
    odd  = np.where(gc == GOAL_ODDBALL)[0]
    viol = np.where(gc == GOAL_VIOLATING)[0]
    if len(cong) < 6 or len(viol) < 4:
        return None

    # Gate 0 — split-half reference stability (RESEARCH_v8 Gate 0), runs first
    rng = np.random.default_rng(rng_state)
    perm = rng.permutation(cong); h = len(cong) // 2
    C_sh = mean_riemann(covs[perm[:h]])
    d_sh = np.array([distance_riemann(C_sh, covs[i]) for i in perm[h:]])
    d_loo_sh = np.array([
        distance_riemann(mean_riemann(covs[np.delete(cong, np.where(cong == si)[0][0])]), covs[si])
        for si in perm[h:]
    ])
    pooled_sd = np.std(np.concatenate([d_sh, d_loo_sh])) + 1e-12
    stable = bool(abs(d_sh.mean() - d_loo_sh.mean()) / pooled_sd < 0.50)

    # LOO congruent; violating & oddball vs full reference
    d_cong = np.array([distance_riemann(mean_riemann(covs[np.delete(cong, k)]), covs[i])
                       for k, i in enumerate(cong)])
    C_ref = mean_riemann(covs[cong])
    d_viol = np.array([distance_riemann(C_ref, covs[i]) for i in viol])
    d_odd  = np.array([distance_riemann(C_ref, covs[i]) for i in odd]) if len(odd) else np.array([])

    return {
        'stable': stable,
        'gap': cohens_d(d_viol, d_cong),                      # violation vs congruent
        'g_odd_cong': cohens_d(d_odd, d_cong) if len(odd) else np.nan,   # want ~0
        'g_viol_odd': cohens_d(d_viol, d_odd) if len(odd) else np.nan,   # want >0
        'covs': covs, 'gc': gc,
    }

# ════════════════════════════════════════════════════════════════════════
# THE GENERATOR  (generic, firewalled — must NOT mirror the paradigm)
# ════════════════════════════════════════════════════════════════════════

def make_subject_raw(n_chan, T, n_per_cond, effect, response_confound, motor_arm, rng,
                     artifact=0.0, artifact_leak=0.0):
    """
    Generate one subject's RAW windowed signal (N, n_chan, T) + labels.

    GENERIC model only:
      - base trials = correlated Gaussian noise with a random per-subject channel
        covariance (nuisance; stands in for individual cortical/skull geometry).
      - VIOLATING trials get a generic separable covariance perturbation of size
        `effect` along a random low-rank direction — NOT a paradigm-shaped pattern.
      - response label is assigned INDEPENDENTLY of goal label (LAW 1 — orthogonal).

    EEG REALISM (ENGINEERING_v8 §0a) — models POST-preprocessing RESIDUAL artifact,
    i.e. what ICA/ASR leaves behind, NOT pristine Gaussian:
      artifact:      strength of residual ocular (frontal, low-rank, high-variance) +
                     EMG (lateral, broadband) + line (fixed spatial) components, added
                     INDEPENDENTLY of condition (nuisance — firewall-safe).
      artifact_leak: STRESS TEST ONLY. If >0, the ocular component is made stronger on
                     VIOLATING trials (artifact correlated with condition). A correct
                     instrument MUST then show a gap that DIES under the ocular covariate
                     (Gate 3). This arm exists to find the danger zone, not to pass.
    """
    conds = np.repeat([GOAL_CONGRUENT, GOAL_ODDBALL, GOAL_VIOLATING], n_per_cond)
    N = len(conds)
    resp = rng.integers(0, 2, size=N)                       # orthogonal to goal — LAW 1

    base_cov = draw_base_cov(n_chan, rng)       # real nuisance if CALIB_POOL loaded, else generic

    u = rng.standard_normal((n_chan, 1)); u /= np.linalg.norm(u)   # generic signal direction
    P_perturb = u @ u.T

    # residual-artifact spatial templates (frontal ocular, lateral EMG, fixed line)
    ocular = np.zeros((n_chan, 1)); ocular[:max(1, n_chan // 4)] = 1.0   # frontal channels
    ocular /= np.linalg.norm(ocular)
    emg = rng.standard_normal((n_chan, 1)); emg[:n_chan // 2] *= 0.2      # lateral-weighted
    emg /= np.linalg.norm(emg)
    P_ocular, P_emg = ocular @ ocular.T, emg @ emg.T

    X = np.empty((N, n_chan, T))
    for i in range(N):
        cov_i = base_cov.copy()
        if not motor_arm and conds[i] == GOAL_VIOLATING and response_confound == 0.0:
            cov_i = cov_i + effect * P_perturb               # signal on the GOAL axis
        if response_confound > 0.0 and resp[i] == 1:
            cov_i = cov_i + response_confound * P_perturb     # signal on the RESPONSE axis
        if artifact > 0.0:                                   # residual artifact (nuisance)
            oc = artifact * (1.0 + (artifact_leak if conds[i] == GOAL_VIOLATING else 0.0))
            cov_i = cov_i + oc * P_ocular + 0.5 * artifact * P_emg
        Li = np.linalg.cholesky(cov_i + np.eye(n_chan) * 1e-6)
        x = Li @ rng.standard_normal((n_chan, T))
        if artifact > 0.0:                                   # fixed-spatial line component
            x += artifact * 0.3 * (ocular @ np.sin(2 * np.pi * 0.25 * np.arange(T))[None, :])
        X[i] = x
    return X, conds, resp

def covs_from_raw(X, gc, estimator='erp'):
    """
    ENGINEERING_v8 §2/§4: scale → covariance → condition_spd → EA.
    estimator='erp'  : ERPCovariances(congruent prototype) — more sensitive to elicited_response
                       signal, but the RAW gap carries a prototype-membership bias
                       (non-prototype classes look farther for free). Only the ORDINAL
                       (violation vs oddball, both non-prototype) is bias-free.
    estimator='plain': Covariances — unbiased raw gap, may be less sensitive to elicited_response.
    The choice is a PRE-REGISTERED decision the recovery curve adjudicates (see report).
    """
    from pyriemann.estimation import Covariances
    Xs = X * SCALE
    if estimator == 'erp':
        covs = ERPCovariances(classes=[GOAL_CONGRUENT], estimator='lwf').fit_transform(Xs, gc)
    else:
        covs = Covariances(estimator='lwf').fit_transform(Xs)
    covs = condition_spd(covs)
    return per_subject_euclidean_alignment(covs)

# ════════════════════════════════════════════════════════════════════════
# COHORT + CONTROLS + RECOVERY CURVE
# ════════════════════════════════════════════════════════════════════════

def run_cohort(n_subj, n_chan, T, n_per_cond, effect,
               response_confound=0.0, motor_arm=False, estimator='erp',
               artifact=0.0, artifact_leak=0.0, rng_state=rng_state):
    rng = np.random.default_rng(rng_state)
    gaps, g_oc, g_vo, stab = [], [], [], []
    for _ in range(n_subj):
        X, gc, resp = make_subject_raw(n_chan, T, n_per_cond, effect,
                                       response_confound, motor_arm, rng,
                                       artifact=artifact, artifact_leak=artifact_leak)
        res = subject_gap(covs_from_raw(X, gc, estimator), gc)
        if res is None:
            continue
        gaps.append(res['gap']); g_oc.append(res['g_odd_cong'])
        g_vo.append(res['g_viol_odd']); stab.append(res['stable'])
    gaps = np.array(gaps); g_vo_arr = np.array(g_vo)
    return {
        'mean_d': float(gaps.mean()) if len(gaps) else np.nan,        # raw gap (ERP: biased)
        'mean_ord': float(np.nanmean(g_vo_arr)) if len(g_vo_arr) else np.nan,  # PRIMARY, bias-free
        'n_pos_ord': int((g_vo_arr > 0).sum()),
        'n': len(gaps),
        'wilcoxon_p_ord': float(wilcoxon(g_vo_arr).pvalue) if len(g_vo_arr) >= 6 and np.any(g_vo_arr != 0) else np.nan,
        'g_odd_cong': float(np.nanmean(g_oc)) if len(g_oc) else np.nan,
        'frac_stable': float(np.mean(stab)) if len(stab) else np.nan,
        'gaps': gaps, 'ord': g_vo_arr,
    }

def check_negatives(n_subj, n_chan, T, n_per_cond, estimator='erp'):
    """The instrument MUST pass these on the ORDINAL (bias-free) or it is broken."""
    print(f"\n── NEGATIVE CONTROLS  [estimator={estimator}]  (judged on bias-free ORDINAL) ──")
    def line(name, r, want_null=True):
        v = r['mean_ord']
        ok = abs(v) < 0.15 if want_null else v > 0.15
        print(f"  {name:<24}: ordinal d={v:+.3f}  raw={r['mean_d']:+.3f}  "
              f"{'OK' if ok else 'FAIL'}")
        return ok
    a = line("no-violation null", run_cohort(n_subj, n_chan, T, n_per_cond, 0.0, estimator=estimator))
    b = line("response-confounded", run_cohort(n_subj, n_chan, T, n_per_cond, 0.0, response_confound=2.0, estimator=estimator))
    c = line("motor-arm null", run_cohort(n_subj, n_chan, T, n_per_cond, 0.0, motor_arm=True, estimator=estimator))
    # EEG residual artifact, independent of condition → must stay null (nuisance only)
    d = line("residual-artifact null", run_cohort(n_subj, n_chan, T, n_per_cond, 0.0, artifact=1.5, estimator=estimator))
    # STRESS: artifact CORRELATED with violation → instrument SHOULD show a (false) gap here.
    # This is the danger zone §0a names; it must be killed by the Gate-3 ocular covariate on
    # real data. Here we only confirm the leak is detectable (a non-null is EXPECTED).
    leak = run_cohort(n_subj, n_chan, T, n_per_cond, 0.0, artifact=1.5, artifact_leak=1.5, estimator=estimator)
    print(f"  {'condition-leak STRESS':<24}: ordinal d={leak['mean_ord']:+.3f}  "
          f"← EXPECTED non-null; on real data this MUST die under the ocular covariate (Gate 3)")
    print("  (raw gap under ERP is expected to be inflated by prototype membership;")
    print("   the ordinal is the readout that must be null on true nuisance — see report.)")
    return a and b and c and d

def recovery_curve(n_chan=16, T=120, effects=(0.0, 0.25, 0.5, 1.0, 2.0),
                   trial_counts=(15, 30, 60), n_subj=40, estimator='erp'):
    """
    The deliverable (RESEARCH_v8 §XI): ORDINAL detection reliability over
    (effect size × trials/condition). This SIZES the real EEG collection.
    Reports the bias-free ordinal (violation vs oddball), not the biased raw gap.
    """
    print("=" * 66)
    print(f"RECOVERY CURVE [estimator={estimator}] — ORDINAL detection vs (effect × trials)")
    print(f"  n_subj={n_subj}  n_chan={n_chan}  T={T}")
    print("=" * 66)
    print(f"  {'effect':>7} | " + " | ".join(f"n={nc:<3}" for nc in trial_counts))
    for eff in effects:
        cells = []
        for nc in trial_counts:
            r = run_cohort(n_subj, n_chan, T, nc, effect=eff, estimator=estimator)
            cells.append(f"ord={r['mean_ord']:+.2f}({r['n_pos_ord']}/{r['n']})")
        print(f"  {eff:>7.2f} | " + " | ".join(f"{c:<15}" for c in cells))
    print("\n  Read: smallest (effect, n) cell with ordinal clearly >0 AND majority")
    print("  in-direction is the floor the real EEG paradigm must clear → sets")
    print("  trials/condition and n_subjects for the collection.")

# ════════════════════════════════════════════════════════════════════════

def main():
    import sys
    n_subj, n_chan, T, n_per_cond = 40, 16, 120, 30

    # Optional state-(1) calibration: `python synth_validation_scaffold.py <batch_dir>`
    if len(sys.argv) > 1:
        n_loaded = load_calibration_pool(sys.argv[1])
        if n_loaded:
            n_chan = min(n_chan, int(np.asarray(CALIB_POOL[0]).shape[0]))
            print(f"[calibration] using real nuisance; n_chan set to {n_chan}")
    else:
        print("[calibration] STATE 0 (generic Gaussian). Pass a batch_dir to calibrate. "
              "Recovery-curve NUMBERS are not collection-grade until EEG-pilot calibration.")

    print("=" * 66)
    print("PHASE A — SYNTHETIC INSTRUMENT VALIDATION")
    print("  Proves the instrument, NOT the phenomenon. Firewall: generic shift only.")
    print("  PRIMARY readout = the bias-free ORDINAL (violation vs oddball).")
    print("=" * 66)

    # 1) sanity: instrument recovers a clear injected signal (ordinal)
    pos = run_cohort(n_subj, n_chan, T, n_per_cond, effect=1.0)
    print(f"\n── POSITIVE (effect=1.0, estimator=erp) ──")
    print(f"  ORDINAL d={pos['mean_ord']:+.3f}  in-direction {pos['n_pos_ord']}/{pos['n']}  "
          f"wilcoxon p={pos['wilcoxon_p_ord']:.2e}  stable {pos['frac_stable']*100:.0f}%")
    print(f"  (raw gap={pos['mean_d']:+.3f} — inflated by ERP prototype bias; ordinal is the truth)")

    # 2) negatives must be null ON THE ORDINAL, under both estimators
    erp_ok   = check_negatives(n_subj, n_chan, T, n_per_cond, estimator='erp')
    plain_ok = check_negatives(n_subj, n_chan, T, n_per_cond, estimator='plain')
    print(f"\n  Negatives null (ordinal): ERP={'YES' if erp_ok else 'NO'}  "
          f"plain={'YES' if plain_ok else 'NO'}")

    # 3) the deliverable — adjudicate estimator via detection on the ordinal
    print()
    recovery_curve(n_chan=n_chan, T=T, n_subj=n_subj, estimator='erp')
    print()
    recovery_curve(n_chan=n_chan, T=T, n_subj=n_subj, estimator='plain')

    print("\n" + "=" * 66)
    print("ESTIMATOR DECISION (pre-register from the two curves above):")
    print("  ERP   = more sensitive to elicited_response signal, raw gap biased → ordinal-only.")
    print("  plain = unbiased throughout, possibly less sensitive to elicited_response.")
    print("  Pick the one with higher ordinal detection at the lowest (effect,n) cell.")
    print("NEXT when extending:")
    print("  - [Pending]: calibrate base_cov from REAL covariances (keep injection generic)")
    print("  - add cohort PERMUTATION on the ordinal (>=1000)")
    print("  - add the perceptual-response covariate check (LAW 1, Gate 3)")
    print("  - FIREWALL stays: at no point shape the injected signal like the paradigm.")
    print("=" * 66)

if __name__ == '__main__':
    main()
