"""
c2_rgd_three_gate.py — Canonical C2 RGD three-gate validation pipeline
======================================================================
Implements the FULL c02RESEARCH.md §III cascade exactly as specified:

  GATE 0 — Noise floor:        held-out 0-back probe RGD distribution
  GATE 1 — Load effect:        Correct-2back  vs  Probe-0back  (MW one-sided >)
  GATE 2 — Friction effect:    Incorrect-2back vs Correct-2back (MW one-sided >)
                                + monotonic ordering Null < Correct < Incorrect

Two tiers per gate:
  tier 1 (per subject): Mann-Whitney U + Cohen's d (pooled SD). p<0.05, d>0.3.
  tier 2 (cohort):      Wilcoxon signed-rank on per-subject d's. median(d)>0.3.

This REPLACES rgd_sprint1.py. The fixes it codifies vs that script:
  [FIX A] Disjoint initialized 50/50 anchor/probe split of the 0-back pool (§7).
          rng_state = 42 + int(subject_id). at no point re-randomized. No trial in both.
  [FIX B] Gate 0 noise floor computed from the probe pool (was entirely absent).
  [FIX C] Gate 1 load effect (Correct-2back > Probe-0back) (was absent).
  [FIX D] Monotonic three-tier ordering enforced as a structural falsifier.
  [FIX E] tier-2 Wilcoxon + formal PASS/FALSIFIED/INCONCLUSIVE classification.
  [FIX F] Correct 0-back-only anchor (anchor purity, §7 rationale).

  ANCHOR MODE SWITCH (testable hypothesis, not assumption):
    mode="full_zone"   — anchor = Frechet mean of 43-sample full-zone covariances
                          (the ORIGINAL construction with the OAS mismatch)
    mode="sliding"     — anchor = Frechet mean of 12-window 20-sample covariances
                          (the Sprint-1 OAS-mismatch FIX)
  Both run; the comparison IS the test of whether the OAS mismatch mattered.

Locked constants (c02ENGINEERING.md / c02RESEARCH.md §VI):
  Spatial mask:   12 Tier-1 ROIs (indices below)
  Temporal zone:  samples 132..175 (528..700ms @ 250Hz)
  Sliding window: W=20 (80ms), step=2 (8ms), 12 windows/trial
  Shrinkage:      OAS (Ledoit-Wolf family)  -- see observation on OAS below
  Distance:       AIRM (distance_riemann)
  Min 0-back:     >=20;  Min 2-back per class: >=5;  Min admissible: 30

[Detail]: ON OAS vs LWF:
  c02ENGINEERING.md §5 mandates OAS. pyriemann's Covariances(estimator='oas')
  exposes sklearn's OAS. We use it for the probe/sliding matrices. For the
  full-zone anchor we likewise use OAS so the mismatch is purely window-size,
  not estimator-type. (rgd_sprint1 used 'lwf' — corrected here to 'oas'.)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import json
import time
import numpy as np
import scipy.stats as ss
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

# ---------------------------------------------------------------- config
DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".data", "nsvd_fusion")),
)
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", "c2")),
)
os.makedirs(RESULTS_DIR, exist_ok=True)

rng_state = 42
EXCLUDED_SUBJECTS = {'140117', '204521'}  # assessd but flagged; c2AGENTS §I NonNeg 6

# 12 Tier-1 Executive Control ROIs (DK atlas, L-then-R order)
TIER1_ROI_INDICES = [1, 2, 6, 25, 26, 29, 35, 36, 40, 59, 60, 63]

ZONE_START = 132   # 528ms
ZONE_END = 175     # 700ms (exclusive in slices we treat as [...175)
WINDOW_SIZE = 20   # 80ms
WINDOW_STEP = 2    # 8ms
MIN_0BACK = 20
MIN_PER_CLASS_2BACK = 5
MIN_ADMISSIBLE_COHORT = 30

# Observation: ZONE covers samples 132..174 inclusive => 43 samples. 12 sliding windows:
# starts = 132,134,...,155; last window [155,175) => 20 samples. Confirm.
ZONE_W = ZONE_END - ZONE_START  # 43
N_WINDOWS = (ZONE_W - WINDOW_SIZE) // WINDOW_STEP + 1
assert N_WINDOWS == 12, f"expected 12 windows, got {N_WINDOWS}"


# ---------------------------------------------------------------- helpers
def condition_spd(C, eps=1e-6):
    C = (C + C.T) * 0.5
    tr = np.trace(C)
    C += np.eye(C.shape[0]) * (tr * eps if tr > 0 else eps)
    return C


def oas_cov(X_windows):
    """X_windows: (M, 12, W) -> (M,12,12) OAS-shrunk SPD, conditioned."""
    cov = Covariances(estimator='oas').fit_transform(X_windows)
    for i in range(cov.shape[0]):
        cov[i] = condition_spd(cov[i])
    return cov


def trial_windows(X_trial_tier1):
    """Return (12, 12, 20) sliding windows for one trial."""
    return np.stack([
        X_trial_tier1[:, s:s + WINDOW_SIZE]
        for s in range(ZONE_START, ZONE_END - WINDOW_SIZE + 1, WINDOW_STEP)
    ])


def rgd_curve_max(X_trial_tier1, anchor):
    """Peak RGD = max over 12 windows of AIRM(C_w, anchor)."""
    wins = trial_windows(X_trial_tier1)         # (12,12,20)
    covs = oas_cov(wins)                         # (12,12,12)
    return float(max(distance_riemann(c, anchor) for c in covs))


def cohens_d_indep(a, b):
    """Pooled-SD Cohen's d (directional: a - b)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float('nan')
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return float('nan')
    return float((a.mean() - b.mean()) / pooled)


def mw_one_sided_greater(x, y):
    """Mann-Whitney U one-sided alternative: x > y. Returns (U, p, direction_ok)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 1 or len(y) < 1:
        return float('nan'), float('nan'), False
    try:
        U, p = ss.mannwhitneyu(x, y, alternative='greater')
    except ValueError:
        return float('nan'), float('nan'), False
    return float(U), float(p), True


# ---------------------------------------------------------------- per-subject
def load_subject(sid):
    X = np.load(os.path.join(DATA_DIR, f"{sid}_X.npy"))
    y_sem = np.load(os.path.join(DATA_DIR, f"{sid}_y_semantic.npy"))
    y_load = np.load(os.path.join(DATA_DIR, f"{sid}_y.npy"))
    assert len(X) == len(y_sem) == len(y_load), f"len mismatch {sid}"
    return X, y_sem, y_load


def split_zero_back(zero_idx, sid):
    """Disjoint 50/50 anchor/probe split, initialized per subject. §7."""
    rng = np.random.RandomState(rng_state + int(sid))
    perm = rng.permutation(len(zero_idx))
    half = len(zero_idx) // 2
    anchor_idx = zero_idx[perm[:half]]
    probe_idx = zero_idx[perm[half:]]
    return anchor_idx, probe_idx


def build_anchor(X_t1, anchor_trial_idx, mode):
    """Construct the 12x12 Fréchet-mean anchor.
    mode='sliding':   pool all 12 windows per anchor trial -> Frechet mean (Sprint-1 fix)
    mode='full_zone': one OAS cov per trial over full 43-sample zone -> Frechet mean (original)
    """
    if mode == 'sliding':
        win_list = []
        for i in anchor_trial_idx:
            win_list.append(trial_windows(X_t1[i]))   # (12,12,20)
        allwins = np.concatenate(win_list, axis=0)    # (12*nA, 12, 20)
        covs = oas_cov(allwins)
        return mean_riemann(covs)
    elif mode == 'full_zone':
        zone = X_t1[anchor_trial_idx][:, :, ZONE_START:ZONE_END]  # (nA,12,43)
        covs = oas_cov(zone)
        return mean_riemann(covs)
    else:
        assess ValueError(mode)


def process_subject(sid, mode):
    t0 = time.time()
    X, y_sem, y_load = load_subject(sid)
    task = y_sem[:, 0] != 0
    iscorrect = y_sem[task, 2]
    load = y_load[task]
    X_task = X[task]

    zero_idx = np.where((load == 0) & (iscorrect == 1))[0]
    corr_idx = np.where((load == 1) & (iscorrect == 1))[0]
    inc_idx = np.where((load == 1) & (iscorrect == 0))[0]

    out = {
        "subject_id": sid, "mode": mode, "banned": sid in EXCLUDED_SUBJECTS,
        "n_0back_correct": int(len(zero_idx)),
        "n_0back_total": int(np.sum(load == 0)),
        "n_2back_correct": int(len(corr_idx)),
        "n_2back_incorrect": int(len(inc_idx)),
        "admissible": False,
    }

    enough = (len(zero_idx) >= MIN_0BACK and
              len(corr_idx) >= MIN_PER_CLASS_2BACK and
              len(inc_idx) >= MIN_PER_CLASS_2BACK)
    if not enough:
        out["exclusion"] = "INSUFFICIENT_TRIALS"
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    X_t1 = X_task[:, TIER1_ROI_INDICES, :]
    anchor_idx, probe0_idx = split_zero_back(zero_idx, sid)
    anchor = build_anchor(X_t1, anchor_idx, mode)

    probe0 = np.array([rgd_curve_max(X_t1[i], anchor) for i in probe0_idx])
    corr = np.array([rgd_curve_max(X_t1[i], anchor) for i in corr_idx])
    inc = np.array([rgd_curve_max(X_t1[i], anchor) for i in inc_idx])

    g1_U, g1_p, _ = mw_one_sided_greater(corr, probe0)
    g2_U, g2_p, _ = mw_one_sided_greater(inc, corr)

    out.update({
        "admissible": True,
        "exclusion": None,
        "gate0_median": float(np.median(probe0)),
        "gate0_iqr": float(ss.iqr(probe0)),
        "gate0_n": int(len(probe0)),
        "med_probe0": float(np.median(probe0)),
        "med_correct": float(np.median(corr)),
        "med_incorrect": float(np.median(inc)),
        "gate1_U": g1_U, "gate1_p": g1_p, "gate1_d": cohens_d_indep(corr, probe0),
        "gate2_U": g2_U, "gate2_p": g2_p, "gate2_d": cohens_d_indep(inc, corr),
        "elapsed_s": round(time.time() - t0, 2),
    })
    return out


# ---------------------------------------------------------------- tier-2
def wilcoxon_one_sided_positive(dvals):
    """Wilcoxon signed-rank, H0: median(d)=0 vs H1: median(d)>0.
    dvals: per-subject effect sizes. Returns (W, p, median, n)."""
    dvals = np.asarray([d for d in dvals if np.isfinite(d)])
    n = len(dvals)
    if n < 1:
        return float('nan'), float('nan'), float('nan'), 0
    med = float(np.median(dvals))
    try:
        W, p = ss.wilcoxon(dvals, alternative='greater', zero_method='wilcox')
    except ValueError:
        # all zeros or constant
        return float('nan'), float('nan'), med, n
    return float(W), float(p), med, n


def classify_gate(dvals, cohort_medians_ordered):
    """cohort_medians_ordered = (med_null, med_correct, med_incorrect) for monotonic check.
    Returns dict with PASS/FALSIFIED/INCONCLUSIVE per c02RESEARCH.md §V."""
    W, p, med, n = wilcoxon_one_sided_positive(dvals)
    monotonic = (cohort_medians_ordered[0] < cohort_medians_ordered[1] < cohort_medians_ordered[2])
    if n < MIN_ADMISSIBLE_COHORT:
        verdict = "INCONCLUSIVE"
    elif med > 0.3 and p < 0.05 and (cohort_medians_ordered is None or monotonic):
        verdict = "PASS"
    elif med < 0:
        verdict = "FALSIFIED"   # wrong direction
    else:
        verdict = "FAIL"        # not significant / effect too small, right direction
    return {"wilcoxon_W": W, "p": p, "median_d": med, "n": n,
            "monotonic_ordering": bool(monotonic), "verdict": verdict}


# ---------------------------------------------------------------- main
def run_mode(mode, sids, checkpoint_path):
    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    completed[e["subject_id"]] = e
        print(f"  [ckpt] resumed {len(completed)} subjects for mode={mode}")

    with open(checkpoint_path, "a") as ck:
        for i, sid in enumerate(sids, 1):
            if sid in completed:
                continue
            print(f"  [{mode}] {i:02d}/{len(sids)} {sid} ...", end=" ", flush=True)
            try:
                res = process_subject(sid, mode)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            if res.get("admissible"):
                print(f"d1={res['gate1_d']:+.3f} d2={res['gate2_d']:+.3f} "
                      f"(2bc={res['n_2back_correct']},2bi={res['n_2back_incorrect']}) "
                      f"[{res['elapsed_s']:.1f}s]")
            else:
                print(f"SKIP ({res.get('exclusion')})")
            ck.write(json.dumps(res) + "\n"); ck.flush(); os.fsync(ck.fileno())
            completed[sid] = res
    return list(completed.values())


def summarize(mode, results):
    adm = [r for r in results if r.get("admissible")]
    print(f"\n{'='*64}\nMODE={mode}  admissible={len(adm)}/{len(results)}\n{'='*64}")
    if not adm:
        print("  no admissible subjects"); return
    d1 = [r["gate1_d"] for r in adm]
    d2 = [r["gate2_d"] for r in adm]
    med_null = float(np.median([r["gate0_median"] for r in adm]))
    med_corr = float(np.median([r["med_correct"] for r in adm]))
    med_inc = float(np.median([r["med_incorrect"] for r in adm]))
    g1 = classify_gate(d1, (med_null, med_corr, med_inc))
    g2 = classify_gate(d2, (med_null, med_corr, med_inc))
    # tier-1 pass counts
    g1_l1 = sum(1 for r in adm if r["gate1_p"] < 0.05 and r["gate1_d"] > 0.3)
    g2_l1 = sum(1 for r in adm if r["gate2_p"] < 0.05 and r["gate2_d"] > 0.3)
    print(f"  cohort median RGD:  null={med_null:.4f}  correct={med_corr:.4f}  incorrect={med_inc:.4f}")
    print(f"  monotonic Null<Correct<Incorrect: {med_null<med_corr<med_inc}")
    print(f"  --- GATE 1 (load) ---")
    print(f"    tier1 pass: {g1_l1}/{len(adm)}   tier2: W={g1['wilcoxon_W']:.1f} "
          f"p={g1['p']:.3g} median(d)={g1['median_d']:+.3f} n={g1['n']}  -> {g1['verdict']}")
    print(f"  --- GATE 2 (friction) ---")
    print(f"    tier1 pass: {g2_l1}/{len(adm)}   tier2: W={g2['wilcoxon_W']:.1f} "
          f"p={g2['p']:.3g} median(d)={g2['median_d']:+.3f} n={g2['n']}  -> {g2['verdict']}")
    return {
        "mode": mode, "n_admissible": len(adm), "n_total": len(results),
        "cohort_medians": {"null": med_null, "correct": med_corr, "incorrect": med_inc},
        "monotonic": bool(med_null < med_corr < med_inc),
        "gate1": g1, "gate2": g2,
        "gate1_tier1_pass": g1_l1, "gate2_tier1_pass": g2_l1,
    }


def main():
    if not os.path.isdir(DATA_DIR):
        sys.exit(f"DATA_DIR not found: {DATA_DIR}")
    sids_all = sorted(f.split("_")[0] for f in os.listdir(DATA_DIR) if f.endswith("_X.npy"))
    sids = sids_all  # process allone incl banned for transparency; banned flagged in output
    print(f"Subjects on disk: {len(sids_all)}  (banned tracked: {sorted(EXCLUDED_SUBJECTS)})")
    print(f"Modes: sliding (Sprint-1 fix), full_zone (original/OAS-mismatch)\n")

    summaries = {}
    for mode in ("sliding", "full_zone"):
        ckpt = os.path.join(RESULTS_DIR, f"three_gate_{mode}.jsonl")
        print(f"\n>>> running mode={mode}")
        results = run_mode(mode, sids, ckpt)
        summaries[mode] = summarize(mode, results)

    out = os.path.join(RESULTS_DIR, "three_gate_summary.json")
    with open(out, "w") as f:
        json.dump({"rng_state": rng_state, "min_0back": MIN_0BACK,
                   "min_2back_per_class": MIN_PER_CLASS_2BACK,
                   "n_windows": N_WINDOWS, "summaries": summaries}, f, indent=2)
    print(f"\nSummary written -> {out}")


if __name__ == "__main__":
    main()
