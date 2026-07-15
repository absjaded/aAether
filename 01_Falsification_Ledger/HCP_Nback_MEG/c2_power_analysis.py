"""
c2_power_analysis.py — Statistical power of the C2 friction Gate 2
==================================================================
Pure simulation. No neural data. Answers:

  Given the REAL per-subject trial counts from c2_trial_census.json, what is the
  minimum true Cohen's d that the per-subject Mann-Whitney U test (tier 1) can
  reliably detect, and what does that imply for the tier-2 Wilcoxon?

This quantifies the starvation risk: median incorrect-2back = 6 trials.
Mann-Whitney with n=6 vs n=140 has very low resolution for small effects.

We simulate Gaussian draws with a known true d, run the exact tier-1 test
the pipeline uses, aggregate via Wilcoxon (tier 2), and report power = P(pass)
as a function of true d. This gives the detectable-effect-size floor.
"""

import os
import json
import numpy as np
import scipy.stats as ss

CENSUS = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", ".data", "results", "c2_trial_census.json"))

rng_state = 42
MIN_PER_CLASS = 5
MIN_COHORT = 30


def load_real_counts():
    with open(CENSUS) as f:
        c = json.load(f)
    # use the actual per-subject (2bc, 2bi) counts; restrict to gate-2-ready
    ready = [(s["n_2back_correct"], s["n_2back_incorrect"])
             for s in c["subjects"] if s["subject_id"] not in {"140117", "204521"}
             and s["n_2back_correct"] >= MIN_PER_CLASS and s["n_2back_incorrect"] >= MIN_PER_CLASS]
    return ready


def simulate_one_subject(nc, ni, true_d, rng):
    """Draw 2back-correct and 2back-incorrect RGD-ish values with a known d.
    incorrect mean shifted UP by true_d * pooled_sd. Run the pipeline's test."""
    sd = 1.0
    corr = rng.normal(0.0, sd, nc)
    inc = rng.normal(true_d * sd, sd, ni)
    if nc < 2 or ni < 2:
        return np.nan, np.nan, np.nan
    try:
        U, p = ss.mannwhitneyu(inc, corr, alternative='greater')
    except ValueError:
        return np.nan, np.nan, np.nan
    v1, v2 = inc.var(ddof=1), corr.var(ddof=1)
    pooled = np.sqrt(((ni - 1) * v1 + (nc - 1) * v2) / (ni + nc - 2))
    d = (inc.mean() - corr.mean()) / pooled if pooled > 0 else np.nan
    return p, d, U


def run(true_d, counts, n_mc=2000, rng_state=rng_state):
    rng = np.random.RandomState(rng_state)
    l1_pass = np.zeros(n_mc)
    l2_pass = np.zeros(n_mc)
    l2_falsified = np.zeros(n_mc)
    for mc in range(n_mc):
        ds = []
        for (nc, ni) in counts:
            p, d, _ = simulate_one_subject(nc, ni, true_d, rng)
            if np.isfinite(d):
                ds.append(d)
                if p < 0.05 and d > 0.3:
                    l1_pass[mc] += 1
        ds = np.array(ds)
        # tier 2: Wilcoxon on d's, one-sided greater
        try:
            W, p = ss.wilcoxon(ds, alternative='greater', zero_method='wilcox')
            med = np.median(ds)
            l2_pass[mc] = 1.0 if (p < 0.05 and med > 0.3) else 0.0
            l2_falsified[mc] = 1.0 if med < 0 else 0.0
        except ValueError:
            pass
    return dict(true_d=true_d,
                l1_pass_frac=float(np.mean(l1_pass) / max(len(counts), 1)),
                l2_power=float(np.mean(l2_pass)),
                l2_falsified_frac=float(np.mean(l2_falsified)),
                median_n_subj=len(counts))


def main():
    counts = load_real_counts()
    print(f"Gate-2-ready subjects (real counts): {len(counts)}")
    nc = [c for c, _ in counts]; ni = [i for _, i in counts]
    print(f"  2back-correct:  min {min(nc)}  median {int(np.median(nc))}  max {max(nc)}")
    print(f"  2back-incorrect:min {min(ni)}  median {int(np.median(ni))}  max {max(ni)}\n")

    print(f"{'true_d':>7} {'L1pass/subj':>12} {'L2 power':>9} {'L2 falsif%':>10}")
    print("-" * 42)
    table = []
    for d in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        r = run(d, counts, n_mc=1500)
        print(f"{d:>7.2f} {r['l1_pass_frac']*100:>11.1f}% {r['l2_power']*100:>8.1f}% "
              f"{r['l2_falsified_frac']*100:>9.1f}%")
        table.append(r)

    print("\nInterpretation:")
    print("  - Row true_d=0.0 is the false-positive rate (should be ~5% at L2).")
    print("  - L2 power >= 80% defines the MINIMUM detectable true effect size.")
    print("  - If a biologically plausible friction d (~0.2-0.4) gives <50% L2 power,")
    print("    Gate 2 is underpowered by design regardless of any geometry fix.")

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", "c2"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "power_analysis.json"), "w") as f:
        json.dump({"counts_used": {"n_subjects": len(counts),
                                    "median_nc": int(np.median(nc)),
                                    "median_ni": int(np.median(ni))},
                   "table": table}, f, indent=2)
    print(f"\nSaved -> {os.path.join(out,'power_analysis.json')}")


if __name__ == "__main__":
    main()
