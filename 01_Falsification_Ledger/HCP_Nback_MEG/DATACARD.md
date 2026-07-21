# HCP N-back MEG Falsification Datacard

## Dataset

Name: Human Connectome Project MEG working-memory N-back task.

Primary access: https://www.humanconnectome.org/study/hcp-young-adult

Data terms: HCP data requires the HCP data-use agreement. This folder does not redistribute raw MEG, source arrays, extracted NumPy tensors, credential material, logs, or batch outputs.

## Question

Can a Riemannian source-space geometry isolate rule-violation intent in the 2-back task, rather than response execution?

## Analysis Summary

The analysis used source-localized MEG epochs at 68 Desikan-Killiany ROIs with t=0 at the target sample. The core representation was ERPCovariances plus Riemannian/geodesic distances to a target reference. The decisive control compared two withhold-response conditions: Lure and Non-Target.

## Result

| Test | Result | Interpretation |
|---|---:|---|
| Headline Lure vs Target contrast | d approx +0.327, p < 0.0012 over 833 permutations | Looked significant before the motor-matched control. |
| Lure vs Non-Target control | d = +0.008, p = 0.75 | The two withhold conditions were geometrically identical. |
| Lag regression on withhold trials | mean beta = -0.011, p = 0.62 | No continuous cognitive gradient survived. |
| Positive lag slopes | 32 / 69 subjects | Not a stable cohort effect. |

## Verdict

Falsified for intent isolation. The geometry separated press from withhold, not Lure intent from Non-Target novelty. The confound is in the task design: the rule-violation event and motor suppression are physically entangled.

## Kept Scripts

| File | Purpose |
|---|---|
| `aether_fusion.py` | Core source-space ERPCovariance/Riemannian contrast code. |
| `mvp0_closeout.py` | Closeout harness for ordinal spacing, RT, motor-control, and permutation checks. |
| `negative_control_matched.py` | Dimensionality-matched motor/spatial negative control. |
| `mvp0_reanalysis.py` | Reduce-side reanalysis of saved covariance batches. |
| `c2_power_analysis.py` | Power calculation support for the HCP arm. |
| `c2_trial_census.py` | Trial-count and label census. |
| `gate2_saliency_analysis.py` | Secondary feature/saliency inspection. |

## Excluded From Public Package

Environment-specific launchers, phase snapshots, credential-loading extractors, raw arrays, batch outputs, and result logs are not part of this public ledger.
