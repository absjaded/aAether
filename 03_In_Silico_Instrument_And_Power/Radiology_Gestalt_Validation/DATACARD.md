# In-Silico Radiology Probe

This folder implements a synthetic radiologist initial-read covariance probe.

## Paradigm

- Domain: radiologist initial read.
- Held state: fast pre-verbal gestalt that a film is clean or has something present.
- Violation: rung-3 subject-state violation, meaning the reader's initial gestalt diverges from later ground truth.
- Constant response: a pathology-orthogonal perceptual response, represented synthetically as a 3-level image-quality response.
- Labels are assigned post-hoc; covariance trials are generated at read time.

## Stage 1

`synthetic_radiology.py` generates covariance trials for five modes:

| mode | purpose |
|---|---|
| `signal` | gestalt-violation signal only |
| `N1_null` | no planted signal |
| `N2_response_confound` | response/motor signal only, no gestalt signal |
| `N3_motor_null` | gestalt plus response signal, used to verify separation |
| `N4_entangled_response` | response-only signal whose scalar response drive is correlated with violation; used to measure false-positive limits |

Two generator leaks were found and removed during smoke testing:

- Trial difficulty no longer changes violation probability, so `N1_null` cannot decode violation through a scalar difficulty proxy.
- Subject error rate is fixed for Stage 1, so subject covariance baselines cannot accidentally proxy subject-level violation rates.

## Stage 1b

`stage1b_extract.py` runs a blind Riemannian-EA-log-tangent extraction. It does not receive planted axes.

Medium run command:

```bash
python 03_In_Silico_Instrument_And_Power/Radiology_Gestalt_Validation/stage1b_extract.py --subjects 20 --trials 140 --channels 28 --components 24 --gestalt-effect 0.10 --response-effect 0.24 --out 03_In_Silico_Instrument_And_Power/Radiology_Gestalt_Validation/outputs/stage1b_medium.json
```

Medium run results:

| mode | violation rate | gestalt AUC | gestalt balanced acc | response AUC | response balanced acc | verdict |
|---|---:|---:|---:|---:|---:|---|
| `signal` | 0.087857 | 1.000000 | 1.000000 | 0.495275 | 0.487064 | pass |
| `N1_null` | 0.089643 | 0.491231 | 0.511814 | 0.508643 | 0.507917 | pass |
| `N2_response_confound` | 0.096429 | 0.512177 | 0.519777 | 0.999988 | 0.998417 | pass |
| `N3_motor_null` | 0.090000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |

Interpretation: Stage 1b passes as an instrument check. The blind extractor recovers the planted gestalt geometry in `signal`, stays flat on `N1_null`, ignores the response-only confound for gestalt in `N2_response_confound`, and detects both axes in `N3_motor_null`.

This proves the extraction pipeline can read the synthetic signal when it exists. It does not prove the human phenomenon exists.

## Planned Extensions

- Stage 2: feed recovered per-trial contrast into GW plus anchors on synthetic data with known correspondence.
- Stage 3: sweep effect size, trials, subjects, and violation rate into an IRB/power curve.

The current medium run is local CPU work. Larger grids use the same command-line entry points.

## Stage 1c Full Sweep

Full sweep command:

```bash
python stage1c_sweep.py --reps 5 --workers 8 --out-dir outputs/stage1c_full
```

Outputs:

- `outputs/stage1c_full/stage1c_sweep.csv`
- `outputs/stage1c_full/stage1c_sweep.json`

Stage 1c used the harder generator settings now in `synthetic_radiology.py`: explicit `violation_rate`, subject-level signal-axis jitter, nuisance covariance axes, and higher trial noise. The matched null is reported beside every row in the `null_*` columns.

Effect-axis thresholds at 20 subjects x 140 trials, violation rate 0.10:

| gestalt effect | per-subject d | gestalt AUC | null AUC | verdict |
|---:|---:|---:|---:|---|
| 0.100 | 1.660555 | 0.851511 | 0.494742 | strong |
| 0.070 | 0.757134 | 0.685621 | 0.494742 | weak_real |
| 0.050 | 0.430797 | 0.609518 | 0.494742 | inconclusive |
| 0.035 | 0.252190 | 0.565578 | 0.494742 | inconclusive |
| 0.025 | 0.158022 | 0.541893 | 0.494742 | chance |
| 0.000 | 0.033170 | 0.509106 | 0.494742 | chance |

Approximate crossings:

- AUC 0.75: effect about 0.083.
- AUC 0.65: effect about 0.061.
- Chance region: effect about 0.025-0.030 and below.
- Weak-effect selected for later axes: effect 0.070, per-subject d about 0.76, AUC about 0.69.

Violation-rate floor at effect 0.070, 20 subjects x 200 trials:

| target violation rate | observed rate | per-subject d | gestalt AUC | verdict |
|---:|---:|---:|---:|---|
| 0.05 | 0.051200 | 0.574681 | 0.648064 | inconclusive |
| 0.08 | 0.080100 | 0.685469 | 0.669526 | weak_real |
| 0.10 | 0.101000 | 0.729711 | 0.680750 | weak_real |
| 0.20 | 0.195900 | 1.026756 | 0.739545 | weak_real |
| 0.30 | 0.295050 | 1.407728 | 0.810270 | strong |

Current floor: below about 8% violations the result falls below the weak-real AUC threshold. Curating hard cases to raise the gestalt-violation rate is probably necessary.

Cohort-size curve at effect 0.070, 140 trials:

| subjects | per-subject d | gestalt AUC | cohort perm p | verdict |
|---:|---:|---:|---:|---|
| 5 | 0.445001 | 0.622891 | 0.042421 | inconclusive |
| 8 | 0.642554 | 0.655889 | 0.007664 | weak_real |
| 20 | 0.757134 | 0.685621 | 0.000244 | weak_real |
| 50 | 1.315229 | 0.793429 | 0.000244 | strong |

Current cohort floor: 8 radiologists reaches weak-real by both AUC and permutation p under the synthetic assumptions. A stricter collection should still plan above that because this is synthetic.

Important limitation: the trials-per-subject axis did not collapse at low trial counts. At effect 0.070 and violation rate 0.10, even 40 trials produced AUC 0.702526 with p 0.000293. That is not a credible claim that 40 reads is enough for a real study. It means the current Stage 1c extractor is mostly recovering a cohort-shared direction and does not yet force per-subject calibration/reference estimation. The next architecture iteration should add a calibration-constrained analysis where each subject must contribute enough violation trials to estimate their own reference.

Practical read: Stage 1c now gives a useful recovery curve for the cohort-shared extractor. It is not yet the final IRB power curve for per-radiologist calibration cost.

## Stage 1d Calibration-Constrained Sweep

Question: Stage 1c's low-trial result looked too generous because the extractor could use a cohort-shared direction. Stage 1d forces the harder operational constraint: for each subject, sample a natural calibration set from that same subject, build that subject's own intact/violated centroids from the calibration reads only, and test on the remaining reads. A subject-repeat is unusable if the calibration set or test set lacks enough examples from both classes. Calibration size `0` is retained as the group leave-one-subject-out baseline.

Primary run command:

```bash
python stage1d_calibration.py --subjects 20 --trials 300 --channels 28 --components 24 --gestalt-effect 0.07 --violation-rate 0.10 --response-effect 0.24 --reps 3 --splits 50 --workers 8 --out-dir outputs/stage1d_full
```

Primary outputs:

- `outputs/stage1d_full/stage1d_calibration_summary.csv`
- `outputs/stage1d_full/stage1d_calibration_trials.csv`
- `outputs/stage1d_full/stage1d_calibration.json`

Primary 300-read calibration curve:

| calibration reads | usable fraction | mean violated cal reads | mean violated test reads | per-subject d | gestalt AUC | null AUC | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.000000 | 0.000000 | 30.750000 | 0.692044 | 0.681437 | 0.488550 | weak_real |
| 5 | 0.080000 | 2.104167 | 30.650000 | 0.081176 | 0.519933 | 0.500160 | calibration_undercovered |
| 10 | 0.273667 | 2.364190 | 29.880633 | 0.086988 | 0.523366 | 0.497035 | calibration_undercovered |
| 20 | 0.618333 | 2.876011 | 28.950943 | 0.098561 | 0.527174 | 0.496611 | calibration_undercovered |
| 40 | 0.918667 | 4.370102 | 26.729681 | 0.133118 | 0.536921 | 0.495793 | chance |
| 70 | 0.993000 | 7.208795 | 23.605237 | 0.167180 | 0.546001 | 0.494974 | chance |
| 100 | 0.999000 | 10.249249 | 20.511178 | 0.200574 | 0.554964 | 0.495180 | inconclusive |
| 140 | 1.000000 | 14.389333 | 16.360667 | 0.233339 | 0.564125 | 0.492949 | inconclusive |

Primary finding: the rare-base-rate collapse is real under natural calibration. With 10% violations, 5 and 10 calibration reads mostly do not contain enough violated examples to build a subject reference. Even after the calibration set becomes usable, the pure subject-reference estimator stays below weak-real. At 140 calibration reads it reaches only AUC `0.564125` and d `0.233339`.

Extended run command:

```bash
python stage1d_calibration.py --subjects 20 --trials 1000 --channels 28 --components 24 --gestalt-effect 0.07 --violation-rate 0.10 --response-effect 0.24 --reps 3 --splits 25 --workers 8 --calibration-values 0,70,140,200,300,500,700 --out-dir outputs/stage1d_large_calibration
```

Extended outputs:

- `outputs/stage1d_large_calibration/stage1d_calibration_summary.csv`
- `outputs/stage1d_large_calibration/stage1d_calibration_trials.csv`
- `outputs/stage1d_large_calibration/stage1d_calibration.json`

Extended 1000-read calibration curve:

| calibration reads | usable fraction | mean violated cal reads | mean violated test reads | per-subject d | gestalt AUC | null AUC | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.000000 | 0.000000 | 101.466667 | 0.749685 | 0.692329 | 0.492956 | weak_real |
| 70 | 0.991333 | 7.074647 | 94.451917 | 0.169505 | 0.546457 | 0.497805 | chance |
| 140 | 1.000000 | 14.201333 | 87.265333 | 0.234910 | 0.563896 | 0.499646 | inconclusive |
| 200 | 1.000000 | 20.316000 | 81.150667 | 0.275196 | 0.574542 | 0.499438 | inconclusive |
| 300 | 1.000000 | 30.185333 | 71.281333 | 0.325832 | 0.587566 | 0.498075 | inconclusive |
| 500 | 1.000000 | 50.861333 | 50.605333 | 0.385452 | 0.603344 | 0.497790 | inconclusive |
| 700 | 1.000000 | 71.142000 | 30.324667 | 0.448587 | 0.619237 | 0.496950 | inconclusive |

Extended finding: the pure subject-reference estimator still does not cross the weak-real AUC threshold of `0.65`, even with 700 natural calibration reads. The group leave-one-subject-out baseline at the same weak effect remains weak-real, AUC `0.692329`, while subject-only calibration tops out at AUC `0.619237`.

Interpretation: Stage 1d confirms that Stage 1c was measuring cohort-shared recoverability, not per-radiologist reference calibration cost. Natural per-subject calibration is both sample-inefficient and statistically weak in this architecture. The next architecture should not be a pure subject-only centroid reference. It should use the cohort-shared geometry as the backbone and add subject-level calibration or shrinkage on top of it.

## Stage 1d Shrinkage Architecture Iteration

Question: Stage 1d showed that a pure subject-only reference does not recover the weak effect, even with large natural calibration sets. The architecture iteration tests a hierarchical estimator: keep the leave-one-subject-out cohort backbone, build sparse subject centroids from calibration reads, and blend them class-wise with empirical-Bayes shrinkage weights estimated from other subjects. This run uses the established `synthetic_radiology.py` generator and the same Riemannian-EA/PCA centroid-gap metric as Stage 1c/1d, not the separate draft generator.

Run command:

```bash
python stage1d_shrinkage.py --subjects 20 --trials 1000 --channels 28 --components 24 --gestalt-effect 0.07 --violation-rate 0.10 --response-effect 0.24 --calibration-values 0,70,140,200,300,500,700 --reps 3 --splits 25 --workers 8 --out-dir outputs/stage1d_shrinkage
```

Outputs:

- `outputs/stage1d_shrinkage/stage1d_shrinkage_summary.csv`
- `outputs/stage1d_shrinkage/stage1d_shrinkage_trials.csv`
- `outputs/stage1d_shrinkage/stage1d_shrinkage.json`

Shrinkage curve:

| calibration reads | cohort AUC | subject-only AUC | shrinkage AUC | shrinkage d | shrinkage null AUC | lambda violated | lambda intact | shrinkage verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.692329 | 0.692329 | 0.692329 | 0.749685 | 0.492956 | 0.000000 | 0.000000 | weak_real |
| 70 | 0.692511 | 0.532461 | 0.659289 | 0.605543 | 0.496169 | 0.084879 | 0.113528 | weak_real |
| 140 | 0.692141 | 0.545177 | 0.647534 | 0.558062 | 0.495303 | 0.156861 | 0.204065 | inconclusive |
| 200 | 0.692376 | 0.552583 | 0.641937 | 0.535297 | 0.495493 | 0.206601 | 0.268001 | inconclusive |
| 300 | 0.692229 | 0.564703 | 0.637986 | 0.519803 | 0.497330 | 0.283136 | 0.354771 | inconclusive |
| 500 | 0.692947 | 0.578073 | 0.631658 | 0.493785 | 0.496393 | 0.397452 | 0.477508 | inconclusive |
| 700 | 0.693807 | 0.590954 | 0.634882 | 0.507263 | 0.494854 | 0.477539 | 0.561950 | inconclusive |

Finding: shrinkage helps relative to pure subject-only, but it does not beat the cohort-only backbone. At 70 calibration reads, shrinkage reaches AUC `0.659289`, but cohort-only is still higher at AUC `0.692511`. As calibration reads increase, the shrinkage weights rise and the blended estimator moves toward the weaker subject-only reference, so shrinkage AUC falls to `0.634882` at 700 reads while cohort-only remains about `0.694`.

Matched `N1_null` remains clean for shrinkage, with null AUC between `0.494854` and `0.497330` across nonzero calibration sizes.

Architecture verdict: hierarchical shrinkage does not rescue per-subject calibration in this synthetic regime. The useful signal is in the cohort-shared geometry. Sparse subject references add noise unless the paradigm supplies richer subject evidence or a higher violation rate. The current architecture should therefore be treated as a cohort-backbone instrument with optional light subject calibration, not as a solved per-radiologist reference estimator.

## Stage 3 Power Curve

Reason for moving to Stage 3 before Stage 2: Stage 1d and the shrinkage iteration showed that per-subject reference estimation is not the current bottleneck to solve with alignment. The working detector is the cohort-backbone geometry. Stage 3 therefore sizes the collection around the detector that actually worked: leave-one-subject-out cohort reference, with matched `N1_null` at every design cell.

Wide-grid command:

```bash
python stage3_power_curve.py --effect-values 0.05,0.07,0.10 --violation-values 0.05,0.08,0.10,0.15,0.20,0.30 --subjects-values 8,12,20,30,50 --trials-values 70,140,300 --channels 28 --components 24 --response-effect 0.24 --reps 12 --workers 8 --out-dir outputs/stage3_power_curve
```

Wide-grid outputs:

- `outputs/stage3_power_curve/stage3_power_summary.csv`
- `outputs/stage3_power_curve/stage3_recommendations.csv`
- `outputs/stage3_power_curve/stage3_power_curve.json`

Criterion: `stage3_powered` means weak-real pass rate at least `0.80`, matched-null weak false-positive rate at most `0.05`, and matched-null clean rate at least `0.90`. The wide grid has `270` cells. Verdict counts: `154` powered, `20` borderline, `55` inconclusive, `41` underpowered.

Wide-grid recommendations by effect and violation rate:

| effect | violation rate | recommendation | subjects | reads/subject | total reads | expected violations/subject | mean d | mean AUC | weak-power | null FP |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.05 | not powered in grid | 50 | 300 | 15000 | 15.0 | 0.581877 | 0.641513 | 0.25 | 0.0 |
| 0.05 | 0.08 | not powered in grid | 50 | 300 | 15000 | 24.0 | 0.660476 | 0.659197 | 0.50 | 0.0 |
| 0.05 | 0.10 | not powered in grid | 50 | 300 | 15000 | 30.0 | 0.703981 | 0.669396 | 0.75 | 0.0 |
| 0.05 | 0.15 | powered | 50 | 70 | 3500 | 10.5 | 0.852759 | 0.699495 | 1.00 | 0.0 |
| 0.05 | 0.20 | powered | 50 | 70 | 3500 | 14.0 | 0.974246 | 0.725097 | 1.00 | 0.0 |
| 0.05 | 0.30 | powered | 30 | 70 | 2100 | 21.0 | 0.874114 | 0.705334 | 1.00 | 0.0 |
| 0.07 | 0.05 | powered | 50 | 70 | 3500 | 3.5 | 0.972848 | 0.724182 | 0.916667 | 0.0 |
| 0.07 | 0.08 | powered | 30 | 70 | 2100 | 5.6 | 0.881288 | 0.709290 | 0.833333 | 0.0 |
| 0.07 | 0.10 | powered | 12 | 70 | 840 | 7.0 | 0.733108 | 0.674621 | 0.833333 | 0.0 |
| 0.07 | 0.15 | powered | 8 | 70 | 560 | 10.5 | 0.819528 | 0.695563 | 0.916667 | 0.0 |
| 0.07 | 0.20 | powered | 8 | 70 | 560 | 14.0 | 0.879321 | 0.706361 | 0.833333 | 0.0 |
| 0.07 | 0.30 | powered | 8 | 70 | 560 | 21.0 | 0.972791 | 0.723589 | 0.916667 | 0.0 |
| 0.10 | 0.05 | powered | 12 | 70 | 840 | 3.5 | 1.132395 | 0.767573 | 1.00 | 0.0 |
| 0.10 | 0.08 | powered | 12 | 70 | 840 | 5.6 | 1.353881 | 0.796348 | 1.00 | 0.0 |
| 0.10 | 0.10 | powered | 12 | 70 | 840 | 7.0 | 1.493241 | 0.821513 | 1.00 | 0.0 |
| 0.10 | 0.15 | powered | 8 | 70 | 560 | 10.5 | 1.546168 | 0.832645 | 1.00 | 0.0 |
| 0.10 | 0.20 | powered | 8 | 70 | 560 | 14.0 | 1.779603 | 0.862879 | 1.00 | 0.0 |
| 0.10 | 0.30 | powered | 8 | 70 | 560 | 21.0 | 2.270195 | 0.917533 | 1.00 | 0.0 |

The wide grid is useful for topology, but `12` reps makes the 80% boundary coarse. A focused confirmation was therefore run around the target weak effect `0.07`.

Focused confirmation command:

```bash
python stage3_power_curve.py --effect-values 0.07 --violation-values 0.08,0.10 --subjects-values 8,12,20,30,50 --trials-values 70,140 --channels 28 --components 24 --response-effect 0.24 --reps 36 --workers 8 --out-dir outputs/stage3_power_confirm_effect007
```

Focused confirmation outputs:

- `outputs/stage3_power_confirm_effect007/stage3_power_summary.csv`
- `outputs/stage3_power_confirm_effect007/stage3_recommendations.csv`
- `outputs/stage3_power_confirm_effect007/stage3_power_curve.json`

Focused confirmation, effect `0.07`:

| violation rate | subjects | reads/subject | total reads | expected violations/subject | weak-power | mean AUC | mean d | null FP | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.08 | 8 | 70 | 560 | 5.6 | 0.472222 | 0.642352 | 0.605438 | 0.055556 | inconclusive |
| 0.08 | 12 | 70 | 840 | 5.6 | 0.500000 | 0.652153 | 0.635103 | 0.0 | inconclusive |
| 0.08 | 20 | 70 | 1400 | 5.6 | 0.750000 | 0.679442 | 0.751736 | 0.0 | borderline |
| 0.08 | 30 | 70 | 2100 | 5.6 | 0.861111 | 0.711737 | 0.903400 | 0.0 | powered |
| 0.08 | 20 | 140 | 2800 | 11.2 | 0.805556 | 0.683658 | 0.763037 | 0.0 | powered |
| 0.08 | 30 | 140 | 4200 | 11.2 | 0.944444 | 0.712414 | 0.900964 | 0.0 | powered |
| 0.10 | 8 | 70 | 560 | 7.0 | 0.583333 | 0.654643 | 0.645748 | 0.027778 | inconclusive |
| 0.10 | 12 | 70 | 840 | 7.0 | 0.750000 | 0.667329 | 0.695959 | 0.0 | borderline |
| 0.10 | 20 | 70 | 1400 | 7.0 | 0.777778 | 0.702101 | 0.861217 | 0.0 | borderline |
| 0.10 | 30 | 70 | 2100 | 7.0 | 0.944444 | 0.737805 | 1.042448 | 0.0 | powered |
| 0.10 | 20 | 140 | 2800 | 14.0 | 0.944444 | 0.701871 | 0.845170 | 0.0 | powered |
| 0.10 | 30 | 140 | 4200 | 14.0 | 1.000000 | 0.735363 | 1.016795 | 0.0 | powered |

Confirmed Stage 3 recommendation for the target weak effect `0.07`: if the study can curate to at least `8%` violations, the smallest confirmed total-read design in this grid is `30 radiologists x 70 reads` (`2100` total reads, weak-power `0.861111`). A lower-radiologist alternative is `20 x 140` (`2800` total reads, weak-power `0.805556` at 8% and `0.944444` at 10%). At 10% violations, `12 x 70` and `20 x 70` are still borderline, not confirmed.

Practical sentence: to detect the target weak synthetic effect around `d = 0.90` at roughly 80%+ simulation power with the cohort-backbone detector, plan around `30` radiologists x `70` reads each at a violation rate of at least `8%`, or `20` radiologists x `140` reads each if fewer radiologists are available. Below `8%` violations, the design becomes subject-count heavy; at the smaller effect `0.05`, the grid did not power below `15%` violations.
## Stage 1e Entangled-Response Limit Probe

Question: Stage 3 assumes the pathology-orthogonal response is not statistically entangled with the violation label. Stage 1e tests that assumption directly. `N4_entangled_response` plants no gestalt-violation axis. It plants only the response axis, then correlates the continuous `response_drive` with the violation label at controlled rho values. Any violation detection in N4 is therefore a response-confound false positive.

Implementation facts:

- `synthetic_radiology.py` now includes `N4_entangled_response`.
- `response_drive` is continuous and public in labels; `response_quality` remains a 3-bin compatibility label.
- N4 uses `gestalt_effect = 0.0`; the only planted task axis is `response_axis`.
- The sweep uses the same Riemannian-EA/PCA/cohort-centroid detector as Stage 3.

Run command:

```bash
python stage1e_entanglement_sweep.py --rho-values 0,0.1,0.2,0.3,0.4,0.5,0.8 --violation-values 0.08,0.10 --subjects-values 20,30 --trials-values 70,140 --channels 28 --components 24 --response-effect 0.24 --reps 36 --workers 8 --out-dir outputs/stage1e_entanglement
```

Outputs:

- `outputs/stage1e_entanglement/stage1e_entanglement_summary.csv`
- `outputs/stage1e_entanglement/stage1e_entanglement_limits.csv`
- `outputs/stage1e_entanglement/stage1e_entanglement.json`

Criterion: N4 weak false positive means AUC at least `0.65` and cohort permutation p at most `0.05`. N4 is a confound-only mode, so a weak pass is a failure of specificity.

N4 limit table:

| violation rate | design | max clean rho tested | first non-clean rho | first leaky rho | first severe rho |
|---:|---|---:|---:|---:|---:|
| 0.08 | 20 x 70 | 0.1 | 0.2 | 0.3 | 0.3 |
| 0.08 | 20 x 140 | 0.1 | 0.2 | 0.2 | 0.3 |
| 0.08 | 30 x 70 | 0.1 | 0.2 | 0.3 | 0.3 |
| 0.08 | 30 x 140 | 0.1 | 0.2 | 0.2 | 0.3 |
| 0.10 | 20 x 70 | 0.1 | 0.2 | 0.3 | 0.4 |
| 0.10 | 20 x 140 | 0.1 | 0.2 | 0.2 | 0.3 |
| 0.10 | 30 x 70 | 0.1 | 0.2 | 0.3 | 0.4 |
| 0.10 | 30 x 140 | 0.1 | 0.2 | 0.2 | 0.3 |

Target Stage 3 design, `30 x 70`:

| violation rate | rho | N4 weak false-positive rate | mean N4 AUC | mean N4 d | measured response-drive corr | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 0.08 | 0.0 | 0.000000 | 0.493677 | -0.024329 | 0.000373 | clean |
| 0.08 | 0.1 | 0.000000 | 0.571563 | 0.276363 | 0.100065 | clean |
| 0.08 | 0.2 | 0.777778 | 0.666920 | 0.600081 | 0.195638 | borderline |
| 0.08 | 0.3 | 1.000000 | 0.765695 | 0.981029 | 0.296659 | severe_leak |
| 0.10 | 0.0 | 0.000000 | 0.494609 | -0.014877 | -0.000778 | clean |
| 0.10 | 0.1 | 0.000000 | 0.567143 | 0.254592 | 0.101573 | clean |
| 0.10 | 0.2 | 0.500000 | 0.651721 | 0.547232 | 0.196099 | borderline |
| 0.10 | 0.3 | 1.000000 | 0.745107 | 0.899799 | 0.297575 | leaky |

Finding: the detector's null behavior is clean when response and violation are independent (`rho = 0.0`) and remains clean at `rho = 0.1`. The architecture becomes unstable at about `rho = 0.2`, and it is broken by `rho = 0.3` in the confirmed Stage 3 target design. More trials per subject make the confound easier to detect, not safer: both `20 x 140` and `30 x 140` become leaky already at `rho = 0.2`.

Math verdict: the cohort-backbone geometry is specific only under low response-label entanglement. The current math cannot separate a true gestalt axis from a response axis whose scalar drive is correlated with violation. A real elicitation must measure response proxies and either keep observed response/violation correlation near the clean regime, match/stratify it away, or add an explicit residualization test. The practical guardrail from this sweep is: `rho <= 0.1` tested clean; `rho = 0.2` is danger; `rho >= 0.3` is a hard failure.