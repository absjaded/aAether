# ROAMM ds007629 Datacard

## Dataset

Name: ROAMM / ReMind

OpenNeuro accession: `ds007629`

Snapshot used: `1.1.1`

Primary data link: https://openneuro.org/datasets/ds007629/versions/1.1.1

Synced derivative prefix used for cohort EEG/eye/label tables: https://s3.amazonaws.com/openneuro.org/ds007629/derivatives/synced/

REVE model link: https://huggingface.co/brain-bzh/reve-base

REVE channel-position link: https://huggingface.co/brain-bzh/reve-positions

## Redistribution

This folder does not redistribute ROAMM data. It excludes raw EEG, synced pickle files, extracted epoch tables, tensor `.dat` files, NumPy prediction arrays, logs, credentials, SSH details, local machine paths, and remote execution paths.

Included files are scripts and this datacard only.

## Data Checked

Synced derivative files checked: `220` pickle files under `derivatives/synced/`.

Subjects in synced derivatives: `44`.

EEG columns in synced derivatives: `64`.

Labels used: synced `is_mw` mind-wandering labels.

EEG unit check: synced EEG values are in volts and were converted to microvolts before EEG/REVE feature extraction.

REVE sample rate: synced EEG was resampled to `200 Hz` for REVE.

Channel-name fix: `Afz` was renamed to `AFz` for the REVE position bank.

## Label Audit

Balanced 10s subset label alignment:

| Label rule | Agreement | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|
| center_mw | `1.000000` | `6923` | `6923` | `0` | `0` |
| majority_mw | `1.000000` | `6923` | `6923` | `0` | `0` |
| any_mw | `0.996028` | `6923` | `6868` | `55` | `0` |

## Local Eye/Behavior Residual Screen

Input table used for the local residual screen: balanced ROAMM epoch table derived from synced derivatives. The table is not included in this package.

Usable balanced rows after feature/label filtering: `13,846`.

Subjects retained: `40`.

Model: histogram gradient boosting checker on eye/reading features, with held-out-subject predictions.

| Metric | Value |
|---|---:|
| Baseline accuracy | `0.500000` |
| Checker accuracy | `0.742164` |
| Checker AUC | `0.803522` |
| Checker log loss | `0.487116` |
| Residual split-half r | `0.656753` |
| Residual permutation p | `0.001996` |
| Per-subject recalibration LL gain | `+0.019682` |
| Recalibration subjects improved | `32 / 40` |
| Recalibration Wilcoxon p | `5.371e-06` |
| Full personal feature-model LL gain | `-0.026593` |
| Full personal feature-model Wilcoxon p | `0.170088` |

Finding: the reliable subject-specific effect is calibration/gain-threshold structure, not a clean full personal eye-feature mapping improvement.

## Idiosyncrasy Check

Question tested: whether each subject's own calibration beats a generic calibration learned from other subjects' calibration halves.

| Metric | Value |
|---|---:|
| Global LOSO calibration LL gain over raw checker | `+0.008179` |
| Global LOSO calibration p | `0.030970` |
| Subject calibration LL gain over raw checker | `+0.019682` |
| Subject calibration p | `5.371e-06` |
| Subject calibration over global LOSO LL gain | `+0.011503` |
| Subjects improved over global LOSO | `29 / 40` |
| Subject-over-global Wilcoxon p | `0.000123` |
| Subject-over-global sign-flip p | `0.000060` |
| Subject slope mean / SD | `0.786828 / 0.321812` |
| Subject intercept mean / SD | `0.182432 / 0.375877` |

Finding: ROAMM contains stable subject-specific calibration structure beyond generic group calibration.

## Local Eye/Reading Reference Tests

These tests use the local ROAMM eye/reading epoch table, not REVE and not EEG tensors.

### Shared 39-to-1 Reference

Method: train the eye/reading checker on 39 subjects and test on the held-out subject. The table reports pooled held-out predictions across the 40 retained subjects.

| Feature set | Rows | Accuracy | Balanced accuracy | AUC | Log loss | On-task accuracy | MW accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| eye/reading + run,t | `13,846` | `0.743103` | `0.743103` | `0.805936` | `0.481706` | `0.632963` | `0.853243` |
| eye/reading only | `13,846` | `0.732053` | `0.732053` | `0.793760` | `0.489930` | `0.606529` | `0.857576` |

Finding: the eye/reading reference transfers across subjects. A held-out subject's on-task/MW state is classifiable from a reference built without that subject.

### Calibration-Cost Curve

Method: use the 39-to-1 group predictions, then fit a one-dimensional subject calibrator from a small labeled calibration set sampled within each held-out subject. Values are subject/repeat means over 40 subjects and 200 repeats per calibration size.

Feature set: eye/reading + run,t.

| Calibration trials | Subjects | Accuracy | AUC | Log loss | On-task accuracy | MW accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | `40` | `0.733225` | `0.801605` | `0.493030` | `0.619255` | `0.847195` |
| `5` | `40` | `0.671137` | `0.772420` | `0.573610` | `0.703481` | `0.639287` |
| `10` | `40` | `0.736162` | `0.794532` | `0.519572` | `0.576352` | `0.895972` |
| `20` | `40` | `0.744498` | `0.801158` | `0.492509` | `0.588632` | `0.900364` |
| `50` | `40` | `0.748276` | `0.801476` | `0.481633` | `0.595988` | `0.900564` |

Feature set: eye/reading only.

| Calibration trials | Subjects | Accuracy | AUC | Log loss | On-task accuracy | MW accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | `40` | `0.716427` | `0.789090` | `0.503796` | `0.594250` | `0.838604` |
| `5` | `40` | `0.655647` | `0.756851` | `0.587178` | `0.697218` | `0.614661` |
| `10` | `40` | `0.722401` | `0.780450` | `0.531280` | `0.549516` | `0.895286` |
| `20` | `40` | `0.730862` | `0.788805` | `0.502601` | `0.559928` | `0.901796` |
| `50` | `40` | `0.733728` | `0.788835` | `0.492629` | `0.565503` | `0.901954` |

Finding: 5 labeled trials is unstable and worsens the calibrated model on average. Ten trials recovers accuracy but worsens log loss. Twenty trials roughly matches the uncalibrated reference. Fifty trials gives the first clean improvement in log loss and a small accuracy gain.

### Run Stability

Method: for each held-out subject, train the group reference on runs 1-2 from the other subjects and test on runs 4-5 from the held-out subject. The table reports pooled held-out predictions for subjects with enough run 4-5 data after balancing.

| Feature set | Rows | Accuracy | Balanced accuracy | AUC | Log loss | On-task accuracy | MW accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| eye/reading + run,t | `6,220` | `0.713505` | `0.713505` | `0.780845` | `0.535297` | `0.653698` | `0.773312` |
| eye/reading only | `6,220` | `0.705466` | `0.705466` | `0.780561` | `0.540588` | `0.662058` | `0.748875` |

Finding: the reference weakens across runs but remains above chance. The no-run/no-time result is similar in AUC, so the run-stability result is not just a numeric run-index effect.

## Local 10s EEG Riemannian Reference Tests

These tests used a local 10s EEG tensor built from ROAMM synced derivatives. The tensor is not redistributed. They do not use REVE and do not require a GPU.

Pipeline: covariance -> SPD conditioning -> Euclidean Alignment -> subject-balanced Riemannian class references -> affine-invariant geodesic score.

Input files used locally: non-redistributed EEG tensor, epoch index, and metadata artifacts derived from the ROAMM synced derivatives.

Cached local intermediates: non-redistributed covariance and alignment arrays.

### Shared 39-to-1 EEG Reference

Method: for each held-out subject, build on-task and mind-wandering Riemannian references from the other 39 subjects, then score the held-out subject by geodesic distance to the two references.

| Rows | Accuracy | Balanced accuracy | AUC | Log loss | On-task accuracy | MW accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| `13,822` | `0.516712` | `0.516796` | `0.517582` | `0.696130` | `0.565154` | `0.468439` |

Finding: this covariance/EA/Riemannian EEG geometry does not transfer cleanly across subjects. It is near chance, unlike the eye/reading reference.

### EEG Calibration-Cost Curve

Method: use the 39-to-1 Riemannian geodesic score, then fit a one-dimensional subject calibrator from a small labeled calibration set sampled within each held-out subject. Values are subject/repeat means over 40 subjects and 200 repeats per calibration size.

| Calibration trials | Subjects | Accuracy | AUC | Log loss | On-task accuracy | MW accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | `40` | `0.510078` | `0.508796` | `0.697367` | `0.582859` | `0.437528` |
| `5` | `40` | `0.496778` | `0.508792` | `0.714838` | `1.000000` | `0.000000` |
| `10` | `40` | `0.523335` | `0.508862` | `0.693095` | `0.500837` | `0.545745` |
| `20` | `40` | `0.532343` | `0.509187` | `0.693046` | `0.505631` | `0.558984` |
| `50` | `40` | `0.546799` | `0.508922` | `0.692901` | `0.508402` | `0.585041` |

Finding: calibration changes the operating point but does not add discrimination. AUC remains about `0.509` across calibration sizes.

### EEG Run Stability

Method: align and build the group Riemannian reference on runs 1-2 from the other subjects, then align/test runs 4-5 from the held-out subject.

| Rows | Accuracy | Balanced accuracy | AUC | Log loss | On-task accuracy | MW accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| `5,642` | `0.452145` | `0.503180` | `0.530260` | `0.725967` | `0.992877` | `0.013483` |

Finding: the cross-run EEG geometry does not provide a useful balanced classifier in this configuration. It predicts nearly all test epochs as on-task after calibration.

Conclusion for this arm: the tested Riemannian covariance backbone does not support a shared, calibration-free neural reference on ROAMM 10s EEG. The eye/reading proxy transfers; this EEG geometry mostly does not.

### Broadband Within-Subject Control

Method: for each subject, fit that subject's own broadband covariance/EA/Riemannian references on runs 1-3, then test runs 4-5.

| Rows | Accuracy | Balanced accuracy | AUC | Log loss | On-task accuracy | MW accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| `5,571` | `0.445521` | `0.486054` | `0.541376` | `0.741176` | `0.894380` | `0.077727` |

Finding: broadband covariance is weak even within subject. The failed 39-to-1 result is not just a transfer failure.

## Local 10s EEG Filter-Bank Riemannian Tests

These tests use theta `4-8 Hz`, alpha `8-13 Hz`, and beta `13-30 Hz` bandpass covariances. Each band uses covariance -> SPD conditioning -> Euclidean Alignment -> Riemannian class references -> geodesic score. A logistic combiner is fit over the three band scores.

Cached local intermediates: non-redistributed covariance and alignment arrays.

### Filter-Bank 39-to-1 EEG Reference

| Rows | Accuracy | Balanced accuracy | AUC | Log loss | On-task accuracy | MW accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| `13,822` | `0.534583` | `0.534680` | `0.551993` | `0.688197` | `0.590665` | `0.478694` |

### Filter-Bank Calibration-Cost Curve

| Calibration trials | Subjects | Accuracy | AUC | Log loss | On-task accuracy | MW accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | `40` | `0.528584` | `0.551627` | `0.690537` | `0.581433` | `0.475874` |
| `5` | `40` | `0.531984` | `0.541133` | `0.693012` | `0.558567` | `0.505717` |
| `10` | `40` | `0.539923` | `0.555564` | `0.692879` | `0.533906` | `0.545883` |
| `20` | `40` | `0.549571` | `0.571188` | `0.692641` | `0.545102` | `0.553991` |
| `50` | `40` | `0.561734` | `0.589459` | `0.691925` | `0.559575` | `0.563803` |

### Filter-Bank Within-Subject Control

Method: for each subject, fit that subject's own filter-bank Riemannian references on runs 1-3, then test runs 4-5.

| Rows | Accuracy | Balanced accuracy | AUC | Log loss | On-task accuracy | MW accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| `5,571` | `0.509244` | `0.521456` | `0.539864` | `0.718114` | `0.644480` | `0.398432` |

Finding: filter-bank covariance improves cross-subject EEG AUC from `0.518` to `0.552`, and the 50-trial calibration condition reaches AUC `0.589`, but the within-subject control remains weak. This is still not a usable neural reference on ROAMM.

## REVE EEG Feature Extraction

Balanced 2s EEG extraction:

| Artifact type | Shape | Notes |
|---|---:|---|
| EEG windows | `(13846, 64, 400)` | `2s`, `200 Hz`, microvolts |
| REVE features | `(13846, 64, 2, 512)` | float16 features |

Balanced 10s EEG extraction:

| Artifact type | Shape | Notes |
|---|---:|---|
| EEG windows | `(13822, 64, 2000)` | `10s`, `200 Hz`, microvolts |
| REVE features | `(13822, 64, 11, 512)` | float16 features |
| Edge skips | `24` | windows skipped at recording boundaries |

Large tensors are not included in this package.

## REVE Probes Before PCA

Eye baseline on the 10s subset: log loss `0.473373`, AUC `0.811088`.

Best pooled 10s REVE add-on before PCA:

| Model | Log loss | AUC | LL gain over eye |
|---|---:|---:|---:|
| eye_reve10s_mean | `0.473461` | `0.824595` | `-0.000088` |

Full-token streaming probe:

| Model | Log loss | AUC | LL gain over eye |
|---|---:|---:|---:|
| reve10s_full_stream | `1.678692` | `0.5571` | `-1.205319` |
| eye_reve10s_full_stream | `1.685058` | `0.5568` | `-1.211686` |

Finding: pooled 10s REVE improved AUC but not calibrated log loss; full-token dense REVE was negative.

## Final REVE Patch-PCA Probe

Input representation: saved 10s REVE tensor `(13822, 64, 11, 512)`.

Feature view: mean over the 11 REVE patches, keeping channel-resolved `64 x 512 = 32768` features.

Scaling: fit split only.

PCA: fit split only, maximum `300` components.

Model selection: `k in {100, 200, 300}` and weight decay selected by validation log loss.

Final evaluation: held-out test split.

Null: `500` within-subject shuffles of the REVE PCA block.

| Model | k | Validation LL | Test LL | Test AUC | LL gain over eye | AUC gain over eye |
|---|---:|---:|---:|---:|---:|---:|
| reve10s_patchpca100 | `100` | `0.679103` | `0.670921` | `0.613635` | `-0.197548` | `-0.197453` |
| eye_reve10s_patchpca100 | `100` | `0.477229` | `0.466419` | `0.825380` | `+0.006953` | `+0.014292` |
| reve10s_patchpca200 | `200` | `0.668200` | `0.660056` | `0.636385` | `-0.186684` | `-0.174703` |
| eye_reve10s_patchpca200 | `200` | `0.474965` | `0.465456` | `0.828273` | `+0.007916` | `+0.017185` |
| reve10s_patchpca300 | `300` | `0.658375` | `0.648866` | `0.660688` | `-0.175494` | `-0.150400` |
| eye_reve10s_patchpca300 | `300` | `0.472895` | `0.466552` | `0.833227` | `+0.006821` | `+0.022139` |

Selected model: `eye_reve10s_patchpca300`, selected by validation log loss.

Permutation result for selected model:

| Metric | Value |
|---|---:|
| Observed LL gain over eye | `+0.006821` |
| Observed AUC gain over eye | `+0.022139` |
| p(LL gain >= observed) | `0.001996` |
| p(AUC gain >= observed) | `0.001996` |
| Null LL gain mean | `-0.032657` |
| Null LL gain 95th percentile | `-0.025908` |
| Null AUC gain mean | `-0.020374` |
| Null AUC gain 95th percentile | `-0.015502` |

Finding: frozen REVE did not work as a strong standalone EEG detector, but 10s channel-resolved REVE features compressed by train-only PCA added a small statistically supported marginal gain over the calibrated eye baseline.

## Leakage Checks

The saved run log showed `fit=5166`, `val=1721`, and `test=6935`.

The patch-PCA script standardizes features using the fit split only.

The patch-PCA script fits PCA using the fit split only.

The selected `k=300` model was selected by validation log loss, not by test log loss. The best test log loss among the three eye+REVE PCA models was `k=200`, but `k=300` had the best validation log loss and was the selected model.

The permutation p-value is conditional on the selected `k=300` model. A stricter follow-up would re-run model selection inside each permutation or use a fresh held-out test split.

## Scripts

`scripts/residual_screen.py`: shared residual-screen implementation. It fits held-out-subject checker predictions, measures residual split-half reliability, and tests per-subject intercept/recalibration/full personal-model gains.

`scripts/replicate_roamm.py`: ROAMM runner for the balanced eye/reading residual screen.

`scripts/confirm_roamm_idiosyncrasy.py`: compares subject-specific calibration against global leave-one-subject-out calibration.

`scripts/preflight_reve_roamm.py`: checks REVE model access, channel-position resolution, synced derivative shape/units, and GPU/RAM availability.

`scripts/probe_reve_10s_full_stream_gpu.py`: streams full 10s REVE token features through a GPU linear probe without materializing a dense standardized matrix.

`scripts/probe_reve_10s_patchpca_perm_gpu.py`: runs the final 10s channel-resolved patch-PCA probe and within-subject REVE-block permutation test.

`scripts/roamm_eye_reference_tests.py`: runs the local 39-to-1 shared-reference test, calibration-cost curve, and run-stability test on eye/reading epoch features.

`scripts/roamm_riemann_eeg10s_tests.py`: runs the local 10s EEG covariance/Euclidean-Alignment/Riemannian shared-reference, calibration-cost, and run-stability tests.
`scripts/roamm_riemann_within_subject_broadband.py`: runs the within-subject broadband covariance/Euclidean-Alignment/Riemannian control.

`scripts/roamm_riemann_filterbank_eeg10s.py`: runs theta/alpha/beta filter-bank Riemannian shared-reference, calibration-cost, and within-subject tests.

## Expected Local Inputs For Reproduction

The scripts expect local copies of ROAMM-derived inputs. These files are not included here.

For eye/behavior scripts:

```text
data/roamm_epochs_44subj.csv
```

For EEG Riemannian scripts, set `--root` to a directory containing:

```text
balanced_epoch_index_10s.csv
non-redistributed balanced 10s EEG tensor
roamm_balanced_eeg10s_uV_200hz_meta.json
```
For REVE probe scripts, set `ROAMM_RESULTS_DIR` to a directory containing:

```text
balanced_epoch_index_10s.csv
p_eye_personal_cal.npy
non-redistributed 10s REVE feature tensor
reve_base_features_10s_balanced_f16_meta.json
```




