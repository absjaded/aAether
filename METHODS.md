# Methods and evidence

## Scientific target

Aether investigates structure in a person's judgment before report or action.
The target is operational: a measured distinction should add information beyond
the task, timing, response and cheaper observable behavior. A decoder's label
accuracy alone cannot establish that interpretation.

The project developed through public-data probes, controlled instrument tests,
an empirically calibrated observation model and a first-collection specification.
Those components answer different questions. This package gives each a separate
evidence status rather than treating them as one completed neural pipeline.

## Public-data probes

### HCP N-back MEG

The original experiment used source-space MEG and a within-subject geometric
distance, rather than the portable package's subject-held-out classifier.
The retained closeout harness selects the 2-back condition, uses a fixed
250–500 ms window and 22 working-memory/conflict ROIs, and constructs
ERP-augmented covariances. Target distances use leave-one-out covariance
references; Non-Target and Lure trials use the full Target reference. Additional
code implements reference-stability, response-time and dimensionality-matched
spatial controls. The presence of these implementations is distinct from a
verified successful outcome for every control.

The original HCP N-back MEG analysis reported a Lure-versus-Target contrast of
approximately $d=0.327$. Its response-matched Lure-versus-Non-Target control was
$d=0.008$, $p=0.75$. That control undermined the intended judgment-specific
interpretation of the initial contrast. It does not establish that all public
neural data are unsuitable or that motor activity was the sole cause.

These HCP numbers are retained as historical context from
`01_Falsification_Ledger/HCP_Nback_MEG/DATACARD00.md` and
`AETHER_PHASE5_6_CLOSEOUT.md` in the research workspace. Their underlying numeric
outputs are absent from this release; `records` therefore does not certify or
recompute them. HCP participant data remain under their original access terms.

Two unresolved checks limit the headline inference. Previously recovered gap
arrays contain 70 observations, whereas the later specificity report describes
69 stable subjects; their eligibility transition has not been reconciled.
Also, the retained `mvp0_closeout.py` constructs a Target-labelled ERP prototype
before its distance-reference leave-one-out loop. Its permutation worker shuffles
labels on already-constructed covariances. Leaving a trial out of the distance
reference therefore does not establish that it was excluded from the ERP
prototype, and the saved null is conditional on those constructed features.
An end-to-end check of prototype fitting and permutation would be required
before endorsing the headline p-value as validation of the complete procedure.
This consolidation records the issue without modifying historical code or scores.

### ROAMM

ROAMM provided a complementary test of cheap behavior versus tested EEG
representations. Its surviving aggregate outputs are described in the evidence
catalog and included byte-identically. The eye/reading model excludes run and
time as predictor inputs and balances observations within subject. Its pooled
leave-one-subject-out AUC is 0.79376 over 13,846 observations. The EEG 10-second
broadband covariance readout gives 0.51758 over 13,822 observations; the
filterbank variant gives 0.55199. These pooled AUCs concatenate held-out
predictions rather than averaging subject AUCs.

The shared EEG LOSO analyses use each held-out subject's unlabelled all-run
covariances for Euclidean alignment. The cross-run variants calibrate train and
test blocks separately on their unlabelled covariances. The within-subject EEG
analyses train on runs 1–3, test on runs 4–5 and fit alignment on training data
only. The no-time eye cross-run LOSO result trains on runs 1–2 and tests on
runs 4–5. Calibration curves average subject/repetition metrics, using a different
aggregation from the pooled headline AUCs.

Different modalities, windows, available epochs and calibration contracts make
these parallel requirement probes rather than an exact matched-row comparison.
`records` checks the integrity and contents of the aggregate CSVs; it cannot
reconstruct ROC curves or refit classifiers without the original predictions and
features. Feature extraction and participant arrays remain outside this package.

### Earlier exploratory branches

The research workspace also retains EEG deception and Flanker explorations,
a domain-adversarial EEG prototype, an HCP resting-reference assessment and
earlier fMRI-to-symbolic decoder work. Their surviving artifacts vary between
code, saved outputs and retrospective datacards. They provide historical context;
this release does not count each branch as an independently verified experiment.
Decoder and robotics implementation belongs with the corresponding compiler and
NISP packages. The selected measurement evidence remains this repository's focus.

## Portable covariance instrument

### Input contract and calibration

Inputs are real, finite, symmetric positive-definite trial covariance matrices,
binary labels and subject identifiers. The caller is responsible for epoch
selection, artifact handling, covariance estimation, regularization and label
validity. This package starts at the covariance boundary.

For each subject, the instrument forms an arithmetic mean covariance $R_s$ over
all supplied trials. It computes the inverse square root by symmetric
eigendecomposition, whitens each covariance, and applies the symmetric matrix
logarithm. The upper triangle becomes a feature vector, with off-diagonal entries
multiplied by $\sqrt{2}$ to preserve the Frobenius inner product.

This is **unlabelled batch calibration**. A held-out subject contributes their
whole unlabelled covariance batch to their own reference, so a single future
trial cannot be scored under the same contract without a calibration batch.
There is no silent absolute eigenvalue floor: non-positive matrices fail, and
input scale is preserved. A positive scalar rescaling cancels in whitening.

### Readout and scoring

`GroupKFold` holds out whole subjects, using up to six folds by default. PCA,
standardization and class-balanced logistic regression are fitted separately in
each training fold. PCA uses at most 12 components, bounded by feature dimension
and training sample count. Constant training features use an intercept-only
readout. A training fold missing either class fails explicitly.

Every trial receives exactly one held-out probability in original row order.
The report includes pooled held-out AUC, balanced accuracy, per-fold AUC and
subject assignments. Pooled AUC compares scores from different fitted models;
the fold scores allow inspection of between-fold variability. The threshold for
balanced accuracy is 0.5. The portable runner does not calculate significance,
perform feature selection against test labels or declare a scientific gate.

### Relationship to historical code

The generator mechanisms derive from
`03_In_Silico_Instrument_And_Power/Radiology_Gestalt_Validation/synthetic_radiology.py`.
The geometric workflow derives from the adjacent `stage1b_extract.py`.
The portable demonstration uses six subjects, 72 trials each, eight channels and
three nuisance axes; it exposes four named constructions with fixed defaults.
Its random draws, vectorization and readout are a new executable demonstration.

The historical stage-1b runner fitted PCA across the complete batch before
grouped readout. The portable runner fits PCA inside training folds and weights
off-diagonals explicitly. Consequently, fresh demo AUC values are not expected
to match saved historical experiments. No historical result is overwritten.

The response-entanglement construction correlates a scalar response drive with
the standardized target label while retaining geometrically orthogonal axes.
The parameter is a latent construction coefficient; it is not guaranteed to equal
the realized finite-sample Pearson correlation after all transformations. Neither
the demo nor the saved synthetic records are observations of a human state.

## Relation to existing work

Covariance-based BCI decoding has an established geometric foundation.
[Barachant et al. (2012)](https://pubmed.ncbi.nlm.nih.gov/22010143/)
classify EEG covariance matrices using their positive-definite geometry and
tangent-space features. Aether's portable instrument uses an arithmetic-mean
subject reference, whitening, matrix-log coordinates and a grouped readout.
It combines established measurement tools with a specific investigation of
judgment-related information.

The whitening step has the same covariance transformation as
[Euclidean Alignment](https://arxiv.org/abs/1808.05464)
(He and Wu, 2020): the subject's mean covariance becomes the identity.
The [later EA study](https://arxiv.org/abs/2502.09203)
(Wu, 2025) further examines placement within decoding pipelines. In this
package, alignment uses each subject's entire supplied unlabelled batch,
including held-out subjects. Learned PCA, scaling and classifier stages use
training subjects. This is the declared calibration contract of the instrument.

The substantive measurement question is what drives a successful readout.
[Snoek et al. (2019)](https://pubmed.ncbi.nlm.nih.gov/30268846/)
show how confound-control choices and their placement in cross-validation can
bias decoding results. Aether examines candidate explanations using behavioral
comparators, empirical specificity checks and controlled generators; the
portable instrument does not automatically remove a supplied confound.

The [Same Analysis Approach](https://arxiv.org/abs/1703.06670)
(Görgen et al., 2018) motivates applying an analysis pipeline to design
variables, control data and simulations. The four fresh constructions here
use the same portable instrument to expose recovery, null and response-driven
behavior. Saved historical experiments retain their own documented protocol.
In those records, temporal structure produces apparent AUC 0.712, falling to
0.490 after detrending. The response-entanglement example further distinguishes
geometrically orthogonal axes from statistically independent variables.

The released contribution is an instrument and an inspectable set of
measurement tests. The public-data results, controlled examples and calibrated
observation model each inform the next collection needed to test the intended
pre-report distinction.

## Saved instrument experiments

`records` verifies source-file sizes and SHA-256 values before loading the assets.
It reaggregates the saved replicate rows using the historical summary definitions
and verdict logic, then compares every stored summary field at numerical tolerance.
This is an aggregation replay, separate from rerunning feature extraction and
model fitting on the original trial arrays.

The temporal-control record contains 108 replicate rows spanning clean signal,
independent null and temporal-confound null at three drift levels. The behavioral
record contains 144 rows spanning signal, independent null, response-only null
and three response-label entanglement levels. Both use 30 synthetic subjects,
70 trials per subject and 28 channels; the configuration is preserved in each file.
The power record reports 23/24 planted detections and 0/24 null detections for
that retained setting. Zero observed false positives in 24 replicates is a
finite-sample result, not a zero population false-positive rate.

The README quotes arithmetic means from these particular records. Earlier
project summaries include other run versions, such as a planted AUC near 0.686;
those values are not substituted for the packaged record's mean of 0.732.

## Observation reliability and utility

The Phase-4 assets describe a planted signal observed through a calibration
derived from COGITATE/TRIBE analyses. They preserve the observation contract and
the exported readout, gate, incremental-effect, observation-cost and utility
surfaces. The grid sizes are 28, 6, 17, 4 and 17 respectively. `records` checks
those counts and all 17 combined-AUC correspondences between effect and utility
surfaces.

The utility target fixes prevalence at 8% and flags the highest-scoring 20%.
Under an equal-variance binormal score model, AUC approximately 0.67764 produces
2× enrichment among flagged trials. The best primary-reliability combined AUC
in the saved grid is 0.62027, giving modeled lift 1.634×. The alternative
Spearman–Brown setting reaches 0.64806 and 1.806×. These are model-dependent
projections from calibrated simulations, not clinical or behavioral benefits.

Original assets retain their source-path strings and configuration metadata.
The catalog maps them to the new package filenames. Packaging hashes establish
which bytes were consolidated; the original runs lack a complete environment
lock and source-hash closure for bit-identical historical reproduction.

## Correspondence work

The historical synthetic correspondence datacard reports 54 design cells:
38 recovering without measured anchors, 12 requiring one anchor and four requiring
two. The full numerical grid has not been recovered for this release. These counts
remain reported context, excluded from the replayed numerical evidence.
The old benchmark explored identifiability of constructed shared structure;
it did not establish correspondence between a measured pre-report human state
and a computational model. The portable measurement package does not depend on
that unfinished correspondence claim.

## Behavioral ledgers and baseline repair

`ledger` and `baseline` adapt the public runtime's event, covariate and ground-truth
contracts. They check schema, identities, duplicate trials, finite values and
timing consistency before scoring. A row contributes to the baseline only when
`usable_trial` and `usable_covariates` are both true. The report gives joined,
included and excluded counts. Initial templates are deliberately empty.

The recovered cheap-baseline code placed predictions into the wrong trial rows:
test features followed original row order while destination indices were grouped
by label. The adapted code uses one shared test-index ordering. Regression tests
cover interleaved labels and known predictions. That repair changes this runtime;
it neither recalculates prior research scores nor alters the independent saved
stage-1g experiment.

The inherited baseline uses trial-stratified folds and global label permutations.
It does not demonstrate participant- or session-held-out generalization. Small
samples use explicitly labelled apparent AUC; a single-class target yields an
undefined metric represented as JSON `null`. Utility targeting uses the same
binormal assumption described above. The retained decision labels and thresholds
are scaffold conventions that require review for a new collection protocol.

## Reproduction and distribution

`python -m pytest -q` covers geometry, malformed inputs, fold isolation, score
placement, saved-record integrity and behavioral ledger behavior. `demo` and
`measure` generate fresh outputs under `runs/`; the packaged evidence is read-only
input to `records`. No participant-level data, neural caches, credentials,
cloud-volume contents or legacy robotics dependencies are included.

The assembled package passed 57 tests and an installed-wheel check on Python
3.12/Windows, including the fresh demo, NPZ input analysis and saved-record replay.
`requirements-tested.txt` records the tested scientific/test dependency versions;
`pyproject.toml` gives the supported dependency ranges. The original research
environment is distinct from this portable verification environment.

Those tests establish the implemented software contracts and saved-record
checks. They do not retroactively validate the HCP inference, establish a new
biological finding or resolve the uncollected pre-report measurement.

The software is available under the [Apache-2.0 license](LICENSE). Source
datasets retain their own terms. The original research workspaces are preserved;
the [earlier public snapshot](https://github.com/absjaded/aAether/tree/86f199716eed5a269409f7be72a78be867995dad)
remains accessible in this repository's history.
