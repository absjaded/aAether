# Aether

Measuring the structure of a judgment before it becomes a report.

A person's answer is an observable outcome. The neural activity preceding it
may contain additional structure, mixed with preparation to respond, task
difficulty and individual variation. Aether investigates whether that structure
can be isolated and made computationally usable.

This repository brings together selected experiments and portable measurement
tools from that investigation: covariance-based readouts, signal and confound
constructions, saved results, and behavioral collection tools.

## The measurement problem

A readout can predict a label for the wrong reason. Aether makes those alternative
explanations part of the experiment: compare a candidate neural readout against
response measures, cheap behavioral predictors and time-related controls.

For trial covariance $C_{s,t}$, the portable instrument first removes a subject's
average covariance scale and maps the remaining geometry into matrix-log features:

$$
R_s = \frac{1}{T_s}\sum_t C_{s,t},
\qquad
x_{s,t} = \mathrm{vec}_{\mathrm{sym}}\!\left[
\log\!\left(R_s^{-1/2}C_{s,t}R_s^{-1/2}\right)\right].
$$

A classifier then scores held-out subjects. Each subject supplies an unlabelled
batch for the reference $R_s$; PCA, scaling and classifier fitting use training
subjects only. The implementation is in [geometry.py](src/aether_measurement/geometry.py)
and [instrument.py](src/aether_measurement/instrument.py).

## What the experiments established

The earlier **HCP N-back MEG investigation** built a within-subject measurement:
geometric distance from a person's Target reference, with reference-stability,
response-time and spatial controls. It then tested whether Lure trials separated
from Non-Target trials when both required withholding a response. The reported
control failed to support the intended judgment-specific interpretation of the
initial Lure-versus-Target separation. This established a concrete limitation of
the tested measurement and motivated separating target structure from response
demands. [METHODS.md](METHODS.md#hcp-n-back-meg) describes the historical evidence
and outstanding checks. The portable classifier above is a separate analysis;
it does not replay the HCP distance experiment.

In the ROAMM public dataset, the saved no-time eye/reading readout achieved
pooled held-out-subject AUC **0.794**. The tested broadband and filterbank EEG
covariance readouts achieved **0.518** and **0.552**. These are parallel probes
with different windows and usable trial counts; the EEG readouts also use
unlabelled held-out-subject calibration. Their contrast motivated measuring
cheap behavior before assigning a neural interpretation. The packaged aggregate
CSVs preserve the results and split labels; [METHODS.md](METHODS.md) explains
the comparison.

The saved synthetic experiments show both recovery and failure modes:

| Construction | Saved result | What it tells us |
|---|---|---|
| Planted covariance signal | Mean readout AUC **0.732**; cheap behavioral baseline **0.493** | The instrument can recover a planted distinction without relying on those cheap predictors. |
| Matched independent null | Mean readout AUC **0.507** | Removing the planted signal removes the useful separation in this control. |
| Time-confounded null | Apparent AUC **0.712**, falling to **0.490** after detrending | Time structure alone can create a convincing false result. |
| Response-label entanglement | Both covariance and cheap readouts rise as their dependence increases | Geometrically distinct activity can still carry the target label through a confound. |

These are historical runs, with 30 synthetic subjects and 70 trials per subject.
The temporal experiment contains 108 replicate records; the behavioral comparison
contains 144. `records` checks their integrity and recomputes the saved summaries.
The small `demo` below runs fresh constructions with a separate, portable protocol.

Later observation-model experiments used COGITATE/TRIBE-derived calibration to
ask how measurement reliability changes the usefulness of a planted signal.
Under the primary reliability setting, the best tested combined readout reached
AUC **0.620**, corresponding to a modeled **1.63×** enrichment. The specified
**2×** target required AUC **0.678** under the assumed score model. This gives a
concrete observation-cost constraint for designing the next measurement.

## Run it

Python 3.10 or newer, on an ordinary CPU:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q

# Fresh signal, independent-null, response-only and entangled-response trials.
aether-measurement demo --out runs/demo.json --save-trials runs/trials

# Run the same measurement pipeline on one saved construction.
aether-measurement measure --input runs/trials/signal.npz --out runs/measurement.json

# Verify and reaggregate the packaged historical records.
aether-measurement records --out runs/record-check.json
```

The demo reports covariance-readout and response-proxy AUC side by side. In the
entangled construction, the response axis is orthogonal to the generator's
signal axis and no target signal is planted. Its scalar drive still correlates
with the label. This is the distinction
between geometric separation and statistical independence.

### Analyze your own covariance trials

Prepare an NPZ containing `covariances` with shape `(trials, channels, channels)`,
binary `labels` with shape `(trials,)`, and matching `subjects` identifiers.
Matrices must be symmetric and strictly positive definite; any regularization
belongs in your declared preprocessing. The package accepts compatible empirical
inputs as well as constructed data.

```python
import numpy as np
from aether_measurement.instrument import measure

with np.load("covariance_trials.npz", allow_pickle=False) as data:
    result = measure(data["covariances"], data["labels"], data["subjects"])

print(result["auc"])
# Original-row-order held-out scores and fold assignments are also available.
scores, fold_ids = result["scores"], result["fold_ids"]
```

### Prepare a behavioral pilot

The collection tools join event timing, behavioral covariates and ground truth
through trial identifiers. Start with blank templates:

```bash
aether-measurement ledger init --out runs/pilot
```

After collecting and completing the three CSV ledgers:

```bash
aether-measurement ledger validate --event runs/pilot/event_ledger.csv --covariate runs/pilot/covariate_ledger.csv --ground-truth runs/pilot/ground_truth_ledger.csv
aether-measurement baseline --event runs/pilot/event_ledger.csv --covariate runs/pilot/covariate_ledger.csv --ground-truth runs/pilot/ground_truth_ledger.csv --out runs/baseline.json
```

The baseline validates before scoring and includes trials only when both
usability flags are true. Its inherited trial-level cross-validation differs
from the covariance instrument's subject holdout. Small samples and undefined
targets receive explicit report states.

## Scope and next experiment

The contribution is a measurement investigation: implemented analyses, empirical
specificity checks, controlled recovery and confound experiments, and a collection
specification. The central empirical question remains whether a stable pre-report
judgment distinction can survive those controls in a purpose-designed collection.
Behavioral calibration is the next scientific dependency. This repository
consolidates the existing work; it does not launch that collection.

[METHODS.md](METHODS.md) details the empirical evidence, calibration assumptions,
historical provenance and changes made for this portable implementation. The
[evidence catalog](src/aether_measurement/data/catalog.json) identifies each
packaged source artifact.

The software uses the [Apache-2.0 license](LICENSE). The
[earlier research layout](https://github.com/absjaded/aAether/tree/86f199716eed5a269409f7be72a78be867995dad)
is preserved in Git history, including the exploratory scripts and datacards.
