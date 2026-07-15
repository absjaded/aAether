# Aether Research Map

## Claim Chain

Aether's funding claim is a sequence of separately falsifiable dependencies:

1. A pre-verbal task state has a measurable representation.
2. The representation contains state geometry rather than response, identity, or acquisition geometry.
3. Relevant geometry transfers across people, sessions, tasks, and measurement systems.
4. Human and model spaces can be aligned without correspondence, or with a measurable number of anchors.
5. The aligned state can causally steer a model while preserving unrelated capabilities.

Evidence at one layer does not establish a later layer.

## Current Evidence

| Layer | Evidence | Current conclusion |
|---|---|---|
| Cross-person measurement | ROAMM eye/reading, 39-to-1 | Behavioral proxy transfers; held-out AUC about 0.81. |
| Neural measurement | ROAMM 10 s EEG | Tested covariance geometry is weak: broadband AUC about 0.52 and filter-bank AUC about 0.55. |
| Personal calibration | ROAMM and synthetic radiology | Sparse calibration is unstable; cohort geometry is currently stronger than subject-only references. |
| Cross-session stability | ROAMM runs 1-2 to 4-5 | Eye/reading transfer weakens but remains above chance; tested EEG geometry does not. |
| Correspondence-free alignment | IBL RepeatedSite | GW was close to its null with high coupling degeneracy; richer labels helped only slightly. |
| Alignment identifiability | Aether synthetic full grid | 38 / 54 cells were correspondence-free; symmetric binary/compositional cells required 1-2 true anchors; no paired upper-bound failures. |
| Instrument recovery | Synthetic radiology | The cohort-backbone extractor recovers weak planted geometry under matched clean nulls. |
| Confound specificity | Synthetic N4 | Clean through response-label correlation rho 0.1, unstable around 0.2, broken by 0.3. |
| Causal model steering | None yet | This is the largest unsupported link in the claim chain. |

## Decision Questions And Current Answers

| Rank | Question | Current answer | Status |
|---:|---|---|---|
| 1 | Under what conditions is cross-space alignment identifiable? | In the synthetic full grid, rich or asymmetric geometry usually aligns without correspondence; symmetric binary and symmetric compositional geometry needs 1-2 true anchors. Paired recovery did not fail. | Answered in silico; needs external validation |
| 2 | Can a public human neural task space align to an LLM task space? | Not answered. ROAMM/IBL do not test human-to-model latent alignment. | Unanswered |
| 3 | Does an aligned state support selective causal steering? | Not answered. No steering experiment has been run yet. | Unanswered |
| 4 | Can explicit nuisance control move the N4 confound boundary? | Partially answered. The current synthetic N4 boundary is clean through response-label correlation rho 0.1, unstable around 0.2, and broken by 0.3; explicit nuisance-control mitigation has not been tested. | Partially answered |
| 5 | Which state geometry survives subject, session, task, and site shifts? | Partially answered. ROAMM eye/reading transfers cross-subject and weakly cross-run; tested ROAMM EEG geometry does not. Task, site, and model shifts remain open. | Partially answered |
| 6 | What are the anchor, trial, and subject cost curves? | Partially answered. Synthetic alignment anchor cost is 1-2 true anchors in the blocked regimes; empirical ROAMM/IBL trial, subject, and session cost curves remain open. | Partially answered |

## Experiment Gate

Every side quest must specify the claim layer, falsifier, matched null, architecture decision, locked metrics, compute cost, and stopping rule before a substantial run.

Abstraction checkpoints:

1. Before implementation: does the experiment test a general dependency or only improve one dataset score?
2. Before compute: will every plausible outcome change a decision?
3. During analysis: is the signal state, response, identity, acquisition, or leakage?
4. After analysis: what transfers across subjects, sessions, tasks, modalities, and models?
5. Before presentation: is the result evidence for measurement, alignment, or intervention?

## Completed Experiment: Alignment Identifiability

Question: when two spaces encode the same latent state, which combinations of relational richness, symmetry, distortion, noise, and paired anchors permit recovery?

Locked design:

- Relational regimes: binary, ordinal, and compositional.
- Geometry: symmetric versus asymmetric.
- Distortion: isometric through nonlinear deformation.
- Noise: low through high.
- Correspondence-free methods: entropic GW, distance-signature matching, and signature-fused GW.
- Calibrated method: true point-paired orthogonal Procrustes at a locked anchor curve.
- Upper bounds: held-out paired Procrustes and paired ridge regression.
- Firewall: state labels and true point pairs are unavailable to correspondence-free methods.
- Null: each fitted map is scored against 99 within-space label permutations; exact pair recovery is evaluated against 1 / N.

Verdicts:

- `world_b`: a correspondence-free method recovers state and non-degenerate relational structure.
- `world_a_k`: correspondence-free methods fail, but k paired anchors recover.
- `paired_only`: the paired upper bound works, but the tested anchor range does not.
- `representation_limit`: even the flexible paired upper bound fails.

The deliverable is a failure-boundary table, not a best score.

## Result: Alignment Identifiability Full Grid

Run location: `04_Alignment_Identifiability/Synthetic_Geometry_Benchmark/outputs/identifiability_full_grid/`.

Configuration: 3 relational regimes x 2 symmetry regimes x 3 distortion levels x 3 noise levels x 8 replicates = 432 replicate cells. Each cell used 100 latent points, 8 observation dimensions, anchors `1,2,4,8,16,32`, and 99 label-permutation nulls.

Validity fixes before the full run:

- B-space rows are independently permuted; same row index is no longer a hidden point-pair cue.
- Observation spaces use centering plus one global scale after rotation, not per-coordinate standardization.
- The pre-observation embedding is direct and metric-preserving; random projection no longer breaks intended symmetries.
- The symmetric compositional regime is a true square-grid geometry; nonlinear signed features are reserved for the asymmetric regime.

Verdict counts across 54 design cells:

| Verdict | Cells | Interpretation |
|---|---:|---|
| `world_b` | 38 | Correspondence-free geometry recovered state under the locked thresholds. |
| `world_a_1` | 12 | One true paired anchor was enough after correspondence-free failure. |
| `world_a_2` | 4 | Two true paired anchors were needed. |
| `paired_only` | 0 | The tested anchor range always matched the paired upper bound when needed. |
| `representation_limit` | 0 | The paired upper bounds never failed in this constructed grid. |

Boundary by regime:

| Regime | Result |
|---|---|
| Binary asymmetric | 9 / 9 `world_b`; class imbalance and class-specific spread made the state identifiable without anchors. |
| Binary symmetric | 2 / 9 `world_b`, 7 / 9 `world_a_1`; pure balanced two-state geometry generally needs one anchor. |
| Ordinal symmetric | 9 / 9 `world_b`; ordered one-dimensional geometry carries semantic position even without paired samples. |
| Ordinal asymmetric | 9 / 9 `world_b`; asymmetry strengthens an already identifiable ordered geometry. |
| Compositional symmetric | 5 / 9 `world_a_1`, 4 / 9 `world_a_2`; grid symmetries block free semantic orientation. |
| Compositional asymmetric | 9 / 9 `world_b`; relational asymmetry resolves correspondence. |

Winning methods: GW won 20 cells, distance-signature matching won 15, signature-fused GW won 3, and anchor Procrustes won 16. Among `world_b` cells, mean label-edge / exact-pair / structure-r were: distance signature 0.336 / 0.101 / 0.812, GW 0.640 / 0.553 / 0.975, and signature-fused GW 0.553 / 0.344 / 0.967.

Interpretation: the architecture problem is not whether a paired map can recover the constructed relation; paired Procrustes and paired ridge passed every cell. The hard boundary is identifiability without anchors. Aether should treat relational richness, symmetry, and anchor cost as required measurements, not as implementation details.

Funding-safe claim from this run: in silico, when two observation spaces share a latent state, the Aether alignment math recovers correspondence-free geometry in rich or asymmetric regimes and quantifies the paired-anchor cost when symmetry blocks free recovery. This does not prove neural-to-model alignment or causal steering.

## Claim Discipline

The current permitted claim is: Aether has a falsifiable measurement and alignment architecture with quantified recovery, power, calibration, and confound limits in public and synthetic probes.

The current evidence does not establish that a human pre-verbal intent manifold has been aligned to a model or that steering from such a manifold works.

