# Aether: Falsification Ledger and In-Silico Validation

Aether tests whether a pre-verbal task state can be measured before it is collapsed into a motor response, a verbal report, or an after-the-fact explanation.

The current repository is not a claim that this has been demonstrated in humans end-to-end. It is the evidence package for what has been ruled out, what has been instrumented in silico, and what still has to be measured with a consented human collection.

## Current Claim Discipline

What is supported here:

- Standard public neural datasets are mostly unsuitable for the core claim because they entangle intent, response, timing, or task design.
- The synthetic radiology program recovers a planted pre-verbal gestalt axis under matched nulls and quantifies power, calibration, and response-confound limits.
- Public proxy probes show what real data must provide: ROAMM eye/reading transfers across subjects, ROAMM EEG covariance geometry is weak, and IBL cross-lab GW alignment is close to null under the tested configuration.
- The Aether alignment benchmark quantifies when latent spaces align without correspondence and when true anchors are needed.

What is not supported yet in-silico:

- A human pre-verbal intent manifold has not been directly measured.
- A human neural task space has not been aligned to an LLM latent space.
- No causal steering result has been demonstrated.

## Folder Map

| Folder | Role | Main finding |
|---|---|---|
| `01_Falsification_Ledger/` | Public-data kill reasons. | Prior public paradigms do not isolate intent from response/timing confounds. |
| `02_Empirical_Requirement_Probes/ROAMM_ds007629/` | Human public proxy probe. | Eye/reading transfers cross-subject; tested EEG geometry is weak. |
| `02_Empirical_Requirement_Probes/IBL_RepeatedSite/` | Public neural geometry requirement probe. | GW alignment stays weak/high-degenerate under the tested 100 ms pre-movement setup. |
| `03_In_Silico_Instrument_And_Power/Radiology_Gestalt_Validation/` | Synthetic instrument, calibration, power, and confound limits. | Cohort-backbone detector works in silico; target weak-effect design is about 30 radiologists x 70 reads at >=8% violations, or 20 x 140 reads. |
| `04_Alignment_Identifiability/Synthetic_Geometry_Benchmark/` | Cross-space alignment identifiability benchmark. | 38 / 54 cells align correspondence-free; symmetric low-information regimes need 1-2 true anchors. |
| `90_Legacy_Synthetic_Validation/Synthetic_Validation/` | Earlier synthetic sanity checks. | Retained as historical instrument validation, not the main positive evidence. |

## Key Numbers

| Probe | Result |
|---|---|
| HCP 2-back motor-confound check | Lure vs Non-Target shared the same withheld response; geodesic gap was near zero: delta d = +0.0033, Cohen d = +0.008, p = 0.75. |
| ROAMM eye/reading 39-to-1 | Held-out-subject AUC about 0.81 with run/time features, about 0.79 without them. |
| ROAMM 10 s EEG covariance | Shared-reference AUC about 0.52 broadband and about 0.55 filter-bank. |
| IBL RepeatedSite | Corrected 100 ms label sweep remained close to null with high coupling degeneracy. |
| Radiology Stage 3 | For effect 0.07, confirmed powered designs start at 30 x 70 reads for 8% violations or 20 x 140 reads. |
| Radiology N4 confound | Clean through response-label correlation rho 0.1, unstable around 0.2, broken by 0.3. |
| Aether identifiability | `world_b=38`, `world_a_1=12`, `world_a_2=4`, `paired_only=0`, `representation_limit=0`. |

## Data Policy

Raw datasets, synced derivatives, tensor caches, credentials, SSH details, and local execution logs are not part of the clean evidence package. Local scratch material is ignored through `.gitignore`, especially `files/`, `files.zip`, `.env`, and conversation scratch folders.

Dataset links and redistribution notes live in the per-probe datacards or run cards.