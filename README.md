# Aether: Public Evidence Ledger

Aether studies the brief interval where a judgment exists before it becomes a sentence, a button press, or an after-the-fact explanation.

This repository is the public evidence ledger: public-data probes, in-silico instrument checks, datacards, and result summaries. It is not an operating protocol.

## Public Claim Discipline

What this repo supports:

- Public neural datasets were useful as filters: most available paradigms entangle state, response, timing, or task design.
- The synthetic radiology program recovers a planted pre-report gestalt axis under matched nulls and quantifies calibration and response-confound limits.
- Public proxy probes show why cheap observable behavior has to be measured first: ROAMM eye/reading transfers across subjects, while tested EEG covariance geometry stays weak.
- The correspondence benchmark maps when shared latent structure can transfer across spaces and when measured anchors are needed.

What this repo does not contain:

- Raw neural datasets, synced derivatives, tensor caches, credentials, or execution logs.
- Private collection protocols or subject-facing task designs.
- A claim that public data has already solved Aether's biological measurement problem.

## Folder Map

| Folder | Role | Main finding |
|---|---|---|
| `01_Falsification_Ledger/` | Public-data kill reasons. | Prior public paradigms do not isolate state from response/timing confounds. |
| `02_Empirical_Requirement_Probes/ROAMM_ds007629/` | Human public proxy probe. | Eye/reading transfers cross-subject; tested EEG geometry is weak. |
| `02_Empirical_Requirement_Probes/IBL_RepeatedSite/` | Public neural geometry requirement probe. | GW recovery stays weak/high-degenerate under the tested 100 ms pre-movement setup. |
| `03_In_Silico_Instrument_And_Power/Radiology_Gestalt_Validation/` | Synthetic instrument, calibration, power, and confound limits. | Cohort-backbone detector works in silico under the tested weak-effect regime. |
| `04_Correspondence_Identifiability/Synthetic_Geometry_Benchmark/` | Cross-space correspondence benchmark. | 38 / 54 cells transfer correspondence-free; symmetric low-information regimes need 1-2 true anchors. |
| `90_Legacy_Synthetic_Validation/Synthetic_Validation/` | Earlier synthetic sanity checks. | Retained as historical instrument validation, not the main positive evidence. |

## Key Numbers

| Probe | Result |
|---|---|
| HCP 2-back motor-confound check | Lure vs Non-Target shared the same withheld response; geodesic gap was near zero: delta d = +0.0033, Cohen d = +0.008, p = 0.75. |
| ROAMM eye/reading 39-to-1 | Held-out-subject AUC about 0.81 with run/time features, about 0.79 without them. |
| ROAMM 10 s EEG covariance | Shared-reference AUC about 0.52 broadband and about 0.55 filter-bank. |
| IBL RepeatedSite | Corrected 100 ms label sweep remained close to null with high coupling degeneracy. |
| Radiology weak-effect sizing | Effect 0.07 was used as the synthetic sizing target under matched nulls. |
| Radiology N4 confound | Clean through response-label correlation rho 0.1, unstable around 0.2, broken by 0.3. |
| Aether correspondence | `world_b=38`, `world_a_1=12`, `world_a_2=4`, `paired_only=0`, `representation_limit=0`. |

## Data Policy

Raw datasets, synced derivatives, tensor caches, credentials, SSH details, local execution logs, private notes, and collection planning material are not part of the clean evidence package.

Dataset links and redistribution notes live in the per-probe datacards or run cards.
