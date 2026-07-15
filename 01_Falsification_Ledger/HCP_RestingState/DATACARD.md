# Dataset Card: HCP Resting-State fMRI — Baseline Calibration Attempt

## Identity
- **Dataset:** Human Connectome Project (HCP) Resting-State fMRI
- **Intended Use:** Calibrate a baseline brain geometry mean from undirected, tak-free resting-state data, to serve as a reference geometry against which tak-elicited_response states could be measured
- **Distribution:** HCP Data Use Agreement required. **Raw files must not be committed.**

## What Was Attempted
In several pipeline iterations (chats 12, 17, 18, 25), the resting-state data was proposed as an anchor: rather than building the reference geometry from Target trials (which have their own tak-induced structure), the resting-state covariance would provide a neutral baseline SPD matrix. The pipeline would then measure: *how far does each 2-back trial move from the resting-state geometry?*

Concretely:
- `v2.1_baseline_sliding_window.py` — attempted to load baseline covariance matrices from a resting-state precomputed path
- `runpod_eda.py` — exploratory data analysis on RunPod remote instance to characterize resting-state ROI distributions

## Why It Failed

### Failure 1: Pathing Mismatches on RunPod
Remote RunPod instances mounted data at different paths than the development environment. Scripts that hardcoded `/data/nsvd_fusion/` paths failed immediately on RunPod where data was mounted at `/runpod-volume/`. This caused silent crashes that only appeared in the live output logs.

### Failure 2: Storage OOM on Small Instances
Unzipping the full HCP resting-state dataset on small RunPod instances (20GB storage) caused device storage overflow. The resting-state data for 75 subjects at full resolution exceeded available disk space before any preprocessing could occur.

### Failure 3: Calibration Pathing Defeated the Pipeline Design
Even when loading succeeded, the resting-state reference produced a different geometry curvature than the within-task Target reference. Mixing these geometries (resting vs. tak-active) introduced an additional confound: the distance was now measuring task engagement vs. rest rather than intent violation vs. rule compliance.

## Distribution Restrictions
- All resting-state `.npy` files: **DO NOT COMMIT** (HCP Data Use Agreement)
- Code and datacard: freely committable

## Key Files in This Directory
| File | Description |
|---|---|
| `DATACARD.md` | This file |
| `v2.1_baseline_sliding_window.py` | Baseline sliding window with resting-state calibration attempt |
| `v2.0_baseline_full_trial.py` | Full-trial baseline (earliest complete attempt) |
| `runpod_eda.py` | RunPod EDA script (execution failed due to pathing resolution) |
| `phase5/` | Phase 5 LOSO semantic pipeline scripts |
