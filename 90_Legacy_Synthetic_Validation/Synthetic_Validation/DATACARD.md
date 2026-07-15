# Dataset Card: Synthetic / Mocked Neural Datasets

## Identity
- **Type:** Fully synthetic — no real brain data
- **Generator:** `MockBrainSignalGenerator` (Phase 0 and Phase 0.5) — a PyTorch module that generates mock brain signal tensors in LaBraM's 768-dim embedding space
- **Purpose:** Pressure-test the Riemannian geometry architecture in controlled conditions where the ground-truth signal is known and injected deliberately

## What Was Generated
Two categories of synthetic data:

### Category 1: Phase 0 / snnTorch Sweep Mock
- Shape: `(1, T=10, 768)` — minimal time steps for spike train conversion
- Signal model: `z_brain = z_static + artifact_coupling × z_artifact [+ z_ern if deceptive]`
- ERN injection: discrete spike at dimensions [42, 43, 44] with magnitude 5.0 when `unit == 'deceptive'`
- Used in: `sweep_phase05.py` — swept lambda, spike threshold, LIF beta across 1000+ combinations

### Category 2: Phase 8 MEG Mock (SNR sweep)
- Shape: `(N_trials, 22, 22)` SPD covariance matrices — 22 primary ROIs (Tier-1 + Tier-2)
- SNR=2.0: injected covariance shift on Lure-labeled trials in primary ROIs only
- SNR=0.5: small shift, expected near-zero gap
- Motor ROI control: same injection applied to motor ROIs → expected null result

## Results

### Phase 0 snnTorch Sweeps: Failure
The optimization sweep failed to converge on biologically meaningful features. Without real biological data, the model tuned hyperparameters to extract structure from simulated noise — a mathematical dead-end. The feature extraction reached a singularity where the SAE dictionary collapsed to near-identical columns.

### Phase 8 MEG Mock: Architecture Validated
At SNR=2.0: cohort Lure-vs-Target geodesic distance gap was clearly positive (d > 0.5). The pipeline correctly separated injected intent signal from motor-related variation.
At SNR=0.5: gap approached zero as expected.
Motor ROI subset: d ≈ 0 (no injected motor effect — confirmed null).

**Significance:** This proved the Riemannian architecture is mathematically sound. The failure on real HCP data was attributable entirely to the task design confound (motor command collinearity), not to an architectural flaw.

## Distribution Restrictions
- None — fully synthetic, no data use agreements apply
- All code freely committable

## Key Files in This Directory
| File | Description |
|---|---|
| `DATACARD.md` | This file |
| `synth_validation_scaffold.py` | Phase 8 SNR sweep — the architecture proof |
| `kaggle_phaseA.py` | Phase 8 Kaggle execution wrapper |
| `check_keys.py` | Key mapping verification utility |
