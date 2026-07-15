# Dataset Card: Naturalistic Scene-Viewing fMRI (NHA / Gallant-Planck)

## Identity
- **Common Name:** NHA fMRI — Naturalistic Hyperalignment Scene-Viewing Dataset
- **Source:** Derived from continuous naturalistic video-fMRI paradigms (Gallant Lab style); processed via the TRIBE v2 foundation model and Omni-fMRI AdaptiveMAE encoder
- **Distribution:** **NOT for redistribution.** Raw fMRI activations are derived from internal lab recordings. The `.npy` files are excluded from this repository via `.gitignore`.

## What Was Attempted
A full Neural-eDSL decoder pipeline: continuous naturalistic video scenes (3 scenes, ~13 min each) were encoded through:
1. **TRIBE v2** — tri-modal (text, audio, video) → 20,484 cortical vertex predictions in fsaverage5 space
2. **Subcortical diffusion** — Gaussian blur into MNI space, Harvard-Oxford atlas extraction (8,802 voxels)
3. **Omni-fMRI AdaptiveMAE** — dynamic patch unitization into (W, N_max, 768) embeddings
4. **Aether-Gamma decoder** — attempted to decode Lean 4 formal grammar expressions from brain state via RL (PPO + PIG reward)

Architecture: `network.py` (Perceiver IO + 3D spatial attention), `train_gamma.py` (PPO loop), `train_delta.py` (delta-HRF correction via Wiener deconvolution).

## Why It Failed for Pre-Verbal Intent Capture

### Failure 1: Hemodynamic Delay (4–7 TR temporal misalignment)
TRIBE v2 applies a canonical −5s HRF correction calibrated for early sensory cortex. The ToM/TPJ regions relevant to intent have HRF peaks at 9–12s. This leaves a 4–7 TR misalignment between the model's output and true neural intent signals. The Wiener deconvolution delta-HRF correction was implemented but at no point converged.

### Failure 2: VRAM OOM at O(P²) Spatial Attention
The `latents.npy` tensors are shape `(W, N_max=4345, 768)`. Full self-attention across 4,345 patches is O(P²) = ~19M attention weights per window. At fp32, this requires ~72MB per window per layer. With 40-TR windows and multi-layer architecture, this consistently triggered VRAM OOM on both local and Kaggle GPUs (P100 16GB).

### Failure 3: PIG Collapsed to 0.0000 nats
The Predictive Information Gain metric (measuring how much the brain state predicts the model's future output) converged to 0.0000 nats across 8+ training runs (see `EXPERIMENT_LOG.md`). The Lean 4 formal proof discharge rate collapsed to 0.0% after the first few runs. The decoder learned nothing beyond the baseline.

## Distribution Restrictions
- Raw `.npy` tensor files: **DO NOT COMMIT** (gitignored)
- Code, datacards, logs, architecture documents: freely committable
- No ethics_assessment or HCP-specific restrictions apply here (these are model-predicted, not scanned activations)

## Key Files in This Directory
| File | Description |
|---|---|
| `DATACARD.md` | This file |
| `network.py` | Perceiver IO decoder with 3D spatial attention |
| `train_gamma.py` | PPO training loop with PIG reward |
| `train_delta.py` | Delta-HRF correction training |
| `run_experiment.py` | Experiment orchestration |
