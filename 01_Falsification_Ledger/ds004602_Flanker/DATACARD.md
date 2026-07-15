# Dataset Card: OpenNeuro ds004602 — EEG Flanker Task (ERN Proxy)

## Identity
- **OpenNeuro Accession:** `ds004602`
- **Full Name:** "A large electroencephalography dataset for cognitive flexibility and error monitoring"
- **Task:** Eriksen Flanker Task — congruent/incongruent arrow trials capturing Error-Related Negativity (ERN) at ~150ms post-response
- **Acquisition:** 128-channel EGI HydroCel GSN EEG, 500 Hz, N=20 subjects
- **Distribution:** OpenNeuro CC0 License — freely redistributable, but raw EEG `.npy` files are gitignored due to file size

## What Was Attempted
1. **LaBraM feature extraction:** Frozen Large Brain Model (LaBraM, 768-dim) applied to bandpass-filtered, epoch-sliced EEG windows (−400ms to +600ms around stimulus)
2. **Sparse Autoencoder (SAE):** Dictionary learning (768→4096) to extract monosemantic features; fixed `DECEPTION_FEATURE_IDX = 42` as a stand-in for the ERN signal
3. **Spiking Neural Network (SNN):** snnTorch Leaky Integrate-and-Fire network to convert SAE feature activations into spike trains, then RL penalties
4. **Global centroid classification:** Cosine similarity between trial embeddings and a population-average centroid to classify deceptive vs. honest cognitive states
5. **Contrastive k-NN geometry:** k-nearest neighbors on the LaBraM embedding space

## Why It Failed for Pre-Verbal Intent Capture

### Failure 1: Cross-Subject Skull Geometry Warping
EEG signals are volume-conducted through skull, scalp, and cerebrospinal fluid before reaching the electrodes. Physical skull geometry, scalp impedance, and cortical folding patterns are unique to each individual. This means a spatial centroid learned from one subject is geometrically meaningless for another. Zero-shot cross-subject generalization was impossible.

### Failure 2: SAE Temporal Window Superposition Paradox
The ERN signal is a 50–100ms transient occurrence within a 1000ms epoch. Averaging LaBraM hidden states across the full temporal window (`[90:130]ms` slice, then mean-pooled) collapsed the continuous neural dynamics into a single vector. Distinct cognitive states at different timepoints superimposed and cancelled, collapsing True Positive Rate to ~1.6%.

### Failure 3: The ERN Is Post-Articulatory
Even if the EEG signal could be recovered, the ERN fires ~50–100ms *after* response execution. This is definitionally post-verbal — the error has already been committed and registered. There is no pre-verbal intent signal in this occurrence-related potential.

## Distribution Restrictions
- Raw `.set`, `.fdt`, `.npy` preprocessed files: **DO NOT COMMIT** (gitignored)
- Code and datacard: freely committable

## Key Files in This Directory
| File | Description |
|---|---|
| `DATACARD.md` | This file |
| `kaggle_labram_ingestion.py` | Pipeline: download ds004602, bandpass, epoch, LaBraM forward pass |
| `kaggle_temporal_eda.py` | EDA of temporal dynamics in LaBraM embeddings |
| `d2_sae.py` | Sparse Autoencoder (768→4096 dictionary) |
| `d3_snn.py` | snnTorch Leaky I&F SNN spike train converter |
| `d1_mock_generator.py` | MockBrainSignalGenerator (Phase 0.5 synthetic EEG) |
| `d1_5_nuisance_regressor.py` | Nuisance regressor to suppress artifact coupling |
| `preprocess.py` | Raw EEG preprocessing (bandpass, epoch, resample) |
