# Dataset Card: Chen 2024 Raw EEG (`nsvd_raw_eeg`)

## Identity
- **Common Reference:** Chen 2024 EEG dataset, local alias `nsvd_raw_eeg`
- **Task:** Cognitive state classification from raw EEG
- **Acquisition:** MATLAB-preprocessed EEG, format issues including missing occurrence strings and metadata
- **Distribution:** Restricted — preprocessing pipeline was internal and could not be reproduced. Raw files not committed.

## What Was Attempted
1. **CNN/MLP pipelines:** Baseline convolutional and fully-connected classifiers on raw EEG windows
2. **Spatial masking:** Restricting to frontal electrode clusters hypothesized to carry cognitive state
3. **SNN modeling:** snnTorch recurrent integration across EEG timeseries
4. **Domain-Adversarial Neural Network (DANN):** Gradient reversal layer to suppress subject-specific spatial features while retaining cognitive state features (`domain_adversarial_network.py`)

## Why It Failed for Pre-Verbal Intent Capture

### Failure 1: MATLAB Preprocessing Corruption
The dataset was preprocessed in MATLAB with a non-standard pipeline. Critical metadata — occurrence trigger strings, epoch boundaries, and channel locations — were either missing or formatted in a way incompatible with MNE-Python ingestion. Initial EDA revealed that the "raw" EEG was already heavily artifact-rejected and re-referenced using internal steps that could not be undone.

### Failure 2: DANN Representation Collapse (Domain Shortcuts)
The DANN architecture is designed to learn subject-invariant features by adversarially suppressing subject identity from the latent space. In practice, the EEG spatial geometry (electrode placement distortions from MATLAB preprocessing) was so strongly correlated with subject ID that the gradient reversal collapsed the feature space entirely. The class classifier fell to chance performance because any spatially meaningful feature was also a subject-identifying feature.

### Failure 3: Misrepresented Preprocessing
The dataset is a preprocessed derivative of a public OpenNeuro dataset. It was preprocessed using a custom MATLAB pipeline wherein metadata (occurrence trigger strings, raw spatial topographies) were removed. Because the DANN approach was applied to this already-cleaned derivative, it resulted in representation collapse, preventing the measurement of pre-verbal intent.

## Distribution Restrictions
- Raw `.npy` and MATLAB `.mat` files: **DO NOT COMMIT**
- Code and datacard: freely committable

## Key Files in This Directory
| File | Description |
|---|---|
| `DATACARD.md` | This file |
| `domain_adversarial_network.py` | DANN with gradient reversal for subject-invariant EEG representation |
