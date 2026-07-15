# Dataset Card: OpenNeuro ds004511 — Deception and Cognitive Control EEG

## Identity
- **OpenNeuro Accession:** `ds004511`
- **Full Name:** "A Multimodal Dataset for Deception and Cognitive Control"
- **DOI:** `doi:10.18112/openneuro.ds004511.v1.0.2`
- **Task:** Deception paradigm — subjects lie or tell the truth in response to biographical questions while EEG is recorded
- **Acquisition:** Standard 64-channel EEG setup, ~500 Hz
- **Distribution:** OpenNeuro (see DOI). Raw `.npy` files gitignored due to size.

## What Was Attempted
After abandoning the MATLAB-corrupted Chen 2024 dataset (chat 13), this dataset was selected as a clean replacement specifically because it targets deception — a cognitive state that is closer to intent violation than the Flanker task. The pipeline attempted:
1. **Data ingestion and formatting:** `kaggle_ds004511_ingestion.py` — downloaded raw data on Kaggle, standardized format
2. **Classifier oracle test:** Verified whether any signal existed at all via a simple logistic classifier
3. **snnTorch training loop:** Leaky Integrate-and-Fire SNN with continuous membrane potential integration across EEG timeseries
4. **Hyperparameter sweep:** `sweep_phase05.py` — swept lambda (L1 sparsity), spike thresholds, and LIF beta parameters

## Why It Failed for Pre-Verbal Intent Capture

### Failure 1: Low Spatial Resolution Overfitting
Standard 64-channel EEG has approximately 3–4 cm spatial resolution at best. Cognitive processes related to deception and intent involve coordinated prefrontal-limbic circuits at millimeter scale. The spatial resolution is fundamentally insufficient to isolate these circuits. Standard CNN models overfit to channel-tier noise patterns that are unique to individual subjects, achieving high training accuracy but near-chance generalization across subjects.

### Failure 2: EEG Cannot Distinguish Deception Intent from Execution
The EEG ERN and frontal negativity signals that activate during deception occur *during or after* the lying response, not before it. The "intent to deceive" — the pre-verbal commitment — precedes any measurable EEG deflection by design. The dataset captures the aftermath of the cognitive act, not the pre-verbal intent.

## Distribution Restrictions
- OpenNeuro CC0 — technically public, but large raw files gitignored
- Code and datacard: freely committable

## Key Files in This Directory
| File | Description |
|---|---|
| `DATACARD.md` | This file |
| `kaggle_ds004511_ingestion.py` | Dataset download, formatting, Kaggle ingestion |
| `sweep_phase05.py` | 35KB hyperparameter sweep script (LIF beta, lambda, thresholds) |
| `run_mini_sweep_lambda.py` | Minimal lambda sweep launcher |
