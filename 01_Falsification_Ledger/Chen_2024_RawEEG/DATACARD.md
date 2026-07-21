# Chen 2024 Raw EEG Datacard

## Dataset
- Source: Chen 2024 EEG derivative, local alias used during exploration: nsvd_raw_eeg
- Modality: EEG windows from a preprocessed derivative
- Repository policy: no raw EEG arrays, MATLAB files, or derivative data are redistributed here

## Question Tested
Could subject-invariant neural features be recovered from a preprocessed EEG derivative using domain-adversarial learning?

## Work Performed
The retained script, `domain_adversarial_network.py`, implements a gradient-reversal domain-adversarial neural network. The intended test was to suppress subject identity while preserving cognitive-state information.

## Finding
The run failed as a substrate test. Important metadata needed for reproducible EEG ingestion and interpretation was missing or transformed by earlier preprocessing. Subject identity and spatial channel structure were too entangled: suppressing subject information also removed the features that the class head needed.

## Technical Verdict
The DANN failure is evidence about this derivative, not about the general impossibility of EEG. It shows that preprocessed EEG without complete acquisition metadata is too brittle for the intended subject-invariant measurement.

## Files Kept
| File | Purpose |
|---|---|
| `domain_adversarial_network.py` | Gradient-reversal DANN prototype for subject-invariant EEG features |