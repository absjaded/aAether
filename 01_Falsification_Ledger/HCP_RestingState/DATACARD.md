# HCP Resting-State Baseline Datacard

## Dataset
- Source: Human Connectome Project resting-state fMRI
- Data link: https://www.humanconnectome.org/study/hcp-young-adult
- Modality: resting-state fMRI
- Repository policy: no HCP data, derivatives, arrays, subject files, or manifests are redistributed here

## Question Tested
Could resting-state geometry serve as a neutral baseline for measuring task-evoked N-back states?

## Work Performed
This branch explored whether an off-task resting covariance reference could replace an in-task target reference. The motivation was to avoid building the reference geometry from task trials that already contain response and stimulus structure.

## Finding
The branch was rejected for the falsification ledger. Resting and N-back states live under different task regimes, so distances from rest primarily measure task engagement versus rest. That is a different contrast than lure violation versus rule-compliant withholding.

## Technical Verdict
Resting-state geometry may be useful for calibration research, but it is not a clean control for the HCP N-back intent question. The cleaner negative control is within-task: compare lure trials against non-target trials, because both are withhold-response conditions.

## Files Kept
Only this datacard is kept in the public candidate folder. Runtime scripts and intermediate baseline attempts were archived.