# OpenNeuro ds004511 Deception EEG Datacard

## Dataset
- Source: OpenNeuro ds004511, "A multimodal dataset for deception and cognitive control"
- Data link: https://openneuro.org/datasets/ds004511
- DOI: 10.18112/openneuro.ds004511.v1.0.2
- Modality: 64-channel EEG during truth/deception responses
- Repository policy: no raw EEG arrays, subject files, or downloaded data are redistributed here

## Question Tested
Could a public deception EEG task act as a proxy for pre-verbal intent violation?

## Work Performed
The earlier run tested whether deception labels could produce a stable neural representation using standard EEG windows, simple classifier probes, and recurrent/SNN-style modeling. The goal was not just classification. The relevant test was whether a representation survived across subjects without collapsing into subject-specific channel geometry or response-locked artifacts.

## Finding
The dataset was rejected as an intent substrate. Its strongest EEG features are response monitoring and deception-execution signals, not a clean pre-action commitment signal. Standard EEG spatial resolution also made the learned representation vulnerable to subject and channel-placement shortcuts.

## Technical Verdict
This dataset is useful as a negative example: deception is closer to intent than generic attention tasks, but the recording and task timing still observe consequences of the act. It does not isolate the pre-verbal state Aether needs.

## Files Kept
No executable scripts are kept in this public candidate folder. Older ingestion and sweep scripts were archived because they were runtime-specific and not needed to document the result.