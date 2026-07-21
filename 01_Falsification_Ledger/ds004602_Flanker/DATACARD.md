# OpenNeuro ds004602 Flanker EEG Datacard

## Dataset

Name: OpenNeuro ds004602, EEG Flanker / error-monitoring task.

Primary data link: https://openneuro.org/datasets/ds004602

Data terms: OpenNeuro public dataset. This folder does not redistribute raw EEG, preprocessed arrays, model features, or local outputs.

## Question

Can an error-monitoring EEG task provide a pre-verbal intent signal, or is it post-response error processing?

## Analysis Summary

The attempted path used EEG preprocessing, frozen brain-model style embeddings, sparse autoencoding, and a spiking readout on Flanker/error-related activity.

## Result

The task was excluded as a pre-verbal intent substrate. Error-related negativity is locked to response/error processing, not to a private pre-response state. Temporal pooling over long epochs also collapsed short transient structure and made the representation unsuitable for the intended claim.

## Verdict

Structurally excluded. This dataset is useful as an error-monitoring/response-control reference, but not as evidence for pre-verbal intent measurement.

## Kept Scripts

| File | Purpose |
|---|---|
| `preprocess.py` | Local EEG preprocessing scaffold. |
| `d2_sae.py` | Sparse autoencoder readout prototype. |
| `d3_snn.py` | Simple spiking-network readout prototype. |

## Excluded From Public Package

Remote notebook ingestion scripts, raw/preprocessed arrays, and older synthetic/nuisance phase helpers were archived outside the commit candidate.
