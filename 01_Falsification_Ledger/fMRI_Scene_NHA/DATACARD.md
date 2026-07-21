# Naturalistic Scene fMRI Decoder Datacard

## Dataset
- Source type: naturalistic video-fMRI style scene-viewing derivative
- Modality: derived fMRI-like latent tensors and decoder scaffolds
- Repository policy: no activations, tensors, subject files, checkpoints, or logs are redistributed here

## Question Tested
Could a naturalistic fMRI-derived latent sequence support a formal decoder from brain state into a symbolic action or proof grammar?

## Work Performed
The pipeline scaffold combined adaptive fMRI latents, a Perceiver-style neural decoder, grammar-constrained output, predictive-information objectives, and a delta-HRF correction path. The retained scripts document the architecture attempt, not a redistributable dataset.

## Main Failure Modes
- Timing: the hemodynamic response created a several-TR lag between stimulus, cortical response, and decoder target.
- Scaling: full attention over thousands of latent patches was memory-heavy and unstable for the available run envelope.
- Objective collapse: predictive information did not rise above the baseline in the attempted runs, and the formal-output path did not produce a useful discharge rate.

## Technical Verdict
This branch was rejected as an early intent substrate. It was valuable architecturally because it exposed the need for explicit temporal alignment, smaller latent geometry, and synthetic validation before returning to expensive biological data.

## Files Kept
| File | Purpose |
|---|---|
| `network.py` | Decoder architecture scaffold |
| `train_gamma.py` | Grammar-decoder training loop scaffold |
| `train_delta.py` | Delta-HRF correction scaffold |
| `run_experiment.py` | Local experiment orchestration scaffold |
| `grammar.py`, `parser.py`, `inference.py` | Symbolic grammar and inference utilities |
| `autoencoder.py`, `delta_loss.py` | Latent compression and objective utilities |