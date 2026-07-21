# Robot Telemetry and UCF Video Proxy Datacard

## Dataset
- Source A: simulated Universal Robots UR10e telemetry generated from reproducible robot scenarios
- Source B: UCF Crime / road-accident style anomaly-video benchmark used as a public video proxy
- Modalities: robot telemetry, video anomaly features, formal safety-verdict scaffolds
- Repository policy: no videos, extracted latents, checkpoints, telemetry dumps, or large generated arrays are redistributed here

## Question Tested
Could a video-derived brain-like latent proxy support a formal safety stack for anticipatory intent or anomaly veto?

## Work Performed
The retained scripts represent a five-layer proxy architecture: video-to-latent transduction, grammar-masked decoding, proof aggregation, formal verification, and final proceed/replan/abort arbitration. Synthetic scenarios were used to test the safety-stack mechanics.

## Result
The stack succeeded as a safety/proof-system prototype: simulated geofence, torque, hesitation, and combined-critical cases routed to abort, while nominal cases routed to proceed. The verification layer ran at microsecond-scale latency in the recorded prototype run.

## Why It Was Rejected for Intent
The proxy detects consequences visible in robot or video state. It does not observe the human pre-verbal decision that caused, permitted, or would have corrected that state. Different human intentions can produce identical video frames and telemetry, so the proxy cannot identify intent itself.

## Technical Verdict
This branch remains useful as formal-systems evidence, but it is not evidence that pre-verbal intent has been measured. It helped separate two claims: the safety stack can execute, but the proxy substrate is not the target signal.

## Files Kept
| File | Purpose |
|---|---|
| `generate_synthetic.py` | Synthetic robot scenario generator |
| `l1_transducer.py` | Video-to-latent transducer interface |
| `l2_decoder_network.py` | Grammar-masked decoder scaffold |
| `lean_bridge.py` | Formal-verification bridge |
| `orchestrator.py` | Full proxy safety-stack orchestration |
| `proxy_prep.py` | Feature preparation bridge |