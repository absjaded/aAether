# Dataset Card: UR10e Cobot Telemetry + UCF Crime Video Anomaly Benchmark

## Identity
- **Dataset A:** Universal Robots UR10e collaborative robot (cobot) sensor telemetry — 15 simulated operational cycles, 5 scenarios (Nominal, Geofence, Torque, Hesitant, Combined Critical)
- **Dataset B:** UCF Crime / Road Accidents video anomaly benchmark — publicly available surveillance video clips, augmented with 5x augmentation
- **Source A:** MuJoCo simulation (mujoco_menagerie UR10e model) — see `assets/models/mujoco_menagerie/universal_robots_ur10e/` in `nisp-v0.2`
- **Source B:** UCF Dataset — publicly available. Latent embeddings stored in local data structures.
- **Distribution:** MuJoCo simulation data freely reproducible. UCF latents are model-extracted embeddings — raw video not committed, embeddings not committed (gitignored due to size).

## What Was Attempted
The NISP (Neural Intent Safety Protocol) architecture was built to use brain-like latent representations from video and robot sensor streams as a proxy for human anticipatory intent — the hypothesis being that if an observer's brain state (represented as a TRIBE v2 latent) predicted an anomaly, that pre-verbal prediction could be read out and used to veto the action before execution.

Full 5-layer architecture:
- **L1 Transducer:** `l1_transducer.py` — TRIBE v2 processes video → 1024-dim brain latent
- **L2 Decoder:** Grammar-masked Lean 4 eDSL decoder — generates formal safety verdicts
- **L3 proof:** Aggregates multi-source proof
- **L4 Verifier:** `lean_bridge.py` — SHA-256 Merkle certificate chain
- **L5 Arbiter:** Final PROCEED/REPLAN/ABORT verdict

Results (from `RESULTS_table_ii.md`):
- Geofence, Torque, Hesitant, Combined Critical: 100% ABORT (correct)
- Nominal: 100% PROCEED (correct)
- Lean latency: 0.9–1.9 µs

## Why It Failed for Pre-Verbal Intent Capture

### The Core Issue: Proxy Divergence
The system correctly detected anomalies in robot behavior. But it was measuring the physical consequence of a decision (the robot moving into a forbidden zone) not the pre-verbal cognitive state of the human operator *before* they authorized that movement. The "brain latent" from TRIBE v2 is a predicted fMRI response to a video — a model-predicted proxy, not an actual brain recording. It cannot distinguish between:
- A human who intended the robot to proceed (and the robot is wrong)
- A human who was distracted (and didn't notice the robot was wrong)
- A human who was about to intervene (and the robot is about to self-correct)

All three produce the same video input and therefore the same brain latent proxy. The signal is orthogonal to pre-verbal human intent.

### What It Did Prove
The architecture and the Lean 4 formal verification layer work correctly. The safety system attains 100% fault isolation on simulated scenarios. This is a valid proof of the formal verification layer, not a proof of pre-verbal intent reading.

## Distribution Restrictions
- UCF video clips: refer to UCF dataset license
- MuJoCo simulation data: freely reproducible (no raw data committed)
- NISP code and results: freely committable

## Key Files in This Directory
| File | Description |
|---|---|
| `DATACARD.md` | This file |
| `l1_transducer.py` | TRIBE v2 video → brain latent transducer |
| `l2_decoder_network.py` | Grammar-masked Lean 4 decoder |
| `lean_bridge.py` | SHA-256 Merkle certificate chain verifier |
| `orchestrator.py` | Full 5-layer NISP orchestration |
| `proxy_prep.py` | Bridge: TRIBE v2 latents → NISP input format |
| `generate_synthetic.py` | Synthetic cobot scenario generator |
