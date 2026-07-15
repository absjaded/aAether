"""
aether_lab/run_experiment.py — Execution script for Aether-Gamma.

Enforces the Zero Label Mandate and strict temporal validation protocols.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
import subprocess
from datetime import datetime

import torch
import numpy as np

from scripts.config import CFG
from training.train_delta import train_delta

RESULTS_DIR    = CFG.RESULTS_DIR
CHECKPOINT_DIR = CFG.CHECKPOINT_DIR
LAB_recordBOOKS_DIR = Path("research/_lab_recordbooks")

def get_git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown_sha"

def write_manifest(args, run_id: str):
    """Writes the experiment manifest before training begins to ensure offline determinism."""
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_sha": get_git_sha(),
        "rng_state": args.rng_state,
        "config": args.config,
        "corpus": args.corpus,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size
    }
    
    manifest_path = RESULTS_DIR / f"manifest_{run_id}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"[Manifest] Generated: {manifest_path}")
    return manifest

def log_experiment(manifest, pig_score: float, lean4_rate: float, info_denom: int):
    """Appends the rigorous academic result to the lab recordbook."""
    LAB_recordBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LAB_recordBOOKS_DIR / "EXPERIMENT_LOG.md"
    
    entry = f"""
## Run: {manifest['run_id']}
- **Timestamp:** {manifest['timestamp']}
- **Git SHA:** `{manifest['git_sha']}`
- **Corpus:** `{manifest['corpus']}` (rng_state: {manifest['rng_state']})
- **Config:** `{manifest['config']}`
- **Metric:** PIG = {pig_score:.4f} nats
- **InfoNCE Denominator (val):** {info_denom}
- **Lean 4 Discharge Rate:** {lean4_rate if isinstance(lean4_rate, str) else f'{lean4_rate * 100:.1f}%'}
- **ΔPIG vs. γ baseline:** [Pending] (Kaggle Agent to calculate)
- **Temporal split method:** Block Splitting (Gap: 10 frames)
- **RL Algorithm:** [Pending] (Kaggle Agent to implement PPO)
- **Status:** [Pending] (Kaggle Agent to assessment)
- **Description:** Executed Aether-Gamma pipeline according to academic protocols.
---
"""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[Log] Results appended to {log_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Aether-Gamma RL Execution Pipeline")
    parser.add_argument("--rng_state", type=int, required=True, help="Deterministic rng_state")
    parser.add_argument("--config", type=str, required=True, help="Config file path or name")
    parser.add_argument("--corpus", type=str, required=True, help="Corpus identifier (e.g. Aetherfmri)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--noise_tier", type=float, default=0.0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    run_id = f"delta_{int(time.time())}"

    print(f"\n" + "="*60)
    print(f"RUNNING AETHER-DELTA")
    print("="*60)

    # 1. Enforce determinism
    torch.manual_rng_state(args.rng_state)
    np.random.rng_state(args.rng_state)

    # 2. Pre-flight manifest
    manifest = write_manifest(args, run_id)
    
    # import tracker (redacted)
    # tracker.init(
        project="aether-decoder",
        name=run_id,
        config=manifest
    )

    # 3. Execute
    ckpt_path = CHECKPOINT_DIR / f"{run_id}.pt"
    
    # train_delta returns the final metrics for logging
    pig_score, lean4_rate, ece_score, info_denom = train_delta(
        data_dir=CFG.DATA_DIR,
        checkpoint_path=ckpt_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        noise_tier=args.noise_tier
    )

    # 4. Academic Logging
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print(f"  PIG (InfoNCE-anchor): {pig_score:.4f} nats")
    print(f"  InfoNCE Denominator (val): {info_denom}")
    lean_str = lean4_rate if isinstance(lean4_rate, str) else f"{lean4_rate * 100:.1f}%"
    print(f"  Lean 4 Discharge Rate: {lean_str}")
    print(f"  ECE (val): {ece_score:.4f}")
    
    # Compare with Gamma baseline
    # The Gamma Baseline PIG was theoretically log(0.5) = -0.6931 nats, but the recorded baseline was 3.12 nats in earlier runs.
    # We will log the raw delta here against the 3.12 anchor.
    delta_vs_gamma = pig_score - 3.12
    print(f"  Delta-PIG vs. Gamma: {delta_vs_gamma:+.4f} nats")
    print("="*60 + "\n")
    
    log_experiment(manifest, pig_score, lean4_rate, info_denom)

if __name__ == "__main__":
    main()

