"""
research/aether_lab/data/proxy_prep.py — Bridge v0.2 latents to Aether Lab.

This script:
1. Loads real TRIBE v2 latents (1024,) from nisp-v0.2.
2. Uses the v1.5 adapter (difumo_to_v1_latent) to create (17, 16) Yeo-like latents.
3. Maps v0.2 labels to Grammar v1 social cognition eDSL expressions.
4. Saves a new ucf_cobot_labels.jsonl and .npy latents into research/aether_lab/data/
"""
import sys
import json
import numpy as np
from pathlib import Path

# Add the project root and v0.2 root to path
LAVA_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(LAVA_ROOT))

from lab.datasets.v1_real import load_real_ucf, load_real_cobot

DATA_DIR = LAVA_ROOT / "src" / "aether_lab" / "data"
LATENT_DIR = DATA_DIR / "latents"

def better_adapter(e_1024: np.ndarray) -> np.ndarray:
    """Max-pool (1024,) -> (17, 16) without discarding features."""
    # Reshape to (16 timepoints, 64 features)
    x = e_1024.reshape(16, 64)
    # Split 64 features into 17 groups and take max of each feature group
    # We want 17 features per timepoint.
    # axis=1 is features.
    groups = np.array_split(x, 17, axis=1)
    # Each g is (16, N_feat). Max over N_feat -> (16,)
    pooled = np.stack([g.max(axis=1) for g in groups], axis=0) # (17, 16)
    return pooled.astype(np.float32)
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LATENT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw v0.2 latents (Signal-Preserving Mode)...")
    try:
        from v1_assets.loader import load_ucf_corpus, load_cobot_cycles
        ucf_raw = load_ucf_corpus()
        cobot_raw = load_cobot_cycles()
    except Exception as e:
        print(f"Error loading v0.2 assets: {e}")
        return

    new_records = []

    # Map UCF-Crime
    print(f"Mapping {len(ucf_raw)} UCF samples with 5x augmentation...")
    for c in ucf_raw:
        for aug in range(5):
            scene_id = f"ucf_{c['video_id']}_aug{aug}"
            edsl = "(detect scene (anomaly))" if c["anomaly"] else "(detect scene (normal))"
            
            # Add jitter to raw latent
            jitter = np.random.normal(0, 0.01 * np.abs(c["E"]).mean(), c["E"].shape)
            e_jittered = c["E"] + jitter
            
            latent_17x16 = better_adapter(e_jittered)
            np.save(LATENT_DIR / f"{scene_id}.npy", latent_17x16)
            
            new_records.append({
                "scene_id": scene_id,
                "edsl": edsl,
                "metadata": {"domain_id": 0, "study": c["study"], "aug": aug}
            })

    # Map Cobot
    print(f"Mapping {len(cobot_raw)} Cobot samples with 5x augmentation...")
    for c in cobot_raw:
        for aug in range(5):
            scene_id = f"cobot_cycle_{c['cycle_id']}_aug{aug}"
            if c["expected_verdict"] == "PROCEED":
                edsl = "(command robot (proceed))"
            else:
                edsl = "(command robot (abort))"
            
            # Add jitter
            jitter = np.random.normal(0, 0.01 * np.abs(c["E"]).mean(), c["E"].shape)
            e_jittered = c["E"] + jitter
            
            latent_17x16 = better_adapter(e_jittered)
            np.save(LATENT_DIR / f"{scene_id}.npy", latent_17x16)
            
            new_records.append({
                "scene_id": scene_id,
                "edsl": edsl,
                "metadata": {"domain_id": 1, "predicate": c["predicate_name"], "aug": aug}
            })

    # Write new JSONL
    output_path = DATA_DIR / "ucf_cobot_labels.jsonl"
    with open(output_path, "w") as f:
        for rec in new_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nSuccess! Signal-Preserving Proxy dataset created at {DATA_DIR}")
    print(f"Total samples: {len(new_records)}")

if __name__ == "__main__":
    # Ensure nisp-v0.2 is in path for loader imports
    sys.path.append(str(LAVA_ROOT / "src"))
    main()
