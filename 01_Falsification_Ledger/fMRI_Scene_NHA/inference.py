import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from models.network import TacitExecutor, ResidualIsolation, AetherGammaFlowModel
from models.autoencoder import DecoupledTextAutoencoder
from models.grammar import VOCAB_SIZE, UNIT_TO_IDX, UNIT_OPEN, TOK_EOS, VOCAB
from dataloader.preprocess import DeltaHRFCorrection

class AetherGammaPipeline(torch.nn.Module):
    """Reconstructed Pipeline for Inference."""
    def __init__(self, fmri_dim=768, sensory_dim=128, num_parcels=3712, latent_dim=128):
        super().__init__()
        self.delta_hrf = DeltaHRFCorrection()
        self.executor = TacitExecutor(latent_dim, num_parcels)
        self.isolation = ResidualIsolation(latent_dim, sensory_dim)
        self.flow_net = AetherGammaFlowModel(latent_dim)
        self.patch_proj = torch.nn.Linear(fmri_dim, latent_dim)

    def forward(self, fmri_patches, padding_mask, v_sensory):
        z_tpj_raw = self.delta_hrf(fmri_patches)
        z_tpj_proj = self.patch_proj(z_tpj_raw)
        z_executed = self.executor(z_tpj_proj, padding_mask)
        v_sens_pooled = v_sensory.mean(dim=1) if len(v_sensory.shape) == 3 else v_sensory
        z_fmri_grounded = self.isolation(z_executed, v_sens_pooled)
        return z_fmri_grounded, self.flow_net

def run_inference(data_dir: Path, checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Models
    pipeline = AetherGammaPipeline().to(device)
    autoencoder = DecoupledTextAutoencoder(vocab_size=VOCAB_SIZE, d_model=128).to(device)

    # Load checkpoint if exists
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        pipeline.load_state_dict(ckpt['pipeline'], strict=False)
        autoencoder.load_state_dict(ckpt['autoencoder'], strict=False)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Running with untrained weights!")

    pipeline.eval()
    autoencoder.eval()

    # 2. Load Sample fMRI Data
    latents_path = data_dir / "omni/cortical/nha_Scene1_latents.npy"
    mask_path = data_dir / "omni/cortical/nha_Scene1_mask.npy"

    if latents_path.exists() and mask_path.exists():
        latents = np.load(latents_path)[0:1] # Take only first batch (1 slice)
        padding_mask = np.load(mask_path)[0:1]
        print("Loaded valid fMRI slice from disk.")
    else:
        print("fMRI data not found. Generating mock fMRI slice for testing...")
        latents = np.random.randn(1, 3712, 768).astype(np.float32)
        padding_mask = np.ones((1, 3712), dtype=bool)

    v_sensory = np.random.randn(1, 17, 128).astype(np.float32)

    # Move to device
    lat = torch.tensor(latents, dtype=torch.float32).to(device)
    p_mask = torch.tensor(padding_mask, dtype=torch.bool).to(device)
    v_sens = torch.tensor(v_sensory, dtype=torch.float32).to(device)

    with torch.no_grad():
        # 3. Continuous fMRI Physics Pass
        print("Extracting Tacit Knowledge (Continuous Physics Pass)...")
        z_fmri_grounded, flow_net = pipeline(lat, p_mask, v_sens)

        # 4. Flow Matching ODE Solver (Euler Method)
        print("Solving Flow Matching ODE (t=0 -> t=1)...")
        x_t = z_fmri_grounded
        steps = 20
        dt = 1.0 / steps
        for step in range(steps):
            t_val = step * dt
            t_tensor = torch.full((x_t.size(0), 1), t_val, device=device)
            v_pred = flow_net(x_t, t_tensor)
            x_t = x_t + v_pred * dt

        z_edsl_pred = x_t  # The final predicted continuous language embedding

        # 5. Autoregressive Text Generation
        print("Discretizing and Generating Lean 4 Logic...")
        # Map into FSQ Bottleneck
        z_edsl_seq = z_edsl_pred.unsqueeze(1)
        z_q, _, _ = autoencoder.fsq(z_edsl_seq)

        # Initialize sequence with UNIT_OPEN
        input_units = [UNIT_TO_IDX[UNIT_OPEN]]
        max_len = 10
        
        for _ in range(max_len):
            unit_tensor = torch.tensor([input_units], dtype=torch.long, device=device)
            unit_emb = autoencoder.unit_emb(unit_tensor)
            
            # Combine [FSQ_Slot] + [Units]
            seq = torch.cat([z_q, unit_emb], dim=1)
            seq = seq + autoencoder.pos_emb[:, :seq.shape[1], :]
            
            mask = torch.nn.Transformer.generate_square_subsequent_mask(seq.shape[1], device=device)
            out = autoencoder.transformer(seq, mask=mask)
            
            # Predict next unit from the last position
            logits = autoencoder.vocab_head(out[:, -1, :])
            next_tok = int(logits.argmax(dim=-1).item())
            
            input_units.append(next_tok)
            
            if VOCAB[next_tok] == TOK_EOS:
                break

        # Decode units to string
        output_string = " ".join([VOCAB[idx] for idx in input_units])
        print("Aether Inference Output:")
        print(f"Generated Lean 4 Logic: {output_string}")
        print("==================================================\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aether-Gamma Inference Pipeline")
    parser.add_argument("--data_dir", type=str, default="Aetherfmri", help="Path to fMRI data directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/aether_gamma.pth", help="Path to trained weights")
    args = parser.parse_args()
    
    run_inference(Path(args.data_dir), Path(args.checkpoint))
