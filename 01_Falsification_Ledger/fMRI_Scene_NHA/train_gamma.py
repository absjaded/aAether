import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from pathlib import Path
import numpy as np
import random
import itertools

from models.network import ResidualIsolation, NeuRONAExecutor, NeuroVAE, SiTVelocityNet, neuroflow_total_loss
from dataloader.preprocess import DeltaHRFCorrection
from dataloader.split import TemporalBlockSplitter
from models.grammar import VOCAB_SIZE, UNIT_TO_IDX, TOK_FACTS, TOK_VALENCE, UNIT_OPEN, TOK_EOS
from models.parser import LeanParser
from models.autoencoder import DecoupledTextAutoencoder

# Restore InfoNCE anchor loss
def infonce_anchor_loss(z_fmri, z_future, negatives=None):
    """
    Computes InfoNCE-anchor (PIG) loss. 
    z_fmri: Current state (B, D)
    z_future: Positive anchor (Future state) (B, D)
    negatives: Negative bank (N, D). If None, uses z_future.
    """
    temperature = 0.1
    if negatives is None:
        negatives = z_future
        
    # We must ensure the positive anchor is included in the negatives.
    # If negatives is exactly z_future, it's already there at index i.
    # If negatives is a larger bank that INCLUDES z_future, we must provide the correct labels.
    # For simplicity, we can prepend the positive anchor to the negatives for EACH sample.
    
    # Actually, a simpler way when using a global negative bank:
    # sim_pos: (B, 1)
    sim_pos = torch.sum(z_fmri * z_future, dim=-1, keepdim=True) / temperature
    # sim_neg: (B, N)
    sim_neg = torch.matmul(z_fmri, negatives.transpose(0, 1)) / temperature
    
    # logits: (B, 1 + N)
    logits = torch.cat([sim_pos, sim_neg], dim=-1)
    # The true class is always at index 0 (the positive anchor)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    
    return nn.CrossEntropyLoss()(logits, labels)

class AetherGammaPipeline(nn.Module):
    def __init__(self, fmri_dim=768, sensory_dim=20484, num_parcels=4345, latent_dim=1664):
        super().__init__()
        self.delta_hrf = DeltaHRFCorrection()
        self.isolation = ResidualIsolation(sensory_dim, num_parcels)
        self.neuro_vae = NeuroVAE(fmri_dim, latent_dim)
        
        # eDSL Embedding (shared with parser mapping)
        self.edsl_emb = nn.Embedding(VOCAB_SIZE, latent_dim)
        
        self.executor = NeuRONAExecutor(d_parcel=fmri_dim, d_concept=latent_dim, n_parcels=num_parcels)
        self.flow_net = SiTVelocityNet(latent_dim)

    def forward(self, fmri_patches, padding_mask, v_sensory, subj_emb=None, pred_emb=None, obj_emb=None):
        # 1. Delta-HRF Correction
        z_tpj_raw = self.delta_hrf(fmri_patches) # (B, P, F)
        
        # 2. Residual Isolation (uses sensory features to filter noise)
        v_sens_pooled = v_sensory.mean(dim=1) if len(v_sensory.shape) == 3 else v_sensory
        X_res = self.isolation(z_tpj_raw, v_sens_pooled) # (B, P, F)
        
        # 3. NeuroVAE
        mask = padding_mask.unsqueeze(-1).float()
        x_res_pooled = (X_res * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        z_brain, _, kl_loss = self.neuro_vae(x_res_pooled) # (B, 1664)
        
        # 4. NEURONA Executor
        executor_out = None
        if subj_emb is not None and pred_emb is not None and obj_emb is not None:
            executor_out = self.executor(X_res, subj_emb, pred_emb, obj_emb)
            
        return z_brain, kl_loss, executor_out

def train_gamma(
    data_dir: Path,
    checkpoint_path: Path,
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    noise_tier: float = 0.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ce_loss_fn = nn.CrossEntropyLoss()
    parser = LeanParser()
    
    # Load fMRI Data
    latents_train_list, mask_train_list, sens_train_list, valid_train_list = [], [], [], []
    latents_val_list, mask_val_list, sens_val_list, valid_val_list = [], [], [], []
    
    # We load 3 scenes as provided by the user
    scenes = [1, 2, 3]
    for scene_idx in scenes:
        latents_path = data_dir / f"omni/whole_brain/scene{scene_idx}_latents.npy"
        mask_path = data_dir / f"omni/whole_brain/scene{scene_idx}_mask.npy"
        tribe_path = data_dir / f"tribe/nha_Scene{scene_idx}_tribe_cortical.npy"
        
        if not latents_path.exists() or not tribe_path.exists():
            print(f"Warning: Data for scene {scene_idx} not found. Skipping.")
            continue
            
        latents = np.load(latents_path)  # (B, 4345, 768)
        padding_mask = np.load(mask_path) # (B, 4345) bool
        b_size = latents.shape[0]
        
        # Load Real Sensory Audio-Visual features (Tribe v2)
        tribe_raw = np.load(tribe_path) # e.g. (1104, 20484)
        chunks = np.array_split(tribe_raw, b_size)
        v_sensory = np.stack([c.mean(axis=0) for c in chunks]).astype(np.float32) # (B, 20484)
        
        # TEMPORAL BLOCK SPLITTER (applied independently per scene)
        splitter = TemporalBlockSplitter(b_size, train_ratio=0.70, hrf_gap_frames=10)
        
        lat_train, lat_val = splitter.split(torch.tensor(latents, dtype=torch.float32))
        mask_train, mask_val = splitter.split(torch.tensor(padding_mask, dtype=torch.bool))
        sens_train, sens_val = splitter.split(torch.tensor(v_sensory, dtype=torch.float32))
        
        latents_train_list.append(lat_train[0])
        mask_train_list.append(mask_train[0])
        sens_train_list.append(sens_train[0])
        
        # Create valid_pair mask: 1 for all except the last window in the scene
        valid_train = torch.ones(lat_train[0].shape[0], dtype=torch.bool)
        if valid_train.shape[0] > 0:
            valid_train[-1] = False
        valid_train_list.append(valid_train)
        
        latents_val_list.append(lat_val[0])
        mask_val_list.append(mask_val[0])
        sens_val_list.append(sens_val[0])
        
        valid_val = torch.ones(lat_val[0].shape[0], dtype=torch.bool)
        if valid_val.shape[0] > 0:
            valid_val[-1] = False
        valid_val_list.append(valid_val)

    if not latents_train_list:
        raise FileNotFoundError(f"Real data not found at {data_dir}. Halting execution.")
        
    # Pad latents and masks to the global maximum N_max across all scenes
    max_N = max(l.shape[1] for l in latents_train_list + latents_val_list)
    
    pipeline = AetherGammaPipeline(fmri_dim=768, sensory_dim=20484, num_parcels=max_N, latent_dim=1664).to(device)
    autoencoder = DecoupledTextAutoencoder(vocab_size=VOCAB_SIZE, d_model=1664).to(device)
    autoencoder.unit_emb = pipeline.edsl_emb
    
    if torch.cuda.device_count() > 1:
        print(f"[Hardware] Detected {torch.cuda.device_count()} GPUs. Wrapping models in DataParallel.")
        pipeline = nn.DataParallel(pipeline)
        autoencoder = nn.DataParallel(autoencoder)
    
    opt_fMRI = optim.AdamW(pipeline.parameters(), lr=lr, weight_decay=1e-4)
    opt_Text = optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=1e-4)
    
    def pad_to_max(tensor_list, pad_val=0.0):
        padded = []
        for t in tensor_list:
            pad_size = max_N - t.shape[1]
            if pad_size > 0:
                if t.dim() == 3: # latents (B, N, F)
                    p = torch.nn.functional.pad(t, (0, 0, 0, pad_size), value=pad_val)
                elif t.dim() == 2: # mask (B, N)
                    p = torch.nn.functional.pad(t, (0, pad_size), value=pad_val)
            else:
                p = t
            padded.append(p)
        return padded
        
    latents_train_padded = pad_to_max(latents_train_list, pad_val=0.0)
    latents_val_padded = pad_to_max(latents_val_list, pad_val=0.0)
    # The mask is boolean, pad with False
    mask_train_padded = pad_to_max(mask_train_list, pad_val=0) 
    mask_val_padded = pad_to_max(mask_val_list, pad_val=0)
        
    lat_train_all = torch.cat(latents_train_padded, dim=0)
    mask_train_all = torch.cat(mask_train_padded, dim=0)
    sens_train_all = torch.cat(sens_train_list, dim=0)
    valid_train_all = torch.cat(valid_train_list, dim=0)
    
    lat_val_all = torch.cat(latents_val_padded, dim=0)
    mask_val_all = torch.cat(mask_val_padded, dim=0)
    sens_val_all = torch.cat(sens_val_list, dim=0)
    valid_val_all = torch.cat(valid_val_list, dim=0)
    
    train_dataset = TensorDataset(lat_train_all, mask_train_all, sens_train_all, valid_train_all)
    val_dataset = TensorDataset(lat_val_all, mask_val_all, sens_val_all, valid_val_all)
    
    # We use the provided batch_size to proccurrence OOM on large N_max
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SequentialSampler(train_dataset))
    # For validation, we use a small batch size but accumulate the outputs globally to compute cross-scene InfoNCE
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SequentialSampler(val_dataset))
    
    # Base b_size on the full combined dataset for mock text length
    b_size_total = lat_train_all.shape[0] + lat_val_all.shape[0]
    
    # Pre-generate mock texts for Autoencoder
    facts = list(TOK_FACTS)
    valences = list(TOK_VALENCE)
    
    # Ensure stable length for text dataset to pair with fMRI
    # Let's generate a full set of texts matching b_size_total
    mock_strings = [f"({random.choice(valences)} {random.choice(facts)})" for _ in range(b_size_total)]
    sub_idx, pred_idx = parser.parse(mock_strings)
    
    tok_open_idx, tok_eos_idx = UNIT_TO_IDX[UNIT_OPEN], UNIT_TO_IDX[TOK_EOS]
    seq_inputs = torch.stack([torch.full((b_size_total,), tok_open_idx, dtype=torch.long), pred_idx, sub_idx], dim=1)
    seq_targets = torch.stack([pred_idx, sub_idx, torch.full((b_size_total,), tok_eos_idx, dtype=torch.long)], dim=1)
    
    text_dataset = TensorDataset(sub_idx, pred_idx, seq_inputs, seq_targets)
    text_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)
    
    tau = torch.tensor(0.07, device=device) # Initial SoftCLIP temp
    
    for epoch in range(epochs):
        pipeline.train()
        autoencoder.train()
        total_flow_loss = 0.0
        total_pig_loss = 0.0
        total_ce_loss = 0.0
        
        # PASS 1: Decoupled Text Autoencoder
        for t_sub, t_pred, s_in, s_tgt in text_loader:
            t_sub, t_pred, s_in, s_tgt = t_sub.to(device), t_pred.to(device), s_in.to(device), s_tgt.to(device)
            opt_Text.zero_grad()
            
            p_module = pipeline.module if hasattr(pipeline, "module") else pipeline
            z_edsl_text = p_module.edsl_emb(t_sub) + p_module.edsl_emb(t_pred)
            logits, commit_loss = autoencoder(z_edsl_text, s_in)
            loss_ce = ce_loss_fn(logits.view(-1, VOCAB_SIZE), s_tgt.reshape(-1)) + commit_loss.mean()
            loss_ce.backward()
            nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
            opt_Text.step()
            total_ce_loss += loss_ce.item()
            
        # PASS 2: NeuroFlow matching (fMRI to eDSL semantics)
        # Cycle through text_loader infinitely so train_loader doesn't get truncated
        text_iter = itertools.cycle(text_loader)
        
        for lat, p_mask, v_sens, v_pair in train_loader:
            t_sub, t_pred, _, _ = next(text_iter)
            curr_b = min(lat.size(0), t_sub.size(0))
            lat, p_mask, v_sens = lat[:curr_b].to(device), p_mask[:curr_b].to(device), v_sens[:curr_b].to(device)
            v_pair = v_pair[:curr_b]
            t_sub, t_pred = t_sub[:curr_b].to(device), t_pred[:curr_b].to(device)
            
            opt_fMRI.zero_grad()
            opt_Text.zero_grad()
            
            # Embed the text features to act as XFM targets
            p_module = pipeline.module if hasattr(pipeline, "module") else pipeline
            z_edsl_target = p_module.edsl_emb(t_sub) + p_module.edsl_emb(t_pred)
            
            # Add fake obj_emb (just sub_emb again for PoC) to satisfy NeuRONA signature
            z_brain, kl_loss, exec_out = pipeline(
                lat, p_mask, v_sens,
                subj_emb=p_module.edsl_emb(t_sub),
                pred_emb=p_module.edsl_emb(t_pred),
                obj_emb=p_module.edsl_emb(t_sub)
            )
            
            loss_flow, loss_dict = neuroflow_total_loss(z_brain, z_edsl_target.detach(), kl_loss.mean(), p_module.flow_net, tau)
            
            if z_brain.shape[0] > 1:
                valid_mask = v_pair[:-1]
                z_pres = z_brain[:-1][valid_mask]
                z_fut = z_brain[1:][valid_mask]
                if z_pres.shape[0] > 0:
                    loss_pig = infonce_anchor_loss(z_pres, z_fut)
                    loss = loss_flow + loss_pig
                else:
                    loss = loss_flow
                    loss_pig = torch.tensor(0.0)
            else:
                loss = loss_flow
                loss_pig = torch.tensor(0.0)
                
            # Phase 5: PPO Reinforcement Learning for Lean 4 validation
            if exec_out is not None:
                assertion_bool = exec_out['assertion_bool'].detach() # (B,)
                gumbel_logits = exec_out['gumbel_logits'] # (B, 2)
                log_probs = F.log_softmax(gumbel_logits, dim=-1)
                
                # action_idx: 0 for True, 1 for False
                action_idx = (1 - assertion_bool).long()
                old_log_probs = log_probs.gather(1, action_idx.unsqueeze(-1)).squeeze(-1).detach()
                
                # Mock Lean 4 Reward
                # True if the sum of text embedding indices is even (arbitrary mock logic)
                correct_bool = (t_sub.sum(dim=-1) % 2 == 0).float()
                reward = torch.where(assertion_bool == correct_bool, 1.0, -1.0).to(device)
                
                # Advantage estimation (mean baseline)
                advantage = (reward - reward.mean()) / (reward.std() + 1e-8)
                
                # PPO Clipped Surrogate
                curr_log_probs = log_probs.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
                ratio = torch.exp(curr_log_probs - old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantage
                
                loss_ppo = -torch.min(surr1, surr2).mean()
                loss = loss + loss_ppo
                
            loss = loss.mean()
            loss.backward()
            
            # Add gradient clipping to proccurrence exploding gradients and NaN asserts
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            
            opt_fMRI.step()
            opt_Text.step()
            
            total_flow_loss += loss_flow.item()
            total_pig_loss += loss_pig.item()
            
        if epoch % 10 == 0 or epoch == epochs - 1:
            flow_loss_avg = total_flow_loss / len(train_loader)
            pig_loss_avg = total_pig_loss / len(train_loader)
            ce_loss_avg = total_ce_loss / len(text_loader)
            print(f"Epoch {epoch:2d} | Flow Loss: {flow_loss_avg:.4f} | PIG Loss: {pig_loss_avg:.4f} | Text CE: {ce_loss_avg:.4f}")
            # External experiment tracking removed from the public scaffold.
            
    # VALIDATION
    pipeline.eval()
    val_pig = 0.0
    val_lean4 = 0.0
    val_batches = 0
    
    # We will accumulate z_present and z_future across all validation batches
    z_present_list = []
    z_future_list = []
    
    val_text_iter = itertools.cycle(text_loader)
    with torch.no_grad():
        for lat, p_mask, v_sens, v_pair in val_loader:
            t_sub, t_pred, _, _ = next(val_text_iter)
            curr_b = min(lat.size(0), t_sub.size(0))
            lat, p_mask, v_sens = lat[:curr_b].to(device), p_mask[:curr_b].to(device), v_sens[:curr_b].to(device)
            v_pair = v_pair[:curr_b]
            t_sub, t_pred = t_sub[:curr_b].to(device), t_pred[:curr_b].to(device)
            
            p_module = pipeline.module if hasattr(pipeline, "module") else pipeline
            z_brain, _, exec_out = pipeline(
                lat, p_mask, v_sens,
                subj_emb=p_module.edsl_emb(t_sub),
                pred_emb=p_module.edsl_emb(t_pred),
                obj_emb=p_module.edsl_emb(t_sub)
            )
            
            if z_brain.shape[0] > 1:
                valid_mask = v_pair[:-1]
                z_pres = z_brain[:-1][valid_mask]
                z_fut = z_brain[1:][valid_mask]
                if z_pres.shape[0] > 0:
                    z_present_list.append(z_pres)
                    z_future_list.append(z_fut)
            
            if exec_out is not None:
                assertion_bool = exec_out['assertion_bool']
                correct_bool = (t_sub.sum(dim=-1) % 2 == 0).float()
                acc = (assertion_bool == correct_bool).float().mean().item()
                val_lean4 += acc
                
            val_batches += 1
            
    if len(z_present_list) > 0:
        Z_present = torch.cat(z_present_list, dim=0)
        Z_future = torch.cat(z_future_list, dim=0)
        # Using all futures as the negative bank for cross-scene InfoNCE
        val_pig = -infonce_anchor_loss(Z_present, Z_future, negatives=Z_future).item()
            
    if val_batches > 0:
        val_lean4 /= val_batches
        
        # Log InfoNCE denominator specifically as requested
        infoNCE_denominator = Z_future.shape[0] if len(z_present_list) > 0 else 0
        print(f"InfoNCE Denominator (val): {infoNCE_denominator}")
        # External experiment tracking removed from the public scaffold.
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'pipeline': pipeline.state_dict(), 'autoencoder': autoencoder.state_dict()}, checkpoint_path)
    
    print(f"Saved NeuroFlow checkpoint to {checkpoint_path}")
    
    # We might not have infoNCE_denominator if no val batches ran
    info_denom = infoNCE_denominator if 'infoNCE_denominator' in locals() else 0
    return val_pig, val_lean4, info_denom


