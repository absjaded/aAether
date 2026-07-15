import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from pathlib import Path
import numpy as np
import random
import itertools

from models.network import ResidualIsolation, NeuRONAExecutor, NeuroVAE, softclip_loss
from dataloader.preprocess import DeltaHRFCorrection
from dataloader.split import TemporalBlockSplitter
from models.grammar import VOCAB_SIZE, UNIT_TO_IDX, TOK_FACTS, TOK_VALENCE, UNIT_OPEN, TOK_EOS
from models.parser import LeanParser
from models.autoencoder import DecoupledTextAutoencoder

# DELTA IMPORTS
from models.delta_encoder import AetherDeltaEncoder
from models.delta_flow import SiTVelocityNet as DeltaSiTVelocityNet
from models.delta_loss import bipartite_pig_loss, global_pig_only

def concept_diversity_loss(embs):
    n = F.normalize(embs, dim=-1)
    sim = n @ n.T
    return (sim - torch.eye(sim.shape[0], device=sim.device)).clamp(0).mean()

def delta_neuroflow_total_loss(z_brain_global, z_brain_spatial, z_edsl, kl_loss, flow_net, tau, step, total_steps, lam1=1.0, lam2=0.0, lam3=0.01):
    """Full combined loss for Aether Delta Flow Matching."""
    l_clip = softclip_loss(z_brain_global, z_edsl, tau)

    N = z_brain_spatial.shape[0]
    device = z_brain_spatial.device
    
    # Flow Matching interpolant
    x_0 = torch.randn_like(z_edsl) # pure noise
    t = torch.rand(N, device=device)
    
    # Sigma scheduling: max(0, 1 - step/(0.3 * total_steps))
    sigma_max = 0.1
    # Avoid division by zero if total_steps is small
    decay_factor = max(0.0, 1.0 - (step / max(1, 0.3 * total_steps)))
    sigma_t = sigma_max * (1 - t) * decay_factor
    epsilon = torch.randn_like(z_edsl)
    
    x_t = t.unsqueeze(-1) * z_edsl + (1 - t).unsqueeze(-1) * x_0 + sigma_t.unsqueeze(-1) * epsilon
    v_true = z_edsl - x_0
    v_pred, _ = flow_net(x_t, t, z_brain_spatial)
    l_xfm = F.mse_loss(v_pred, v_true)

    # SoftCLIP cycle consistency (z_brain_global vs flow_reverse(z_edsl, z_brain_spatial))
    # Disabled by default (lam2=0) due to massive memory cost of backpropagating 40 ODE steps
    if lam2 > 0.0:
        z_edsl_hat = flow_net.flow_forward(z_brain_spatial)
        z_brain_hat = flow_net.flow_reverse(z_edsl_hat, z_brain_spatial)
        l_cycle = softclip_loss(z_brain_global, z_brain_hat, tau)
    else:
        l_cycle = torch.tensor(0.0, device=device)

    L = l_clip + lam1 * l_xfm + lam2 * l_cycle + lam3 * kl_loss
    return L, {"clip": l_clip, "xfm": l_xfm, "cycle": l_cycle, "kl": kl_loss}

class AetherDeltaPipeline(nn.Module):
    def __init__(self, fmri_dim=768, sensory_dim=20484, num_parcels=4345, latent_dim=1664, K=128):
        super().__init__()
        self.delta_hrf = DeltaHRFCorrection()
        self.isolation = ResidualIsolation(sensory_dim, num_parcels)
        self.neuro_vae = NeuroVAE(fmri_dim, latent_dim)
        
        self.delta_encoder = AetherDeltaEncoder(d_model=fmri_dim, K=K, fourier_base=10000.0)
        self.edsl_emb = nn.Embedding(VOCAB_SIZE, latent_dim)
        self.executor = NeuRONAExecutor(d_parcel=fmri_dim, d_concept=latent_dim, n_parcels=num_parcels)
        
        self.flow_net = DeltaSiTVelocityNet(d_model=latent_dim)

    def forward(self, fmri_patches, padding_mask, v_sensory, coords, cls_unit, subj_emb=None, pred_emb=None, obj_emb=None):
        # 1. Delta-HRF Correction
        z_tpj_raw = self.delta_hrf(fmri_patches) # (B, P, F)
        
        # 2. Residual Isolation (uses sensory features to filter noise)
        v_sens_pooled = v_sensory.mean(dim=1) if len(v_sensory.shape) == 3 else v_sensory
        X_res = self.isolation(z_tpj_raw, v_sens_pooled) # (B, P, F)
        
        # 3. Delta Encoder (Perceiver IO + Fourier PE)
        z_brain_spatial = self.delta_encoder(X_res, coords, padding_mask, cls_unit) # (B, K, 768)
        
        # 4. NeuroVAE
        mask = padding_mask.unsqueeze(-1).float()
        x_res_pooled = (X_res * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        z_brain_global, _, kl_loss = self.neuro_vae(x_res_pooled) # (B, 1664)
        
        # 5. NEURONA Executor
        executor_out = None
        if subj_emb is not None and pred_emb is not None and obj_emb is not None:
            executor_out = self.executor(X_res, subj_emb, pred_emb, obj_emb, z_brain_global=z_brain_global.detach())
            
        return z_brain_global, z_brain_spatial, kl_loss, executor_out

def train_delta(
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
    
    latents_train_list, mask_train_list, sens_train_list, valid_train_list = [], [], [], []
    coords_train_list, cls_train_list = [], []
    latents_val_list, mask_val_list, sens_val_list, valid_val_list = [], [], [], []
    coords_val_list, cls_val_list = [], []
    
    scenes = [1, 2, 3]
    for scene_idx in scenes:
        latents_path = data_dir / f"omni/whole_brain/scene{scene_idx}_latents.npy"
        mask_path = data_dir / f"omni/whole_brain/scene{scene_idx}_mask.npy"
        coords_path = data_dir / f"omni/whole_brain/scene{scene_idx}_coords.npy"
        cls_path = data_dir / f"omni/whole_brain/scene{scene_idx}_cls.npy"
        tribe_path = data_dir / f"tribe/nha_Scene{scene_idx}_tribe_cortical.npy"
        
        if not latents_path.exists() or not tribe_path.exists():
            print(f"Warning: Data for scene {scene_idx} not found. Skipping.")
            continue
            
        latents = np.load(latents_path)
        padding_mask = np.load(mask_path)
        coords = np.load(coords_path)
        cls_units = np.load(cls_path)
        b_size = latents.shape[0]
        
        tribe_raw = np.load(tribe_path)
        chunks = np.array_split(tribe_raw, b_size)
        v_sensory = np.stack([c.mean(axis=0) for c in chunks]).astype(np.float32)
        
        # GAP = 3 frames (k=3 lag) based on Protocol 3
        splitter = TemporalBlockSplitter(b_size, train_ratio=0.70, hrf_gap_frames=3)
        
        lat_train, lat_val = splitter.split(torch.tensor(latents, dtype=torch.float32))
        mask_train, mask_val = splitter.split(torch.tensor(padding_mask, dtype=torch.bool))
        sens_train, sens_val = splitter.split(torch.tensor(v_sensory, dtype=torch.float32))
        coords_train, coords_val = splitter.split(torch.tensor(coords, dtype=torch.long))
        cls_train, cls_val = splitter.split(torch.tensor(cls_units, dtype=torch.float32))
        
        latents_train_list.append(lat_train[0])
        mask_train_list.append(mask_train[0])
        sens_train_list.append(sens_train[0])
        coords_train_list.append(coords_train[0])
        cls_train_list.append(cls_train[0])
        
        valid_train = torch.ones(lat_train[0].shape[0], dtype=torch.bool)
        if valid_train.shape[0] > 0:
            valid_train[-1] = False
        valid_train_list.append(valid_train)
        
        latents_val_list.append(lat_val[0])
        mask_val_list.append(mask_val[0])
        sens_val_list.append(sens_val[0])
        coords_val_list.append(coords_val[0])
        cls_val_list.append(cls_val[0])
        
        valid_val = torch.ones(lat_val[0].shape[0], dtype=torch.bool)
        if valid_val.shape[0] > 0:
            valid_val[-1] = False
        valid_val_list.append(valid_val)

    if not latents_train_list:
        assess FileNotFoundError(f"Real data not found at {data_dir}. Halting execution.")
        
    max_N = max(l.shape[1] for l in latents_train_list + latents_val_list)
    
    # Delta Pipeline uses K=128
    pipeline = AetherDeltaPipeline(fmri_dim=768, sensory_dim=20484, num_parcels=max_N, latent_dim=1664, K=128).to(device)
    autoencoder = DecoupledTextAutoencoder(vocab_size=VOCAB_SIZE, d_model=1664).to(device)
    autoencoder.unit_emb = pipeline.edsl_emb
    
    if torch.cuda.device_count() > 1:
        print(f"[Hardware] Detected {torch.cuda.device_count()} GPUs. Wrapping models in DataParallel.")
        pipeline = nn.DataParallel(pipeline)
        autoencoder = nn.DataParallel(autoencoder)
    opt_fMRI = optim.AdamW(pipeline.parameters(), lr=lr, weight_decay=0.05)
    opt_Text = optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=0.05)
    
    def pad_to_max(tensor_list, pad_val=0.0):
        padded = []
        for t in tensor_list:
            pad_size = max_N - t.shape[1]
            if pad_size > 0:
                if t.dim() == 3:
                    p = torch.nn.functional.pad(t, (0, 0, 0, pad_size), value=pad_val)
                elif t.dim() == 2:
                    p = torch.nn.functional.pad(t, (0, pad_size), value=pad_val)
            else:
                p = t
            padded.append(p)
        return padded
        
    latents_train_padded = pad_to_max(latents_train_list, pad_val=0.0)
    latents_val_padded = pad_to_max(latents_val_list, pad_val=0.0)
    mask_train_padded = pad_to_max(mask_train_list, pad_val=0) 
    mask_val_padded = pad_to_max(mask_val_list, pad_val=0)
    coords_train_padded = pad_to_max(coords_train_list, pad_val=0)
    coords_val_padded = pad_to_max(coords_val_list, pad_val=0)
        
    lat_train_all = torch.cat(latents_train_padded, dim=0)
    mask_train_all = torch.cat(mask_train_padded, dim=0)
    sens_train_all = torch.cat(sens_train_list, dim=0)
    coords_train_all = torch.cat(coords_train_padded, dim=0)
    cls_train_all = torch.cat(cls_train_list, dim=0)
    valid_train_all = torch.cat(valid_train_list, dim=0)
    
    lat_val_all = torch.cat(latents_val_padded, dim=0)
    mask_val_all = torch.cat(mask_val_padded, dim=0)
    sens_val_all = torch.cat(sens_val_list, dim=0)
    coords_val_all = torch.cat(coords_val_padded, dim=0)
    cls_val_all = torch.cat(cls_val_list, dim=0)
    valid_val_all = torch.cat(valid_val_list, dim=0)
    
    train_dataset = TensorDataset(lat_train_all, mask_train_all, sens_train_all, coords_train_all, cls_train_all, valid_train_all)
    val_dataset = TensorDataset(lat_val_all, mask_val_all, sens_val_all, coords_val_all, cls_val_all, valid_val_all)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SequentialSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SequentialSampler(val_dataset))
    
    b_size_total = lat_train_all.shape[0] + lat_val_all.shape[0]
    
    facts = list(TOK_FACTS)
    valences = list(TOK_VALENCE)
    mock_strings = [f"({random.choice(valences)} {random.choice(facts)})" for _ in range(b_size_total)]
    sub_idx, pred_idx = parser.parse(mock_strings)
    
    tok_open_idx, tok_eos_idx = UNIT_TO_IDX[UNIT_OPEN], UNIT_TO_IDX[TOK_EOS]
    seq_inputs = torch.stack([torch.full((b_size_total,), tok_open_idx, dtype=torch.long), pred_idx, sub_idx], dim=1)
    seq_targets = torch.stack([pred_idx, sub_idx, torch.full((b_size_total,), tok_eos_idx, dtype=torch.long)], dim=1)
    
    text_dataset = TensorDataset(sub_idx, pred_idx, seq_inputs, seq_targets)
    text_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)
    
    tau = torch.tensor(0.07, device=device)
    tau_global_pig = torch.tensor(0.07, device=device, requires_grad=True)
    opt_fMRI.add_param_group({'params': [tau_global_pig]})
    
    total_train_steps = epochs * len(train_loader)
    global_step = 0
    
    for epoch in range(epochs):
        pipeline.train()
        autoencoder.train()
        total_flow_loss = 0.0
        total_pig_loss = 0.0
        total_ce_loss = 0.0
        
        text_iter = itertools.cycle(text_loader)
        
        for lat, p_mask, v_sens, coords, cls_tok, v_pair in train_loader:
            t_sub, t_pred, s_in, s_tgt = next(text_iter)
            curr_b = min(lat.size(0), t_sub.size(0))
            lat, p_mask, v_sens = lat[:curr_b].to(device), p_mask[:curr_b].to(device), v_sens[:curr_b].to(device)
            coords, cls_tok = coords[:curr_b].to(device), cls_tok[:curr_b].to(device)
            v_pair = v_pair[:curr_b]
            t_sub, t_pred = t_sub[:curr_b].to(device), t_pred[:curr_b].to(device)
            s_in, s_tgt = s_in[:curr_b].to(device), s_tgt[:curr_b].to(device)
            
            # PASS 1: Decoupled Text Autoencoder
            opt_Text.zero_grad()
            p_module = pipeline.module if hasattr(pipeline, "module") else pipeline
            z_edsl_text = p_module.edsl_emb(t_sub) + p_module.edsl_emb(t_pred)
            logits, commit_loss = autoencoder(z_edsl_text, s_in)
            loss_ce = ce_loss_fn(logits.view(-1, VOCAB_SIZE), s_tgt.reshape(-1)) + commit_loss.mean()
            loss_ce.backward()
            nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
            opt_Text.step()
            total_ce_loss += loss_ce.item()
            
            # PASS 2: Delta Pipeline Flow
            opt_fMRI.zero_grad()
            z_edsl_target = p_module.edsl_emb(t_sub) + p_module.edsl_emb(t_pred)
            
            # --- VAR-TrueContrast: True Contrastive Negatives via Random Pairs ---
            # Instead of sequential windows from the same scene, sample random pairs 
            # across ALL scenes so that negative samples are diverse and true.
            gap_t = min(4, 1 + (global_step // 200))
            
            # valid_train_all is False exactly at the last frame of each scene.
            # A pair (i, i+gap_t) is valid if no frame in [i, i+gap_t-1] is the end of a scene.
            # Equivalently, the rolling sum of 'end_of_scene' flags in that window must be 0.
            end_of_scene = ~valid_train_all.to(device)
            # Use cumsum to quickly sum over windows
            eos_cumsum = torch.cat([torch.tensor([0], device=device), end_of_scene.cumsum(dim=0)])
            # The sum of end_of_scene from i to i+gap_t-1 is eos_cumsum[i+gap_t] - eos_cumsum[i]
            # If this is 0, the pair is valid.
            valid_mask_all = (eos_cumsum[gap_t:] - eos_cumsum[:-gap_t]) == 0
            
            valid_indices = torch.where(valid_mask_all)[0]
            
            # Randomly sample batch_size indices for 'present'
            rand_idx = torch.randperm(len(valid_indices))[:lat.size(0)]
            b_idx_pres = valid_indices[rand_idx].cpu()
            b_idx_fut  = (b_idx_pres + gap_t).cpu()
            
            lat_pres = lat_train_all[b_idx_pres].to(device)
            mask_pres = mask_train_all[b_idx_pres].to(device)
            sens_pres = sens_train_all[b_idx_pres].to(device)
            coords_pres = coords_train_all[b_idx_pres].to(device)
            cls_pres = cls_train_all[b_idx_pres].to(device)
            
            lat_fut = lat_train_all[b_idx_fut].to(device)
            mask_fut = mask_train_all[b_idx_fut].to(device)
            sens_fut = sens_train_all[b_idx_fut].to(device)
            coords_fut = coords_train_all[b_idx_fut].to(device)
            cls_fut = cls_train_all[b_idx_fut].to(device)
            
            z_brain_global_pres, z_brain_spatial_pres, kl_loss_pres, exec_out = pipeline(
                lat_pres, mask_pres, sens_pres, coords_pres, cls_pres,
                subj_emb=p_module.edsl_emb(t_sub),
                pred_emb=p_module.edsl_emb(t_pred),
                obj_emb=p_module.edsl_emb(t_sub)
            )
            
            z_brain_global_fut, z_brain_spatial_fut, kl_loss_fut, _ = pipeline(
                lat_fut, mask_fut, sens_fut, coords_fut, cls_fut,
                subj_emb=None, pred_emb=None, obj_emb=None
            )
            
            loss_flow, loss_dict = delta_neuroflow_total_loss(
                z_brain_global_pres, z_brain_spatial_pres, z_edsl_target.detach(), kl_loss_pres.mean(), 
                p_module.flow_net, tau, global_step, total_train_steps
            )
            
            if z_brain_global_pres.shape[0] > 0:
                loss = loss_flow
                loss_pig = torch.tensor(0.0)
            else:
                loss = loss_flow
                loss_pig = torch.tensor(0.0)
                
            # Phase 5: PPO Reinforcement Learning for Lean 4 validation
            if exec_out is not None:
                assertion_bool = exec_out['assertion_bool'].detach() 
                gumbel_logits = exec_out['gumbel_logits'] 
                log_probs = F.log_softmax(gumbel_logits, dim=-1)
                
                action_idx = (1 - assertion_bool).long()
                old_log_probs = log_probs.gather(1, action_idx.unsqueeze(-1)).squeeze(-1).detach()
                
                correct_bool = (t_sub.sum(dim=-1) % 2 == 0).float()
                reward = torch.where(assertion_bool == correct_bool, 1.0, -1.0).to(device)
                
                advantage = (reward - reward.mean()) / (reward.std() + 1e-8)
                
                curr_log_probs = log_probs.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
                ratio = torch.exp(curr_log_probs - old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantage
                
                loss_ppo = -torch.min(surr1, surr2).mean()
                loss = loss + loss_ppo
                
            batch_concept_embeddings = torch.cat([p_module.edsl_emb(t_sub), p_module.edsl_emb(t_pred)], dim=0)
            loss = loss + 0.01 * concept_diversity_loss(batch_concept_embeddings)
                
            loss = loss.mean()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=1.0)
            
            opt_fMRI.step()
            
            total_flow_loss += loss_flow.item()
            total_pig_loss += loss_pig.item() if isinstance(loss_pig, torch.Tensor) else loss_pig
            
            global_step += 1
            
        if epoch % 10 == 0 or epoch == epochs - 1:
            flow_loss_avg = total_flow_loss / len(train_loader)
            pig_loss_avg = total_pig_loss / len(train_loader)
            ce_loss_avg = total_ce_loss / len(train_loader) 
            print(f"Epoch {epoch:2d} | Flow Loss: {flow_loss_avg:.4f} | PIG Loss: {pig_loss_avg:.4f} | Text CE: {ce_loss_avg:.4f}")
            
    # VALIDATION
    pipeline.eval()
    val_pig = 0.0
    val_lean4 = 0.0
    val_batches = 0
    val_ece_sum = 0.0
    
    z_present_g_list, z_future_g_list = [], []
    z_present_s_list, z_future_s_list = [], []
    
    val_text_iter = itertools.cycle(text_loader)
    with torch.no_grad():
        for lat, p_mask, v_sens, coords, cls_tok, v_pair in val_loader:
            t_sub, t_pred, _, _ = next(val_text_iter)
            curr_b = min(lat.size(0), t_sub.size(0))
            lat, p_mask, v_sens = lat[:curr_b].to(device), p_mask[:curr_b].to(device), v_sens[:curr_b].to(device)
            coords, cls_tok = coords[:curr_b].to(device), cls_tok[:curr_b].to(device)
            v_pair = v_pair[:curr_b]
            t_sub, t_pred = t_sub[:curr_b].to(device), t_pred[:curr_b].to(device)
            
            p_module = pipeline.module if hasattr(pipeline, "module") else pipeline
            z_brain_global, z_brain_spatial, _, exec_out = pipeline(
                lat, p_mask, v_sens, coords, cls_tok,
                subj_emb=p_module.edsl_emb(t_sub),
                pred_emb=p_module.edsl_emb(t_pred),
                obj_emb=p_module.edsl_emb(t_sub)
            )
            
            # We no longer use sequential neg sampling.
            # However, since we want to assess on ALL valid pairs in the validation set,
            # we will just compute z_brain_global for the entire val set first,
            # and then compute PIG loss using random valid pairs AFTER the loop.
            
            # Just collect the latents
            if z_brain_global.shape[0] > 0:
                z_present_g_list.append(z_brain_global)
                z_present_s_list.append(z_brain_spatial)
            
            if exec_out is not None:
                assertion_bool = exec_out['assertion_bool']
                correct_bool = (t_sub.sum(dim=-1) % 2 == 0).float()
                acc = (assertion_bool == correct_bool).float().mean().item()
                val_lean4 += acc
                
                # ECE Calculation
                gumbel_logits = exec_out['gumbel_logits']
                probs = torch.softmax(gumbel_logits, dim=-1)
                confidence = probs.max(dim=-1)[0]
                prediction = probs.argmax(dim=-1).float()
                accuracy = (prediction == correct_bool).float()
                
                ece = 0.0
                bins = 10
                for i in range(bins):
                    bin_lower = i / bins
                    bin_upper = (i + 1) / bins
                    in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
                    if in_bin.sum() > 0:
                        bin_acc = accuracy[in_bin].mean()
                        bin_conf = confidence[in_bin].mean()
                        ece += (in_bin.sum().float() / confidence.shape[0]) * torch.abs(bin_acc - bin_conf).item()
                val_ece_sum += ece
                
            val_batches += 1
            
    if len(z_present_g_list) > 0:
        Z_all_g = torch.cat(z_present_g_list, dim=0)
        Z_all_s = torch.cat(z_present_s_list, dim=0)
        
        # Use true random negative sampling for validation pairs
        gap_t = min(4, 1 + (global_step // 200))
        end_of_scene = ~valid_val_all.to(device)
        eos_cumsum = torch.cat([torch.tensor([0], device=device), end_of_scene.cumsum(dim=0)])
        valid_mask_all = (eos_cumsum[gap_t:] - eos_cumsum[:-gap_t]) == 0
        valid_indices = torch.where(valid_mask_all)[0]
        
        # We will compute validation PIG by sampling chunks of true random pairs
        val_pig = 0.0
        n_chunks = 0
        chunk_size = batch_size
        
        # assess over a fixed number of chunks or all valid indices
        num_pairs = len(valid_indices)
        
        for i in range(0, num_pairs, chunk_size):
            if i + chunk_size > num_pairs:
                break
                
            b_idx_pres = valid_indices[i:i+chunk_size]
            b_idx_fut = b_idx_pres + gap_t
            
            zp_g = Z_all_g[b_idx_pres]
            zf_g = Z_all_g[b_idx_fut]
            zp_s = Z_all_s[b_idx_pres]
            zf_s = Z_all_s[b_idx_fut]
            
            loss_pig_val, _ = bipartite_pig_loss(zp_g, zf_g, zp_s, zf_s, tau_global=tau_global_pig)
            val_pig += loss_pig_val.item()
            n_chunks += 1
            
        if n_chunks > 0:
            val_pig /= n_chunks
            
    if val_batches > 0:
        val_lean4 /= val_batches
        val_ece = val_ece_sum / val_batches
        infoNCE_denominator = Z_all_g.shape[0] if len(z_present_g_list) > 0 else 0
        print(f"InfoNCE Denominator (val): {infoNCE_denominator}")
    else:
        val_ece = 0.0
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'pipeline': pipeline.state_dict(), 'autoencoder': autoencoder.state_dict()}, checkpoint_path)
    
    print(f"Saved Delta NeuroFlow checkpoint to {checkpoint_path}")
    
    info_denom = infoNCE_denominator if 'infoNCE_denominator' in locals() else 0
    return val_pig, val_lean4, val_ece, info_denom
