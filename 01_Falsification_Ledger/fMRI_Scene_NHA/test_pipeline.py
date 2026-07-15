import torch
import torch.nn.functional as F
from models.network import neuroflow_total_loss
from training.train_gamma import AetherGammaPipeline
from models.autoencoder import DecoupledTextAutoencoder
from models.grammar import VOCAB_SIZE

def test_pipeline_dry_run():
    print("--- Starting AetherGammaPipeline Synthetic Dry-Run ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize models
    print("Initializing models...")
    pipeline = AetherGammaPipeline(fmri_dim=768, sensory_dim=20484, num_parcels=4345, latent_dim=1664).to(device)
    autoencoder = DecoupledTextAutoencoder(vocab_size=VOCAB_SIZE, d_model=1664).to(device)
    autoencoder.unit_emb = pipeline.edsl_emb
    
    # 2. Generate Synthetic Data
    B = 2
    P = 4345
    F_dim = 768
    S_dim = 20484
    
    print(f"Generating synthetic batch (B={B})...")
    lat = torch.randn(B, P, F_dim, device=device)
    p_mask = torch.ones(B, P, dtype=torch.bool, device=device) # False means masked, but let's use all True for testing
    v_sens = torch.randn(B, S_dim, device=device)
    
    t_sub = torch.randint(0, VOCAB_SIZE, (B,), device=device)
    t_pred = torch.randint(0, VOCAB_SIZE, (B,), device=device)
    
    s_in = torch.randint(0, VOCAB_SIZE, (B, 3), device=device)
    s_tgt = torch.randint(0, VOCAB_SIZE, (B, 3), device=device)
    
    tau = torch.tensor(0.07, device=device)
    
    # 3. Test Text Autoencoder Pass
    print("Testing DecoupledTextAutoencoder Forward + Backward Pass...")
    z_edsl_text = pipeline.edsl_emb(t_sub) + pipeline.edsl_emb(t_pred)
    logits, commit_loss = autoencoder(z_edsl_text, s_in)
    
    loss_ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), s_tgt.reshape(-1)) + commit_loss
    loss_ce.backward()
    print(f"  -> Autoencoder Success! Loss: {loss_ce.item():.4f}")
    
    # 4. Test AetherGammaPipeline Pass
    print("Testing AetherGammaPipeline Forward + Backward Pass...")
    z_brain, kl_loss, flow_net, exec_out = pipeline(
        lat, p_mask, v_sens,
        subj_emb=pipeline.edsl_emb(t_sub),
        pred_emb=pipeline.edsl_emb(t_pred),
        obj_emb=pipeline.edsl_emb(t_sub) # Mock obj
    )
    
    z_edsl_target = pipeline.edsl_emb(t_sub) + pipeline.edsl_emb(t_pred)
    loss_flow, loss_dict = neuroflow_total_loss(z_brain, z_edsl_target.detach(), kl_loss, flow_net, tau)
    
    # 5. Test PPO RL Loop Pass
    print("Testing PPO RL Loop Logic...")
    if exec_out is not None:
        assertion_bool = exec_out['assertion_bool'].detach()
        gumbel_logits = exec_out['gumbel_logits']
        log_probs = F.log_softmax(gumbel_logits, dim=-1)
        
        action_idx = (1 - assertion_bool).long()
        old_log_probs = log_probs.gather(1, action_idx.unsqueeze(-1)).squeeze(-1).detach()
        
        correct_bool = (t_sub % 2 == 0).float()
        reward = torch.where(assertion_bool == correct_bool, 1.0, -1.0).to(device)
        
        advantage = (reward - reward.mean()) / (reward.std() + 1e-8)
        
        curr_log_probs = log_probs.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(curr_log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantage
        
        loss_ppo = -torch.min(surr1, surr2).mean()
        loss_flow = loss_flow + loss_ppo
        print(f"  -> PPO Loss appended: {loss_ppo.item():.4f}")
    else:
        print("  -> ERROR: exec_out is None!")
        
    loss_flow.backward()
    print(f"  -> Pipeline Success! Total Flow+RL Loss: {loss_flow.item():.4f}")
    
    print("--- Dry-Run Completed Successfully ---")

if __name__ == "__main__":
    test_pipeline_dry_run()
