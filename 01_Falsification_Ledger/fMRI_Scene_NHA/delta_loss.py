import torch
import torch.nn.functional as F

def bipartite_pig_loss(z_present: torch.Tensor,    # (B, D) NeuroVAE mu, present
                       z_future:  torch.Tensor,    # (B, D) NeuroVAE mu, future
                       slots_present: torch.Tensor, # (B, K, d) Perceiver latents, present
                       slots_future:  torch.Tensor, # (B, K, d) Perceiver latents, future
                       tau_global: torch.Tensor,    # learnable scalar temperature
                       alpha: float = 0.5,
                       beta:  float = 0.01
                       ) -> tuple:
    B, K, d = slots_present.shape

    # - Global InfoNCE on NeuroVAE latent -
    gp = F.normalize(z_present, dim=-1)    # (B, D)
    gf = F.normalize(z_future,  dim=-1)
    tau_safe = tau_global.clamp(0.01, 5.0)
    logits_g = gp @ gf.T / tau_safe      # (B, B)
    labels_g = torch.arange(B, device=gp.device)
    L_global = F.cross_entropy(logits_g, labels_g)

    # - Bipartite Unit-to-Unit InfoNCE -
    # Reshape to (B*K, d) - process all K slots simultaneously
    # Positive: slot k of item b at time t matches slot k of item b at t+tau
    # Negatives: slot k of item b matches slot k of ALL OTHER items j!=b
    sp = F.normalize(slots_present.reshape(B * K, d), dim=-1)  # (B*K, d)
    sf = F.normalize(slots_future.reshape(B * K, d),  dim=-1)  # (B*K, d)

    # Block-diagonal positive structure: (B*K, B*K) similarity matrix
    # Positive pair for row (b*K + k) is column (b*K + k) - diagonal
    logits_l = sp @ sf.T / 0.07            # fixed eta=0.07 for local slots
    labels_l = torch.arange(B * K, device=sp.device)
    L_local  = F.cross_entropy(logits_l, labels_l)

    # - Spatial Decorrelation (SRC enforcement) -
    # Compute Gram matrix of normalized slot means across batch
    slot_n  = F.normalize(slots_present, dim=-1)        # (B, K, d)
    gram    = torch.bmm(slot_n, slot_n.transpose(1, 2)) # (B, K, K)
    diag    = torch.eye(K, device=gram.device).unsqueeze(0)
    L_SRC   = (gram * (1 - diag)).pow(2).sum(dim=(1,2)).mean() / (K * (K - 1))

    L_PIG = L_global + alpha * L_local + beta * L_SRC

    return L_PIG, {
        'pig_global':   L_global.item(),
        'pig_local':    L_local.item(),
        'pig_src':      L_SRC.item(),
    }

def global_pig_only(z_present, z_future, tau):
    zp = F.normalize(z_present, dim=-1)
    zf = F.normalize(z_future,  dim=-1)
    logits = zp @ zf.T / tau
    labels = torch.arange(len(zp), device=zp.device)
    return F.cross_entropy(logits, labels)
