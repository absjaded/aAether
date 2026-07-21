"""
Aether causal transformer for aether_lab.

Input:  raw torch.Tensor shape (batch, 17, 16) - Yeo-17 x 16-frame window
Output: dict(surface=str, confidence=float)

Zero imports from nisp/. Uses Grammar v1 vocabulary from grammar.py.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn

from models.grammar import VOCAB, VOCAB_SIZE, UNIT_TO_IDX, UNIT_OPEN, TOK_EOS
import torch.nn.functional as F

class ResidualIsolation(nn.Module):
    """
    Simpler alternative: projects out only low-tier audio-visual sensory variance using TRIBE features.
    Does NOT include physiological regressors - use as PoC fallback only.
    """
    def __init__(self, sensory_dim: int, num_parcels: int, n_parcels_early: int = 256):
        super().__init__()
        self.sensory_proj = nn.Linear(sensory_dim, n_parcels_early, bias=True)
        # Learnable cross-parcel leakage: early sensory -> target parcels
        self.leakage = nn.Linear(n_parcels_early, num_parcels, bias=False)
        nn.init.uniform_(self.leakage.weight, -0.01, 0.01)
        self.scale = nn.Parameter(torch.ones(num_parcels, 1))

    def forward(self, z_tpj: torch.Tensor, v_sensory: torch.Tensor) -> torch.Tensor:
        # v_sensory: (B, sensory_dim)
        yhat_early = self.sensory_proj(v_sensory)
        contamination = self.leakage(yhat_early).unsqueeze(-1) # (B, P, 1)
        return (z_tpj - contamination) * self.scale

class TacitExecutor(nn.Module):
    """
    Tacit Executor Neuro-Symbolic Differentiable Executor (Autonomous Query Version).
    Discovers topological clusters autonomously via learnable queries.
    """
    def __init__(self, latent_dim: int, num_parcels: int):
        super().__init__()
        # Autonomous queries driven by InfoNCE
        self.q_sub_prior = nn.Parameter(torch.randn(1, latent_dim))
        self.q_pred_prior = nn.Parameter(torch.randn(1, latent_dim))
        
        self.q_sub_proj = nn.Linear(latent_dim, latent_dim)
        self.q_pred_proj = nn.Linear(latent_dim, latent_dim)
        
        # Topological priors
        self.vis_mask_logits = nn.Parameter(torch.randn(num_parcels))
        self.tpj_mask_logits = nn.Parameter(torch.randn(num_parcels))
        
        self.composer = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, parcel_embeddings: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = parcel_embeddings.shape[0]
        
        # Expand autonomous queries for the batch
        q_sub = self.q_sub_prior.expand(B, -1)
        q_pred = self.q_pred_prior.expand(B, -1)
        
        # Calculate role-conditioned attention scores
        score_sub = torch.bmm(self.q_sub_proj(q_sub).unsqueeze(1), parcel_embeddings.transpose(-1, -2)).squeeze(1)
        score_pred = torch.bmm(self.q_pred_proj(q_pred).unsqueeze(1), parcel_embeddings.transpose(-1, -2)).squeeze(1)
        
        # Add spatial mask priors
        score_sub = score_sub + self.vis_mask_logits
        score_pred = score_pred + self.tpj_mask_logits
        
        # Apply padding mask penalty (-1e9) to padded space
        # Observation: padding_mask == False means padded (zeroes). True means valid brain data.
        if padding_mask is not None:
            penalty = -1e9
            score_sub = score_sub.masked_fill(~padding_mask, penalty)
            score_pred = score_pred.masked_fill(~padding_mask, penalty)
        
        # Inject structural priors (Zero Label Mandate compliant)
        alpha_sub = F.softmax(score_sub, dim=-1)
        alpha_pred = F.softmax(score_pred, dim=-1)
        
        # Soft Aggregation
        grounded_sub = torch.bmm(alpha_sub.unsqueeze(1), parcel_embeddings).squeeze(1)
        grounded_pred = torch.bmm(alpha_pred.unsqueeze(1), parcel_embeddings).squeeze(1)
        
        # Composition for downstream flow matching
        execution_state = self.composer(torch.cat([grounded_sub, grounded_pred], dim=-1))
        return execution_state

class AetherGammaFlowModel(nn.Module):
    """
    The Aether-Gamma flow network v_theta(x_t, t) for NeuroFlow matching.
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: (batch, latent_dim), t: (batch, 1)
        inp = torch.cat([x_t, t], dim=-1)
        return self.net(inp)




class FiniteScalarQuantization(nn.Module):
    """Block 1: Discretization (The FSQ Bottleneck). structurally immune to collapse."""
    def __init__(self, tiers: list[int], dim: int):
        super().__init__()
        self.tiers = tiers
        self.dim = dim
        self.n_codes = int(torch.tensor(tiers).prod().item())
        self.project_in = nn.Linear(dim, len(tiers))
        self.project_out = nn.Linear(len(tiers), dim)
        self.register_buffer("basis", torch.tensor(tiers, dtype=torch.float32))
        self.is_warmup = False

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(z)
        half_width = (self.basis - 1) / 2
        z_scaled = z * half_width
        if self.is_warmup:
            z_quantized = z_scaled
        else:
            z_quantized = torch.round(z_scaled)
            z_quantized = z_scaled + (z_quantized - z_scaled).detach()
        return z_quantized / half_width

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        z = self.project_in(x)
        z_q = self.quantize(z)
        x_q = self.project_out(z_q)
        
        # L_commit: Distance between continuous projection and quantized target
        # For ProVQ, this should be annealing-aware in the training loop
        commit_loss = torch.mean((z_q.detach() - z) ** 2)
        
        return x_q, commit_loss, None


# =====================================================================
# PHASE 3: NEURONA Executor
# =====================================================================

class SpatiallyMaskedCrossAttention(nn.Module):
    """
    Cross-attention where eDSL concept units attend to brain parcels,
    guided by soft anatomical prior masks (learnable, initialized from atlas).
    """
    def __init__(self, d_concept: int, d_parcel: int, n_heads: int = 8,
                 n_parcels: int = 17, prior_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_concept, n_heads, batch_first=True)
        self.kv_proj = nn.Linear(d_parcel, d_concept)
        # Learnable soft spatial mask - initialized from atlas priors
        mask_init = prior_mask if prior_mask is not None else torch.zeros(n_parcels)
        self.spatial_log_mask = nn.Parameter(mask_init)  # log-space for stability

    def forward(self, concept_emb: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        KV = self.kv_proj(E)                          # (B, P, d_concept)
        spatial_mask = self.spatial_log_mask.unsqueeze(0)  # (1, P) additive bias
        out, _ = self.attn(concept_emb, KV + spatial_mask.unsqueeze(-1), KV)
        return out                                    # (B, n_concepts, d_concept)

class H5RelationalGrounding(nn.Module):
    """H5: Full argument-guided parcel-pair relational grounding for Predicate."""
    def __init__(self, d_parcel: int, d_concept: int, n_parcels: int, hidden: int = 256):
        super().__init__()
        self.d_parcel = d_parcel
        self.d_concept = d_concept
        
        self.W_q = nn.Linear(d_parcel, hidden)
        self.W_k = nn.Linear(d_parcel, hidden)
        self.W_ctx = nn.Linear(2 * d_concept, hidden)
        self.hidden = hidden

    def forward(self, E: torch.Tensor, z_subj: torch.Tensor,
                z_obj: torch.Tensor) -> torch.Tensor:
        B, P, d = E.shape
        
        Q = self.W_q(E)
        K = self.W_k(E)
        
        ctx = self.W_ctx(torch.cat([z_subj, z_obj], dim=-1)).unsqueeze(1)
        
        Q = Q + ctx
        K = K + ctx
        
        score = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden ** 0.5)
        
        attn = F.softmax(score.reshape(B, -1), dim=-1).reshape(B, P, P)
        
        attn_i = attn.sum(dim=2)  # (B, P)
        attn_j = attn.sum(dim=1)  # (B, P)
        
        z_pred = (attn_i.unsqueeze(-1) * E).sum(dim=1) + (attn_j.unsqueeze(-1) * E).sum(dim=1)
        return z_pred

class NeuRONAExecutor(nn.Module):
    """
    Full NEURONA Differentiable Executor for Aether-Gamma.
    Implements:
      - Spatially masked cross-attention (Q=eDSL, K=V=brain parcels)
      - H5 full argument-guided relational grounding for predicates
      - Gumbel-Softmax -> discrete Boolean -> Lean 4 terminal reward
    """
    def __init__(self, d_parcel: int, d_concept: int, n_parcels: int,
                 predicate_prior: torch.Tensor = None,
                 subject_prior: torch.Tensor = None):
        super().__init__()
        self.concept_proj = nn.Linear(d_concept, d_concept)

        self.subj_attn = SpatiallyMaskedCrossAttention(
            d_concept, d_parcel, n_parcels=n_parcels, prior_mask=subject_prior)
        self.obj_attn  = SpatiallyMaskedCrossAttention(
            d_concept, d_parcel, n_parcels=n_parcels, prior_mask=predicate_prior)

        self.pred_grounder = H5RelationalGrounding(d_parcel, d_concept, n_parcels)
        self.pred_proj = nn.Linear(d_parcel, d_concept)

        self.log_tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, E: torch.Tensor,
                subj_emb: torch.Tensor,
                pred_emb: torch.Tensor,
                obj_emb:  torch.Tensor,
                z_brain_global: torch.Tensor = None) -> dict:
        z_subj = self.subj_attn(subj_emb.unsqueeze(1), E).squeeze(1)
        z_obj  = self.obj_attn(obj_emb.unsqueeze(1),   E).squeeze(1)

        z_pred_parcel = self.pred_grounder(E, z_subj, z_obj)
        z_pred = self.pred_proj(z_pred_parcel)

        S_unary = F.cosine_similarity(z_subj, z_obj,  dim=-1)
        S_binary = F.cosine_similarity(z_pred, obj_emb, dim=-1)
        
        if z_brain_global is not None:
            z_edsl_target = subj_emb + pred_emb
            S_global = F.cosine_similarity(z_brain_global, z_edsl_target, dim=-1)
            S_final = torch.sigmoid((S_unary + S_binary + S_global) / 3)
        else:
            S_final = torch.sigmoid((S_unary + S_binary) / 2)

        tau = torch.exp(self.log_tau).clamp(0.1, 5.0)
        logits = torch.stack([1 - S_final, S_final], dim=-1)
        # Gumbel-Softmax straight-through gradient
        assertion_bool = F.gumbel_softmax(logits, tau=tau, hard=True)[:, 1]

        return {
            'S_final':        S_final,
            'assertion_bool': assertion_bool,
            'gumbel_logits':  logits
        }

# =====================================================================
# PHASE 4: NeuroFlow Latent Space
# =====================================================================

D_LATENT = 1664

class NeuroVAE(nn.Module):
    """Variational backbone: fMRI residual -> compact CLIP-aligned latent z_c."""
    def __init__(self, d_fmri: int, d_latent: int = D_LATENT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_fmri, 2048), nn.GELU(),
            nn.Linear(2048, 1024),   nn.GELU()
        )
        self.mu_head      = nn.Linear(1024, d_latent)
        self.log_var_head = nn.Linear(1024, d_latent)
        self.decoder      = nn.Sequential(
            nn.Linear(d_latent, 1024), nn.GELU(),
            nn.Linear(1024, d_fmri)
        )

    def encode(self, y):
        h = self.encoder(y)
        return self.mu_head(h), self.log_var_head(h)

    def reparameterize(self, mu, log_var):
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    def forward(self, y):
        mu, log_var = self.encode(y)
        z = self.reparameterize(mu, log_var)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        return z, self.decoder(z), kl

class SiTVelocityNet(nn.Module):
    """Scalable Interpolation Transformer velocity network v_theta(z_t, t)."""
    def __init__(self, d_latent: int = D_LATENT, n_heads: int = 8,
                 n_layers: int = 4, d_hidden: int = 1024):
        super().__init__()
        self.t_embed  = nn.Linear(1, d_hidden)
        self.in_proj  = nn.Linear(d_latent, d_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden, nhead=n_heads, dim_feedforward=d_hidden * 4,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_hidden, d_latent)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embed(t.unsqueeze(-1))
        z_emb = self.in_proj(z_t)
        seq = torch.stack([z_emb, t_emb], dim=1)
        out = self.transformer(seq)[:, 0]
        return self.out_proj(out)

    def flow_forward(self, z0, n_steps=20):
        z = z0.clone()
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((z.shape[0],), i * dt, device=z.device)
            z = z + self.forward(z, t) * dt
        return z

    def flow_reverse(self, z1, n_steps=20):
        z = z1.clone()
        dt = 1.0 / n_steps
        for i in range(n_steps, 0, -1):
            t = torch.full((z.shape[0],), i * dt, device=z.device)
            z = z - self.forward(z, t) * dt
        return z

def softclip_loss(z_brain: torch.Tensor, z_edsl: torch.Tensor,
                  tau: torch.Tensor, soft_labels: torch.Tensor = None) -> torch.Tensor:
    """Bidirectional SoftCLIP with semantic soft labels."""
    N = z_brain.shape[0]
    zb = F.normalize(z_brain, dim=-1)
    ze = F.normalize(z_edsl,  dim=-1)
    logits = zb @ ze.T / tau
    if soft_labels is None:
        soft_labels = (ze @ ze.T).detach()
        soft_labels = F.softmax(soft_labels, dim=-1)
    loss_b2e = -(soft_labels * F.log_softmax(logits,   dim=-1)).sum(-1).mean()
    loss_e2b = -(soft_labels * F.log_softmax(logits.T, dim=-1)).sum(-1).mean()
    return (loss_b2e + loss_e2b) / 2

def neuroflow_total_loss(z_brain, z_edsl, kl_loss, flow_net, tau,
                         lam1=1.0, lam2=0.5, lam3=0.01):
    """Full combined loss for NeuroFlow."""
    l_clip = softclip_loss(z_brain, z_edsl, tau)

    N = z_brain.shape[0]
    t  = torch.rand(N, device=z_brain.device)
    zt = (1 - t).unsqueeze(-1) * z_brain + t.unsqueeze(-1) * z_edsl
    v_true = z_edsl - z_brain
    v_pred = flow_net(zt, t)
    l_xfm  = F.mse_loss(v_pred, v_true)

    z_edsl_hat  = flow_net.flow_forward(z_brain)
    z_brain_hat = flow_net.flow_reverse(z_edsl_hat)
    l_cycle = softclip_loss(z_brain, z_brain_hat, tau)

    L = l_clip + lam1 * l_xfm + lam2 * l_cycle + lam3 * kl_loss
    return L, {"clip": l_clip, "xfm": l_xfm, "cycle": l_cycle, "kl": kl_loss}



