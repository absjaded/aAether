"""
LiveL2Decoder - Aether-beta v2.0 (FSQ + Slot Attention + Grounded-Symbolic).

Architecture (v2.0 spec, NISP_v2_Delta_Report.md):
    Input:  CognitiveLatent.data of shape (17, 16) - Yeo-17 networks x 16-frame window
    Block 0: Slot Attention (K=17) binds Yeo networks to distinct latent slots.
    Block 1: Finite Scalar Quantization (FSQ) eliminates codebook collapse.
    Block 2: Unified Causal Transformer over [Domain Tag] + [17 Neural Slots] + [Units].
    Block 3: Dual Heads for EDSLExpression surface and confidence.
"""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nisp.contracts import GRAMMAR_VERSION, CognitiveLatent, EDSLExpression
from nisp.l2_decoder.constrained import (
    VOCAB,
    UNIT_OPEN,
    TOK_EOS,
    units_to_surface,
)

log = logging.getLogger(__name__)


class SlotAttention(nn.Module):
    """
    Block 0: Object-centric binding for Yeo-17 networks.
    Each slot query competes for fMRI feature map activations.
    """
    def __init__(self, num_slots: int, input_dim: int, slot_dim: int, iters: int = 3):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 2, slot_dim)
        )
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, N, D_in) where N=16 frames, D_in=17 networks
        b, n, d = inputs.shape
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        # Initialize slots from shared distribution
        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + 1e-8
            attn = attn / attn.sum(dim=2, keepdim=True)

            updates = torch.einsum('bij,bjd->bid', attn, v)
            
            slots = self.gru(updates.reshape(-1, q.shape[-1]), slots_prev.reshape(-1, q.shape[-1]))
            slots = slots.reshape(b, self.num_slots, q.shape[-1])
            slots = slots + self.mlp(self.norm_pre_mlp(slots))

        return slots


class FiniteScalarQuantization(nn.Module):
    """
    Block 1: Discretization (The FSQ Bottleneck).
    Implicit codebook via scalar rounding. structurally immune to collapse.
    """
    def __init__(self, tiers: list[int], dim: int):
        super().__init__()
        self.tiers = tiers
        self.dim = dim
        self.n_codes = int(torch.tensor(tiers).prod().item())
        
        # Projection to the low-dimensional scalar space (e.g. 4D or 8D)
        self.project_in = nn.Linear(dim, len(tiers))
        self.project_out = nn.Linear(len(tiers), dim)
        
        # Scalar tiers for rounding (centered around 0)
        self.register_buffer("basis", torch.tensor(tiers))
        self.is_warmup = False # If True, skip rounding
        
    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        # Bounded projection to [-1, 1]
        z = torch.tanh(z)
        
        # Scale to tiers
        half_width = (self.basis - 1) / 2
        z_scaled = z * half_width
        
        if self.is_warmup:
            # Continuous warm-up (no rounding)
            z_quantized = z_scaled
        else:
            # Discrete rounding (straight-through)
            z_quantized = torch.round(z_scaled)
            z_quantized = z_scaled + (z_quantized - z_scaled).detach()
        
        # Normalize back to [-1, 1] relative to the grid
        return z_quantized / half_width

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # x: (B, L, D)
        z = self.project_in(x)
        z_q = self.quantize(z)
        x_q = self.project_out(z_q)
        
        # FSQ has no codebook loss, but we provide a zero loss for API compatibility
        loss = torch.tensor(0.0, device=x.device)
        
        # Compute implicit indices for debugging/logging if needed
        indices = None # FSQ indices are typically not needed for the forward pass
        
        return x_q, loss, indices


class Aether(nn.Module):
    """
    Aether-beta (v2.0): Slot Attention -> FSQ -> Causal Transformer.
    Unified Sequence: [Domain Tag] + [17 Grounded Slots] + [eDSL Units]
    """
    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        fsq_tiers: list[int] | None = None,
        orthogonal_reg_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.orthogonal_reg_weight = orthogonal_reg_weight
        
        # Block 0: Object-centric binding (17 slots for 17 Yeo networks)
        self.slots = SlotAttention(num_slots=17, input_dim=17, slot_dim=d_model)
        
        # Block 1: Discretization (FSQ replaces VQ)
        # tiers=[8, 8, 5, 5] gives 1600 codes (implicit)
        tiers = fsq_tiers or [8, 8, 5, 5]
        self.fsq = FiniteScalarQuantization(tiers, d_model)
        
        # Block 2: Sequence Engine (Unified Causal Transformer)
        self.domain_emb = nn.Embedding(2, d_model)
        self.unit_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + 17 + 32, d_model) * 0.02)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=n_layers)

        # Block 3: Dual Heads
        self.vocab_head = nn.Linear(d_model, vocab_size)
        self.confidence_head = nn.Linear(d_model, 1)

    def compute_orthogonal_loss(self, slots: torch.Tensor) -> torch.Tensor:
        """Forces slots to be distinct directions in latent space."""
        # slots: (B, K, D)
        b, k, d = slots.shape
        # Normalize slots to unit sphere
        slots_norm = F.normalize(slots, p=2, dim=-1)
        # Compute pairwise cosine similarities: (B, K, K)
        sim = torch.einsum('bid,bjd->bij', slots_norm, slots_norm)
        # Identity matrix: (K, K)
        eye = torch.eye(k, device=slots.device).unsqueeze(0).expand(b, k, k)
        # Loss: Frobenius norm of (sim - I)
        loss = torch.mean((sim - eye)**2)
        return loss

    def prepare_sequence(
        self, latent: torch.Tensor, units: torch.Tensor, domain_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Constructs the unified sequence: [Domain] + [FSQ Slots] + [Units]"""
        # latent: (B, 17, 16) -> (B, 16, 17) for slot attention (frames as temporal seq)
        x = latent.transpose(1, 2) 
        
        # 1. Bind to 17 slots
        slots = self.slots(x) # (B, 17, d_model)
        
        # Orthogonal Regularization
        ortho_loss = self.compute_orthogonal_loss(slots)
        
        # 2. Quantize slots
        quantized, fsq_loss, _ = self.fsq(slots)
        
        # 3. Embedding fusion
        dom = self.domain_emb(domain_id).unsqueeze(1)
        tok = self.unit_emb(units)
        
        seq = torch.cat([dom, quantized, tok], dim=1)
        seq = seq + self.pos_emb[:, :seq.shape[1], :]
        
        mask = nn.Transformer.generate_square_subsequent_mask(seq.shape[1], device=latent.device)
        return seq, ortho_loss, mask

    def forward(self, latent: torch.Tensor, units: torch.Tensor, domain_id: torch.Tensor | None = None):
        """
        domain_id: defaults to 1 (Cobot) if not provided.
        Returns: logits (B, T, vocab_size), confidence (B,)
        """
        if domain_id is None:
            domain_id = torch.ones(latent.shape[0], dtype=torch.long, device=latent.device)
            
        seq, ortho_loss, mask = self.prepare_sequence(latent, units, domain_id)
        out = self.transformer(seq, mask=mask)
        
        # Logits from the unit portion [18:] (1 domain + 17 slots)
        unit_out = out[:, 18:, :]
        logits = self.vocab_head(unit_out)
        
        # Confidence from the neural slots [1:18]
        conf_feat = out[:, 1:18, :].mean(dim=1)
        confidence = torch.sigmoid(self.confidence_head(conf_feat)).squeeze(-1)
        
        return logits, confidence, ortho_loss


class LiveL2Decoder:
    """Trained constrained-generation decoder. Loads Aether-beta v2.0 checkpoint."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model: Aether | None = None
        self._weight_hash: str | None = None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._checkpoint_path is None or not self._checkpoint_path.exists():
            raise NotImplementedError(f"Aether-beta v2.0 checkpoint missing at {self._checkpoint_path}")

        self._model = Aether(vocab_size=len(VOCAB))
        state = torch.load(self._checkpoint_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
        
        buf = io.BytesIO()
        torch.save(state, buf)
        self._weight_hash = hashlib.sha256(buf.getvalue()).hexdigest()

    async def decode(self, latent: CognitiveLatent) -> EDSLExpression:
        if latent.degraded:
            return EDSLExpression(
                surface="(hold)",
                grammar_version=GRAMMAR_VERSION,
                confidence=0.0,
                degraded=True,
                reason=latent.reason,
            )
        try:
            return await self._live(latent)
        except Exception as exc:
            log.warning("L2 fallback to degraded", extra={"reason": str(exc)})
            return EDSLExpression(
                surface="(hold)",
                grammar_version=GRAMMAR_VERSION,
                confidence=0.0,
                degraded=True,
                reason=str(exc),
            )

    async def _live(self, latent: CognitiveLatent) -> EDSLExpression:
        self.ensure_loaded()
        assert self._model is not None
        device = next(self._model.parameters()).device
        
        lat_tensor = torch.tensor(latent.data, dtype=torch.float32, device=device).unsqueeze(0)
        dom_id = torch.tensor([1], device=device) # Assume Cobot for live robot runs

        # 1. Prefix processing
        with torch.no_grad():
            x = lat_tensor.transpose(1, 2)
            slots = self._model.slots(x)
            quantized, _, _ = self._model.fsq(slots)
            dom_emb = self._model.domain_emb(dom_id).unsqueeze(1)
            prefix = torch.cat([dom_emb, quantized], dim=1) # (1, 18, d_model)

        # 2. Native unconstrained decoding
        units = [VOCAB.index(UNIT_OPEN)]
        
        last_out = prefix
        for _ in range(30):
            with torch.no_grad():
                unit_emb = self._model.unit_emb(torch.tensor([units], device=device))
                seq = torch.cat([prefix, unit_emb], dim=1)
                seq = seq + self._model.pos_emb[:, :seq.shape[1], :]
                
                attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq.shape[1]).to(device)
                last_out = self._model.transformer(seq, mask=attn_mask)
                logits = self._model.vocab_head(last_out[:, -1, :])
            
            next_idx = int(logits.argmax(dim=-1).item())
            units.append(next_idx)
            if VOCAB[next_idx] == TOK_EOS:
                break
        
        # 3. Confidence from final state
        conf_feat = last_out[:, 1:18, :].mean(dim=1)
        confidence = torch.sigmoid(self._model.confidence_head(conf_feat)).item()

        return EDSLExpression(
            surface=units_to_surface([VOCAB[i] for i in units]),
            grammar_version=GRAMMAR_VERSION,
            confidence=confidence,
        )
