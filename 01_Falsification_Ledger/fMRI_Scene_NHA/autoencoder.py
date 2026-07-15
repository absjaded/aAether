import torch
import torch.nn as nn
from models.network import FiniteScalarQuantization
from models.grammar import VOCAB_SIZE

class DecoupledTextAutoencoder(nn.Module):
    """
    Self-Supervised Generative Decoder.
    Takes parsed eDSL embeddings -> FSQ -> Autoregressive Sequence Reconstruction.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_heads=8, n_layers=4, fsq_tiers=[8, 8, 8, 8]):
        super().__init__()
        self.d_model = d_model
        
        # Continuous projection to FSQ bottleneck
        self.fsq = FiniteScalarQuantization(fsq_tiers, d_model)
        
        # Unit Embeddings for autoregressive inputs
        self.unit_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        
        # Autoregressive Causal Transformer
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=n_layers)
        
        # Prediction Head
        self.vocab_head = nn.Linear(d_model, vocab_size)

    def forward(self, z_edsl: torch.Tensor, input_units: torch.Tensor):
        """
        z_edsl: (B, d_model) - the continuous embedding from the parser
        input_units: (B, seq_len) - the target text sequence to reconstruct
        """
        B, seq_len = input_units.shape
        
        # Discretize continuous semantic space into FSQ slots
        # We unsqueeze because FSQ might expect sequences, or FSQ operates on last dim.
        # FSQ quantize expects (B, d_model) or (B, N, d_model)
        # z_edsl is (B, d_model), so let's keep it (B, 1, d_model) for sequence prep
        z_edsl_seq = z_edsl.unsqueeze(1)
        z_q, commit_loss, _ = self.fsq(z_edsl_seq)
        
        # Prepare units
        tok = self.unit_emb(input_units)
        
        # Sequence: [FSQ_Slot] + [Units]
        seq = torch.cat([z_q, tok], dim=1) # (B, 1 + seq_len, d_model)
        seq = seq + self.pos_emb[:, :seq.shape[1], :]
        
        # Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq.shape[1], device=seq.device)
        
        # Autoregressive Decoding
        out = self.transformer(seq, mask=mask)
        
        # Predict logits for the unit sequence (shift by 1 because FSQ slot is at index 0)
        # unit_out is the output at positions 0 to seq_len-1, which predicts input_units 1 to seq_len
        unit_out = out[:, :-1, :]
        logits = self.vocab_head(unit_out)
        
        return logits, commit_loss
