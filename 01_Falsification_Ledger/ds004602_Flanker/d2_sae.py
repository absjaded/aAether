import torch
import torch.nn as nn
import torch.nn.functional as F
class SAE(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=4096):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()
    def forward(self, z_brain):
        features = self.activation(self.encoder(z_brain))
        reconstruction = self.decoder(features)
        return features, reconstruction
def sae_loss(z_brain, features, reconstruction, lambda_l1: float):
    loss_reconstruction = F.mse_loss(reconstruction, z_brain)
    loss_sparsity = lambda_l1 * features.abs().mean()
    return loss_reconstruction + loss_sparsity
DECEPTION_FEATURE_IDX = 42
ACTIVATION_THRESHOLD = 0.5
def audit_feature(features: torch.Tensor) -> dict:
    activation = features[0, :, DECEPTION_FEATURE_IDX].mean().item()
    return {
        'activation': activation,
        'fired': activation > ACTIVATION_THRESHOLD,
        'label': 'Deception Detected' if activation > ACTIVATION_THRESHOLD else 'Benign'
    }
