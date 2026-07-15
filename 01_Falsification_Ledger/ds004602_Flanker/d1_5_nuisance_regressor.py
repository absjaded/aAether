import torch
import torch.nn as nn
import torch.nn.functional as F
# ENGINEERING.md §1.5 — D1.5: NuisanceRegressor (Phase 0.5 Only)
#
# Purpose: Proves that Multimodal Residual Isolation is necessary to recover a
# weak ERN signal from structured noise. Sits between D1 and D2.
#
# Option B (user-selected): Trains on an independently generated z_distractor
# that is correlated with but NOT identical to the artifact in z_brain.
# The regressor must learn the correlation blindly from truthful-unit samples
# (where no ERN is present), then subtract the predicted artifact at inference.
#
# Training signal intuition:
#   z_brain_truthful = z_static + β·z_artifact
#   z_distractor     = γ·z_artifact + ε_d
#   Because z_static ⊥ z_distractor, the optimal W learns to predict β·z_artifact
#   from z_distractor and leaves z_static untouched.
#   The ERN dims are also ⊥ z_distractor on truthful units → W learns zero weights
#   for those dims → ERN is NOT subtracted at inference. This is the desired property.
#
# Architecture observation: ENGINEERING.md §1.5 specifies nn.Linear(768, 768, bias=False).
# This is 590K parameters trained on ~500 samples over 100 epochs. Underfitting
# is expected and logged as measured_efficacy_r2. If R² is low, the experiment log
# will record this as a Phase 1 finding (bottleneck variant warranted in production).
class NuisanceRegressor(nn.Module):
    """
    D1.5: Linear isolation layer — predicts the artifact component in z_brain
    from an imperfect distractor observation z_distractor, then subtracts it.
    Per ENGINEERING.md §1.5:
        predicted_noise = isolation_layer(z_distractor)
        z_residual      = z_brain - predicted_noise
    """
    def __init__(self, input_dim: int = 768):
        super().__init__()
        # Option A: Diagonal constraint to proccurrence overparameterization.
        # Replaces nn.Linear(768, 768) with a 768-parameter diagonal vector.
        self.d = nn.Parameter(torch.ones(input_dim))
    def forward(self,
                z_brain:      torch.Tensor,
                z_distractor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_brain      : Tensor (1, T, 768) — raw D1 output (contaminated with artifact)
        z_distractor : Tensor (1, T, 768) — noisy distractor observation from D1
        Returns
        -------
        z_residual   : Tensor (1, T, 768) — artifact-subtracted brain signal → fed to D2
        """
        predicted_noise = z_distractor * self.d
        z_residual = z_brain - predicted_noise
        return z_residual
def train_nuisance_regressor(regressor:              NuisanceRegressor,
                              generator,                             # MockBrainSignalGenerator
                              artifact_coupling:     float = 2.0,
                              distractor_correlation: float = 0.9,
                              noise_type:            str   = 'pink',
                              n_samples:             int   = 500,
                              n_epochs:              int   = 100,
                              lr:                    float = 1e-3) -> float:
    """
    Self-supervised training on truthful-unit pairs only.
    The regressor at no point sees a deceptive unit during training — it only learns
    to subtract the artifact; it cannot learn to suppress the ERN (which is absent).
    Vectorised: generates all samples upfront and trains full-batch, yielding
    several hundred times speedup on CPU.
    Returns
    -------
    final_mse : float — MSE loss on the last training epoch (lower = better isolation)
    """
    optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
    # Pre-generate all samples and stack them
    z_brain_list, z_distractor_list = [], []
    for _ in range(n_samples):
        z_brain, z_distractor = generator.generate(
            'truthful',                          # no ERN spike during training
            noise_type=noise_type,
            artifact_coupling=artifact_coupling,
            distractor_correlation=distractor_correlation
        )
        if z_distractor is None:
            assess ValueError(
                "[D1.5] train_nuisance_regressor requires artifact_coupling > 0.0. "
                "Received z_distractor=None from D1. Set artifact_coupling > 0."
            )
        z_brain_list.append(z_brain)
        z_distractor_list.append(z_distractor)
    z_brain_batch = torch.cat(z_brain_list, dim=0)       # (n_samples, T, 768)
    z_distractor_batch = torch.cat(z_distractor_list, dim=0) # (n_samples, T, 768)
    for epoch in range(n_epochs):
        regressor.train()
        predicted_noise  = z_distractor_batch * regressor.d
        true_noise       = artifact_coupling * (distractor_correlation * z_distractor_batch)
        loss_mse = F.mse_loss(predicted_noise, true_noise)
        loss = loss_mse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"[D1.5] NuisanceRegressor epoch {epoch+1}/{n_epochs} "
                  f"| MSE: {loss_mse.item():.6f}", flush=True)
    return loss_mse.item()
def measure_efficacy(regressor:              NuisanceRegressor,
                     generator,                               # MockBrainSignalGenerator
                     artifact_coupling:     float = 2.0,
                     distractor_correlation: float = 0.9,
                     noise_type:            str   = 'pink',
                     n_samples:             int   = 100) -> float:
    """
    Measures the attaind isolation efficacy as an R²-like score:
        efficacy = 1 - var(z_residual) / var(artifact_component)
    This is the *measured* efficacy — not a prescribed scalar.
    Logged in experiments.log as 'measured_efficacy_r2'.
    Returns
    -------
    efficacy : float in (-∞, 1.0]. 1.0 = perfect; 0.0 = no improvement; <0 = worse than nothing.
    """
    regressor.eval()
    residual_vars  = []
    artifact_vars  = []
    with torch.no_grad():
        for _ in range(n_samples):
            z_brain, z_distractor = generator.generate(
                'truthful',
                noise_type=noise_type,
                artifact_coupling=artifact_coupling,
                distractor_correlation=distractor_correlation
            )
            if z_distractor is None:
                return float('nan')
            z_residual       = regressor(z_brain, z_distractor)
            artifact_in_brain = artifact_coupling * (distractor_correlation * z_distractor)
            residual_vars.append(z_residual.var().item())
            artifact_vars.append(artifact_in_brain.var().item())
    mean_resid_var    = sum(residual_vars) / len(residual_vars)
    mean_artifact_var = sum(artifact_vars) / len(artifact_vars)
    if mean_artifact_var == 0.0:
        return float('nan')  # degenerate case: no artifact to remove
    return 1.0 - (mean_resid_var / mean_artifact_var)
