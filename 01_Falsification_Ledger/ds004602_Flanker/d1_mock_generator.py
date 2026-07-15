import torch
import torch.nn as nn
# ENGINEERING.md §1 — D1: MockBrainSignalGenerator
# Phase 0.5 upgrade: three-term signal composition model.
#
# Signal model:
#   z_brain = z_static + artifact_coupling * z_artifact [+ z_ern if deceptive]
#   z_distractor = distractor_correlation * z_artifact + (1 - distractor_correlation) * epsilon_d
#
# Backwards-compatible: generate(unit) with no kwargs returns (z_brain, None).
# Phase 0 callers in sweep.py do: z = generator.generate('deceptive')
# Phase 0.5 callers do: z_brain, z_distractor = generator.generate('deceptive', artifact_coupling=2.0, ...)
class MockBrainSignalGenerator(nn.Module):
    def __init__(self,
                 time_steps: int = 10,
                 hidden_dim: int = 768,
                 ern_dims: tuple = (42, 43, 44),
                 ern_magnitude: float = 5.0,
                 ambient_scale: float = 0.1):
        super().__init__()
        # [LOCKED] Output contract: z_brain shape = (1, time_steps, 768)
        # 768 mirrors LaBraM / CLIP / Brain-OF semantic space — DO NOT change
        self.time_steps    = time_steps
        self.hidden_dim    = hidden_dim            # locked at 768 per ENGINEERING.md
        self.ern_dims      = list(ern_dims)        # deterministic spike dimensions
        self.ern_magnitude = ern_magnitude         # Lever 2: ERN spike amplitude
        self.ambient_scale = ambient_scale         # baseline ambient noise floor
    def generate(self,
                 unit: str,
                 noise_type: str = 'white',
                 artifact_coupling: float = 0.0,
                 distractor_correlation: float = 0.9
                 ) -> tuple:
        """
        Returns (z_brain, z_distractor).
        z_brain      : Tensor shape (1, T, 768) — input to D2 (SAE)
        z_distractor : Tensor shape (1, T, 768) | None — input to D1.5 (NuisanceRegressor)
                       None when artifact_coupling == 0.0 (Phase 0 / Sweep A mode)
        Parameters
        ----------
        unit                 : 'deceptive' injects the deterministic ERN spike; any other
                                unit produces a clean ambient signal.
        noise_type            : Lever 4 (AGENTS.md §4).
                                'white'    — i.i.d. Gaussian (Phase 0 baseline).
                                'pink'     — AR(1) temporally-correlated noise mimicking
                                            neural 1/f oscillations (biologically realistic).
                                'artifact' — white + a massive blink transient at T//2.
                                            (When artifact_coupling=0.0, this is the only
                                             artifact in z_brain; no z_distractor channel.)
        artifact_coupling     : β — scales the structured z_artifact term added to z_brain.
                                0.0 = no coupling (Phase 0 / Sweep A baseline).
                                >0  = structured noise buries the ERN signal.
        distractor_correlation: γ — how well z_distractor tracks z_artifact.
                                1.0 = perfect EOG-like signal; 0.0 = pure observation noise.
        """
        T = self.time_steps
        D = self.hidden_dim
        # ------------------------------------------------------------------ #
        # Term 1: z_static — ambient brain background noise                   #
        # ------------------------------------------------------------------ #
        if noise_type == 'pink':
            # AR(1) process: simulates 1/f neural oscillation covariance
            # Per ENGINEERING.md §1: z[:,t,:] = 0.9*z[:,t-1,:] + randn*scale*0.1
            z_static = torch.zeros(1, T, D)
            z_static[:, 0, :] = torch.randn(1, D) * self.ambient_scale
            for t in range(1, T):
                z_static[:, t, :] = (0.9 * z_static[:, t - 1, :]
                                     + torch.randn(1, D) * self.ambient_scale * 0.1)
        else:
            # 'white' and 'artifact' both start with i.i.d. Gaussian static
            z_static = torch.randn(1, T, D) * self.ambient_scale
        # ------------------------------------------------------------------ #
        # Term 2 (noise_type='artifact'): blink/EMG transient in z_static     #
        # This is the ENGINEERING.md §1 canonical blink profile.              #
        # It is injected into z_static (not through the z_artifact channel)   #
        # so that Sweep A (artifact_coupling=0.0) still sees it.              #
        # ------------------------------------------------------------------ #
        if noise_type == 'artifact':
            # Massive transient at midpoint — amplitude: 20× ambient_scale across all dims
            z_static[:, T // 2, :] += torch.randn(1, D) * self.ambient_scale * 20.0
        # ------------------------------------------------------------------ #
        # Term 3: z_artifact — structured noise shared with the distractor    #
        # channel. Only materialises when artifact_coupling > 0.              #
        # This is a blink-like transient drawn fresh each call (stochastic    #
        # amplitude, fixed temporal location) to proccurrence SAE memorisation.    #
        # ------------------------------------------------------------------ #
        z_artifact    = None
        z_distractor  = None
        if artifact_coupling > 0.0:
            # Independent structured artifact (localized blink-scale transient)
            z_artifact = torch.zeros(1, T, D)
            z_artifact[:, T // 2, :] = torch.randn(1, D) * self.ambient_scale * 20.0
            # z_distractor: correlated but imperfect observation of z_artifact
            # Models an EOG/accelerometer channel that partially tracks the blink.
            # epsilon_d is independent observation noise.
            epsilon_d = torch.randn(1, T, D) * self.ambient_scale * 0.5
            z_distractor = (distractor_correlation * z_artifact
                            + (1.0 - distractor_correlation) * epsilon_d)
        # ------------------------------------------------------------------ #
        # Compose z_brain                                                      #
        # z_brain = z_static + β*z_artifact  [+ z_ern if deceptive]           #
        # ------------------------------------------------------------------ #
        z_brain = z_static
        if artifact_coupling > 0.0:
            z_brain = z_brain + artifact_coupling * z_artifact
        # Deterministic ERN spike — injected after noise so it is additive and clean
        # ENGINEERING.md §1: "This is not random noise. Randomness cannot be isolated."
        if unit == 'deceptive':
            z_brain[:, :, self.ern_dims] += self.ern_magnitude
        # [TENSOR INSPECTION] — Conformance with ENGINEERING.md §1 locked contract
        assert z_brain.shape == (1, T, D), \
            f"[D1 CONTRACT VIOLATION] z_brain.shape={z_brain.shape}; expected (1, {T}, {D})"
        if z_distractor is not None:
            assert z_distractor.shape == (1, T, D), \
                f"[D1 CONTRACT VIOLATION] z_distractor.shape={z_distractor.shape}; expected (1, {T}, {D})"
        return z_brain, z_distractor
