import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
def build_snntorch_snn(beta=0.9, threshold=1.0):
    """
    Builds the SNN Phase 0 integrator using snnTorch.
    The weights are fixed to 1.0 to simply integrate the input spikes.
    """
    net = nn.Sequential(
        nn.Linear(1, 1, bias=False),
        snn.Leaky(beta=beta, threshold=threshold)
    )
    # Fix the linear weight to 1.0 so spikes pass through directly to the integrator
    with torch.no_grad():
        net[0].weight.data = torch.tensor([[1.0]])
    return net
def run_snn(snn_net, spike_train: np.ndarray) -> tuple[bool, int]:
    """
    Runs the input spike train through the snnTorch network.
    Returns (spiked: bool, convergence_step: int).
    """
    # Initialize membrane potential
    mem = snn_net[1].init_leaky()
    
    # Convert spike_train to tensor of shape (Time, Batch, InputDim) -> (Time, 1, 1)
    spk_in = torch.tensor(spike_train, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    
    for t in range(len(spike_train)):
        cur = snn_net[0](spk_in[t])
        spk_out, mem = snn_net[1](cur, mem)
        
        if spk_out.item() > 0:
            return True, t
            
    return False, len(spike_train)
def build_spike_train(features, feature_idx: int, threshold: float, n_timesteps: int) -> np.ndarray:
    """
    Converts a continuous SAE feature activation into a binary spike train.
    """
    activation = features[0, :, feature_idx].mean().item()
    spike = 1 if activation > threshold else 0
    return np.array([spike] * n_timesteps, dtype=float)
def process_snn_output(snn_spiked: bool) -> float:
    """
    Translates the SNN spike into an RL penalty for the ToyLLM.
    """
    if snn_spiked:
        return -10.0
    return 0.0
LAVA_AVAILABLE = False
