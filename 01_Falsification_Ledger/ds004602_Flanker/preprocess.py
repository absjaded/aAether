import torch
import torch.nn as nn
import torch.nn.functional as F
class DeltaHRFCorrection(nn.Module):
    """
    Learnable Causal Convolution for Delta-HRF Correction.
    Since TRIBE v2 latents already contain a 5-second offset, this module targets
    the remaining sluggishness (e.g., 4-7 extra seconds) in the TPJ/dmPFC.
    """
    def __init__(self, tr_duration=1.5, max_extra_delay=9.0):
        super().__init__()
        # Calculate kernel size based on TR and expected max sluggishness
        self.kernel_size = int(max_extra_delay / tr_duration)
        if self.kernel_size < 1:
            self.kernel_size = 1
            
        # Initialize with a decaying gamma-like weight distribution
        weights = torch.linspace(1, 0, self.kernel_size).view(1, 1, -1)
        self.delta_kernel = nn.Parameter(weights / weights.sum())
        
    def forward(self, fmri_seq):
        """
        fmri_seq: [batch, voxels, time] (Already TRIBE-offset by 5s)
        """
        if self.kernel_size == 1:
            return fmri_seq
            
        # Causal padding to proccurrence looking into the future
        padded = F.pad(fmri_seq, (self.kernel_size - 1, 0))
        b, v, t = fmri_seq.shape
        x = padded.view(b * v, 1, -1)
        corrected = F.conv1d(x, self.delta_kernel)
        return corrected.view(b, v, t)
