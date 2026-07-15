import torch
import torch.nn as nn
import torch.nn.functional as F

TR = 1.5  # seconds per volume

def double_gamma_hrf(peak_seconds: float, T: int, tr: float = TR) -> torch.Tensor:
    t = torch.arange(T, dtype=torch.float32) * tr
    a1, b1 = peak_seconds, 1.0          # primary peak
    a2, b2 = peak_seconds + 8.0, 1.0   # undershoot
    hrf = (t ** (a1 - 1) * torch.exp(-t / b1) / torch.lgamma(torch.tensor(a1)).exp()
         - 0.35 * t ** (a2 - 1) * torch.exp(-t / b2) / torch.lgamma(torch.tensor(a2)).exp())
    hrf = hrf / (hrf.abs().max() + 1e-8)
    return hrf

def build_delta_hrf_filter(T: int, canonical_peak_s: float = 5.0, tpj_peak_s: float = 10.0, lam: float = 0.1) -> torch.Tensor:
    h_can = double_gamma_hrf(canonical_peak_s, T)
    h_tpj = double_gamma_hrf(tpj_peak_s, T)

    H_can = torch.fft.rfft(h_can)
    H_tpj = torch.fft.rfft(h_tpj)

    W_can = H_can.conj() / (H_can.abs().pow(2) + lam)
    W_tpj = H_tpj.conj() / (H_tpj.abs().pow(2) + lam)

    delta_filter = W_can / (W_tpj + 1e-8)
    return delta_filter

class DeltaHRFCorrection(nn.Module):
    """
    Wiener Deconvolution in the Frequency Domain (Tikhonov-Regularized).
    Analytically corrects the 4-7 TR sluggishness of the TPJ.
    Zero-Label compliant: requires no training data.
    """
    def __init__(self, canonical_peak_s=5.0, tpj_peak_s=10.0, lam=0.1):
        super().__init__()
        self.canonical_peak_s = canonical_peak_s
        self.tpj_peak_s = tpj_peak_s
        self.lam = lam
        
    def forward(self, fmri_seq):
        """
        fmri_seq: [B, P, F] where B is time/batch, P is parcels, F is features.
        We apply FFT along the time axis (dim=0).
        """
        T = fmri_seq.shape[0]
        # Build filter on the fly for the given sequence length T
        delta_filter = build_delta_hrf_filter(T, self.canonical_peak_s, self.tpj_peak_s, self.lam).to(fmri_seq.device)
        
        # FFT along time axis (dim=0)
        X_f = torch.fft.rfft(fmri_seq, dim=0)           # (T//2+1, P, F)
        
        # Apply filter (broadcast over P and F)
        X_f_corr = X_f * delta_filter.unsqueeze(-1).unsqueeze(-1)  # (T//2+1, P, F)
        
        # IFFT back to time domain
        X_corrected = torch.fft.irfft(X_f_corr, n=T, dim=0)      # (T, P, F)
        return X_corrected

