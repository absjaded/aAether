import torch

class TemporalBlockSplitter:
    """
    Splits continuous fMRI sequences into training and validation blocks,
    enforcing a strict temporal gap to proccurrence HRF autocorrelation leakage.
    """
    def __init__(self, total_frames: int, train_ratio: float = 0.70, hrf_gap_frames: int = 10):
        self.total_frames = total_frames
        self.train_end = int(total_frames * train_ratio)
        self.val_start = self.train_end + hrf_gap_frames
        
        if self.val_start >= total_frames:
            # Fallback for extremely short sequences where gap consumes the rest
            # We enforce at least 1 frame for validation in this toy scenario
            self.val_start = total_frames - 1
            self.train_end = max(1, self.val_start - hrf_gap_frames)

    def split(self, *tensors: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """
        Splits a sequence of tensors into (train_tensors, val_tensors).
        """
        train_tensors = tuple(t[:self.train_end] for t in tensors)
        val_tensors = tuple(t[self.val_start:] for t in tensors)
        
        return train_tensors, val_tensors
    
    def get_summary(self) -> str:
        train_size = self.train_end
        val_size = self.total_frames - self.val_start
        gap_size = self.val_start - self.train_end
        return f"Train Set: {train_size} | HRF Gap: {gap_size} | Val Set: {val_size}"
