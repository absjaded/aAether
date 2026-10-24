"""Unlabelled, batch-calibrated covariance geometry.

Each subject supplies its own arithmetic covariance reference. Whitening and
matrix-log features use no labels; fitting this reference on the entire subject
batch is an explicit calibration requirement, including for held-out subjects.
"""

from __future__ import annotations

import numpy as np


def subject_ids(subjects: np.ndarray, n_rows: int) -> np.ndarray:
    """Validate simple scalar identifiers, preserving their input dtype."""
    groups = np.asarray(subjects)
    if groups.shape != (n_rows,) or groups.dtype.kind not in "biufUS":
        raise ValueError("subjects must be a one-dimensional numeric or string array matching trials")
    if groups.dtype.kind in "iuf" and not np.isfinite(groups).all():
        raise ValueError("subjects must contain finite identifiers")
    if groups.dtype.kind in "US" and any(not str(x).strip() for x in groups):
        raise ValueError("subjects must contain nonempty identifiers")
    return groups.astype("U") if groups.dtype.kind == "S" else groups


def validate_covariances(covariances: np.ndarray) -> np.ndarray:
    """Reject malformed, asymmetric or non-positive-definite trial matrices.

    Tolerances scale with each matrix: physical units such as MEG covariance
    magnitudes must not be replaced by an absolute eigenvalue floor.
    """
    raw = np.asarray(covariances)
    if raw.dtype.kind not in "biuf":
        raise ValueError("covariances must be real numeric matrices")
    covs = np.asarray(raw, dtype=np.float64)
    if covs.ndim != 3 or covs.shape[0] == 0 or covs.shape[1] == 0 or covs.shape[1] != covs.shape[2]:
        raise ValueError("covariances must have nonempty shape (trials, channels, channels)")
    if not np.isfinite(covs).all():
        raise ValueError("covariances must be finite")
    scale = np.max(np.abs(covs), axis=(1, 2))
    asymmetry = np.max(np.abs(covs - covs.swapaxes(1, 2)), axis=(1, 2))
    if np.any(asymmetry > 1e-7 * scale):
        raise ValueError("covariances must be symmetric")
    covs = 0.5 * covs + 0.5 * covs.swapaxes(1, 2)
    eigenvalues = np.linalg.eigvalsh(covs)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise ValueError("covariances must be strictly positive definite; regularize upstream explicitly")
    return covs


def _symmetric_function(matrix: np.ndarray, function) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * matrix + 0.5 * matrix.T)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise ValueError("matrix transform encountered a non-positive or non-finite eigenvalue")
    transformed = (eigenvectors * function(eigenvalues)) @ eigenvectors.T
    if not np.isfinite(transformed).all():
        raise ValueError("matrix transform overflowed; check conditioning and physical units")
    return transformed


def covariance_features(covariances: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Return Frobenius-weighted upper triangles of log-whitened covariances.

    C -> log(R_s^(-1/2) C R_s^(-1/2)); off-diagonals receive sqrt(2).
    R_s is the unlabelled mean of the supplied batch for subject s. This is a
    batch transform, not a fitted prospective single-trial decoder.
    """
    covs = validate_covariances(covariances)
    groups = subject_ids(subjects, len(covs))
    rows, cols = np.triu_indices(covs.shape[1])
    weights = np.where(rows == cols, 1.0, np.sqrt(2.0))
    features = np.empty((len(covs), len(rows)), dtype=np.float64)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        # Summing scaled terms avoids overflow in an otherwise finite mean.
        reference = np.sum(covs[indices] / len(indices), axis=0)
        whitening = _symmetric_function(reference, lambda values: 1.0 / np.sqrt(values))
        for index in indices:
            whitened = whitening @ covs[index] @ whitening.T
            log_matrix = _symmetric_function(whitened, np.log)
            features[index] = log_matrix[rows, cols] * weights
    return features
