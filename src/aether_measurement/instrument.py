"""Grouped readout of batch-calibrated covariance features.

PCA, scaling and readout fitting occur inside each training fold. Held-out
subjects provide unlabelled covariance batches for their geometric calibration.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .geometry import covariance_features, subject_ids


def grouped_readout(features, labels, subjects, *, components=12, splits=6) -> dict:
    """Produce one held-out score per row; never reorder predictions by label."""
    raw_x = np.asarray(features)
    raw_y = np.asarray(labels)
    if raw_x.dtype.kind not in "biuf" or raw_x.ndim != 2 or min(raw_x.shape, default=0) == 0:
        raise ValueError("features must be a nonempty real numeric trial-by-feature matrix")
    x = raw_x.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    if raw_y.shape != (len(x),) or raw_y.dtype.kind not in "biuf" or not np.isin(raw_y, [0, 1]).all():
        raise ValueError("labels must be a binary numeric array matching trials")
    y = raw_y.astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError("readout requires both label classes")
    groups = subject_ids(subjects, len(x))
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("readout requires at least two subjects")
    if not isinstance(components, (int, np.integer)) or components < 1:
        raise ValueError("components must be a positive integer")
    if not isinstance(splits, (int, np.integer)) or splits < 2:
        raise ValueError("splits must be an integer of at least two")

    predictions = np.full(len(x), np.nan)
    fold_ids = np.full(len(x), -1, dtype=int)
    folds = []
    for fold_id, (train, test) in enumerate(GroupKFold(min(splits, n_groups)).split(x, y, groups)):
        if len(np.unique(y[train])) != 2:
            raise ValueError(f"training fold {fold_id} contains only one label class")
        dimension = min(components, x.shape[1], len(train) - 1)
        # Constant inputs have a well-defined intercept-only readout. Avoid an
        # undefined PCA explained-variance ratio for that case.
        stages = []
        if np.any(np.ptp(x[train], axis=0) > 0):
            stages.append(PCA(n_components=dimension, svd_solver="full"))
        stages.extend([StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=0)])
        model = make_pipeline(*stages)
        model.fit(x[train], y[train])
        scores = model.predict_proba(x[test])[:, 1]
        predictions[test] = scores
        fold_ids[test] = fold_id
        folds.append({
            "fold": fold_id, "train_rows": len(train), "test_rows": len(test),
            "train_subjects": np.unique(groups[train]).tolist(),
            "test_subjects": np.unique(groups[test]).tolist(),
            "auc": float(roc_auc_score(y[test], scores)) if len(np.unique(y[test])) == 2 else None,
        })
    if not np.isfinite(predictions).all() or np.any(fold_ids < 0):
        raise RuntimeError("cross-validation did not assign exactly one finite score per trial")
    return {
        "auc": float(roc_auc_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions >= 0.5)),
        "trials": len(x), "subjects": n_groups,
        "positives": int(y.sum()), "folds": folds,
        "scores": predictions.tolist(), "fold_ids": fold_ids.tolist(),
        "readout": "GroupKFold; training-fold PCA, scaling and balanced logistic regression",
    }


def measure(covariances, labels, subjects, *, components=12, splits=6) -> dict:
    features = covariance_features(covariances, subjects)
    report = grouped_readout(features, labels, subjects, components=components, splits=splits)
    report.update({
        "feature_dimension": features.shape[1],
        "geometry": "subject arithmetic-mean whitening; matrix log; sqrt(2)-weighted off-diagonals",
        "calibration": "all supplied unlabelled trials per subject, including held-out subject batches",
        "protocol": "portable-batch-covariance-v1",
    })
    return report
