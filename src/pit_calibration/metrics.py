"""Calibration accuracy and top-decile decision ranking metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared error between binary outcomes and predicted probabilities."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and y_prob must have identical non-empty shape.")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10
) -> float:
    """Weighted average absolute gap between bin confidence and empirical event rate."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and y_prob must have identical non-empty shape.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(p, bins, right=True) - 1, 0, n_bins - 1)

    ece = 0.0
    total = float(y.size)
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        emp_rate = float(np.mean(y[mask]))
        avg_conf = float(np.mean(p[mask]))
        ece += (count / total) * abs(emp_rate - avg_conf)
    return float(ece)


def precision_at_k(
    y_true: ArrayLike, y_score: ArrayLike, k_fraction: float = 0.05
) -> float:
    """Positive rate within the top ``k_fraction`` highest-scored observations."""
    if not (0.0 < k_fraction <= 1.0):
        raise ValueError("k_fraction must lie in (0.0, 1.0].")
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    if y.shape != s.shape or y.size == 0:
        raise ValueError("y_true and y_score must have identical non-empty shape.")

    k = max(1, int(np.ceil(y.size * k_fraction)))
    top_idx = np.argsort(-s, kind="mergesort")[:k]
    return float(np.mean(y[top_idx]))


def lift_at_k(y_true: ArrayLike, y_score: ArrayLike, k_fraction: float = 0.05) -> float:
    """Ratio of Precision@K to the overall sample base rate."""
    y = np.asarray(y_true, dtype=np.float64)
    base_rate = float(np.mean(y))
    if base_rate <= 0.0:
        raise ValueError("Cannot compute lift when y_true has zero positive examples.")
    return precision_at_k(y, y_score, k_fraction=k_fraction) / base_rate
