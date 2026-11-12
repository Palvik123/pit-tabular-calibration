"""Hybrid Platt + Isotonic probability calibration and Elkan prior-odds shift adjustment."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _clip_probs(probs: ArrayLike, eps: float = 1e-6) -> NDArray[np.float64]:
    arr = np.asarray(probs, dtype=np.float64)
    return np.clip(arr, eps, 1.0 - eps)


def _to_log_odds(probs: ArrayLike, eps: float = 1e-6) -> NDArray[np.float64]:
    clipped = _clip_probs(probs, eps=eps)
    return np.log(clipped / (1.0 - clipped))


def adjust_prior_odds(
    probs: ArrayLike,
    source_prior: float,
    target_prior: float,
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """Apply Elkan (2001) exact Bayes prior-odds shift correction to calibrated probabilities.

    Given calibrated P(y=1 | x) estimated under base rate ``source_prior`` (pi_0),
    computes posterior probability under ``target_prior`` (pi_1):

        odds_target = odds_source * (pi_1 / (1 - pi_1)) / (pi_0 / (1 - pi_0))
    """
    if not (0.0 < source_prior < 1.0) or not (0.0 < target_prior < 1.0):
        raise ValueError("source_prior and target_prior must lie strictly in (0, 1).")

    p = _clip_probs(probs, eps=eps)
    source_odds = p / (1.0 - p)
    prior_ratio = (target_prior / (1.0 - target_prior)) / (
        source_prior / (1.0 - source_prior)
    )
    adjusted_odds = source_odds * prior_ratio
    return adjusted_odds / (1.0 + adjusted_odds)


class HybridPlattIsotonicCalibrator:
    """Smooth monotonic probability calibrator blending Platt scaling and Isotonic regression.

    Pure Isotonic regression on sparse low-base-rate cohorts creates step-function plateaus
    in the top deciles, hurting ranking tie-breaking. Blending ``alpha * Platt + (1 - alpha) * Isotonic``
    preserves strict monotonicity while keeping upper-quantile probabilities smooth.
    """

    def __init__(self, alpha: float = 0.80, eps: float = 1e-6) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must lie in [0.0, 1.0].")
        self.alpha = alpha
        self.eps = eps
        self._platt: LogisticRegression | None = None
        self._isotonic: IsotonicRegression | None = None

    def fit(
        self, raw_scores: ArrayLike, y_true: ArrayLike
    ) -> HybridPlattIsotonicCalibrator:
        scores = _clip_probs(raw_scores, eps=self.eps)
        y = np.asarray(y_true, dtype=np.int64)
        if scores.ndim != 1 or y.ndim != 1 or scores.shape[0] != y.shape[0]:
            raise ValueError("raw_scores and y_true must be 1D arrays of equal length.")
        if np.unique(y).size < 2:
            raise ValueError(
                "y_true must contain both 0 and 1 classes for calibration."
            )

        log_odds = _to_log_odds(scores, eps=self.eps).reshape(-1, 1)
        platt = LogisticRegression(C=1e4, solver="lbfgs", random_state=42)
        platt.fit(log_odds, y)

        iso = IsotonicRegression(
            y_min=self.eps, y_max=1.0 - self.eps, out_of_bounds="clip"
        )
        iso.fit(scores, y)

        self._platt = platt
        self._isotonic = iso
        return self

    def predict_proba(self, raw_scores: ArrayLike) -> NDArray[np.float64]:
        if self._platt is None or self._isotonic is None:
            raise RuntimeError(
                "Calibrator must be fitted before calling predict_proba."
            )

        scores = _clip_probs(raw_scores, eps=self.eps)
        log_odds = _to_log_odds(scores, eps=self.eps).reshape(-1, 1)

        p_platt = self._platt.predict_proba(log_odds)[:, 1]
        p_iso = np.asarray(self._isotonic.predict(scores), dtype=np.float64)
        blended = self.alpha * p_platt + (1.0 - self.alpha) * p_iso
        return _clip_probs(blended, eps=self.eps)
