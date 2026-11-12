"""Unit tests for Point-in-Time splitting, Walk-Forward backtesting, and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from pit_calibration import (
    HybridPlattIsotonicCalibrator,
    PointInTimeSplitter,
    WalkForwardSplitter,
    adjust_prior_odds,
    brier_score,
    expected_calibration_error,
    lift_at_k,
    precision_at_k,
)


def test_point_in_time_splitter_strict_disjoint_horizons() -> None:
    years = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025])
    splitter = PointInTimeSplitter(train_end=2021, calib_end=2023)
    split = splitter.split(years)

    assert years[split.train_idx].tolist() == [2019, 2020, 2021]
    assert years[split.calib_idx].tolist() == [2022, 2023]
    assert years[split.test_idx].tolist() == [2024, 2025]


def test_point_in_time_splitter_rejects_future_feature_leakage() -> None:
    obs_years = np.array([2020, 2021, 2022, 2023, 2024])
    leaked_feature_years = np.array(
        [2020, 2022, 2022, 2023, 2024]
    )  # index 1 uses 2022 feature in 2021
    splitter = PointInTimeSplitter(train_end=2021, calib_end=2023)

    with pytest.raises(ValueError, match="Future feature leakage"):
        splitter.split(obs_years, feature_timestamps=leaked_feature_years)


def test_walk_forward_splitter_expanding_window() -> None:
    periods = np.repeat([2020, 2021, 2022, 2023, 2024], 4)
    wf = WalkForwardSplitter(min_train_periods=3)
    folds = list(wf.split(periods))

    assert len(folds) == 2
    p0, train0, test0 = folds[0]
    assert p0 == 2023
    assert int(periods[train0].max()) == 2022
    assert np.all(periods[test0] == 2023)

    p1, train1, test1 = folds[1]
    assert p1 == 2024
    assert train1.size > train0.size
    assert int(periods[train1].max()) == 2023
    assert np.all(periods[test1] == 2024)


def test_hybrid_calibrator_reduces_brier_and_preserves_monotonicity() -> None:
    rng = np.random.default_rng(42)
    n = 1500
    signal = rng.normal(0.0, 1.0, size=n)
    true_prob = 1.0 / (1.0 + np.exp(-(1.1 * signal - 2.4)))
    y = (rng.uniform(size=n) < true_prob).astype(int)

    # Deliberately miscalibrated overconfident raw model scores
    raw_scores = 1.0 / (1.0 + np.exp(-(2.2 * signal + 0.3)))

    calibrator = HybridPlattIsotonicCalibrator(alpha=0.80).fit(
        raw_scores[:1000], y[:1000]
    )
    calibrated = calibrator.predict_proba(raw_scores[1000:])

    raw_brier = brier_score(y[1000:], raw_scores[1000:])
    cal_brier = brier_score(y[1000:], calibrated)
    cal_ece = expected_calibration_error(y[1000:], calibrated, n_bins=10)

    assert cal_brier < raw_brier * 0.5
    assert cal_ece < 0.05

    # Monotonicity check on sorted grid
    grid = np.linspace(0.01, 0.99, 50)
    preds = calibrator.predict_proba(grid)
    assert np.all(np.diff(preds) >= -1e-12)


def test_elkan_prior_odds_shift_exact_recovery() -> None:
    probs = np.array([0.10, 0.25, 0.50, 0.80])
    shifted_down = adjust_prior_odds(probs, source_prior=0.20, target_prior=0.02)
    assert np.all(shifted_down < probs)

    # Round-trip back to original prior recovers exact probabilities
    recovered = adjust_prior_odds(shifted_down, source_prior=0.02, target_prior=0.20)
    np.testing.assert_allclose(recovered, probs, rtol=1e-6)


def test_ranking_lift_and_precision_at_k() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.95, 0.91, 0.40, 0.88, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05])

    p_at_20 = precision_at_k(y, scores, k_fraction=0.20)
    lift_20 = lift_at_k(y, scores, k_fraction=0.20)

    assert p_at_20 == pytest.approx(1.0)
    assert lift_20 == pytest.approx(1.0 / 0.30)
