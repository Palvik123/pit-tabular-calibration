# pit-tabular-calibration

Strict **Point-in-Time (`Out-of-Time`)** and **Walk-Forward** validation utilities for longitudinal entity-year panel datasets, paired with **hybrid Platt + Isotonic probability calibration** and **Elkan (2001) Bayes prior-odds shift correction**.

## Why This Exists

Random train/test splits on longitudinal or multi-source corporate/financial datasets silently leak future regime shifts and cross-snapshot entity signals, inflating `ROC-AUC` while producing poorly calibrated probabilities in production.

`pit-tabular-calibration` enforces:
1. **Zero temporal leakage (`PointInTimeSplitter`, `WalkForwardSplitter`):** strict chronological cutoffs (`Train <= T_1 < Calib <= T_2 < Test`) with automated guards ensuring no test timestamp precedes or overlaps calibration/training horizons.
2. **Smooth upper-quantile probability calibration (`HybridPlattIsotonicCalibrator`):** convex combination of parametric sigmoid (`Platt`) and non-parametric monotonic (`Isotonic`) calibration (`alpha * P_platt + (1 - alpha) * P_iso`), eliminating isotonic step-plateaus in low-base-rate top deciles.
3. **Exact prior-odds shift correction (`adjust_prior_odds`):** analytical Bayes update (Elkan, 2001) mapping calibrated probabilities from a sampled cohort base rate $\pi_{\text{train}}$ to the true production population base rate $\pi_{\text{target}}$.
4. **Decision-ranking & calibration diagnostics:** `brier_score`, `expected_calibration_error` (`ECE`), `precision_at_k`, and `lift_at_k`.

## Quickstart

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from pit_calibration import (
    HybridPlattIsotonicCalibrator,
    PointInTimeSplitter,
    adjust_prior_odds,
    brier_score,
    expected_calibration_error,
    lift_at_k,
)

# Longitudinal entity-year observations
years = np.repeat(np.arange(2016, 2025), 200)
rng = np.random.default_rng(42)
X = rng.normal(size=(len(years), 6))
logits = 0.9 * X[:, 0] - 0.6 * X[:, 1] - 2.2
y = (rng.uniform(size=len(years)) < 1.0 / (1.0 + np.exp(-logits))).astype(int)

splitter = PointInTimeSplitter(train_end=2021, calib_end=2023)
split = splitter.split(years)

clf = LogisticRegression(random_state=42).fit(X[split.train_idx], y[split.train_idx])
calibrator = HybridPlattIsotonicCalibrator(alpha=0.80).fit(
    clf.predict_proba(X[split.calib_idx])[:, 1],
    y[split.calib_idx],
)

p_test = calibrator.predict_proba(clf.predict_proba(X[split.test_idx])[:, 1])
p_shifted = adjust_prior_odds(p_test, source_prior=y[split.calib_idx].mean(), target_prior=0.02)

print(f"Brier: {brier_score(y[split.test_idx], p_test):.4f}")
print(f"ECE:   {expected_calibration_error(y[split.test_idx], p_test):.4f}")
print(f"Lift@10%: {lift_at_k(y[split.test_idx], p_shifted, k_fraction=0.10):.2f}x")
```

## Running Tests

```bash
pytest -q
```

## License

MIT
