"""Strict Point-in-Time and Walk-Forward temporal splitters for longitudinal panel data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PointInTimeSplit:
    """Disjoint chronological index partition (Train <= T1 < Calib <= T2 < Test)."""

    train_idx: NDArray[np.intp]
    calib_idx: NDArray[np.intp]
    test_idx: NDArray[np.intp]


class PointInTimeSplitter:
    """Enforce a strict 3-way Out-of-Time split with zero temporal overlap."""

    def __init__(self, train_end: int | float, calib_end: int | float) -> None:
        if calib_end <= train_end:
            raise ValueError(
                f"calib_end ({calib_end}) must be strictly greater than train_end ({train_end})."
            )
        self.train_end = train_end
        self.calib_end = calib_end

    def split(
        self,
        timestamps: ArrayLike,
        feature_timestamps: ArrayLike | None = None,
    ) -> PointInTimeSplit:
        """Partition observation indices chronologically and verify feature availability."""
        ts = np.asarray(timestamps)
        if ts.ndim != 1 or ts.size == 0:
            raise ValueError("timestamps must be a non-empty 1D array.")

        if feature_timestamps is not None:
            feat_ts = np.asarray(feature_timestamps)
            if feat_ts.shape != ts.shape:
                raise ValueError("feature_timestamps must match timestamps shape.")
            leaked = np.flatnonzero(feat_ts > ts)
            if leaked.size > 0:
                raise ValueError(
                    f"Future feature leakage detected at {leaked.size} row(s), first index={int(leaked[0])}."
                )

        train_idx = np.flatnonzero(ts <= self.train_end)
        calib_idx = np.flatnonzero((ts > self.train_end) & (ts <= self.calib_end))
        test_idx = np.flatnonzero(ts > self.calib_end)

        if train_idx.size == 0 or calib_idx.size == 0 or test_idx.size == 0:
            raise ValueError(
                "Empty partition encountered: "
                f"train={train_idx.size}, calib={calib_idx.size}, test={test_idx.size}."
            )

        assert int(ts[train_idx].max()) <= int(ts[calib_idx].min()) - 1 or float(
            ts[train_idx].max()
        ) < float(ts[calib_idx].min())
        assert float(ts[calib_idx].max()) < float(ts[test_idx].min())

        return PointInTimeSplit(
            train_idx=train_idx,
            calib_idx=calib_idx,
            test_idx=test_idx,
        )


class WalkForwardSplitter:
    """Expanding-window chronological backtester across ordered evaluation periods."""

    def __init__(self, min_train_periods: int = 3) -> None:
        if min_train_periods < 1:
            raise ValueError("min_train_periods must be >= 1.")
        self.min_train_periods = min_train_periods

    def split(
        self, periods: ArrayLike
    ) -> Iterator[tuple[int | float, NDArray[np.intp], NDArray[np.intp]]]:
        """Yield (eval_period, train_idx, test_idx) using strictly past periods for training."""
        p = np.asarray(periods)
        unique_periods = np.unique(p)
        if unique_periods.size <= self.min_train_periods:
            raise ValueError(
                f"Need more than {self.min_train_periods} distinct periods, got {unique_periods.size}."
            )

        for idx in range(self.min_train_periods, unique_periods.size):
            eval_period = unique_periods[idx]
            cutoff = unique_periods[idx - 1]
            train_idx = np.flatnonzero(p <= cutoff)
            test_idx = np.flatnonzero(p == eval_period)
            yield eval_period.item(), train_idx, test_idx
