"""
Feature definitions for the ML platform feature store.

Each FeatureDefinition has two compute paths:
- offline: vectorized pandas transform over a full ticker DataFrame (used at training time)
- online:  scalar computation over a rolling window DataFrame (used at serving time)

Both paths must produce the same value at time T given the same underlying price history.
The parity test in test_feature_store.py enforces this numerically.

Design decisions:
- Offline uses pandas rolling/shift for speed over large DataFrames
- Online takes a window DataFrame so the serving layer controls how much history it provides
- required_window is the minimum rows needed for online to return a valid (non-NaN) result
"""

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass
class FeatureDefinition:
    name: str
    offline: Callable[[pd.DataFrame], pd.Series]  # full ticker df → series indexed by date
    online: Callable[[pd.DataFrame], float]         # window df (last N rows) → scalar for last row
    required_window: int                             # min rows needed for online path


@dataclass
class FeatureSet:
    name: str
    version: int
    features: list[FeatureDefinition] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def max_required_window(self) -> int:
        """The largest window any feature needs — used to size stream windows for serving."""
        return max(f.required_window for f in self.features)


# ---------------------------------------------------------------------------
# RSI helper — shared by both compute paths so they can't diverge
# ---------------------------------------------------------------------------

def _rsi_from_closes(closes: pd.Series, period: int = 14) -> float:
    """
    Compute RSI from a Series of close prices (length >= period + 1).

    Uses simple rolling mean (not Wilder's EMA) — sufficient for ML features.
    Returns float in [0, 100], or NaN if insufficient data.
    """
    delta = closes.diff().dropna()
    if len(delta) < period:
        return float("nan")
    recent = delta.tail(period)
    avg_gain = recent.clip(lower=0).mean()
    avg_loss = (-recent).clip(lower=0).mean()
    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


# ---------------------------------------------------------------------------
# Engineered feature set v1
# Features: 1-day return, 5-day return, 5-day volatility, 20-day volatility,
#           14-day RSI, 20-day volume ratio
# ---------------------------------------------------------------------------

ENGINEERED_V1 = FeatureSet(
    name="engineered",
    version=1,
    features=[
        FeatureDefinition(
            name="return_1d",
            offline=lambda df: df["close"].pct_change(1),
            online=lambda w: (w["close"].iloc[-1] - w["close"].iloc[-2]) / w["close"].iloc[-2],
            required_window=2,
        ),
        FeatureDefinition(
            name="return_5d",
            offline=lambda df: df["close"].pct_change(5),
            online=lambda w: (w["close"].iloc[-1] - w["close"].iloc[-6]) / w["close"].iloc[-6],
            required_window=6,
        ),
        FeatureDefinition(
            name="volatility_5d",
            # std of 5 daily returns — rolling(5) on pct_change gives std of [t-4..t]
            offline=lambda df: df["close"].pct_change().rolling(5).std(),
            online=lambda w: w["close"].pct_change().dropna().tail(5).std(),
            required_window=6,  # 6 rows → 5 diffs → std of 5
        ),
        FeatureDefinition(
            name="volatility_20d",
            offline=lambda df: df["close"].pct_change().rolling(20).std(),
            online=lambda w: w["close"].pct_change().dropna().tail(20).std(),
            required_window=21,  # 21 rows → 20 diffs → std of 20
        ),
        FeatureDefinition(
            name="rsi_14",
            offline=lambda df: pd.Series(
                [_rsi_from_closes(df["close"].iloc[: i + 1]) for i in range(len(df))],
                index=df.index,
            ),
            online=lambda w: _rsi_from_closes(w["close"]),
            required_window=15,  # 15 rows → 14 diffs → RSI over 14 periods
        ),
        FeatureDefinition(
            name="volume_ratio_20d",
            offline=lambda df: df["volume"] / df["volume"].rolling(20).mean(),
            online=lambda w: w["volume"].iloc[-1] / w["volume"].tail(20).mean(),
            required_window=20,
        ),
    ],
)


# ---------------------------------------------------------------------------
# MACD helpers — shared by both compute paths
# EWM parity note: online path with required_window=100 approximates the
# offline path to within ~0.5% due to EWM initialization. Tolerance in
# the parity test is relaxed to 1e-2 for macd_signal accordingly.
# ---------------------------------------------------------------------------

def _macd_signal_offline(df: pd.DataFrame) -> pd.Series:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    return macd_line.ewm(span=9, adjust=False).mean()


def _macd_signal_online(window: pd.DataFrame) -> float:
    ema12 = window["close"].ewm(span=12, adjust=False).mean()
    ema26 = window["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    return float(macd_line.ewm(span=9, adjust=False).mean().iloc[-1])


# ---------------------------------------------------------------------------
# Engineered feature set v2
# Adds return_20d and macd_signal on top of the 6 V1 features.
# V1 is preserved unchanged — any model trained on V1 still works with V1 features.
# ---------------------------------------------------------------------------

ENGINEERED_V2 = FeatureSet(
    name="engineered",
    version=2,
    features=[
        FeatureDefinition(
            name="return_1d",
            offline=lambda df: df["close"].pct_change(1),
            online=lambda w: (w["close"].iloc[-1] - w["close"].iloc[-2]) / w["close"].iloc[-2],
            required_window=2,
        ),
        FeatureDefinition(
            name="return_5d",
            offline=lambda df: df["close"].pct_change(5),
            online=lambda w: (w["close"].iloc[-1] - w["close"].iloc[-6]) / w["close"].iloc[-6],
            required_window=6,
        ),
        FeatureDefinition(
            name="return_20d",
            offline=lambda df: df["close"].pct_change(20),
            online=lambda w: (w["close"].iloc[-1] - w["close"].iloc[-21]) / w["close"].iloc[-21],
            required_window=21,
        ),
        FeatureDefinition(
            name="volatility_5d",
            offline=lambda df: df["close"].pct_change().rolling(5).std(),
            online=lambda w: w["close"].pct_change().dropna().tail(5).std(),
            required_window=6,
        ),
        FeatureDefinition(
            name="volatility_20d",
            offline=lambda df: df["close"].pct_change().rolling(20).std(),
            online=lambda w: w["close"].pct_change().dropna().tail(20).std(),
            required_window=21,
        ),
        FeatureDefinition(
            name="rsi_14",
            offline=lambda df: pd.Series(
                [_rsi_from_closes(df["close"].iloc[: i + 1]) for i in range(len(df))],
                index=df.index,
            ),
            online=lambda w: _rsi_from_closes(w["close"]),
            required_window=15,
        ),
        FeatureDefinition(
            name="volume_ratio_20d",
            offline=lambda df: df["volume"] / df["volume"].rolling(20).mean(),
            online=lambda w: w["volume"].iloc[-1] / w["volume"].tail(20).mean(),
            required_window=20,
        ),
        FeatureDefinition(
            name="macd_signal",
            offline=_macd_signal_offline,
            online=_macd_signal_online,
            required_window=100,  # large window to minimize EWM initialization error
        ),
    ],
)
