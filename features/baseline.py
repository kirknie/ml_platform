"""
Baseline feature set for Week 5.

Uses raw OHLCV columns only — no feature engineering.

Rationale: establishing a raw baseline first gives a clean comparison point
when Week 6 introduces engineered features (returns, RSI, volatility, etc.)
via the feature store. If the engineered model doesn't beat this baseline,
something is wrong with the feature definitions.
"""

import pandas as pd

FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]


def get_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df containing only the baseline feature columns.

    No computation needed — these are raw observed values at time T.
    No rolling windows, no leakage risk.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with only FEATURE_COLUMNS retained
    """
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[FEATURE_COLUMNS].copy()
