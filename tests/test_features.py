"""Tests for features/baseline.py"""

import pandas as pd
import pytest

from features.baseline import FEATURE_COLUMNS, get_baseline_features


def make_df(n=10) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000,
            "ticker": "AAPL",
            "label": 1,
        },
        index=dates,
    )


def test_get_baseline_features_returns_correct_columns():
    df = make_df()
    result = get_baseline_features(df)
    assert list(result.columns) == FEATURE_COLUMNS


def test_get_baseline_features_drops_non_feature_columns():
    df = make_df()
    result = get_baseline_features(df)
    assert "ticker" not in result.columns
    assert "label" not in result.columns


def test_get_baseline_features_raises_on_missing_column():
    df = make_df().drop(columns=["volume"])
    with pytest.raises(ValueError, match="Missing required columns"):
        get_baseline_features(df)


def test_get_baseline_features_does_not_modify_input():
    df = make_df()
    original_cols = list(df.columns)
    get_baseline_features(df)
    assert list(df.columns) == original_cols
