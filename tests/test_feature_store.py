"""
Tests for features/definitions.py and features/store.py.

Key tests:
- Offline/online parity: same feature value at time T from both paths
- No forward leakage: feature at T doesn't use data after T
- Backfill idempotency: running backfill twice gives same result
- FeatureStore round-trip: compute → save → load gives same data
- RSI bounds: always in [0, 100]
- Volume ratio: always positive
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.definitions import ENGINEERED_V1, FeatureDefinition
from features.store import FeatureStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_price_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Create a realistic fake OHLCV DataFrame with random walk prices.

    Using a random walk ensures features are non-trivial (not constant).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    close = 100.0 * (1 + rng.normal(0, 0.01, n)).cumprod()
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.uniform(-0.005, 0.005, n)),
            "high": close * (1 + rng.uniform(0, 0.01, n)),
            "low": close * (1 - rng.uniform(0, 0.01, n)),
            "close": close,
            "volume": rng.integers(500_000, 2_000_000, n).astype(float),
            "ticker": "TEST",
        },
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def price_df():
    return make_price_df(n=100)


@pytest.fixture
def store(tmp_path):
    return FeatureStore(ENGINEERED_V1, processed_dir=tmp_path)


# ---------------------------------------------------------------------------
# Offline/online parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("feat", ENGINEERED_V1.features, ids=lambda f: f.name)
def test_offline_online_parity(feat: FeatureDefinition, price_df: pd.DataFrame):
    """
    For each feature, offline[T] must equal online(window ending at T).

    We test at multiple dates (T=50, 60, 70, 80) to avoid accidentally passing
    on a single lucky value.
    """
    offline_series = feat.offline(price_df)

    for t in [50, 60, 70, 80]:
        window_start = t - feat.required_window + 1
        window = price_df.iloc[window_start : t + 1]
        assert len(window) == feat.required_window

        online_val = feat.online(window)
        offline_val = offline_series.iloc[t]

        assert not math.isnan(offline_val), f"{feat.name}: offline value at t={t} is NaN"
        assert not math.isnan(online_val), f"{feat.name}: online value at t={t} is NaN"
        assert abs(online_val - offline_val) < 1e-6, (
            f"{feat.name} parity failed at t={t}: "
            f"offline={offline_val:.8f}, online={online_val:.8f}"
        )


# ---------------------------------------------------------------------------
# No forward leakage
# ---------------------------------------------------------------------------

def test_no_forward_leakage(price_df: pd.DataFrame, store: FeatureStore):
    """
    Features computed at row T must not depend on prices after row T.

    We verify this by computing features on df[:T+1] and df (full) and
    confirming the feature value at T is identical in both.

    If a feature accidentally uses future data (e.g. wrong shift direction),
    truncating the DataFrame changes its value at T.
    """
    T = 60
    features_full = store.compute_offline(price_df, "TEST")
    features_truncated = store.compute_offline(price_df.iloc[: T + 1], "TEST")

    date_at_T = price_df.index[T]

    for feat_name in store.feature_set.feature_names:
        val_full = features_full.loc[date_at_T, feat_name]
        val_truncated = features_truncated.loc[date_at_T, feat_name]
        assert abs(val_full - val_truncated) < 1e-9, (
            f"{feat_name}: leakage detected. "
            f"Value at T changed when future data was removed: "
            f"full={val_full:.8f}, truncated={val_truncated:.8f}"
        )


# ---------------------------------------------------------------------------
# Feature bounds
# ---------------------------------------------------------------------------

def test_rsi_bounds(price_df: pd.DataFrame):
    """RSI must always be in [0, 100]."""
    rsi_feat = next(f for f in ENGINEERED_V1.features if f.name == "rsi_14")
    series = rsi_feat.offline(price_df).dropna()
    assert (series >= 0).all(), "RSI has values below 0"
    assert (series <= 100).all(), "RSI has values above 100"


def test_volume_ratio_positive(price_df: pd.DataFrame):
    """Volume ratio must always be positive."""
    vol_feat = next(f for f in ENGINEERED_V1.features if f.name == "volume_ratio_20d")
    series = vol_feat.offline(price_df).dropna()
    assert (series > 0).all(), "Volume ratio has non-positive values"


# ---------------------------------------------------------------------------
# Feature store round-trip
# ---------------------------------------------------------------------------

def test_compute_offline_drops_nan_rows(price_df: pd.DataFrame, store: FeatureStore):
    """
    compute_offline should return no NaN values.
    Rows with insufficient history (first ~20 rows) are silently dropped.
    """
    features_df = store.compute_offline(price_df, "TEST")
    assert features_df.isnull().sum().sum() == 0, "compute_offline returned NaN values"
    assert len(features_df) < len(price_df), "Expected some rows dropped due to NaN"


def test_feature_store_round_trip(price_df: pd.DataFrame, tmp_path: Path):
    """compute_offline_all → load_offline gives back the same DataFrame."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    price_df.to_parquet(raw_dir / "TEST.parquet")

    store = FeatureStore(ENGINEERED_V1, processed_dir=tmp_path / "processed")
    store.compute_offline_all(tickers=["TEST"], raw_data_dir=raw_dir)
    loaded = store.load_offline("TEST")

    expected = store.compute_offline(price_df, "TEST")
    pd.testing.assert_frame_equal(loaded, expected, check_freq=False)


# ---------------------------------------------------------------------------
# Online path
# ---------------------------------------------------------------------------

def test_compute_online_returns_all_feature_names(price_df: pd.DataFrame, store: FeatureStore):
    """compute_online must return a dict with all expected feature names."""
    window = price_df.iloc[-ENGINEERED_V1.max_required_window :]
    result = store.compute_online(window)
    assert set(result.keys()) == set(ENGINEERED_V1.feature_names)


def test_compute_online_no_nan(price_df: pd.DataFrame, store: FeatureStore):
    """compute_online must not return NaN when given sufficient data."""
    window = price_df.iloc[-ENGINEERED_V1.max_required_window :]
    result = store.compute_online(window)
    for name, val in result.items():
        assert not math.isnan(val), f"compute_online returned NaN for {name}"


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def test_backfill_idempotent(price_df: pd.DataFrame, tmp_path: Path):
    """Running backfill twice over the same range gives identical results."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    price_df.to_parquet(raw_dir / "TEST.parquet")

    store = FeatureStore(ENGINEERED_V1, processed_dir=tmp_path / "processed")
    store.backfill(tickers=["TEST"], raw_data_dir=raw_dir)
    first = store.load_offline("TEST").copy()

    store.backfill(tickers=["TEST"], raw_data_dir=raw_dir)
    second = store.load_offline("TEST")

    pd.testing.assert_frame_equal(first, second)


def test_backfill_preserves_rows_outside_range(price_df: pd.DataFrame, tmp_path: Path):
    """
    Backfilling a sub-range must not delete rows outside that range.

    Scenario:
    - Run full compute_offline_all to populate the store
    - Backfill only the last 30 days
    - Rows before the backfill range must still be present
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    price_df.to_parquet(raw_dir / "TEST.parquet")

    store = FeatureStore(ENGINEERED_V1, processed_dir=tmp_path / "processed")
    store.compute_offline_all(tickers=["TEST"], raw_data_dir=raw_dir)
    full = store.load_offline("TEST")

    split_date = str(full.index[-30].date())
    store.backfill(tickers=["TEST"], start=split_date, raw_data_dir=raw_dir)
    after_backfill = store.load_offline("TEST")

    rows_before_split = full[full.index < split_date]
    assert len(after_backfill) >= len(rows_before_split), (
        "Backfill deleted rows outside the backfill range"
    )
