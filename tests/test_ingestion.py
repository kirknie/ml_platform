"""Tests for ingestion/fetch.py and ingestion/validate.py"""

import pandas as pd
import pytest

from ingestion.validate import validate_raw


def make_valid_df(n=200) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000,
            "ticker": "AAPL",
        },
        index=dates,
    )


def test_validate_raw_passes_valid_data():
    df = make_valid_df()
    validate_raw(df, "AAPL")  # should not raise


def test_validate_raw_fails_on_missing_column():
    df = make_valid_df().drop(columns=["volume"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_raw(df, "AAPL")


def test_validate_raw_fails_on_too_few_rows():
    df = make_valid_df(n=50)
    with pytest.raises(ValueError, match="Only 50 rows"):
        validate_raw(df, "AAPL")


def test_validate_raw_fails_on_null_values():
    df = make_valid_df()
    df.loc[df.index[5], "close"] = None
    with pytest.raises(ValueError, match="Null values"):
        validate_raw(df, "AAPL")


def test_validate_raw_fails_on_non_positive_close():
    df = make_valid_df()
    df.loc[df.index[0], "close"] = -1.0
    with pytest.raises(ValueError, match="Non-positive close"):
        validate_raw(df, "AAPL")


def test_stream_ticker_yields_all_rows(tmp_path):
    """stream_ticker should yield one dict per row in the Parquet file."""
    from ingestion.stream import stream_ticker

    dates = pd.date_range("2021-01-01", periods=10, freq="B")
    df = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100, "ticker": "AAPL"},
        index=dates,
    )
    df.index.name = "date"
    df.to_parquet(tmp_path / "AAPL.parquet")

    rows = list(stream_ticker("AAPL", delay_seconds=0, data_dir=tmp_path))
    assert len(rows) == 10
    assert set(rows[0].keys()) == {"date", "open", "high", "low", "close", "volume", "ticker"}
    assert rows[0]["ticker"] == "AAPL"


def test_stream_ticker_window_yields_correct_shape(tmp_path):
    """Each window should have exactly window_size rows, and windows should be in time order."""
    from ingestion.stream import stream_ticker_window

    dates = pd.date_range("2021-01-01", periods=25, freq="B")
    df = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100, "ticker": "AAPL"},
        index=dates,
    )
    df.index.name = "date"
    df.to_parquet(tmp_path / "AAPL.parquet")

    windows = list(stream_ticker_window("AAPL", window_size=5, delay_seconds=0, data_dir=tmp_path))
    # 25 rows, window_size=5 → 21 windows
    assert len(windows) == 21
    for w in windows:
        assert len(w) == 5
    # Each window's last date should advance by one day
    assert windows[1].index[-1] > windows[0].index[-1]
