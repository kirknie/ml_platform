"""Tests for training/labels.py — focus on leakage correctness."""

import pandas as pd
import pytest

from training.labels import add_labels, add_labels_all_tickers, drop_unlabeled


def make_ticker_df(prices: list[float], ticker="AAPL") -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"close": prices, "ticker": ticker}, index=dates)


def test_label_is_1_when_future_price_higher():
    # close goes: 100, 100, 100, 100, 100, 110
    # label[0] = 1 because close[5] (110) > close[0] (100)
    df = make_ticker_df([100, 100, 100, 100, 100, 110])
    result = add_labels(df)
    assert result["label"].iloc[0] == 1


def test_label_is_0_when_future_price_lower():
    df = make_ticker_df([100, 100, 100, 100, 100, 90])
    result = add_labels(df)
    assert result["label"].iloc[0] == 0


def test_last_n_rows_have_nan_label():
    """The last forward_days rows cannot have a label — there's no future data."""
    df = make_ticker_df([100.0] * 20)
    result = add_labels(df, forward_days=5)
    # Last 5 rows should be NaN
    assert result["label"].iloc[-5:].isna().all()
    # Earlier rows should have labels
    assert result["label"].iloc[:-5].notna().all()


def test_labels_do_not_bleed_across_tickers():
    """
    Shift must be applied per ticker.
    If we shifted across the whole DataFrame, the last row of ticker A
    would pick up the first row of ticker B as its 'future' price.
    """
    # AAPL: prices go down (label should be 0 for first row)
    # MSFT: prices go up (if bleed occurs, AAPL's last row would see MSFT's price)
    aapl = make_ticker_df([100, 99, 98, 97, 96, 95], ticker="AAPL")
    msft = make_ticker_df([50, 51, 52, 53, 54, 999], ticker="MSFT")
    # Sort by (ticker, date) — as the pipeline does; sorting by price would scramble time order
    df = pd.concat([aapl, msft])
    df = df.sort_values(["ticker"]).sort_index(level=0, sort_remaining=True)

    result = add_labels_all_tickers(df)
    aapl_result = result[result["ticker"] == "AAPL"]

    # AAPL prices are always decreasing, so all valid labels should be 0
    valid_aapl_labels = drop_unlabeled(aapl_result)["label"]
    assert (valid_aapl_labels == 0).all(), "AAPL labels should all be 0 (prices falling)"


def test_drop_unlabeled_removes_nan_rows():
    df = make_ticker_df([100.0] * 20)
    labeled = add_labels(df, forward_days=5)
    cleaned = drop_unlabeled(labeled)
    assert cleaned["label"].notna().all()
    assert len(cleaned) == 15  # 20 - 5 = 15
