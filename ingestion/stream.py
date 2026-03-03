"""
Simulated streaming for historical OHLCV data.

Replays a ticker's Parquet file in time order, emitting one row at a time
with a configurable delay. Used to test the online serving path without a
real data feed.

Design decisions:
- Generator-based: caller controls consumption rate; no internal threading
- Configurable delay: set to 0 for tests, >0 to simulate real-time arrival
- Yields plain dicts: easy to consume in the serving layer without a DataFrame dependency
"""

import time
from collections.abc import Generator
from pathlib import Path

import pandas as pd

from ingestion.fetch import RAW_DATA_DIR, load_ticker


def stream_ticker(
    ticker: str,
    delay_seconds: float = 0.0,
    data_dir: Path = RAW_DATA_DIR,
) -> Generator[dict, None, None]:
    """
    Replay a ticker's historical data row by row in time order.

    Args:
        ticker: Ticker symbol, e.g. "AAPL"
        delay_seconds: Pause between emitted rows. Set to 0 for tests.
        data_dir: Directory containing Parquet files from fetch.py

    Yields:
        Dict with keys: date, open, high, low, close, volume, ticker

    Raises:
        FileNotFoundError: If the ticker has not been fetched yet
    """
    df = load_ticker(ticker, data_dir)

    for date, row in df.iterrows():
        yield {
            "date": date,
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
            "ticker": row["ticker"],
        }
        if delay_seconds > 0:
            time.sleep(delay_seconds)


def stream_ticker_window(
    ticker: str,
    window_size: int,
    delay_seconds: float = 0.0,
    data_dir: Path = RAW_DATA_DIR,
) -> Generator[pd.DataFrame, None, None]:
    """
    Replay a ticker's historical data as a rolling window of rows.

    At each step, yields the last `window_size` rows up to and including
    the current row. This is the format expected by the online feature store:
    features at time T require the preceding N rows of price history.

    Args:
        ticker: Ticker symbol
        window_size: Number of rows in each emitted window (e.g. 20 for a 20-day rolling feature)
        delay_seconds: Pause between emitted windows
        data_dir: Directory containing Parquet files

    Yields:
        DataFrame of shape (window_size, columns), sorted by date ascending.
        Rows where fewer than window_size rows are available are skipped.
    """
    df = load_ticker(ticker, data_dir)

    for i in range(window_size, len(df) + 1):
        yield df.iloc[i - window_size : i].copy()
        if delay_seconds > 0:
            time.sleep(delay_seconds)
