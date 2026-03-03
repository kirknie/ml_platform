"""
Download historical OHLCV data and persist to Parquet.

Design decisions:
- One file per ticker: enables partial re-runs without full re-download
- Date range is explicit: forces callers to be intentional about data scope
- No silent failures: raises on bad tickers or empty downloads
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
DEFAULT_START = "2021-01-01"
DEFAULT_END = "2024-12-31"
RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def fetch_ticker(
    ticker: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    output_dir: Path = RAW_DATA_DIR,
) -> Path:
    """
    Download daily OHLCV data for a single ticker and save as Parquet.

    Args:
        ticker: Ticker symbol, e.g. "AAPL"
        start: Start date string, inclusive, format YYYY-MM-DD
        end: End date string, exclusive, format YYYY-MM-DD
        output_dir: Directory to write Parquet files

    Returns:
        Path to the written Parquet file

    Raises:
        ValueError: If download returns empty data
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}.parquet"

    logger.info("Fetching %s from %s to %s", ticker, start, end)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker {ticker!r} ({start} to {end})")

    # Normalize column names to lowercase for consistency
    df = raw.copy()
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"

    # Ensure date index is timezone-naive (yfinance sometimes returns tz-aware)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df["ticker"] = ticker

    df.to_parquet(output_path)
    logger.info("Wrote %d rows to %s", len(df), output_path)
    return output_path


def fetch_all(
    tickers: list[str] = TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    output_dir: Path = RAW_DATA_DIR,
) -> dict[str, Path]:
    """
    Fetch and persist data for all tickers.

    Returns:
        Mapping of ticker -> Parquet file path
    """
    results = {}
    for ticker in tickers:
        path = fetch_ticker(ticker, start=start, end=end, output_dir=output_dir)
        results[ticker] = path
    return results


def load_ticker(ticker: str, data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load a previously fetched ticker from Parquet.

    Raises:
        FileNotFoundError: If the ticker has not been fetched yet
    """
    path = data_dir / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No data file for {ticker!r}. Run fetch_ticker() first."
        )
    return pd.read_parquet(path)


def load_all(tickers: list[str] = TICKERS, data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load all tickers and concatenate into a single DataFrame sorted by (ticker, date).
    """
    frames = [load_ticker(t, data_dir) for t in tickers]
    df = pd.concat(frames).sort_values(["ticker", "date"])
    return df
