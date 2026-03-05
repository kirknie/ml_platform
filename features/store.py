"""
Feature store: offline batch computation, online serving, and incremental backfill.

Storage layout:
    data/processed/{feature_set.name}_v{feature_set.version}/{ticker}.parquet

Offline path (training):
    1. load_ticker() → raw OHLCV DataFrame
    2. compute_offline() → apply each feature's offline function → Parquet

Online path (serving):
    1. stream_ticker_window() yields a rolling window DataFrame
    2. compute_online(window) → apply each feature's online function → dict

Backfill (incremental update):
    1. compute_offline() for full raw history
    2. Filter to requested date range
    3. If Parquet exists: load, concat, deduplicate (keep latest), sort, save
    4. If new: just save

Design decisions:
- One Parquet file per ticker: matches raw data layout, enables per-ticker re-runs
- dropna() in compute_offline: features with large windows (e.g. 20-day vol) produce
  NaN for the first ~20 rows — these are silently dropped. Callers never see NaN features.
- Backfill merges with keep="last" on duplicates: re-running a backfill over an existing
  range overwrites, making the operation idempotent
"""

import logging
from pathlib import Path

import pandas as pd

from features.definitions import FeatureSet
from ingestion.fetch import RAW_DATA_DIR, TICKERS, load_ticker

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


class FeatureStore:
    def __init__(
        self,
        feature_set: FeatureSet,
        processed_dir: Path = PROCESSED_DATA_DIR,
    ) -> None:
        self.feature_set = feature_set
        self.processed_dir = processed_dir

    @property
    def store_dir(self) -> Path:
        """Directory for this feature set's Parquet files."""
        return self.processed_dir / f"{self.feature_set.name}_v{self.feature_set.version}"

    # ------------------------------------------------------------------
    # Offline path
    # ------------------------------------------------------------------

    def compute_offline(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Compute all features for a single-ticker DataFrame.

        Args:
            df: Raw OHLCV DataFrame for one ticker, sorted by date ascending
            ticker: Ticker symbol (written as a column for join convenience)

        Returns:
            DataFrame of feature values, NaN rows dropped.
            Index is the date index from df.
        """
        result = pd.DataFrame(index=df.index)
        for feat in self.feature_set.features:
            result[feat.name] = feat.offline(df)
        result["ticker"] = ticker
        result = result.dropna()
        logger.debug(
            "compute_offline %s: %d rows (dropped %d NaN)",
            ticker,
            len(result),
            len(df) - len(result),
        )
        return result

    def compute_offline_all(
        self,
        tickers: list[str] = TICKERS,
        raw_data_dir: Path = RAW_DATA_DIR,
    ) -> dict[str, Path]:
        """
        Compute and persist features for all tickers.

        Returns:
            Mapping of ticker -> Parquet file path
        """
        self.store_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for ticker in tickers:
            df = load_ticker(ticker, raw_data_dir)
            features_df = self.compute_offline(df, ticker)
            path = self.store_dir / f"{ticker}.parquet"
            features_df.to_parquet(path)
            logger.info(
                "Saved %d rows of %s features for %s → %s",
                len(features_df),
                self.feature_set.name,
                ticker,
                path,
            )
            paths[ticker] = path
        return paths

    def load_offline(self, ticker: str) -> pd.DataFrame:
        """
        Load precomputed features for one ticker.

        Raises:
            FileNotFoundError: If compute_offline_all() has not been run yet
        """
        path = self.store_dir / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No features for {ticker!r} at {path}. Run compute_offline_all() first."
            )
        return pd.read_parquet(path)

    def load_offline_all(self, tickers: list[str] = TICKERS) -> pd.DataFrame:
        """Load all tickers and concatenate, sorted by (ticker, date)."""
        frames = [self.load_offline(t) for t in tickers]
        return pd.concat(frames).sort_values(["ticker", "date"])

    # ------------------------------------------------------------------
    # Online path (used by serving layer in Week 8)
    # ------------------------------------------------------------------

    def compute_online(self, window: pd.DataFrame) -> dict[str, float]:
        """
        Compute features for a single rolling window.

        Args:
            window: DataFrame with the last N rows of OHLCV data for one ticker,
                    sorted by date ascending. N >= feature_set.max_required_window.

        Returns:
            Dict mapping feature name -> scalar value for the last row in the window.
        """
        return {feat.name: feat.online(window) for feat in self.feature_set.features}

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    def backfill(
        self,
        tickers: list[str] = TICKERS,
        start: str | None = None,
        end: str | None = None,
        raw_data_dir: Path = RAW_DATA_DIR,
    ) -> dict[str, Path]:
        """
        Compute features for [start, end] and merge into existing Parquet.

        If the Parquet already exists, new rows are merged in. Rows within the
        backfill range that already exist are overwritten (idempotent).
        Rows outside the backfill range are preserved.

        Args:
            tickers: Tickers to backfill
            start: Start date inclusive (YYYY-MM-DD), or None for beginning of history
            end: End date inclusive (YYYY-MM-DD), or None for end of history

        Returns:
            Mapping of ticker -> Parquet file path
        """
        self.store_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        for ticker in tickers:
            df_raw = load_ticker(ticker, raw_data_dir)
            new_features = self.compute_offline(df_raw, ticker)

            if start is not None:
                new_features = new_features.loc[new_features.index >= start]
            if end is not None:
                new_features = new_features.loc[new_features.index <= end]

            if new_features.empty:
                raise ValueError(
                    f"No features for {ticker!r} in range {start} to {end} after filtering"
                )

            path = self.store_dir / f"{ticker}.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, new_features])
                combined = (
                    combined[~combined.index.duplicated(keep="last")]
                    .sort_index()
                )
            else:
                combined = new_features.sort_index()

            combined.to_parquet(path)
            logger.info(
                "Backfilled %d rows for %s (total: %d rows)",
                len(new_features),
                ticker,
                len(combined),
            )
            paths[ticker] = path

        return paths
