"""
Schema and data quality checks on raw ingested data.

Fail fast: raise immediately on any violation so bad data never reaches
the feature pipeline.
"""

import pandas as pd

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume", "ticker"}
MIN_ROWS_PER_TICKER = 100  # fewer rows than this indicates a bad fetch


def validate_raw(df: pd.DataFrame, ticker: str) -> None:
    """
    Validate a single-ticker raw DataFrame.

    Raises:
        ValueError: On any schema or quality violation
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"[{ticker}] Missing columns: {missing}")

    if len(df) < MIN_ROWS_PER_TICKER:
        raise ValueError(
            f"[{ticker}] Only {len(df)} rows — expected at least {MIN_ROWS_PER_TICKER}"
        )

    null_counts = df[list(REQUIRED_COLUMNS - {"ticker"})].isnull().sum()
    if null_counts.any():
        raise ValueError(f"[{ticker}] Null values found:\n{null_counts[null_counts > 0]}")

    if (df["close"] <= 0).any():
        raise ValueError(f"[{ticker}] Non-positive close prices found")

    if (df["volume"] < 0).any():
        raise ValueError(f"[{ticker}] Negative volume found")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"[{ticker}] Date index is not sorted ascending")
