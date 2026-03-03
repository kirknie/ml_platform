"""
Label generation for the price direction prediction task.

Task: Binary classification — will close price be higher N trading days from now?

Leakage contract:
    label[t] = 1 if close[t+N] > close[t] else 0

    The label at row t depends on close[t+N], which is future data.
    Therefore:
    - label[t] is ONLY used as the target variable, never as a feature
    - The last N rows per ticker have NaN labels and are dropped before training
    - The train/test split respects time order — no shuffling across the split boundary
"""

import pandas as pd

FORWARD_DAYS = 5  # predict N-day forward direction


def add_labels(df: pd.DataFrame, forward_days: int = FORWARD_DAYS) -> pd.DataFrame:
    """
    Add a binary direction label to a single-ticker DataFrame.

    Args:
        df: DataFrame with 'close' column, sorted by date ascending
        forward_days: How many trading days ahead to predict

    Returns:
        DataFrame with added 'label' column (1 = up, 0 = flat/down).
        Rows where the future close is not available have NaN label.
    """
    df = df.copy()
    future_close = df["close"].shift(-forward_days)
    # Compute direction only where future close is known
    has_future = future_close.notna()
    label = pd.array([pd.NA] * len(df), dtype="Int64")
    label[has_future.values] = (future_close[has_future] > df["close"][has_future]).astype(int).values
    df["label"] = label
    return df


def add_labels_all_tickers(
    df: pd.DataFrame, forward_days: int = FORWARD_DAYS
) -> pd.DataFrame:
    """
    Apply label generation per ticker (shift must not bleed across tickers).

    Args:
        df: Multi-ticker DataFrame with 'ticker' and 'close' columns,
            sorted by (ticker, date)

    Returns:
        DataFrame with 'label' column added
    """
    return (
        df.groupby("ticker", group_keys=False)
        .apply(lambda g: add_labels(g, forward_days), include_groups=False)
        .assign(ticker=lambda d: d["ticker"] if "ticker" in d.columns else df["ticker"])
    )


def drop_unlabeled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with no valid label (the last N rows of each ticker).
    """
    return df.dropna(subset=["label"]).copy()
