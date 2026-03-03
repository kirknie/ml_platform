# Week 5 Execution Plan: Data & Baseline

## Goal

By end of week 5 you have:
- A working data ingestion pipeline that fetches and stores historical OHLCV data as Parquet
- A label definition with no lookahead leakage
- A baseline XGBoost model trained on simple features
- Evaluation metrics logged to MLflow
- A test suite covering ingestion and labeling correctness

---

## Repo Structure to Create

```
ml_platform/
├── ingestion/
│   ├── __init__.py
│   ├── fetch.py          # Download raw OHLCV data
│   └── validate.py       # Schema and quality checks on raw data
├── features/
│   ├── __init__.py
│   └── baseline.py       # Simple features for the baseline model only (no feature store yet)
├── training/
│   ├── __init__.py
│   ├── labels.py         # Label generation with leakage-safe logic
│   ├── pipeline.py       # End-to-end: load → features → train → evaluate → log
│   └── evaluation.py     # Metric computation helpers
├── data/
│   └── raw/              # Raw Parquet files land here (gitignored)
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   └── test_labels.py
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Step 1: Environment Setup

### 1.1 Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart your shell or run: source $HOME/.local/bin/env
uv --version  # confirm it's available
```

### 1.2 Initialize the project with uv

```bash
cd /Users/dingnie/Documents/git_repo/ml_platform
uv init --no-workspace
uv python pin 3.13
```

`uv init` creates a minimal `pyproject.toml` with the project name and Python version pre-filled.
`uv python pin 3.13` writes a `.python-version` file that locks the interpreter for this project.

### 1.3 Create and activate the virtual environment

```bash
uv venv
source .venv/bin/activate
```

### 1.4 Add dependencies via uv

Run these commands and uv will add each package to `pyproject.toml`, resolve `uv.lock`, and install into the active venv automatically:

```bash
# Production dependencies
uv add yfinance pandas pyarrow scikit-learn xgboost mlflow structlog prometheus-client

# Dev dependencies (added to the [dependency-groups] dev group)
uv add --dev pytest pytest-cov ruff
```

Also add the ruff config to `pyproject.toml` manually (uv doesn't manage tool config):

```toml
[tool.ruff]
line-length = 100
target-version = "py313"
```

### 1.5 Create `.gitignore`

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/raw/
data/processed/
mlruns/
*.egg-info/
.ruff_cache/
```

Commit `uv.lock` to the repo — it pins exact dependency versions and ensures reproducible installs:
```bash
git add uv.lock pyproject.toml .python-version
```

**Optional: Link to GitHub**
```bash
# Create repo on GitHub first, then:
git remote add origin git@github.com:kirknie/ml_platform.git
git branch -M main
git config --local user.name "kirknie"
git config --local user.email "kirknie@gmail.com"
git config --local commit.gpgsign false
git push -u origin main
```

---

## Step 2: Data Ingestion

### What we're building

`ingestion/fetch.py` downloads daily OHLCV data for a configurable list of tickers using `yfinance` and writes one Parquet file per ticker to `data/raw/`.

### Why Parquet?

- Columnar format — fast for feature computation over many rows
- Schema-preserving — no type ambiguity vs CSV
- Append-friendly — you can write new files per date range and combine later

### 2.1 Create `ingestion/__init__.py` (empty)

### 2.2 Create `ingestion/fetch.py`

```python
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
DEFAULT_START = "2018-01-01"
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
```

### 2.3 Create `ingestion/validate.py`

```python
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
```

---

## Step 3: Label Definition

### The leakage problem — read this carefully

A label defined as "did price go up?" must only use information available **at the time of prediction**. Common mistakes:

- Using `close[t+1]` as a feature (future price as input)
- Computing rolling statistics that include `t+1` or later
- Fitting a scaler on the full dataset before train/test split

### Our label: 5-day forward return binary classification

- At each date `t`, label = 1 if `close[t+5] > close[t]`, else 0
- The label for date `t` is computed using `t+5` close price, which is future data
- **This means the last 5 rows of each ticker have no valid label and must be dropped**
- Features for predicting at date `t` must only use data from `t` and earlier

### 3.1 Create `training/__init__.py` (empty)

### 3.2 Create `training/labels.py`

```python
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
    df["label"] = (future_close > df["close"]).astype("Int64")  # nullable int
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
        .apply(lambda g: add_labels(g, forward_days))
    )


def drop_unlabeled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with no valid label (the last N rows of each ticker).
    """
    return df.dropna(subset=["label"]).copy()
```

---

## Step 4: Baseline Features

### Philosophy

These are intentionally simple. The goal is a working end-to-end pipeline, not a good model. Complexity gets added in the feature store (Week 6).

### Features we'll compute

| Feature | Formula | Why no leakage |
|---|---|---|
| `return_1d` | `(close[t] / close[t-1]) - 1` | Only uses past data |
| `return_5d` | `(close[t] / close[t-5]) - 1` | Only uses past data |
| `return_20d` | `(close[t] / close[t-20]) - 1` | Only uses past data |
| `vol_20d` | Rolling 20-day std of daily returns | Only uses past data |
| `rsi_14` | 14-day RSI | Only uses past data |
| `volume_ratio` | `volume[t] / rolling_mean_volume_20d` | Only uses past data |

### 4.1 Create `features/__init__.py` (empty)

### 4.2 Create `features/baseline.py`

```python
"""
Baseline feature set for Week 5.

All features are point-in-time safe: feature[t] uses only close/volume data
from day t and earlier. No future data is used.

This module is intentionally simple. The Week 6 feature store will replace it
with versioned, registered feature definitions.
"""

import pandas as pd


def add_return(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    df[f"return_{window}d"] = df["close"].pct_change(window)
    return df


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    daily_returns = df["close"].pct_change()
    df[f"vol_{window}d"] = daily_returns.rolling(window).std()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index."""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, float("nan"))
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return df


def add_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    avg_volume = df["volume"].rolling(window).mean()
    df["volume_ratio"] = df["volume"] / avg_volume
    return df


def compute_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all baseline features to a single-ticker DataFrame.

    All rolling operations use only historical data (no center=True, no future).
    Rows with insufficient history for rolling windows will have NaN — these are
    dropped later before training.

    Args:
        df: Single-ticker DataFrame sorted by date ascending

    Returns:
        DataFrame with feature columns added
    """
    df = add_return(df, 1)
    df = add_return(df, 5)
    df = add_return(df, 20)
    df = add_volatility(df, 20)
    df = add_rsi(df, 14)
    df = add_volume_ratio(df, 20)
    return df


def compute_baseline_features_all_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply baseline features per ticker (rolling windows must not bleed across tickers).
    """
    return (
        df.groupby("ticker", group_keys=False)
        .apply(compute_baseline_features)
    )


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "vol_20d",
    "rsi_14",
    "volume_ratio",
]
```

---

## Step 5: Training Pipeline

### 5.1 Create `training/evaluation.py`

```python
"""
Evaluation metric helpers.

Using classification metrics because the task is binary direction prediction.
Accuracy alone is misleading on imbalanced data — always report precision,
recall, and AUC alongside it.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)


def compute_metrics(y_true: pd.Series, y_pred, y_prob=None) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities for class 1 (optional, for AUC)

    Returns:
        Dict of metric name -> value
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Happens if only one class present in y_true (e.g. tiny test sets)
            metrics["roc_auc"] = float("nan")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["precision_class1"] = report.get("1", {}).get("precision", 0.0)
    metrics["recall_class1"] = report.get("1", {}).get("recall", 0.0)
    metrics["f1_class1"] = report.get("1", {}).get("f1-score", 0.0)
    metrics["label_balance"] = float(y_true.mean())

    return metrics
```

### 5.2 Create `training/pipeline.py`

```python
"""
End-to-end baseline training pipeline.

Steps:
1. Load raw data
2. Add labels
3. Compute features
4. Time-based train/test split (no shuffling — respects temporal ordering)
5. Train XGBoost classifier
6. Evaluate on test set
7. Log everything to MLflow

Run this script directly:
    python -m training.pipeline
"""

import logging
from pathlib import Path

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from features.baseline import FEATURE_COLUMNS, compute_baseline_features_all_tickers
from ingestion.fetch import TICKERS, load_all
from ingestion.validate import validate_raw
from training.evaluation import compute_metrics
from training.labels import add_labels_all_tickers, drop_unlabeled

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "ml_platform_baseline"
TEST_SPLIT_DATE = "2023-01-01"  # everything before this is train, after is test


def build_dataset() -> pd.DataFrame:
    """
    Load raw data, add labels and features, drop rows with NaN.

    Returns:
        Clean DataFrame ready for train/test split
    """
    logger.info("Loading raw data for tickers: %s", TICKERS)
    df = load_all(TICKERS)

    for ticker in TICKERS:
        ticker_df = df[df["ticker"] == ticker]
        validate_raw(ticker_df, ticker)

    logger.info("Adding labels (5-day forward direction)")
    df = add_labels_all_tickers(df)

    logger.info("Computing baseline features")
    df = compute_baseline_features_all_tickers(df)

    # Drop rows where label or any feature is NaN
    # These arise from: rolling window warmup period + last N rows (no label)
    before = len(df)
    df = drop_unlabeled(df)
    df = df.dropna(subset=FEATURE_COLUMNS)
    logger.info("Dropped %d rows with NaN (rolling warmup + unlabeled tail)", before - len(df))

    return df


def time_split(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/test by date.

    IMPORTANT: We do NOT shuffle. Shuffling would cause future data to appear
    in the training set when rows near the split boundary are interleaved.
    This is a form of temporal leakage.
    """
    train = df[df.index < split_date]
    test = df[df.index >= split_date]
    logger.info("Train: %d rows, Test: %d rows (split at %s)", len(train), len(test), split_date)
    return train, test


def train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
) -> tuple:
    """
    Train XGBoost classifier and return (model, metrics).
    """
    X_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["label"].astype(int)

    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    return model, metrics, params


def run_pipeline() -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="baseline_xgboost"):
        df = build_dataset()
        train_df, test_df = time_split(df, TEST_SPLIT_DATE)
        model, metrics, params = train(train_df, test_df)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_params({
            "tickers": ",".join(TICKERS),
            "test_split_date": TEST_SPLIT_DATE,
            "feature_columns": ",".join(FEATURE_COLUMNS),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        })
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")

        logger.info("Metrics: %s", metrics)
        print("\n=== Baseline Results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
```

---

## Step 6: Tests

### 6.1 Create `tests/__init__.py` (empty)

### 6.2 Create `tests/test_ingestion.py`

```python
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
```

### 6.3 Create `tests/test_labels.py`

```python
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
    df = pd.concat([aapl, msft]).sort_values(["ticker", "close"])

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
```

---

## Step 7: Running Everything

### 7.1 Run tests first

```bash
cd /Users/dingnie/Documents/git_repo/ml_platform
uv run pytest tests/ -v
```

All tests should pass before you fetch real data.

### 7.2 Fetch data

```bash
uv run python -c "
from ingestion.fetch import fetch_all
import logging
logging.basicConfig(level=logging.INFO)
fetch_all()
"
```

This writes 5 Parquet files to `data/raw/`. Expect ~1700 rows per ticker (2018–2024 trading days).

### 7.3 Run training pipeline

```bash
uv run python -m training.pipeline
```

Expected output:
```
=== Baseline Results ===
  accuracy: ~0.52–0.54  (slightly above random — expected for simple features)
  roc_auc:  ~0.52–0.56
  precision_class1: ~0.52
  recall_class1:    ~0.50
  f1_class1:        ~0.51
  label_balance:    ~0.52  (roughly balanced — markets go up more than down)
```

**Do not chase accuracy.** A baseline near 52% is correct. If you see 70%+ something is wrong (likely leakage).

### 7.4 View MLflow UI

```bash
uv run mlflow ui --port 5000
# Open http://localhost:5000
```

You should see one run logged under the `ml_platform_baseline` experiment with all params and metrics.

---

## Step 8: Validation Checklist

Before calling Week 5 done, verify each item:

- [ ] `pytest tests/ -v` — all tests pass
- [ ] `data/raw/` contains 5 `.parquet` files, one per ticker
- [ ] Each Parquet file has columns: `open, high, low, close, volume, ticker`
- [ ] `python -m training.pipeline` runs without errors
- [ ] MLflow UI shows the baseline run with metrics logged
- [ ] `roc_auc` is between 0.50 and 0.60 (anything higher suggests leakage)
- [ ] `label_balance` is between 0.48 and 0.58 (sanity check on label distribution)
- [ ] No `FutureWarning` or `DeprecationWarning` printed during training

---

## Common Pitfalls

### "My accuracy is 70%+"
Almost certainly leakage. Check:
1. Is the label column accidentally included as a feature?
2. Is `close[t+5]` (or any shifted-forward column) included in features?
3. Was the train/test split done by shuffling instead of by date?

### "All my labels are NaN after labeling"
The `shift(-5)` works per-group. Make sure `add_labels_all_tickers` is called on a DataFrame sorted by `(ticker, date)` and that the groupby is on `ticker`.

### "yfinance returns different column casing"
The fetch code lowercases all columns. If you load Parquet directly and see uppercase column names, you're loading from a different path than expected.

### "MLflow can't find the experiment"
MLflow stores runs in `./mlruns/` relative to where you run the script. Always run from the project root (`ml_platform/`).

---

## Week 5 → Week 6 Handoff

At the end of Week 5 you have a working but naive pipeline. Week 6 replaces `features/baseline.py` with a proper feature store:

- Features will be **registered** with metadata (name, version, description)
- Features will have **offline** (batch Parquet) and **online** (point-in-time dict) compute paths that use the same underlying logic
- Feature sets will be **versioned** so experiments reference exact feature definitions
- Backfill will be supported — recompute features for any date range

The key design challenge for Week 6: the same Python function must work both when given a full historical DataFrame (offline) and when given a single row of current data (online). Plan for this interface early.
