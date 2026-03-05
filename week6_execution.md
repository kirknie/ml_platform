# Week 6 Execution Plan: Feature Store

## Goal

By end of week 6 you have:
- A custom feature store with engineered features (returns, volatility, RSI, volume ratio)
- **Offline/online parity**: the same feature logic runs in batch training and single-row serving
- **Versioned feature sets**: features are grouped, named, and versioned so experiments reference exact definitions
- **Incremental backfill**: recompute and merge features for any date range without rewriting history
- An updated training pipeline that compares baseline (raw OHLCV) vs engineered features in MLflow

---

## Why This Architecture Matters (Read First)

The hardest problem in ML systems is **training/serving skew**: features computed at training time differ from features computed at serving time, causing silent accuracy degradation in production. This happens constantly in real systems.

The fix: **define each feature exactly once**. The feature store exposes two compute paths — `offline` (vectorized pandas for batch training) and `online` (window-based for single-row serving) — both backed by the same underlying definition. A parity test ensures they agree numerically.

The second problem is **reproducibility**: when you retrain 6 months from now, which version of the features did this experiment use? Versioned feature sets solve this — each MLflow run records the feature set name and version.

---

## Repo Changes

New files:
```
features/
├── definitions.py     # FeatureDefinition, FeatureSet, ENGINEERED_V1
└── store.py           # FeatureStore class (offline + online + backfill)
tests/
└── test_feature_store.py   # parity, leakage, backfill, bounds tests
```

Modified files:
```
training/pipeline.py   # add build_dataset_from_store() + run_engineered_pipeline()
```

Data layout after Week 6:
```
data/
├── raw/
│   ├── AAPL.parquet       # from Week 5 (raw OHLCV)
│   └── ...
└── processed/
    └── engineered_v1/     # new this week
        ├── AAPL.parquet   # engineered features, date-indexed
        └── ...
```

---

## Step 1: Feature Definitions

### The core design decision: two compute paths, one definition

Each feature is defined as a `FeatureDefinition` with:
- `offline`: a function `(full_ticker_df) → pd.Series` — vectorized, fast, used for batch training
- `online`: a function `(window_df) → float` — takes the last N rows, used for single-row serving
- `required_window`: minimum number of rows `online` needs to be valid

A `FeatureSet` groups features together with a name and version number. The name+version is logged to MLflow alongside model weights, giving you a full record of what features produced each experiment.

### 1.1 Create `features/definitions.py`

```python
"""
Feature definitions for the ML platform feature store.

Each FeatureDefinition has two compute paths:
- offline: vectorized pandas transform over a full ticker DataFrame (used at training time)
- online:  scalar computation over a rolling window DataFrame (used at serving time)

Both paths must produce the same value at time T given the same underlying price history.
The parity test in test_feature_store.py enforces this numerically.

Design decisions:
- Offline uses pandas rolling/shift for speed over large DataFrames
- Online takes a window DataFrame so the serving layer controls how much history it provides
- required_window is the minimum rows needed for online to return a valid (non-NaN) result
"""

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass
class FeatureDefinition:
    name: str
    offline: Callable[[pd.DataFrame], pd.Series]  # full ticker df → series indexed by date
    online: Callable[[pd.DataFrame], float]         # window df (last N rows) → scalar for last row
    required_window: int                             # min rows needed for online path


@dataclass
class FeatureSet:
    name: str
    version: int
    features: list[FeatureDefinition] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def max_required_window(self) -> int:
        """The largest window any feature needs — used to size stream windows for serving."""
        return max(f.required_window for f in self.features)


# ---------------------------------------------------------------------------
# RSI helpers — extracted because the rolling-mean RSI formula is the same
# in both compute paths; only the input shape differs
# ---------------------------------------------------------------------------

def _rsi_from_closes(closes: pd.Series, period: int = 14) -> float:
    """
    Compute RSI from a Series of close prices (length >= period + 1).

    Uses simple rolling mean (not Wilder's EMA) — sufficient for ML features.
    Returns float in [0, 100], or NaN if insufficient data.
    """
    delta = closes.diff().dropna()
    if len(delta) < period:
        return float("nan")
    recent = delta.tail(period)
    avg_gain = recent.clip(lower=0).mean()
    avg_loss = (-recent).clip(lower=0).mean()
    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


# ---------------------------------------------------------------------------
# Engineered feature set v1
# Features: 1-day return, 5-day return, 5-day volatility, 20-day volatility,
#           14-day RSI, 20-day volume ratio
# ---------------------------------------------------------------------------

ENGINEERED_V1 = FeatureSet(
    name="engineered",
    version=1,
    features=[
        FeatureDefinition(
            name="return_1d",
            offline=lambda df: df["close"].pct_change(1),
            online=lambda w: (w["close"].iloc[-1] - w["close"].iloc[-2]) / w["close"].iloc[-2],
            required_window=2,
        ),
        FeatureDefinition(
            name="return_5d",
            offline=lambda df: df["close"].pct_change(5),
            online=lambda w: (w["close"].iloc[-1] - w["close"].iloc[-6]) / w["close"].iloc[-6],
            required_window=6,
        ),
        FeatureDefinition(
            name="volatility_5d",
            # std of 5 daily returns — rolling(5) on pct_change gives std of [t-4..t]
            offline=lambda df: df["close"].pct_change().rolling(5).std(),
            online=lambda w: w["close"].pct_change().dropna().tail(5).std(),
            required_window=6,  # 6 rows → 5 diffs → std of 5
        ),
        FeatureDefinition(
            name="volatility_20d",
            offline=lambda df: df["close"].pct_change().rolling(20).std(),
            online=lambda w: w["close"].pct_change().dropna().tail(20).std(),
            required_window=21,  # 21 rows → 20 diffs → std of 20
        ),
        FeatureDefinition(
            name="rsi_14",
            offline=lambda df: pd.Series(
                [_rsi_from_closes(df["close"].iloc[: i + 1]) for i in range(len(df))],
                index=df.index,
            ),
            online=lambda w: _rsi_from_closes(w["close"]),
            required_window=15,  # 15 rows → 14 diffs → RSI over 14 periods
        ),
        FeatureDefinition(
            name="volume_ratio_20d",
            offline=lambda df: df["volume"] / df["volume"].rolling(20).mean(),
            online=lambda w: w["volume"].iloc[-1] / w["volume"].tail(20).mean(),
            required_window=20,
        ),
    ],
)
```

### 1.2 Verify the definitions import cleanly

```bash
uv run python -c "
from features.definitions import ENGINEERED_V1
print('Feature set:', ENGINEERED_V1.name, 'v' + str(ENGINEERED_V1.version))
print('Features:', ENGINEERED_V1.feature_names)
print('Max required window:', ENGINEERED_V1.max_required_window)
"
```

Expected output:
```
Feature set: engineered v1
Features: ['return_1d', 'return_5d', 'volatility_5d', 'volatility_20d', 'rsi_14', 'volume_ratio_20d']
Max required window: 21
```

---

## Step 2: Feature Store

### What it does

`FeatureStore` is the runtime class that binds a `FeatureSet` to storage. It:
- Computes offline features for all tickers and saves to `data/processed/{name}_v{version}/`
- Loads precomputed features back
- Computes features online from a rolling window (the path used during serving in Week 8)
- Backfills a date range — compute new features and merge into existing Parquet without rewriting history that hasn't changed

### 2.1 Create `features/store.py`

```python
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

from features.definitions import FeatureDefinition, FeatureSet
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
        logger.debug("compute_offline %s: %d rows (dropped %d NaN)", ticker, len(result), len(df) - len(result))
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
            logger.info("Saved %d rows of %s features for %s → %s",
                        len(features_df), self.feature_set.name, ticker, path)
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

            # Filter to requested date range
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
                # Deduplicate: if a date appears in both, keep the new (last) version
                combined = (
                    combined[~combined.index.duplicated(keep="last")]
                    .sort_index()
                )
            else:
                combined = new_features.sort_index()

            combined.to_parquet(path)
            logger.info("Backfilled %d rows for %s (total: %d rows)", len(new_features), ticker, len(combined))
            paths[ticker] = path

        return paths
```

### 2.2 Verify the store computes features

First confirm the raw data is in place (from Week 5):

```bash
ls data/raw/
# Should show: AAPL.parquet  GOOGL.parquet  MSFT.parquet  NVDA.parquet  TSLA.parquet
```

Then compute offline features:

```bash
uv run python -c "
from features.definitions import ENGINEERED_V1
from features.store import FeatureStore
import logging
logging.basicConfig(level=logging.INFO)

store = FeatureStore(ENGINEERED_V1)
paths = store.compute_offline_all()
for ticker, path in paths.items():
    print(ticker, path)
"
```

Spot-check one ticker:

```bash
uv run python -c "
from features.definitions import ENGINEERED_V1
from features.store import FeatureStore
store = FeatureStore(ENGINEERED_V1)
df = store.load_offline('AAPL')
print(df.shape)
print(df.columns.tolist())
print(df.head(3))
print(df.tail(3))
"
```

Expected: ~980 rows (raw ~1000, minus ~20 NaN rows from the 20-day window features), 7 columns (`return_1d`, `return_5d`, `volatility_5d`, `volatility_20d`, `rsi_14`, `volume_ratio_20d`, `ticker`).

---

## Step 3: Feature Store Tests

### The most important test: offline/online parity

The parity test proves the two compute paths agree. If they diverge, features at training time are different from features at serving time — silent skew.

For each feature, the test:
1. Computes offline features for a known DataFrame
2. Picks a date T in the middle of the DataFrame
3. Extracts the window ending at T (last `required_window` rows)
4. Calls `feat.online(window)` and compares to `offline_series[T]`

### 3.1 Create `tests/test_feature_store.py`

```python
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

import pandas as pd
import pytest

from features.definitions import ENGINEERED_V1, FeatureDefinition, FeatureSet
from features.store import FeatureStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_price_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Create a realistic fake OHLCV DataFrame with random walk prices.

    Using a random walk ensures features are non-trivial (not constant).
    """
    import numpy as np
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
        # Extract window ending at row t (inclusive), sized to required_window
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

    We verify this by computing features on df[:T+1] and df[:T+5] and
    confirming the feature value at T is identical in both.

    If a feature accidentally uses future data (e.g. wrong shift direction),
    truncating the DataFrame changes its value at T.
    """
    T = 60  # test point well inside the DataFrame
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


def test_feature_store_round_trip(price_df: pd.DataFrame, store: FeatureStore, tmp_path: Path):
    """compute_offline_all → load_offline gives back the same DataFrame."""
    df.to_parquet = None  # prevent accidental mutation check
    # Write a Parquet for our fake ticker using the store's raw data path
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    price_df.to_parquet(raw_dir / "TEST.parquet")

    store2 = FeatureStore(ENGINEERED_V1, processed_dir=tmp_path / "processed")
    from ingestion.fetch import RAW_DATA_DIR
    store2.compute_offline_all(tickers=["TEST"], raw_data_dir=raw_dir)
    loaded = store2.load_offline("TEST")

    expected = store2.compute_offline(price_df, "TEST")
    pd.testing.assert_frame_equal(loaded, expected)


# ---------------------------------------------------------------------------
# Compute online
# ---------------------------------------------------------------------------

def test_compute_online_returns_all_feature_names(price_df: pd.DataFrame, store: FeatureStore):
    """compute_online must return a dict with all expected feature names."""
    window = price_df.iloc[-ENGINEERED_V1.max_required_window:]
    result = store.compute_online(window)
    assert set(result.keys()) == set(ENGINEERED_V1.feature_names)


def test_compute_online_no_nan(price_df: pd.DataFrame, store: FeatureStore):
    """compute_online must not return NaN when given sufficient data."""
    window = price_df.iloc[-ENGINEERED_V1.max_required_window:]
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

    # All rows from before the split date must still be present
    rows_before_split = full[full.index < split_date]
    assert len(after_backfill) >= len(rows_before_split), (
        "Backfill deleted rows outside the backfill range"
    )
```

### 3.2 Run the tests

```bash
uv run pytest tests/test_feature_store.py -v
```

Expected: 11 tests pass.

**If parity fails:** The most common cause is the offline function using a different rolling window boundary than the online function. Check the `required_window` — off-by-one errors are common here.

**If leakage fails:** A feature's offline function accidentally includes a `shift(-N)` (negative shift = future data). All shifts in `definitions.py` should be non-negative.

---

## Step 4: Update Training Pipeline

### What changes

The existing `pipeline.py` trains on raw OHLCV. We add a second pipeline function that trains on engineered features from the feature store. Both log to MLflow under the same experiment, letting you compare them in the UI.

We do **not** delete or change `build_dataset()` or `run_pipeline()`. The baseline is preserved for comparison.

New additions to `training/pipeline.py`:

```python
# At the top of the file, add imports:
from features.definitions import ENGINEERED_V1
from features.store import FeatureStore

MLFLOW_EXPERIMENT = "ml_platform_baseline"  # same experiment — both runs visible together


def build_dataset_from_store(store: FeatureStore) -> pd.DataFrame:
    """
    Load labels from raw data and join with precomputed features from the store.

    Labels are derived from raw close prices (Week 5 logic).
    Features come from the offline feature store (Week 6).
    They are joined on (ticker, date) — the index.

    Returns:
        DataFrame with label + feature columns, NaN rows dropped, ready for split.
    """
    logger.info("Loading raw data for labels: %s", TICKERS)
    raw_df = load_all(TICKERS)

    logger.info("Adding labels (5-day forward direction)")
    labeled_df = add_labels_all_tickers(raw_df)
    labeled_df = drop_unlabeled(labeled_df)
    labels = labeled_df[["ticker", "label"]]

    logger.info("Loading engineered features from store: %s v%d",
                store.feature_set.name, store.feature_set.version)
    features_df = store.load_offline_all(TICKERS)
    # Drop the ticker column from features (it's already in labels)
    features_df = features_df.drop(columns=["ticker"])

    # Inner join on date index — only keep dates where both labels and features exist
    df = labels.join(features_df, how="inner")
    df = df.dropna()
    logger.info("Dataset: %d rows, %d feature columns", len(df), len(features_df.columns))
    return df


def run_engineered_pipeline() -> None:
    """
    Train XGBoost on engineered features and log to MLflow.

    The feature set name and version are logged as params so this run is
    reproducible: you can always look up exactly which features produced a result.
    """
    store = FeatureStore(ENGINEERED_V1)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="engineered_xgboost"):
        df = build_dataset_from_store(store)
        train_df, test_df = time_split(df, TEST_SPLIT_DATE)

        feature_cols = store.feature_set.feature_names
        model, metrics, params = train(train_df, test_df, feature_cols=feature_cols)

        mlflow.log_params(params)
        mlflow.log_params({
            "tickers": ",".join(TICKERS),
            "test_split_date": TEST_SPLIT_DATE,
            "feature_set": store.feature_set.name,
            "feature_set_version": store.feature_set.version,
            "feature_columns": ",".join(feature_cols),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        })
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")

        logger.info("Engineered metrics: %s", metrics)
        print("\n=== Engineered Feature Results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
```

Also update `__main__` to run both:

```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- Baseline (raw OHLCV) ---")
    run_pipeline()
    print("\n--- Engineered features ---")
    run_engineered_pipeline()
```

### 4.1 Apply changes to `training/pipeline.py`

Edit the existing file:
1. Add imports at the top: `from features.definitions import ENGINEERED_V1` and `from features.store import FeatureStore`
2. Add `build_dataset_from_store()` function after `build_dataset()`
3. Add `run_engineered_pipeline()` function after `run_pipeline()`
4. Update the `if __name__ == "__main__"` block

### 4.2 Compute features first, then run the pipeline

Step 1 — compute and save engineered features (do this once):

```bash
uv run python -c "
from features.definitions import ENGINEERED_V1
from features.store import FeatureStore
import logging
logging.basicConfig(level=logging.INFO)
store = FeatureStore(ENGINEERED_V1)
store.compute_offline_all()
print('Done. Files in data/processed/engineered_v1/')
"
```

Step 2 — run both pipelines:

```bash
uv run python -m training.pipeline
```

Expected output:
```
--- Baseline (raw OHLCV) ---
=== Baseline Results ===
  accuracy: ~0.52–0.54
  roc_auc:  ~0.52–0.55

--- Engineered features ---
=== Engineered Feature Results ===
  accuracy: ~0.53–0.56
  roc_auc:  ~0.53–0.57
```

The engineered model should have a slightly higher `roc_auc` than the baseline. The gap is usually small (0.02–0.05) because markets are hard to predict — that's expected. **If engineered AUC is lower than baseline, the feature definitions likely have a bug** (most probably the offline/online parity test will tell you where).

---

## Step 5: Run All Tests

```bash
uv run pytest tests/ -v
```

Expected: **27 tests pass** (16 from Week 5 + 11 new feature store tests):
- `tests/test_ingestion.py` — 7 tests
- `tests/test_labels.py` — 5 tests
- `tests/test_features.py` — 4 tests (Week 5 raw baseline)
- `tests/test_feature_store.py` — 11 tests (new)

---

## Step 6: View MLflow Comparison

```bash
uv run mlflow ui --port 5000
# Open http://localhost:5000
```

You should see two runs under `ml_platform_baseline`:
- `baseline_xgboost` — trained on 5 raw OHLCV features
- `engineered_xgboost` — trained on 6 engineered features

Select both runs and click **Compare** to see a side-by-side diff of all params and metrics. Look for `roc_auc` improvement.

---

## Step 7: Validation Checklist

- [ ] `pytest tests/ -v` — all 27 tests pass
- [ ] `data/processed/engineered_v1/` contains 5 Parquet files
- [ ] Each file has columns: `return_1d, return_5d, volatility_5d, volatility_20d, rsi_14, volume_ratio_20d, ticker`
- [ ] `rsi_14` values are all in [0, 100]
- [ ] `volume_ratio_20d` values are all positive
- [ ] Offline/online parity test passes (most important — this is the feature store's core guarantee)
- [ ] MLflow shows both runs under the same experiment
- [ ] Engineered `roc_auc` is in [0.50, 0.65] — above 0.65 suggests leakage
- [ ] No `FutureWarning` or `DeprecationWarning` during the pipeline run

---

## Common Pitfalls

### "Parity test fails for volatility"

The most common cause: `rolling(5).std()` uses ddof=1 (sample std) by default in pandas,
but `.std()` on a plain Series also uses ddof=1. They should match. If they don't,
check that the online path uses exactly 5 diffs (6 rows), not 5 rows.

### "RSI parity fails"

The offline RSI function in `definitions.py` is computed row-by-row using `_rsi_from_closes(df["close"].iloc[:i+1])`. This is intentionally slow but correct — it ensures the online path (which also calls `_rsi_from_closes`) uses identical logic. If you try to vectorize the offline RSI with a pure rolling approach, the boundary handling may differ from the online path.

### "Leakage test fails"

A feature uses future data. Check for:
1. Any use of `shift(-N)` in `offline` functions (negative shift = future)
2. A rolling function that spans past the current row
3. Accidentally joining on a DataFrame that's been forward-filled

### "build_dataset_from_store returns an empty DataFrame"

The inner join between labels and features fails if the date indexes don't align. This can happen if:
- Raw data dates are timezone-aware and feature dates are timezone-naive (or vice versa)
- `compute_offline_all()` hasn't been run yet
- The feature store Parquet was computed from a different raw data fetch than the current one

Fix: make sure `load_offline()` and `load_all()` use the same index dtype. Check with `df.index.dtype`.

### "engineered roc_auc is lower than baseline"

Not necessarily a bug — markets are efficient. But if the gap is large (>0.05 lower),
check for:
1. NaN values in the feature matrix (XGBoost handles them, but they indicate a join issue)
2. A feature defined with the wrong sign (e.g. `return_1d` accidentally negated)
3. `build_dataset_from_store` including a `label` column as a feature (check `df.columns` before training)

---

## Week 6 → Week 7 Handoff

At end of Week 6 you have:
- A feature store with offline/online parity, versioning, and backfill
- Two MLflow runs to compare: baseline vs engineered
- The serving interface (`compute_online`) ready for the inference API

Week 7 builds the **automated training pipeline**:
- Dagster or a simple scheduler triggers retraining when new data arrives
- The trained model is registered in MLflow Model Registry with a version number
- Re-training is fully reproducible: same data range + same feature set version → same model
- A "challenger vs champion" evaluation step: new model only promotes to registry if it beats the current champion's `roc_auc` on a held-out window

Key design question to decide before Week 7: **where is the train/test split boundary?** Week 5 used a fixed date (`2023-01-01`). Week 7 will need to make this dynamic — the split date should be relative to "today" minus some evaluation window, so retraining on new data always evaluates on the most recent unseen period.
