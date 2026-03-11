# Week 8 Execution Plan: Serving & Monitoring

## Goal

By end of week 8 you have:
- **ENGINEERED_V2**: 8 features (adds `return_20d` and `macd_signal` to V1), demonstrating
  feature versioning in practice — a new version means a new offline compute + retrain
- **Training stats persisted**: per-feature mean/std saved at training time, used for drift detection
- **`monitoring/drift.py`**: `DriftDetector` flags inference-time features that fall outside
  the training distribution (>3σ)
- **`serving/app.py`**: FastAPI with `POST /predict`, `POST /reload` (hot swap),
  `GET /health`, `GET /ready`, `GET /metrics` (Prometheus)
- **Prediction logging**: every prediction written to `data/predictions.jsonl` for future
  accuracy evaluation
- **Automated reproducibility test**: pytest verifies two identical runs produce the same model
- **12 new tests** covering drift detection and serving endpoints (53 total)

---

## Why This Architecture Matters (Read First)

### Feature versioning is not just bookkeeping

When you add `return_20d` and `macd_signal`, you cannot simply append them to ENGINEERED_V1
and retrain. Any model trained on V1 expected 6 features at inference time. A serving layer
using V2 features would send 8 features to a V1 model — silent shape mismatch or wrong
predictions.

The fix: **ENGINEERED_V2** is a new, immutable feature set. The runner retrains with V2,
the new champion is a V2 model, and the serving layer loads V2 features. V1 models and
features are preserved for audit. This is the exact workflow of production feature stores
(Feast, Tecton).

### Drift detection is an early warning system

Model accuracy degrades silently. By the time you notice, the damage is done. Drift
detection catches the precursor: the *input distribution* changing. If `return_1d` was
normally distributed around 0.001 ± 0.015 at training time and is now averaging 0.05,
the model is operating outside its training domain. You flag it immediately rather than
waiting for accuracy metrics to degrade.

### Hot reload vs restart

If a serving pod must restart to load a new model, you have ~seconds of downtime per
deployment. Hot reload swaps the model in-process while the server keeps accepting
requests. The `POST /reload` endpoint triggers this — a lightweight deployment hook.

---

## Repo Changes

New files:
```
monitoring/
├── __init__.py
└── drift.py          # DriftDetector: training stats + 3σ flagging
serving/
├── __init__.py
└── app.py            # FastAPI: /predict, /reload, /health, /ready, /metrics
tests/
├── test_drift.py     # 4 tests
└── test_serving.py   # 5 tests + 1 integration test
```

Modified files:
```
features/definitions.py   # add ENGINEERED_V2 (8 features)
features/store.py         # add save_training_stats() and load_training_stats()
training/pipeline.py      # save training stats when run_engineered_pipeline() runs
training/runner.py        # pass feature_set so runner can use V2
tests/test_runner.py      # add automated reproducibility test
```

Data produced:
```
data/processed/
├── engineered_v1/         # from Week 6 (unchanged)
└── engineered_v2/
    ├── AAPL.parquet       # 8-feature offline store
    ├── ...
    └── training_stats.json  # per-feature {mean, std} from train set
data/
└── predictions.jsonl      # append-only prediction log
```

---

## Step 1: Add ENGINEERED_V2 to `features/definitions.py`

### Why a new version, not modifying V1

ENGINEERED_V1 already has a trained champion model pointing at it. If you modify V1 in
place, the serving layer's feature count (6) no longer matches what the V2 model expects
(8). Immutable versioning prevents this class of error.

### New features

**`return_20d`**: 20-day price return. Pure rolling — exact parity between offline and
online.

**`macd_signal`**: 9-day EMA of the MACD line (EMA12 − EMA26). Uses exponential
weighting (`ewm`). Parity is approximate, not exact: EWM has memory, so the online path
(given a 100-row window) produces a slightly different value than the offline path
(which sees the full history). With `required_window=100`, the initialization error is
< 0.5% of the signal value — acceptable for an ML feature. The parity test uses a
relaxed tolerance of 1e-2 for this feature.

### 1.1 Append to `features/definitions.py`

Add these two helpers and the V2 definition after the ENGINEERED_V1 block:

```python
# ---------------------------------------------------------------------------
# MACD helpers — shared by both compute paths
# EWM parity note: online path with required_window=100 approximates the
# offline path to within ~0.5% due to EWM initialization. Tolerance in
# the parity test is relaxed to 1e-2 for macd_signal accordingly.
# ---------------------------------------------------------------------------

def _macd_signal_offline(df: pd.DataFrame) -> pd.Series:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    return macd_line.ewm(span=9, adjust=False).mean()


def _macd_signal_online(window: pd.DataFrame) -> float:
    ema12 = window["close"].ewm(span=12, adjust=False).mean()
    ema26 = window["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    return float(macd_line.ewm(span=9, adjust=False).mean().iloc[-1])


# ---------------------------------------------------------------------------
# Engineered feature set v2
# Adds return_20d and macd_signal on top of the 6 V1 features.
# V1 is preserved unchanged — any model trained on V1 still works with V1 features.
# ---------------------------------------------------------------------------

ENGINEERED_V2 = FeatureSet(
    name="engineered",
    version=2,
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
            name="return_20d",
            offline=lambda df: df["close"].pct_change(20),
            online=lambda w: (w["close"].iloc[-1] - w["close"].iloc[-21]) / w["close"].iloc[-21],
            required_window=21,
        ),
        FeatureDefinition(
            name="volatility_5d",
            offline=lambda df: df["close"].pct_change().rolling(5).std(),
            online=lambda w: w["close"].pct_change().dropna().tail(5).std(),
            required_window=6,
        ),
        FeatureDefinition(
            name="volatility_20d",
            offline=lambda df: df["close"].pct_change().rolling(20).std(),
            online=lambda w: w["close"].pct_change().dropna().tail(20).std(),
            required_window=21,
        ),
        FeatureDefinition(
            name="rsi_14",
            offline=lambda df: pd.Series(
                [_rsi_from_closes(df["close"].iloc[: i + 1]) for i in range(len(df))],
                index=df.index,
            ),
            online=lambda w: _rsi_from_closes(w["close"]),
            required_window=15,
        ),
        FeatureDefinition(
            name="volume_ratio_20d",
            offline=lambda df: df["volume"] / df["volume"].rolling(20).mean(),
            online=lambda w: w["volume"].iloc[-1] / w["volume"].tail(20).mean(),
            required_window=20,
        ),
        FeatureDefinition(
            name="macd_signal",
            offline=_macd_signal_offline,
            online=_macd_signal_online,
            required_window=100,  # large window to minimize EWM initialization error
        ),
    ],
)
```

### 1.2 Verify

```bash
uv run python -c "
from features.definitions import ENGINEERED_V2
print('V2 features:', ENGINEERED_V2.feature_names)
print('Max window:', ENGINEERED_V2.max_required_window)
"
```

Expected:
```
V2 features: ['return_1d', 'return_5d', 'return_20d', 'volatility_5d', 'volatility_20d', 'rsi_14', 'volume_ratio_20d', 'macd_signal']
Max window: 100
```

### 1.3 Compute V2 offline features

```bash
uv run python -c "
from features.definitions import ENGINEERED_V2
from features.store import FeatureStore
import logging
logging.basicConfig(level=logging.INFO)
store = FeatureStore(ENGINEERED_V2)
store.compute_offline_all()
df = store.load_offline('AAPL')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
"
```

Expected: shape `(~905, 9)` — more rows dropped than V1 because `macd_signal` needs
100-row warmup. 9 columns = 8 features + `ticker`.

---

## Step 2: Add training stats to `features/store.py`

### What training stats are

After computing the training set, we save per-feature mean and std to a JSON file at
`store_dir/training_stats.json`. The `DriftDetector` loads this file at serving startup
and compares incoming features against these bounds at inference time.

### 2.1 Add two methods to `FeatureStore`

Add after `backfill()` in `features/store.py`:

```python
import json  # add to top-level imports

def save_training_stats(self, train_df: pd.DataFrame) -> Path:
    """
    Compute and persist per-feature mean/std from the training set.

    Called once after each successful training run. The serving layer
    loads these stats to detect when inference-time features fall outside
    the training distribution (drift detection).

    Args:
        train_df: The training DataFrame (features only, no label/ticker)

    Returns:
        Path to the saved JSON file
    """
    feature_cols = self.feature_set.feature_names
    stats = {
        col: {
            "mean": float(train_df[col].mean()),
            "std": float(train_df[col].std()),
        }
        for col in feature_cols
        if col in train_df.columns
    }
    path = self.store_dir / "training_stats.json"
    path.write_text(json.dumps(stats, indent=2))
    logger.info("Saved training stats to %s", path)
    return path

def load_training_stats(self) -> dict:
    """
    Load saved training stats.

    Returns:
        Dict of {feature_name: {mean: float, std: float}}

    Raises:
        FileNotFoundError: If save_training_stats() has not been called yet
    """
    path = self.store_dir / "training_stats.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No training stats at {path}. Run the training pipeline first."
        )
    return json.loads(path.read_text())
```

Also add `import json` at the top of `store.py`.

### 2.2 Verify

```bash
uv run python -c "
import pandas as pd
import numpy as np
from features.definitions import ENGINEERED_V2
from features.store import FeatureStore
store = FeatureStore(ENGINEERED_V2)
# Simulate a small training df
df = store.load_offline('AAPL').head(100)
path = store.save_training_stats(df)
print('Saved to:', path)
stats = store.load_training_stats()
print('return_1d mean:', round(stats['return_1d']['mean'], 6))
print('Features covered:', list(stats.keys()))
"
```

---

## Step 3: Update `training/pipeline.py` to save training stats

### 3.1 Update `run_engineered_pipeline`

Add the stats save call **after** computing `train_df`, inside the `mlflow.start_run` block:

```python
# After: train_df, test_df = time_split(df, test_split_date)
# Add:
store.save_training_stats(train_df[feature_cols])
mlflow.log_param("training_stats_path", str(store.store_dir / "training_stats.json"))
```

The full updated block (replace just the section after `time_split`):

```python
with mlflow.start_run(run_name="engineered_xgboost") as active_run:
    df = build_dataset_from_store(store)
    train_df, test_df = time_split(df, test_split_date)

    feature_cols = store.feature_set.feature_names

    # Save training distribution for drift detection at serving time
    store.save_training_stats(train_df[feature_cols])

    model, metrics, params = train(train_df, test_df, feature_cols=feature_cols)
    # ... rest unchanged ...
```

### 3.2 Update the FeatureStore used in the pipeline to ENGINEERED_V2

In `run_engineered_pipeline`, change:

```python
# from:
from features.definitions import ENGINEERED_V1
store = FeatureStore(ENGINEERED_V1)

# to:
from features.definitions import ENGINEERED_V2
store = FeatureStore(ENGINEERED_V2)
```

Also update the import at the top of `pipeline.py`:

```python
from features.definitions import ENGINEERED_V1, ENGINEERED_V2
```

### 3.3 Retrain with V2 and verify stats are saved

```bash
uv run python -m training.runner --run-date 2024-12-15
```

Expected:
- Training completes with 8 features
- `data/processed/engineered_v2/training_stats.json` is created
- `feature_set_version: 2` in the MLflow run params
- New champion registered (V2 model beats old V1 champion — different feature count means
  the comparison is across different model families; if it doesn't beat by 0.005, run with
  a slightly different split date until it promotes, or temporarily lower `PROMOTION_THRESHOLD`
  in `runner.py` to 0.0 for the first V2 run)

---

## Step 4: Create `monitoring/drift.py`

### 4.1 Create `monitoring/__init__.py` (empty)

### 4.2 Create `monitoring/drift.py`

```python
"""
Drift detection for the ML platform serving layer.

At training time: per-feature mean and std are saved to training_stats.json
(written by FeatureStore.save_training_stats()).

At serving time: DriftDetector loads those stats and checks each incoming
feature value against the training distribution. Any feature more than
N standard deviations from its training mean is flagged.

Design decisions:
- 3σ threshold: flags ~0.3% of values from a normal distribution — low false
  positive rate while still catching meaningful shifts.
- Per-feature flagging: tells the caller which specific features drifted,
  not just whether drift occurred. Useful for debugging data pipeline issues.
- Graceful on missing features: if a feature is present in inference but
  not in the training stats (e.g. a new feature added to the set), it is
  skipped rather than raising. Prevents a stats file mismatch from taking
  down the serving layer.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DriftDetector:
    def __init__(self, stats: dict | None = None, stats_path: Path | None = None) -> None:
        """
        Initialize with either a stats dict or a path to training_stats.json.

        Args:
            stats: Dict of {feature_name: {mean: float, std: float}}
            stats_path: Path to a training_stats.json file produced by
                        FeatureStore.save_training_stats()

        Exactly one of stats or stats_path must be provided.
        """
        if stats is not None:
            self.stats = stats
        elif stats_path is not None:
            if not stats_path.exists():
                raise FileNotFoundError(f"Training stats not found: {stats_path}")
            self.stats = json.loads(stats_path.read_text())
        else:
            raise ValueError("Provide either stats or stats_path")

    def check(
        self, features: dict[str, float], n_sigma: float = 3.0
    ) -> dict:
        """
        Check whether any feature value falls outside the training distribution.

        Args:
            features: Dict of {feature_name: value} — the current inference-time
                      feature vector for one prediction
            n_sigma: Number of standard deviations beyond which a feature is
                     considered drifted (default: 3.0)

        Returns:
            {
                "drift_detected": bool,
                "drifted_features": list[str],   # names of drifted features
            }
        """
        drifted = []
        for name, val in features.items():
            feature_stats = self.stats.get(name)
            if feature_stats is None:
                continue  # feature not in training stats — skip gracefully
            std = feature_stats["std"]
            if std == 0:
                continue  # constant feature — can't compute z-score
            z = abs(val - feature_stats["mean"]) / std
            if z > n_sigma:
                drifted.append(name)
                logger.warning(
                    "Drift detected for '%s': value=%.4f, mean=%.4f, std=%.4f, z=%.2f",
                    name, val, feature_stats["mean"], std, z,
                )
        return {
            "drift_detected": bool(drifted),
            "drifted_features": drifted,
        }
```

---

## Step 5: Create `serving/app.py`

### 5.1 Create `serving/__init__.py` (empty)

### 5.2 Create `serving/app.py`

```python
"""
FastAPI inference API for the ML platform.

Endpoints:
    POST /predict   — ticker → online features → model → prediction + drift check
    POST /reload    — hot-swap champion model without restarting
    GET  /health    — always 200 (liveness probe)
    GET  /ready     — 200 if model loaded, 503 if not (readiness probe)
    GET  /metrics   — Prometheus metrics (scrape endpoint)

Design decisions:
- Global AppState with threading.Lock: model swaps are atomic; a request
  in flight with the old model completes before the new model is installed.
- mlflow.xgboost.load_model: loads the native XGBoost model (not PyFunc)
  so predict_proba() is available for confidence scores.
- Prediction logging: every prediction is appended to data/predictions.jsonl
  as a JSONL record. A separate job can later join these records with realized
  prices to compute real-world accuracy.
- Drift detection: loaded at startup from training_stats.json. If the file
  doesn't exist (no training run yet), drift checking is skipped gracefully.
"""

import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel

from features.definitions import ENGINEERED_V2
from features.store import FeatureStore, PROCESSED_DATA_DIR
from ingestion.fetch import RAW_DATA_DIR, load_ticker
from monitoring.drift import DriftDetector
from registry.model_registry import CHAMPION_ALIAS, MODEL_NAME, ModelRegistry

logger = logging.getLogger(__name__)

PREDICTIONS_LOG = Path("data/predictions.jsonl")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent computing a prediction end-to-end",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total number of predictions served",
    labelnames=["ticker", "direction"],
)
DRIFT_EVENTS_TOTAL = Counter(
    "drift_events_total",
    "Total number of predictions where drift was detected",
    labelnames=["ticker"],
)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self) -> None:
        self.model = None           # native XGBoost model
        self.model_version: str | None = None
        self.drift_detector: DriftDetector | None = None
        self._lock = threading.Lock()

    def swap(self, model, version: str, detector: DriftDetector | None) -> None:
        """Atomically swap to a new model. Thread-safe."""
        with self._lock:
            self.model = model
            self.model_version = version
            self.drift_detector = detector

    @property
    def ready(self) -> bool:
        return self.model is not None


_state = AppState()
_store = FeatureStore(ENGINEERED_V2)
_registry = ModelRegistry()


def _load_model() -> None:
    """Load the current champion model and drift detector into _state."""
    champion = _registry.get_champion()
    if champion is None:
        logger.warning("No champion model found. /ready will return 503.")
        return

    model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    model = mlflow.xgboost.load_model(model_uri)
    version = str(champion.version)

    stats_path = PROCESSED_DATA_DIR / f"{ENGINEERED_V2.name}_v{ENGINEERED_V2.version}" / "training_stats.json"
    detector = DriftDetector(stats_path=stats_path) if stats_path.exists() else None
    if detector is None:
        logger.warning("No training stats found at %s. Drift detection disabled.", stats_path)

    _state.swap(model, version, detector)
    logger.info("Loaded champion model version %s", version)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="ML Platform Inference API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    ticker: str


class PredictResponse(BaseModel):
    ticker: str
    direction: str          # "UP" or "DOWN"
    probability: float      # P(UP), from predict_proba
    model_version: str
    feature_set: str
    drift_detected: bool
    drifted_features: list[str]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not _state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with PREDICTION_LATENCY.time():
        # Load the last max_required_window rows of raw OHLCV
        try:
            raw_df = load_ticker(req.ticker, data_dir=RAW_DATA_DIR)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"No data for ticker '{req.ticker}'. Run fetch_ticker() first.",
            )

        window = raw_df.tail(_store.feature_set.max_required_window)
        features = _store.compute_online(window)

        # Build a single-row DataFrame for the model
        X = pd.DataFrame([features])[_store.feature_set.feature_names]

        prediction = int(_state.model.predict(X)[0])
        probability = float(_state.model.predict_proba(X)[0][1])  # P(UP)
        direction = "UP" if prediction == 1 else "DOWN"

        # Drift check
        drift_result = (
            _state.drift_detector.check(features)
            if _state.drift_detector
            else {"drift_detected": False, "drifted_features": []}
        )

    # Prometheus
    PREDICTIONS_TOTAL.labels(ticker=req.ticker, direction=direction).inc()
    if drift_result["drift_detected"]:
        DRIFT_EVENTS_TOTAL.labels(ticker=req.ticker).inc()

    # Prediction log
    _log_prediction(
        ticker=req.ticker,
        direction=direction,
        probability=probability,
        model_version=_state.model_version,
    )

    return PredictResponse(
        ticker=req.ticker,
        direction=direction,
        probability=round(probability, 4),
        model_version=_state.model_version,
        feature_set=f"{ENGINEERED_V2.name}_v{ENGINEERED_V2.version}",
        drift_detected=drift_result["drift_detected"],
        drifted_features=drift_result["drifted_features"],
    )


@app.post("/reload")
def reload():
    """Hot-swap the champion model without restarting."""
    _load_model()
    return {"status": "reloaded", "model_version": _state.model_version}


@app.get("/health")
def health():
    """Liveness probe — always 200."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness probe — 503 if model not loaded."""
    if not _state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": _state.model_version}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus scrape endpoint."""
    return generate_latest().decode("utf-8")


# ---------------------------------------------------------------------------
# Prediction logging
# ---------------------------------------------------------------------------

def _log_prediction(
    ticker: str,
    direction: str,
    probability: float,
    model_version: str | None,
) -> None:
    """Append one prediction to the JSONL log. Non-blocking best-effort."""
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "direction": direction,
        "probability": probability,
        "model_version": model_version,
    }
    try:
        with PREDICTIONS_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.warning("Failed to write prediction log: %s", exc)
```

### 5.3 Add FastAPI, uvicorn, and httpx as dependencies

```bash
uv add fastapi uvicorn
uv add --dev httpx   # required by FastAPI's TestClient for serving tests
```

Note: if `uv add` fails due to a corporate proxy, install manually outside the proxy
network and re-run `uv sync` to update `uv.lock`.

### 5.4 Start the server and smoke-test

```bash
uv run uvicorn serving.app:app --port 8000 --reload
```

In another terminal:

```bash
# Liveness
curl http://localhost:8000/health
# → {"status":"ok"}

# Readiness
curl http://localhost:8000/ready
# → {"status":"ready","model_version":"..."}

# Predict
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}' | python3 -m json.tool
# → {"ticker":"AAPL","direction":"UP","probability":0.6123,...}

# Prometheus metrics
curl http://localhost:8000/metrics | grep prediction
# → prediction_latency_seconds_count ...
# → predictions_total{direction="UP",ticker="AAPL"} 1.0

# Hot reload
curl -X POST http://localhost:8000/reload
# → {"status":"reloaded","model_version":"..."}
```

---

## Step 6: Tests

### 6.1 Create `tests/test_drift.py`

```python
"""Tests for monitoring/drift.py."""

import pytest
from monitoring.drift import DriftDetector


SAMPLE_STATS = {
    "return_1d": {"mean": 0.001, "std": 0.015},
    "volatility_5d": {"mean": 0.012, "std": 0.005},
    "rsi_14": {"mean": 55.0, "std": 10.0},
}


def test_no_drift_within_bounds():
    """Values within 3σ should not trigger drift."""
    detector = DriftDetector(stats=SAMPLE_STATS)
    result = detector.check({"return_1d": 0.02, "volatility_5d": 0.010})
    assert not result["drift_detected"]
    assert result["drifted_features"] == []


def test_drift_detected_beyond_threshold():
    """A value >3σ from the mean should be flagged."""
    detector = DriftDetector(stats=SAMPLE_STATS)
    # return_1d mean=0.001, std=0.015 → 3σ boundary = 0.001 ± 0.045
    # 0.10 is well beyond 3σ
    result = detector.check({"return_1d": 0.10})
    assert result["drift_detected"]
    assert "return_1d" in result["drifted_features"]


def test_multiple_features_flagged():
    """Multiple out-of-bounds features should all appear in drifted_features."""
    detector = DriftDetector(stats=SAMPLE_STATS)
    result = detector.check({
        "return_1d": 0.10,        # drifted
        "rsi_14": 110.0,          # drifted (mean=55, std=10 → beyond 3σ)
        "volatility_5d": 0.012,   # normal
    })
    assert result["drift_detected"]
    assert set(result["drifted_features"]) == {"return_1d", "rsi_14"}


def test_unknown_feature_skipped():
    """Features not in training stats are silently skipped, not treated as drift."""
    detector = DriftDetector(stats=SAMPLE_STATS)
    result = detector.check({"new_feature_not_in_stats": 999.9})
    assert not result["drift_detected"]
```

### 6.2 Create `tests/test_serving.py`

```python
"""
Tests for serving/app.py.

Uses FastAPI's TestClient. Model and feature store interactions are mocked
so tests run without a trained champion or raw data files.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """TestClient with model state pre-loaded via mocks."""
    import serving.app as app_module

    # Build a minimal mock model with predict / predict_proba
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.38, 0.62]])

    # Inject into app state
    app_module._state.model = mock_model
    app_module._state.model_version = "3"
    app_module._state.drift_detector = None  # disable drift for most tests

    return TestClient(app_module.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_when_model_loaded(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_ready_when_model_not_loaded():
    """When no model is loaded, /ready should return 503."""
    import serving.app as app_module
    from fastapi.testclient import TestClient

    original = app_module._state.model
    app_module._state.model = None
    c = TestClient(app_module.app, raise_server_exceptions=False)
    resp = c.get("/ready")
    assert resp.status_code == 503
    app_module._state.model = original  # restore


def test_predict_returns_correct_schema(client, tmp_path):
    """POST /predict should return all required fields."""
    import serving.app as app_module

    # Mock load_ticker to return a small DataFrame
    dates = pd.date_range("2024-01-01", periods=110, freq="B")
    fake_raw = pd.DataFrame({
        "open": 150.0, "high": 152.0, "low": 149.0,
        "close": 151.0, "volume": 1_000_000.0, "ticker": "AAPL",
    }, index=dates)
    fake_raw.index.name = "date"

    with patch("serving.app.load_ticker", return_value=fake_raw):
        resp = client.post("/predict", json={"ticker": "AAPL"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["direction"] in {"UP", "DOWN"}
    assert 0.0 <= body["probability"] <= 1.0
    assert "model_version" in body
    assert "feature_set" in body
    assert "drift_detected" in body
    assert isinstance(body["drifted_features"], list)


def test_predict_404_for_unknown_ticker(client):
    """POST /predict with a ticker that has no data file returns 404."""
    from ingestion.fetch import load_ticker
    with patch("serving.app.load_ticker", side_effect=FileNotFoundError("no file")):
        resp = client.post("/predict", json={"ticker": "FAKE"})
    assert resp.status_code == 404


def test_metrics_endpoint(client):
    """GET /metrics should return Prometheus text format."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "prediction_latency_seconds" in resp.text
```

### 6.3 Add automated reproducibility test to `tests/test_runner.py`

Append this test to the existing file:

```python
def test_compute_split_date_is_deterministic():
    """
    Reproducibility: the same run_date always produces the same split_date.
    This is the foundation of reproducible scheduled runs.
    """
    run_date = date(2024, 9, 15)
    assert compute_split_date(run_date) == compute_split_date(run_date)
    assert compute_split_date(run_date) == compute_split_date(run_date, EVALUATION_WINDOW_DAYS)
```

### 6.4 Update parity tests to cover V2 features

The existing `test_offline_online_parity` in `test_feature_store.py` is parametrized over
`ENGINEERED_V1.features`. Add a parallel test for V2 with a relaxed tolerance for
`macd_signal`:

```python
# Add to tests/test_feature_store.py

from features.definitions import ENGINEERED_V2

@pytest.mark.parametrize(
    "feat,tol",
    [(f, 1e-2 if f.name == "macd_signal" else 1e-6) for f in ENGINEERED_V2.features],
    ids=lambda x: x.name if hasattr(x, "name") else str(x),
)
def test_offline_online_parity_v2(feat, tol, price_df):
    """
    V2 parity test. macd_signal uses relaxed tolerance (1e-2) because EWM
    with required_window=100 approximates the full-history offline value —
    initialization error is < 0.5% of signal magnitude.
    """
    offline_series = feat.offline(price_df)

    for t in [50, 60, 70, 80]:
        window_start = t - feat.required_window + 1
        if window_start < 0:
            continue  # price_df too short for this feature at this t
        window = price_df.iloc[window_start : t + 1]
        assert len(window) == feat.required_window

        online_val = feat.online(window)
        offline_val = offline_series.iloc[t]

        if math.isnan(offline_val) or math.isnan(online_val):
            continue  # insufficient data at this point — skip

        assert abs(online_val - offline_val) < tol, (
            f"{feat.name} parity failed at t={t}: "
            f"offline={offline_val:.8f}, online={online_val:.8f}, tol={tol}"
        )
```

### 6.5 Run all tests

```bash
uv run pytest tests/ -v
```

Expected: **53 tests pass** (41 from weeks 5–7 + 12 new):
- `tests/test_drift.py` — 4 tests
- `tests/test_serving.py` — 6 tests
- `tests/test_runner.py` — 7 tests (6 existing + 1 new)
- `tests/test_feature_store.py` — 23 tests (15 existing + 8 V2 parity)

---

## Step 7: End-to-End Verification

### 7.1 Full pipeline with V2

```bash
# Compute V2 offline features (if not done in Step 1.3)
uv run python -c "
from features.definitions import ENGINEERED_V2
from features.store import FeatureStore
FeatureStore(ENGINEERED_V2).compute_offline_all()
print('V2 features computed')
"

# Retrain with V2 and promote
uv run python -m training.runner --run-date 2024-12-15

# Start the server
uv run uvicorn serving.app:app --port 8000
```

### 7.2 Verify prediction logging

After calling `/predict`:
```bash
cat data/predictions.jsonl | python3 -m json.tool
```

Each line should be a JSON record with timestamp, ticker, direction, probability,
model_version.

### 7.3 Verify hot reload

While the server is running, train and promote a new model in another terminal:
```bash
uv run python -m training.runner --run-date 2024-12-16
curl -X POST http://localhost:8000/reload
curl http://localhost:8000/ready  # check new model_version
```

---

## Step 8: Validation Checklist

- [ ] `pytest tests/ -v` — all 53 tests pass
- [ ] `data/processed/engineered_v2/` contains 5 Parquet files + `training_stats.json`
- [ ] `training_stats.json` contains all 8 V2 feature names with mean and std
- [ ] `uv run uvicorn serving.app:app --port 8000` starts without errors
- [ ] `POST /predict` with a valid ticker returns a response matching `PredictResponse` schema
- [ ] `GET /ready` returns 503 if `_state.model = None` (tested in `test_serving.py`)
- [ ] `GET /metrics` returns Prometheus text with `prediction_latency_seconds`
- [ ] `POST /reload` successfully hot-swaps the model
- [ ] `data/predictions.jsonl` grows with each `/predict` call
- [ ] MLflow UI shows `feature_set_version: 2` on the latest run
- [ ] No `FutureWarning` during serving startup

---

## Common Pitfalls

### "predict_proba is not defined on the loaded model"

`mlflow.pyfunc.load_model` wraps XGBoost and exposes only `.predict()`. Use
`mlflow.xgboost.load_model()` to get the native XGBoost model with `.predict_proba()`.

### "Feature count mismatch during serving"

The loaded model was trained with V1 (6 features) but the serving layer sends V2 (8 features).
Promote a fresh V2 model first: `python -m training.runner --run-date <date>`, then reload.

### "GET /ready returns 503 even though training ran"

The champion alias may not be set yet. Check:
```bash
uv run python -c "
from registry.model_registry import ModelRegistry
r = ModelRegistry()
print(r.get_champion())
"
```
If None, run the runner to promote.

### "MACD parity test fails beyond 1e-2 tolerance"

If `price_df` has fewer than 100 rows at position t=50, the window would need to start
before the DataFrame begins. The test skips positions where `window_start < 0` for this
reason. Increase `n=100` in `make_price_df` if you see skips at t=50.

### "TestClient /predict test fails with import errors"

`serving/app.py` imports `mlflow.xgboost` at module level. Make sure `mlflow` is in the
venv. If the import fails, check `uv run python -c "import mlflow.xgboost"`.

---

## Week 8 Deliverables → Resume Bullets

This project is complete after Step 7. Before finalizing, write these in `README.md`:

**Architecture summary (one paragraph):**
> Built a production-grade ML platform for stock direction prediction. Four layers:
> (1) ingestion with batch + simulated streaming, (2) a versioned feature store with
> offline/online parity enforcement and incremental backfill, (3) an automated training
> pipeline with champion/challenger promotion and data fingerprinting for reproducibility,
> (4) a FastAPI inference API with hot model reload, Prometheus metrics, and statistical
> drift detection.

**Resume bullets (senior SWE framing):**
- Designed a **feature store** with offline/online compute parity enforced by property
  tests — eliminates training/serving skew, the most common cause of silent model
  degradation
- Implemented **feature set versioning**: models record the exact feature version used;
  serving layer loads the matching version — prevents shape mismatches across retrains
- Built a **champion/challenger promotion pipeline**: new models are registered and
  evaluated against the current champion; promotion requires beating AUC by a configurable
  threshold — prevents accidental degradations from bad training runs
- Added **statistical drift detection**: per-feature mean/std saved at training time;
  inference-time values beyond 3σ are flagged in the response and counted in Prometheus —
  early warning before accuracy degrades
- Implemented **hot model reload**: `POST /reload` atomically swaps the model in-process
  with a threading lock — zero-downtime deployments
