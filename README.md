# ML Platform

A production-grade ML infrastructure platform for stock direction prediction. The emphasis is on the **engineering that makes ML reliable and reproducible** — not the model accuracy. XGBoost is a vehicle to demonstrate feature stores, experiment tracking, champion/challenger promotion, drift detection, and a hot-reloadable inference API.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          ML Platform                                 │
├──────────────────┬───────────────────┬──────────────┬───────────────┤
│  Layer 1         │  Layer 2          │  Layer 3     │  Layer 4      │
│  Ingestion       │  Feature Store    │  Training    │  Serving      │
├──────────────────┼───────────────────┼──────────────┼───────────────┤
│  yfinance        │  definitions.py   │  pipeline.py │  FastAPI      │
│  ↓               │  (V1: 6 features  │  ↓           │  POST /predict│
│  validate.py     │   V2: 8 features) │  runner.py   │  POST /reload │
│  ↓               │  ↓                │  ↓           │  GET  /health │
│  Parquet         │  offline store    │  MLflow      │  GET  /ready  │
│  data/raw/       │  data/processed/  │  tracking    │  GET  /metrics│
│                  │  ↓                │  ↓           ├───────────────┤
│  stream.py       │  online compute   │  registry/   │  monitoring/  │
│  (simulated      │  (serving parity) │  champion    │  drift.py     │
│  streaming)      │  ↓                │  alias       │  performance  │
│                  │  training_stats   │              │  .py          │
│                  │  (drift baseline) │              │               │
└──────────────────┴───────────────────┴──────────────┴───────────────┘
```

**Data flow at inference time:**
```
POST /predict {"ticker": "AAPL"}
  → load_ticker()          raw OHLCV from Parquet
  → compute_online()       8 V2 features from last 100 rows
  → model.predict_proba()  XGBoost champion (loaded by alias)
  → DriftDetector.check()  compare vs training distribution (3σ)
  → _log_prediction()      append to data/predictions.jsonl
  ← {"direction":"UP", "probability":0.66, "drift_detected":false, ...}
```

---

## Quick Start

```bash
# 1. Environment
uv venv && source .venv/bin/activate

# 2. Fetch 3 years of OHLCV data (requires non-corporate network for yfinance)
uv run python -c "from ingestion.fetch import fetch_all; fetch_all()"

# 3. Compute engineered features (V2: 8 features)
uv run python -c "
from features.definitions import ENGINEERED_V2
from features.store import FeatureStore
FeatureStore(ENGINEERED_V2).compute_offline_all()
"

# 4. Train and register champion model
uv run python -m training.runner --run-date 2024-12-15

# 5. Start inference server
uv run uvicorn serving.app:app --port 8000

# 6. Predict
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}' | python3 -m json.tool

# 7. Run tests
uv run pytest tests/ -v
```

---

## Module Structure

```
ml_platform/
├── ingestion/
│   ├── fetch.py          # yfinance download → Parquet (one file per ticker)
│   ├── validate.py       # fail-fast schema + quality checks
│   └── stream.py         # simulated streaming generator (row-by-row replay)
├── features/
│   ├── definitions.py    # FeatureDefinition, FeatureSet, ENGINEERED_V1/V2
│   └── store.py          # offline compute, online compute, backfill, training stats
├── training/
│   ├── labels.py         # 5-day forward return labels (per-ticker, no leakage)
│   ├── evaluation.py     # accuracy, AUC, precision/recall/F1
│   ├── pipeline.py       # end-to-end: load → label → split → train → MLflow
│   └── runner.py         # CLI: dynamic split date, champion/challenger promotion
├── registry/
│   └── model_registry.py # MLflow registry wrapper: register, promote, load champion
├── monitoring/
│   ├── drift.py          # DriftDetector: 3σ per-feature flagging at inference time
│   └── performance.py    # evaluate logged predictions vs realized prices
├── serving/
│   └── app.py            # FastAPI: /predict, /reload, /health, /ready, /metrics
└── tests/                # 64 tests across all modules
```

---

## Key Engineering Decisions

### Training/serving feature parity
Each feature is defined **once** as a `FeatureDefinition` with two compute paths: `offline` (vectorized pandas for batch training) and `online` (window-based for single-row serving). A property test asserts they produce numerically identical values at any time T. This eliminates training/serving skew — the most common cause of silent model degradation in production.

### Point-in-time correctness
Label for date T uses `close[T+5]` (future data). Features for date T use only data up to and including T. The shift is applied **per ticker** via `groupby` to prevent the last row of AAPL bleeding into MSFT's label window. Verified by a leakage test that truncates the DataFrame at T and asserts feature values are unchanged.

### Feature set versioning
A `FeatureSet` has a `name` and `version`. Every MLflow run records `feature_set_version`. The serving layer loads the matching feature set at startup. Adding features creates a new version (V2) rather than mutating V1 — models trained on V1 and V2 remain independently valid and loadable.

### Champion/challenger promotion
After each training run, the new model's `roc_auc` is compared to the current champion's. Promotion requires beating the champion by `PROMOTION_THRESHOLD = 0.005`. Every run is registered (for audit), but only better models are promoted. This prevents a bad training run from silently replacing a working model in production.

### Reproducibility
The same `--run-date` always produces the same model. Enforced by: deterministic split date (`run_date - 90 days`), pinned `random_state=42`, and a `train_data_fingerprint` (MD5 of the training DataFrame) logged to MLflow. Two runs with the same fingerprint used identical training data.

### Statistical drift detection
At training time, per-feature mean and std are saved to `training_stats.json`. At inference time, `DriftDetector.check()` computes a z-score for each feature. Any feature beyond 3σ is flagged in the response and counted in the `drift_events_total` Prometheus counter. This catches distribution shift before accuracy degrades — TSLA's `macd_signal` was flagged at z=6.4 in testing.

### Hot model reload
`POST /reload` atomically swaps the in-process model using `threading.Lock`. The serving layer loads the champion by MLflow **alias** (`models:/direction_classifier@champion`) rather than by version number — so deploying a new champion only requires calling `/reload`, not restarting.

---

## Test Suite

```
64 tests, 0 failures
├── test_ingestion.py      7   schema validation, streaming generators
├── test_labels.py         5   label correctness, no cross-ticker bleed
├── test_features.py       4   baseline (V1) feature selection
├── test_feature_store.py  23  offline/online parity (V1+V2), leakage, backfill
├── test_drift.py          4   3σ detection, multi-feature flagging, graceful skips
├── test_performance.py    4   correct/incorrect evaluation, pending/skipped logic
├── test_registry.py       4   register, promote, champion alias, replacement
├── test_runner.py         7   date arithmetic, promotion logic, dry-run
├── test_serving.py        6   all endpoints, 503 readiness, schema validation
└── test_integration.py    3   real data end-to-end (skipped in CI without data)
```

Run:
```bash
uv run pytest tests/ -v                    # all tests
uv run pytest tests/ -v -k "not integration"  # exclude integration tests
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Data | pandas, yfinance, pyarrow |
| ML | XGBoost, scikit-learn |
| Experiment tracking | MLflow |
| Serving | FastAPI, uvicorn |
| Metrics | prometheus-client |
| Logging | structlog |
| Tests | pytest |
| Env management | uv |

---

## Resume Bullets

- Designed a **feature store** with offline/online compute parity enforced by property tests — eliminates training/serving skew, the most common cause of silent model degradation in production ML systems

- Implemented **feature set versioning** (V1/V2): models record the exact feature version they were trained on; the serving layer loads the matching version — prevents silent shape mismatches across retrains

- Built a **champion/challenger promotion pipeline**: new models are registered and evaluated against the current champion; promotion requires exceeding AUC threshold — prevents bad training runs from silently replacing a working production model

- Added **statistical drift detection**: per-feature mean/std saved at training time; inference-time values beyond 3σ are flagged in the API response and counted in Prometheus — early warning before accuracy degrades (demonstrated catching a 6.4σ MACD anomaly on TSLA)

- Implemented **hot model reload**: `POST /reload` atomically swaps the in-process model using a threading lock and MLflow alias resolution — zero-downtime deployments without pod restarts

- Enforced **point-in-time correctness** in label generation: shift applied per-ticker via groupby, preventing cross-ticker label bleed; verified by a leakage test that truncates future data and asserts feature values are unchanged

- Guaranteed **reproducibility**: same `--run-date` always produces the same model via deterministic split dates, pinned random seeds, and a training data MD5 fingerprint logged to MLflow
