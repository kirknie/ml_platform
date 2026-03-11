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
        self.model = None
        self.model_version: str | None = None
        self.drift_detector: DriftDetector | None = None
        self._lock = threading.Lock()

    def swap(
        self,
        model,
        version: str,
        detector: DriftDetector | None,
    ) -> None:
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

    stats_path = (
        PROCESSED_DATA_DIR
        / f"{ENGINEERED_V2.name}_v{ENGINEERED_V2.version}"
        / "training_stats.json"
    )
    detector = DriftDetector(stats_path=stats_path) if stats_path.exists() else None
    if detector is None:
        logger.warning(
            "No training stats at %s. Drift detection disabled.", stats_path
        )

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
    direction: str        # "UP" or "DOWN"
    probability: float    # P(UP), from predict_proba
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
        try:
            raw_df = load_ticker(req.ticker, data_dir=RAW_DATA_DIR)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"No data for ticker '{req.ticker}'. Run fetch_ticker() first.",
            )

        window = raw_df.tail(_store.feature_set.max_required_window)
        prediction_date = str(window.index[-1].date())
        features = _store.compute_online(window)

        X = pd.DataFrame([features])[_store.feature_set.feature_names]

        prediction = int(_state.model.predict(X)[0])
        probability = float(_state.model.predict_proba(X)[0][1])  # P(UP)
        direction = "UP" if prediction == 1 else "DOWN"

        drift_result = (
            _state.drift_detector.check(features)
            if _state.drift_detector
            else {"drift_detected": False, "drifted_features": []}
        )

    PREDICTIONS_TOTAL.labels(ticker=req.ticker, direction=direction).inc()
    if drift_result["drift_detected"]:
        DRIFT_EVENTS_TOTAL.labels(ticker=req.ticker).inc()

    _log_prediction(
        ticker=req.ticker,
        direction=direction,
        probability=probability,
        model_version=_state.model_version,
        prediction_date=prediction_date,
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
    prediction_date: str | None = None,
) -> None:
    """Append one prediction to the JSONL log. Best-effort — never raises."""
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "prediction_date": prediction_date,
        "direction": direction,
        "probability": probability,
        "model_version": model_version,
    }
    try:
        with PREDICTIONS_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.warning("Failed to write prediction log: %s", exc)
