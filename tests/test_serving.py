"""
Tests for serving/app.py.

Uses FastAPI's TestClient. Model and feature store interactions are mocked
so tests run without a trained champion or raw data files.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """TestClient with model state pre-loaded via mocks."""
    import serving.app as app_module

    mock_model = app_module._state.model.__class__.__new__(
        app_module._state.model.__class__
    ) if app_module._state.model else None

    # Build a minimal mock model with predict / predict_proba
    from unittest.mock import MagicMock
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.38, 0.62]])

    app_module._state.model = mock_model
    app_module._state.model_version = "5"
    app_module._state.drift_detector = None

    return TestClient(app_module.app)


def _make_fake_raw(n: int = 110) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame large enough for all V2 features."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "open": 150.0, "high": 152.0, "low": 149.0,
            "close": 151.0, "volume": 1_000_000.0, "ticker": "AAPL",
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_when_model_loaded(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    assert resp.json()["model_version"] == "5"


def test_ready_503_when_no_model():
    """When no model is loaded, /ready should return 503."""
    import serving.app as app_module
    original = app_module._state.model
    app_module._state.model = None
    c = TestClient(app_module.app, raise_server_exceptions=False)
    resp = c.get("/ready")
    assert resp.status_code == 503
    app_module._state.model = original  # restore


def test_predict_returns_correct_schema(client):
    """POST /predict should return all required response fields."""
    with patch("serving.app.load_ticker", return_value=_make_fake_raw()):
        resp = client.post("/predict", json={"ticker": "AAPL"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["direction"] in {"UP", "DOWN"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "5"
    assert "feature_set" in body
    assert isinstance(body["drift_detected"], bool)
    assert isinstance(body["drifted_features"], list)


def test_predict_404_for_unknown_ticker(client):
    """POST /predict with a ticker that has no data file returns 404."""
    with patch("serving.app.load_ticker", side_effect=FileNotFoundError("no file")):
        resp = client.post("/predict", json={"ticker": "FAKE"})
    assert resp.status_code == 404


def test_metrics_endpoint(client):
    """GET /metrics should return Prometheus text format."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "prediction_latency_seconds" in resp.text
    assert "predictions_total" in resp.text
