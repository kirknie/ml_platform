"""
Integration tests: real Parquet data → online features → champion model → prediction.

No mocks. These tests exercise the full serving path using files that exist
on disk from the training pipeline. They are skipped automatically if the
required data files or champion model are not present (e.g. in CI).
"""

from pathlib import Path

import pytest

RAW_AAPL = Path("data/raw/AAPL.parquet")
V2_AAPL = Path("data/processed/engineered_v2/AAPL.parquet")

pytestmark = pytest.mark.skipif(
    not RAW_AAPL.exists() or not V2_AAPL.exists(),
    reason="Integration tests require fetched raw data and computed V2 features",
)


def test_v2_feature_store_all_tickers():
    """V2 offline features exist for all tickers, are non-null, and have 8 columns."""
    from features.definitions import ENGINEERED_V2
    from features.store import FeatureStore
    from ingestion.fetch import TICKERS

    store = FeatureStore(ENGINEERED_V2)
    for ticker in TICKERS:
        df = store.load_offline(ticker)
        assert len(df) > 0, f"{ticker}: no rows"
        assert set(ENGINEERED_V2.feature_names).issubset(df.columns)
        null_count = df[ENGINEERED_V2.feature_names].isnull().sum().sum()
        assert null_count == 0, f"{ticker}: {null_count} null feature values"


def test_predict_endpoint_with_real_data():
    """
    Full serving path with real data:
    raw Parquet → online feature compute → champion model → valid response schema.
    """
    from registry.model_registry import ModelRegistry
    import serving.app as app_module
    from fastapi.testclient import TestClient

    registry = ModelRegistry()
    if registry.get_champion() is None:
        pytest.skip("No champion model registered — run training.runner first")

    app_module._load_model()
    client = TestClient(app_module.app)

    resp = client.post("/predict", json={"ticker": "AAPL"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["direction"] in {"UP", "DOWN"}
    assert 0.0 < body["probability"] < 1.0
    assert body["model_version"] is not None
    assert body["feature_set"] == "engineered_v2"
    assert isinstance(body["drift_detected"], bool)
    assert isinstance(body["drifted_features"], list)


def test_drift_detector_loaded_from_training_stats():
    """Drift detector loads real training stats and produces valid output."""
    from features.definitions import ENGINEERED_V2
    from features.store import FeatureStore, PROCESSED_DATA_DIR

    stats_path = PROCESSED_DATA_DIR / "engineered_v2" / "training_stats.json"
    if not stats_path.exists():
        pytest.skip("No training_stats.json — run training pipeline first")

    from monitoring.drift import DriftDetector

    detector = DriftDetector(stats_path=stats_path)
    assert set(detector.stats.keys()) == set(ENGINEERED_V2.feature_names)

    # Values within normal range should not drift
    store = FeatureStore(ENGINEERED_V2)
    sample = store.load_offline("AAPL").iloc[500]
    features = {col: float(sample[col]) for col in ENGINEERED_V2.feature_names}
    result = detector.check(features)
    assert "drift_detected" in result
    assert isinstance(result["drifted_features"], list)
