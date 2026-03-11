"""
Tests for registry/model_registry.py.

Uses a temporary SQLite-backed MLflow server so tests don't touch the
production mlruns directory. Each test gets a fresh isolated registry.
"""

import mlflow
import mlflow.sklearn
import pytest
from sklearn.dummy import DummyClassifier

from registry.model_registry import CHAMPION_ALIAS, MODEL_NAME, ModelRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracking_uri(tmp_path):
    """Isolated MLflow SQLite backend for each test."""
    uri = f"sqlite:///{tmp_path}/test_mlruns.db"
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri("file:./mlruns")


@pytest.fixture
def registry(tracking_uri):
    return ModelRegistry(tracking_uri=tracking_uri)


def _log_dummy_run(roc_auc: float = 0.52) -> str:
    """Log a minimal sklearn model to MLflow and return the run_id."""
    mlflow.set_experiment("test_experiment")
    with mlflow.start_run() as run:
        model = DummyClassifier(strategy="most_frequent")
        model.fit([[1], [2]], [0, 1])
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_metric("roc_auc", roc_auc)
    return run.info.run_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_champion_returns_none_initially(registry):
    """No champion exists on a fresh registry."""
    assert registry.get_champion() is None


def test_register_creates_model_version(registry):
    """register() should create a versioned entry in the model registry."""
    run_id = _log_dummy_run()
    version = registry.register(run_id)

    assert version is not None
    assert version.name == MODEL_NAME
    assert int(version.version) >= 1


def test_promote_sets_champion_alias(registry):
    """After promote(), get_champion() should return the promoted version."""
    run_id = _log_dummy_run(roc_auc=0.52)
    version = registry.register(run_id)
    registry.promote(version)

    champion = registry.get_champion()
    assert champion is not None
    assert champion.version == version.version


def test_promote_replaces_previous_champion(registry):
    """Promoting a new version replaces the old champion alias."""
    run_id_1 = _log_dummy_run(roc_auc=0.52)
    v1 = registry.register(run_id_1)
    registry.promote(v1)

    run_id_2 = _log_dummy_run(roc_auc=0.53)
    v2 = registry.register(run_id_2)
    registry.promote(v2)

    champion = registry.get_champion()
    assert champion.version == v2.version
    assert champion.version != v1.version
