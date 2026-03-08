# Week 7 Execution Plan: Training Pipeline & Model Registry

## Goal

By end of week 7 you have:
- `run_engineered_pipeline()` accepts configurable dates and returns `(run_id, metrics)` — the
  interface the automated runner needs
- A data fingerprint logged to every MLflow run, so you can verify two runs used identical
  training data
- A `ModelRegistry` class wrapping MLflow's registry — registers models, manages the
  **champion alias**, and promotes challengers
- A `training/runner.py` CLI entrypoint that computes a dynamic train/test split, trains,
  registers, and promotes or rejects the challenger in one command
- 9 new tests covering the runner's date logic and the registry's promotion logic

---

## Why This Architecture Matters (Read First)

### Dynamic split dates vs fixed date

Week 5–6 hardcoded `TEST_SPLIT_DATE = "2023-01-01"`. That works for a one-time run but
breaks when you want to retrain on new data: the test set would be the same stale window
forever, and new data would only enter the training set — the model would never be evaluated
on the freshest data.

The fix: compute split date dynamically as `run_date - evaluation_window_days`. If you run
weekly, the test set always covers the most recent 90 days.

### Champion/challenger promotion

Without a registry, every retrain silently replaces the previous model. The serving layer
has no way to know whether the new model is better or worse.

The champion/challenger pattern:
1. Train a **challenger** model
2. Compare its `roc_auc` on the test set to the current **champion's** `roc_auc`
3. If challenger improves by more than a threshold (0.005), promote it to champion
4. Otherwise, register the challenger (for audit) but keep the current champion in production

This prevents silent model degradation from bad retraining runs.

### Reproducibility guarantee

A run is reproducible if: given the same `run_date`, it produces the same training data,
same features, and the same model. We enforce this by:
- Dynamic split date is a pure function of `run_date` (deterministic)
- Feature set version is logged
- Data fingerprint (MD5 of training DataFrame) is logged — two runs with the same fingerprint
  used identical training data

---

## Repo Changes

New files:
```
registry/
├── __init__.py
└── model_registry.py    # ModelRegistry: register, promote, load champion
training/
└── runner.py            # CLI entrypoint: train → register → champion/challenger
tests/
├── test_runner.py       # 5 tests (date logic + promotion logic, all mocked)
└── test_registry.py     # 4 tests (real MLflow operations against tmp_path)
```

Modified files:
```
training/pipeline.py     # run_engineered_pipeline() now accepts params + returns (run_id, metrics)
```

---

## Step 1: Refactor `training/pipeline.py`

### What changes

`run_engineered_pipeline()` currently:
- Hardcodes `TEST_SPLIT_DATE`
- Returns `None`

The runner needs to:
- Pass a dynamic split date
- Capture the MLflow `run_id` to register the model
- Receive `metrics` to decide on promotion

Three changes:
1. Accept `test_split_date: str` and `run_date: str | None` parameters
2. Capture and return `(run_id, metrics)`
3. Log `run_date` and `train_data_fingerprint` as MLflow params

### 1.1 Add fingerprint helper

Add this import and helper **before** `build_dataset_from_store` in `pipeline.py`:

```python
import hashlib
```

```python
def _fingerprint(df: pd.DataFrame) -> str:
    """
    MD5 hash of a DataFrame's values for reproducibility auditing.

    Two runs that produce the same fingerprint used identical training data.
    Logged as a param so you can verify reproducibility from the MLflow UI.
    """
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()
```

### 1.2 Update `run_engineered_pipeline` signature and return value

Replace the existing `run_engineered_pipeline()` with:

```python
def run_engineered_pipeline(
    test_split_date: str = TEST_SPLIT_DATE,
    run_date: str | None = None,
) -> tuple[str, dict]:
    """
    Train XGBoost on engineered features and log to MLflow.

    Args:
        test_split_date: Train/test boundary date (YYYY-MM-DD). Everything before
                         this date is train, on or after is test.
        run_date: Logical run date for metadata (YYYY-MM-DD). Defaults to None
                  (not logged). Pass explicitly for reproducible scheduled runs.

    Returns:
        (run_id, metrics) — the MLflow run ID and evaluation metrics dict.
        run_id is used by ModelRegistry.register() to locate the model artifact.
    """
    store = FeatureStore(ENGINEERED_V1)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="engineered_xgboost") as active_run:
        df = build_dataset_from_store(store)
        train_df, test_df = time_split(df, test_split_date)

        feature_cols = store.feature_set.feature_names
        model, metrics, params = train(train_df, test_df, feature_cols=feature_cols)

        mlflow.log_params(params)
        mlflow.log_params({
            "tickers": ",".join(TICKERS),
            "test_split_date": test_split_date,
            "feature_set": store.feature_set.name,
            "feature_set_version": store.feature_set.version,
            "feature_columns": ",".join(feature_cols),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_data_fingerprint": _fingerprint(train_df),
        })
        if run_date is not None:
            mlflow.log_param("run_date", run_date)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")

        run_id = active_run.info.run_id

    logger.info("Engineered metrics: %s", metrics)
    return run_id, metrics
```

Also update `__main__` since `run_engineered_pipeline` now returns a value:

```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- Baseline (raw OHLCV) ---")
    run_pipeline()
    print("\n--- Engineered features ---")
    run_id, metrics = run_engineered_pipeline()
    print(f"\n  run_id: {run_id}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
```

### 1.3 Verify the refactored pipeline still works

```bash
uv run python -c "
from training.pipeline import run_engineered_pipeline
run_id, metrics = run_engineered_pipeline()
print('run_id:', run_id)
print('roc_auc:', metrics['roc_auc'])
"
```

Expected: a UUID run_id and roc_auc near 0.49–0.53.

---

## Step 2: Create `registry/model_registry.py`

### What it does

`ModelRegistry` is a thin wrapper around `mlflow.tracking.MlflowClient`. It encapsulates
all registry operations behind three meaningful methods:
- `register(run_id)` — creates a new version entry in the registry
- `promote(version)` — sets the `"champion"` alias on a version (MLflow 3.x alias API,
  replaces the deprecated stage API)
- `get_champion()` / `get_champion_metrics()` — look up the current production model
- `load_champion()` — load the champion model for inference (used in Week 8 serving)

### 2.1 Create `registry/__init__.py` (empty)

### 2.2 Create `registry/model_registry.py`

```python
"""
Model registry wrapper for the ML platform.

Uses MLflow's model registry to version and promote models.

Design decisions:
- Alias-based promotion (not stages): MLflow 3.x deprecated stage transitions
  (Staging/Production). We use the `champion` alias instead, which is the
  current recommended approach.
- Single model name: all versions of the direction classifier live under
  MODEL_NAME. The alias "champion" always points to the version in production.
- Separate tracking URI: set explicitly so the registry always hits the same
  backend, regardless of what calling code has configured globally.

Usage pattern (from runner.py):
    registry = ModelRegistry()
    version = registry.register(run_id)      # creates version in registry
    registry.promote(version)                 # sets "champion" alias
    model = registry.load_champion()          # load for serving
"""

import logging

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

TRACKING_URI = "file:./mlruns"
MODEL_NAME = "direction_classifier"
CHAMPION_ALIAS = "champion"


class ModelRegistry:
    def __init__(self, tracking_uri: str = TRACKING_URI) -> None:
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def register(self, run_id: str, artifact_path: str = "model") -> mlflow.entities.model_registry.ModelVersion:
        """
        Register the model artifact from a run into the MLflow model registry.

        Creates the registered model if it doesn't exist yet. Each call creates
        a new version number (auto-incremented by MLflow).

        Args:
            run_id: The MLflow run ID that logged the model artifact
            artifact_path: Path within the run where the model was logged

        Returns:
            ModelVersion object with .version (int) and .run_id
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        version = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME,
            tags={"run_id": run_id},
        )
        logger.info("Registered %s version %s from run %s", MODEL_NAME, version.version, run_id)
        return version

    def get_champion(self) -> mlflow.entities.model_registry.ModelVersion | None:
        """
        Return the ModelVersion currently aliased as champion, or None.

        Returns None both when no model has been registered and when no
        version has been promoted to champion yet.
        """
        try:
            return self.client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        except mlflow.exceptions.MlflowException:
            return None

    def get_champion_metrics(self) -> dict | None:
        """
        Return the evaluation metrics from the champion model's training run.

        Returns None if there is no champion yet.
        """
        champion = self.get_champion()
        if champion is None:
            return None
        run = self.client.get_run(champion.run_id)
        return dict(run.data.metrics)

    def promote(self, version: mlflow.entities.model_registry.ModelVersion) -> None:
        """
        Set the given model version as the champion.

        Uses MLflow's alias API (MLflow 3.x). Safe to call multiple times —
        the alias simply moves to the new version, no cleanup needed.

        Args:
            version: ModelVersion returned by register()
        """
        self.client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=CHAMPION_ALIAS,
            version=version.version,
        )
        logger.info(
            "Promoted %s version %s as '%s'",
            MODEL_NAME,
            version.version,
            CHAMPION_ALIAS,
        )

    def load_champion(self) -> mlflow.pyfunc.PyFuncModel:
        """
        Load the current champion model for inference.

        Returns an MLflow PyFunc model, compatible with any framework.
        Call model.predict(X) where X is a pandas DataFrame.

        Raises:
            RuntimeError: If no champion has been promoted yet
        """
        if self.get_champion() is None:
            raise RuntimeError(
                f"No '{CHAMPION_ALIAS}' alias found for model '{MODEL_NAME}'. "
                "Run `python -m training.runner` first."
            )
        model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
        return mlflow.pyfunc.load_model(model_uri)
```

### 2.3 Verify the registry module imports

```bash
uv run python -c "from registry.model_registry import ModelRegistry, MODEL_NAME; print(MODEL_NAME)"
```

Expected: `direction_classifier`

---

## Step 3: Create `training/runner.py`

### What it does

The runner is the automated entrypoint. It:
1. Computes the split date dynamically from `run_date`
2. Calls `run_engineered_pipeline()` to train and get `(run_id, metrics)`
3. Compares challenger to champion via `evaluate_and_promote()`
4. Prints a clear summary

It's designed to be called by cron: `0 6 * * 1 cd /path/to/ml_platform && uv run python -m training.runner`

### 3.1 Create `training/runner.py`

```python
"""
Automated training runner with champion/challenger promotion.

Usage:
    # Train with today as run_date, promote if challenger beats champion
    python -m training.runner

    # Reproducible historical run (same run_date = same split = same result)
    python -m training.runner --run-date 2024-06-01

    # Train and register but never promote (for testing pipeline)
    python -m training.runner --dry-run

Design decisions:
- Dynamic split date: test set always covers the most recent EVALUATION_WINDOW_DAYS.
  This ensures the model is always evaluated on the freshest available data.
- PROMOTION_THRESHOLD: challenger must beat champion by >0.005 AUC to promote.
  Small random fluctuations shouldn't trigger a promotion.
- Always register, conditionally promote: every run creates a version in the
  registry for audit purposes, even if it doesn't become champion.
- Dry run: runs the full training pipeline and evaluates, but skips registry
  writes. Useful for CI checks and local testing.
"""

import argparse
import logging
from datetime import date, timedelta

from registry.model_registry import ModelRegistry
from training.pipeline import run_engineered_pipeline

logger = logging.getLogger(__name__)

EVALUATION_WINDOW_DAYS = 90  # test set covers this many days before run_date
PROMOTION_THRESHOLD = 0.005  # challenger must beat champion by at least this much


def compute_split_date(run_date: date, eval_window: int = EVALUATION_WINDOW_DAYS) -> str:
    """
    Compute the train/test split date relative to run_date.

    Everything before split_date is training data.
    Everything from split_date to run_date is the evaluation window.

    Example:
        run_date=2024-12-31, eval_window=90 → split_date=2024-10-02

    Args:
        run_date: The logical date of this training run
        eval_window: Number of days to hold out for evaluation

    Returns:
        split_date as YYYY-MM-DD string
    """
    split = run_date - timedelta(days=eval_window)
    return split.strftime("%Y-%m-%d")


def evaluate_and_promote(
    run_id: str,
    challenger_metrics: dict,
    registry: ModelRegistry,
    dry_run: bool = False,
) -> bool:
    """
    Compare challenger to current champion and promote if it's meaningfully better.

    Promotion logic:
    - If no champion exists: always promote (first run)
    - If challenger.roc_auc - champion.roc_auc >= PROMOTION_THRESHOLD: promote
    - Otherwise: register (for audit) but keep the current champion

    Args:
        run_id: MLflow run ID of the challenger training run
        challenger_metrics: Metrics dict from the challenger's evaluation
        registry: ModelRegistry instance
        dry_run: If True, evaluate but skip all registry writes

    Returns:
        True if challenger was promoted (or would have been in dry_run), else False
    """
    challenger_auc = challenger_metrics.get("roc_auc", 0.0)
    champion_metrics = registry.get_champion_metrics()

    if champion_metrics is None:
        logger.info("No champion exists. Promoting challenger as first champion.")
        if not dry_run:
            version = registry.register(run_id)
            registry.promote(version)
        return True

    champion_auc = champion_metrics.get("roc_auc", 0.0)
    improvement = challenger_auc - champion_auc

    logger.info(
        "Challenger roc_auc=%.4f  |  Champion roc_auc=%.4f  |  "
        "improvement=%.4f  (threshold=%.4f)",
        challenger_auc,
        champion_auc,
        improvement,
        PROMOTION_THRESHOLD,
    )

    # Always register for audit — even non-promoted runs are tracked
    if not dry_run:
        version = registry.register(run_id)

    if improvement >= PROMOTION_THRESHOLD:
        logger.info("Challenger beats threshold. Promoting to champion.")
        if not dry_run:
            registry.promote(version)
        return True

    logger.info("Challenger does not beat threshold. Current champion retained.")
    return False


def run(run_date: date | None = None, dry_run: bool = False) -> None:
    """
    Full training run: compute split → train → register → evaluate → maybe promote.

    Args:
        run_date: Logical run date. Defaults to today. Pass explicitly for
                  reproducible historical runs.
        dry_run: If True, train and evaluate but skip registry writes.
    """
    if run_date is None:
        run_date = date.today()

    split_date = compute_split_date(run_date)
    logger.info("run_date=%s  split_date=%s  dry_run=%s", run_date, split_date, dry_run)

    run_id, metrics = run_engineered_pipeline(
        test_split_date=split_date,
        run_date=str(run_date),
    )

    registry = ModelRegistry()
    promoted = evaluate_and_promote(run_id, metrics, registry, dry_run=dry_run)

    print("\n" + "=" * 50)
    print("Training run complete")
    print(f"  run_date:  {run_date}")
    print(f"  split:     {split_date}")
    print(f"  run_id:    {run_id}")
    print(f"  roc_auc:   {metrics.get('roc_auc', float('nan')):.4f}")
    print(f"  promoted:  {promoted}")
    if dry_run:
        print("  (dry_run: registry writes skipped)")
    print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train and optionally promote a new model.")
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Logical run date YYYY-MM-DD (default: today). Use for reproducible reruns.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and evaluate but skip all registry writes.",
    )
    args = parser.parse_args()
    run_date = date.fromisoformat(args.run_date) if args.run_date else None
    run(run_date=run_date, dry_run=args.dry_run)
```

### 3.2 Verify end-to-end

First run (no champion exists — should always promote):

```bash
uv run python -m training.runner --run-date 2024-12-01
```

Expected output:
```
==================================================
Training run complete
  run_date:  2024-12-01
  split:     2024-09-02
  run_id:    <uuid>
  roc_auc:   ~0.49–0.53
  promoted:  True
==================================================
```

Second run with the same date (challenger vs champion, should NOT promote since it's identical):

```bash
uv run python -m training.runner --run-date 2024-12-01
```

Expected:
```
  promoted:  False
```

Because the challenger and champion used the same data and features, their AUC values are
identical — improvement = 0, which is below `PROMOTION_THRESHOLD = 0.005`.

### 3.3 Verify reproducibility

Run twice with the same date and confirm identical metrics:

```bash
uv run python -c "
from training.pipeline import run_engineered_pipeline
run_id_1, m1 = run_engineered_pipeline(test_split_date='2024-09-02', run_date='2024-12-01')
run_id_2, m2 = run_engineered_pipeline(test_split_date='2024-09-02', run_date='2024-12-01')
print('Run 1 roc_auc:', round(m1['roc_auc'], 6))
print('Run 2 roc_auc:', round(m2['roc_auc'], 6))
print('Identical:', m1['roc_auc'] == m2['roc_auc'])
"
```

Expected: `Identical: True`. The same `random_state=42`, same data, same features → same model.

Also verify the data fingerprint is the same across both runs in MLflow:

```bash
uv run mlflow ui --port 5000
# Open http://localhost:5000
# Both runs should show the same train_data_fingerprint param
```

---

## Step 4: Tests

### 4.1 Create `tests/test_runner.py`

All 5 tests use `unittest.mock` — no network calls, no MLflow server, fast.

```python
"""
Tests for training/runner.py.

All tests mock the ModelRegistry to isolate runner logic from registry I/O.
"""

from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

from training.runner import (
    EVALUATION_WINDOW_DAYS,
    PROMOTION_THRESHOLD,
    compute_split_date,
    evaluate_and_promote,
)


# ---------------------------------------------------------------------------
# compute_split_date
# ---------------------------------------------------------------------------

def test_compute_split_date_default_window():
    """Split date should be run_date minus EVALUATION_WINDOW_DAYS."""
    run_date = date(2024, 12, 31)
    result = compute_split_date(run_date)
    expected = date(2024, 12, 31) - __import__("datetime").timedelta(days=EVALUATION_WINDOW_DAYS)
    assert result == expected.strftime("%Y-%m-%d")


def test_compute_split_date_custom_window():
    """Custom eval_window overrides the default."""
    run_date = date(2024, 6, 1)
    result = compute_split_date(run_date, eval_window=30)
    assert result == "2024-05-02"


# ---------------------------------------------------------------------------
# evaluate_and_promote
# ---------------------------------------------------------------------------

def test_evaluate_promotes_when_no_champion():
    """With no champion, challenger should always be promoted."""
    registry = MagicMock()
    registry.get_champion_metrics.return_value = None  # no champion
    mock_version = MagicMock()
    registry.register.return_value = mock_version

    promoted = evaluate_and_promote("run-123", {"roc_auc": 0.50}, registry)

    assert promoted is True
    registry.register.assert_called_once_with("run-123")
    registry.promote.assert_called_once_with(mock_version)


def test_evaluate_promotes_when_challenger_wins():
    """Challenger beating champion by >= PROMOTION_THRESHOLD triggers promotion."""
    registry = MagicMock()
    registry.get_champion_metrics.return_value = {"roc_auc": 0.50}
    mock_version = MagicMock()
    registry.register.return_value = mock_version

    challenger_auc = 0.50 + PROMOTION_THRESHOLD  # exactly at threshold
    promoted = evaluate_and_promote("run-456", {"roc_auc": challenger_auc}, registry)

    assert promoted is True
    registry.promote.assert_called_once_with(mock_version)


def test_evaluate_does_not_promote_below_threshold():
    """Improvement below PROMOTION_THRESHOLD: model is registered but not promoted."""
    registry = MagicMock()
    registry.get_champion_metrics.return_value = {"roc_auc": 0.50}
    mock_version = MagicMock()
    registry.register.return_value = mock_version

    challenger_auc = 0.50 + PROMOTION_THRESHOLD - 0.001  # just below threshold
    promoted = evaluate_and_promote("run-789", {"roc_auc": challenger_auc}, registry)

    assert promoted is False
    registry.register.assert_called_once_with("run-789")  # still registered
    registry.promote.assert_not_called()                   # but not promoted


def test_dry_run_skips_registry_writes():
    """dry_run=True should evaluate but make no registry calls."""
    registry = MagicMock()
    registry.get_champion_metrics.return_value = None  # no champion

    promoted = evaluate_and_promote("run-000", {"roc_auc": 0.55}, registry, dry_run=True)

    assert promoted is True  # would have promoted
    registry.register.assert_not_called()
    registry.promote.assert_not_called()
```

### 4.2 Create `tests/test_registry.py`

4 tests using a real (temporary) MLflow tracking server in `tmp_path`. Uses a tiny
`DummyClassifier` to keep the fixture fast — the registry doesn't care about model type.

```python
"""
Tests for registry/model_registry.py.

Uses a temporary SQLite-backed MLflow server so tests don't touch the production
mlruns directory. Each test gets a fresh registry with no existing models.
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
    # Reset to avoid polluting other tests
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
    assert champion.version == v2.version  # champion is now v2
    assert champion.version != v1.version  # v1 is no longer champion
```

### 4.3 Run all tests

```bash
uv run pytest tests/ -v
```

Expected: **40 tests pass** (31 from weeks 5–6 + 9 new):
- `tests/test_runner.py` — 5 tests
- `tests/test_registry.py` — 4 tests
- `tests/test_feature_store.py` — 15 tests
- `tests/test_features.py` — 4 tests
- `tests/test_ingestion.py` — 7 tests
- `tests/test_labels.py` — 5 tests

---

## Step 5: Run All Tests

```bash
uv run pytest tests/ -v
```

---

## Step 6: Validation Checklist

- [ ] `pytest tests/ -v` — all 40 tests pass
- [ ] `python -m training.runner --run-date 2024-12-01` runs end-to-end and prints
      `promoted: True` on first run
- [ ] Second run with same `--run-date` prints `promoted: False` (same data, same AUC,
      no improvement above threshold)
- [ ] `python -m training.runner --dry-run` completes without writing to `mlruns/`
- [ ] MLflow UI shows a `train_data_fingerprint` param on engineered runs
- [ ] MLflow UI shows `run_date` param when passed explicitly
- [ ] Model is visible in MLflow UI under **Models → direction_classifier**
- [ ] The `champion` alias is set on the promoted version
- [ ] Two runs with the same `--run-date` produce identical `roc_auc` (reproducibility)
- [ ] No `FutureWarning` or unhandled exceptions during the runner

---

## Common Pitfalls

### "MlflowException: RESOURCE_DOES_NOT_EXIST: Registered model alias champion not found"

This is the expected exception that `get_champion()` catches and returns `None` for.
If you see it propagating, the `except mlflow.exceptions.MlflowException` block is not
catching it — check you're importing `mlflow.exceptions` correctly.

### "Runner prints promoted: False on first run"

`get_champion_metrics()` returned something (not None), meaning a champion was already
registered from a previous test or manual run. Check `mlruns/` — there may be a leftover
registry entry. You can reset by deleting `mlruns/` and re-running.

### "Second run with same --run-date has different roc_auc"

Check that `random_state=42` is set in `train()`. Also check that the raw data files
haven't been re-fetched (which could introduce slightly different data if yfinance returns
different adjusted prices for historical dates — this is a known yfinance behavior).

### "test_registry.py tests fail with registry errors"

The SQLite fixture may not be resetting the global MLflow tracking URI between tests.
Make sure each test uses the `registry` fixture (which uses the `tracking_uri` fixture),
and the `tracking_uri` fixture resets to `"file:./mlruns"` in its teardown.

---

## Week 7 → Week 8 Handoff

At end of week 7 you have:
- A champion model in the MLflow registry, accessible via `registry.load_champion()`
- `ModelRegistry.load_champion()` returns an MLflow PyFunc model ready for `model.predict(X)`
- The `compute_online(window)` method in `FeatureStore` ready to compute features for a single row

Week 8 builds the **inference API**:
- FastAPI service with a `/predict` endpoint
- Loads the champion model on startup via `ModelRegistry.load_champion()`
- Accepts a ticker symbol, fetches the last `max_required_window` rows via
  `stream_ticker_window`, calls `FeatureStore.compute_online(window)` to get features,
  and returns a prediction
- Prometheus metrics: prediction latency (p50/p99), prediction count, model version served
- Hot reload: re-load the champion when the `/reload` endpoint is called (no restart needed)

Key question to answer before Week 8: **what is the serving contract?** The endpoint
receives a ticker and returns `{"ticker": "AAPL", "prediction": 1, "probability": 0.58,
"model_version": "3", "feature_set": "engineered_v1"}`. Design this schema before writing
the handler.
