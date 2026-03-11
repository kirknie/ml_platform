"""
Model registry wrapper for the ML platform.

Uses MLflow's model registry to version and promote models.

Design decisions:
- Alias-based promotion (not stages): MLflow 3.x deprecated stage transitions
  (Staging/Production). We use the `champion` alias instead, which is the
  current recommended approach.
- Single model name: all versions of the direction classifier live under
  MODEL_NAME. The alias "champion" always points to the version in production.
- Explicit tracking URI: set in the constructor so the registry always hits
  the same backend, regardless of what calling code has configured globally.

Usage pattern (from runner.py):
    registry = ModelRegistry()
    version = registry.register(run_id)      # creates version in registry
    registry.promote(version)                 # sets "champion" alias
    model = registry.load_champion()          # load for serving
"""

import logging

import mlflow
import mlflow.pyfunc
from mlflow.entities.model_registry import ModelVersion
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

TRACKING_URI = "file:./mlruns"
MODEL_NAME = "direction_classifier"
CHAMPION_ALIAS = "champion"


class ModelRegistry:
    def __init__(self, tracking_uri: str = TRACKING_URI) -> None:
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def register(self, run_id: str, artifact_path: str = "model") -> ModelVersion:
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
        logger.info(
            "Registered %s version %s from run %s", MODEL_NAME, version.version, run_id
        )
        return version

    def get_champion(self) -> ModelVersion | None:
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

    def promote(self, version: ModelVersion) -> None:
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

    def load_champion(self) -> PyFuncModel:
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
