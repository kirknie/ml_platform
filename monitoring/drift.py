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
    def __init__(
        self,
        stats: dict | None = None,
        stats_path: Path | None = None,
    ) -> None:
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
        self,
        features: dict[str, float],
        n_sigma: float = 3.0,
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
                "drifted_features": list[str],  # names of features that drifted
            }
        """
        drifted = []
        for name, val in features.items():
            feature_stats = self.stats.get(name)
            if feature_stats is None:
                continue  # feature not in training stats — skip gracefully
            std = feature_stats["std"]
            if std == 0:
                continue  # constant feature — z-score undefined
            z = abs(val - feature_stats["mean"]) / std
            if z > n_sigma:
                drifted.append(name)
                logger.warning(
                    "Drift detected for '%s': value=%.4f, mean=%.4f, std=%.4f, z=%.2f",
                    name,
                    val,
                    feature_stats["mean"],
                    std,
                    z,
                )
        return {
            "drift_detected": bool(drifted),
            "drifted_features": drifted,
        }
