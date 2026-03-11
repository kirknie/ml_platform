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
