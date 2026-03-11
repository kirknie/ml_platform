"""
Tests for training/runner.py.

All tests mock the ModelRegistry to isolate runner logic from registry I/O.
"""

from datetime import date, timedelta
from unittest.mock import MagicMock

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
    expected = (run_date - timedelta(days=EVALUATION_WINDOW_DAYS)).strftime("%Y-%m-%d")
    assert result == expected


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
    registry.get_champion_metrics.return_value = None
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


def test_compute_split_date_is_deterministic():
    """
    Reproducibility: the same run_date always produces the same split_date.
    This is the foundation of reproducible scheduled runs.
    """
    run_date = date(2024, 9, 15)
    assert compute_split_date(run_date) == compute_split_date(run_date)
    assert compute_split_date(run_date) == compute_split_date(run_date, EVALUATION_WINDOW_DAYS)
