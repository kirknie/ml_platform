"""Tests for monitoring/performance.py."""

import json
from pathlib import Path

import pandas as pd
import pytest

from monitoring.performance import evaluate_logged_predictions


def _make_parquet(tmp_path: Path, ticker: str, prices: list[float]) -> Path:
    """Write a minimal Parquet with known close prices."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    df = pd.DataFrame(
        {
            "open": prices, "high": prices, "low": prices,
            "close": prices, "volume": 1_000_000.0, "ticker": ticker,
        },
        index=dates,
    )
    df.index.name = "date"
    path = raw_dir / f"{ticker}.parquet"
    df.to_parquet(path)
    return raw_dir


def _write_log(tmp_path: Path, entries: list[dict]) -> Path:
    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return log_path


def test_correct_prediction_marked_correct(tmp_path):
    """A prediction that matches the realized 5-day direction is marked correct."""
    # Prices go: 100 → 110 at t+5 → UP
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 111.0]
    raw_dir = _make_parquet(tmp_path, "AAPL", prices)

    log = _write_log(tmp_path, [{
        "ticker": "AAPL",
        "prediction_date": "2024-01-01",  # first business day → index 0
        "direction": "UP",
        "probability": 0.65,
        "model_version": "5",
    }])

    result = evaluate_logged_predictions(log, raw_dir, forward_days=5)
    assert result["evaluated"] == 1
    assert result["results"][0]["correct"] is True
    assert result["accuracy"] == 1.0


def test_incorrect_prediction_marked_incorrect(tmp_path):
    """A prediction that contradicts the realized direction is marked incorrect."""
    # Prices go: 100 → 90 at t+5 → DOWN; we predicted UP
    prices = [100.0, 99.0, 98.0, 97.0, 96.0, 90.0, 89.0]
    raw_dir = _make_parquet(tmp_path, "AAPL", prices)

    log = _write_log(tmp_path, [{
        "ticker": "AAPL",
        "prediction_date": "2024-01-01",
        "direction": "UP",
        "probability": 0.55,
        "model_version": "5",
    }])

    result = evaluate_logged_predictions(log, raw_dir, forward_days=5)
    assert result["evaluated"] == 1
    assert result["results"][0]["correct"] is False
    assert result["results"][0]["realized"] == "DOWN"
    assert result["accuracy"] == 0.0


def test_pending_when_future_data_unavailable(tmp_path):
    """Predictions where forward_days data does not exist yet are counted as pending."""
    prices = [100.0, 101.0, 102.0]  # only 3 rows — t+5 unavailable
    raw_dir = _make_parquet(tmp_path, "AAPL", prices)

    log = _write_log(tmp_path, [{
        "ticker": "AAPL",
        "prediction_date": "2024-01-01",
        "direction": "UP",
        "probability": 0.60,
        "model_version": "5",
    }])

    result = evaluate_logged_predictions(log, raw_dir, forward_days=5)
    assert result["pending"] == 1
    assert result["evaluated"] == 0
    assert result["accuracy"] == 0.0


def test_skipped_when_no_prediction_date(tmp_path):
    """Entries without a prediction_date field are counted as skipped."""
    log = _write_log(tmp_path, [{
        "ticker": "AAPL",
        "direction": "UP",
        "probability": 0.60,
        "model_version": "1",
        # no prediction_date
    }])

    result = evaluate_logged_predictions(log, Path(tmp_path / "raw"))
    assert result["skipped"] == 1
    assert result["evaluated"] == 0
