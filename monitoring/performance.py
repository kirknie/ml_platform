"""
Model performance tracking: evaluate logged predictions against realized prices.

At prediction time, the serving layer logs each prediction to data/predictions.jsonl
including the ticker, predicted direction, probability, and prediction_date (the last
OHLCV date available when the prediction was made).

Once FORWARD_DAYS trading days have passed, this module can look up the realized
outcome and compute real-world accuracy.

Usage:
    python -m monitoring.performance
    # → prints accuracy summary for all evaluable predictions
"""

import json
import logging
from pathlib import Path

import pandas as pd

from ingestion.fetch import RAW_DATA_DIR, load_ticker

logger = logging.getLogger(__name__)

FORWARD_DAYS = 5
PREDICTIONS_LOG = Path("data/predictions.jsonl")


def evaluate_logged_predictions(
    predictions_log: Path = PREDICTIONS_LOG,
    raw_data_dir: Path = RAW_DATA_DIR,
    forward_days: int = FORWARD_DAYS,
) -> dict:
    """
    Evaluate accuracy of logged predictions against realized prices.

    For each logged prediction with a valid prediction_date, looks up:
    - close[t]:            closing price on prediction_date
    - close[t+forward]:    closing price forward_days trading days later

    Realized direction = "UP" if close[t+forward] > close[t] else "DOWN".
    A prediction is correct when predicted direction == realized direction.

    Predictions where forward_days of future data is not yet available
    are counted as "pending" and excluded from accuracy.

    Args:
        predictions_log: Path to the JSONL prediction log
        raw_data_dir: Directory containing raw Parquet files
        forward_days: Trading days ahead to check for outcome

    Returns:
        {
            "evaluated": int,     # predictions with known outcomes
            "pending": int,       # predictions awaiting future data
            "skipped": int,       # predictions missing prediction_date field
            "accuracy": float,    # fraction correct (0.0 if evaluated == 0)
            "results": list[dict] # per-prediction detail
        }
    """
    if not predictions_log.exists():
        return {"evaluated": 0, "pending": 0, "skipped": 0, "accuracy": 0.0, "results": []}

    entries = [
        json.loads(line)
        for line in predictions_log.read_text().strip().splitlines()
        if line.strip()
    ]

    results = []
    pending = 0
    skipped = 0

    for entry in entries:
        prediction_date = entry.get("prediction_date")
        if not prediction_date:
            skipped += 1
            continue

        ticker = entry["ticker"]
        predicted = entry["direction"]
        probability = entry.get("probability")

        try:
            df = load_ticker(ticker, raw_data_dir)
        except FileNotFoundError:
            skipped += 1
            continue

        # Find position of prediction_date in the index
        target = pd.Timestamp(prediction_date)
        if target not in df.index:
            # Use nearest available date (handles weekends/holidays)
            idx = df.index.get_indexer([target], method="nearest")[0]
        else:
            idx = df.index.get_loc(target)

        # Check whether forward outcome is available
        if idx + forward_days >= len(df):
            pending += 1
            continue

        close_t = float(df.iloc[idx]["close"])
        close_forward = float(df.iloc[idx + forward_days]["close"])
        realized = "UP" if close_forward > close_t else "DOWN"
        correct = predicted == realized

        results.append({
            "ticker": ticker,
            "prediction_date": prediction_date,
            "predicted": predicted,
            "realized": realized,
            "correct": correct,
            "probability": probability,
            "close_t": round(close_t, 4),
            "close_forward": round(close_forward, 4),
        })

    evaluated = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / evaluated if evaluated > 0 else 0.0

    return {
        "evaluated": evaluated,
        "pending": pending,
        "skipped": skipped,
        "accuracy": round(accuracy, 4),
        "results": results,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = evaluate_logged_predictions()
    print(f"\n=== Prediction Performance ===")
    print(f"  Evaluated : {summary['evaluated']}")
    print(f"  Pending   : {summary['pending']} (future data not yet available)")
    print(f"  Skipped   : {summary['skipped']} (missing prediction_date field)")
    print(f"  Accuracy  : {summary['accuracy']:.1%}")
    if summary["results"]:
        print("\n  Detail:")
        for r in summary["results"]:
            status = "✓" if r["correct"] else "✗"
            print(
                f"    {status} {r['ticker']} {r['prediction_date']} "
                f"predicted={r['predicted']} realized={r['realized']} "
                f"p={r['probability']:.2f}"
            )
