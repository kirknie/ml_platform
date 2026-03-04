"""
Evaluation metric helpers.

Using classification metrics because the task is binary direction prediction.
Accuracy alone is misleading on imbalanced data — always report precision,
recall, and AUC alongside it.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)


def compute_metrics(y_true: pd.Series, y_pred, y_prob=None) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities for class 1 (optional, for AUC)

    Returns:
        Dict of metric name -> value
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Happens if only one class present in y_true (e.g. tiny test sets)
            metrics["roc_auc"] = float("nan")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["precision_class1"] = report.get("1", {}).get("precision", 0.0)
    metrics["recall_class1"] = report.get("1", {}).get("recall", 0.0)
    metrics["f1_class1"] = report.get("1", {}).get("f1-score", 0.0)
    metrics["label_balance"] = float(y_true.mean())

    return metrics
