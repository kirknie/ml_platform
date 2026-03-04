"""
End-to-end baseline training pipeline.

Steps:
1. Load raw data
2. Add labels
3. Select raw OHLCV baseline features
4. Time-based train/test split (no shuffling — respects temporal ordering)
5. Train XGBoost classifier
6. Evaluate on test set
7. Log everything to MLflow

Run this script directly:
    python -m training.pipeline
"""

import logging

import mlflow
import pandas as pd
import xgboost as xgb

from features.baseline import FEATURE_COLUMNS, get_baseline_features
from ingestion.fetch import TICKERS, load_all
from ingestion.validate import validate_raw
from training.evaluation import compute_metrics
from training.labels import add_labels_all_tickers, drop_unlabeled

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "ml_platform_baseline"
TEST_SPLIT_DATE = "2023-01-01"  # everything before this is train, after is test


def build_dataset() -> pd.DataFrame:
    """
    Load raw data, add labels, select raw OHLCV features, drop rows with NaN.

    Returns:
        Clean DataFrame ready for train/test split
    """
    logger.info("Loading raw data for tickers: %s", TICKERS)
    df = load_all(TICKERS)

    for ticker in TICKERS:
        ticker_df = df[df["ticker"] == ticker]
        validate_raw(ticker_df, ticker)

    logger.info("Adding labels (5-day forward direction)")
    df = add_labels_all_tickers(df)

    # Raw OHLCV baseline: no feature engineering, just select the columns
    # Engineered features (RSI, returns, volatility) are added in Week 6
    logger.info("Selecting raw OHLCV baseline features")
    features_df = get_baseline_features(df)
    df = df[["ticker", "label"]].join(features_df)

    # Drop rows with no valid label (last 5 rows per ticker)
    before = len(df)
    df = drop_unlabeled(df)
    logger.info("Dropped %d rows with no label (unlabeled tail)", before - len(df))

    return df


def time_split(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/test by date.

    IMPORTANT: We do NOT shuffle. Shuffling would cause future data to appear
    in the training set when rows near the split boundary are interleaved.
    This is a form of temporal leakage.
    """
    train = df[df.index < split_date]
    test = df[df.index >= split_date]
    logger.info("Train: %d rows, Test: %d rows (split at %s)", len(train), len(test), split_date)
    return train, test


def train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
) -> tuple:
    """
    Train XGBoost classifier and return (model, metrics, params).
    """
    X_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["label"].astype(int)

    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    return model, metrics, params


def run_pipeline() -> None:
    mlflow.set_tracking_uri("file:./mlruns")  # explicit local file-based tracking
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="baseline_xgboost"):
        df = build_dataset()
        train_df, test_df = time_split(df, TEST_SPLIT_DATE)
        model, metrics, params = train(train_df, test_df)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_params({
            "tickers": ",".join(TICKERS),
            "test_split_date": TEST_SPLIT_DATE,
            "feature_columns": ",".join(FEATURE_COLUMNS),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        })
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")

        logger.info("Metrics: %s", metrics)
        print("\n=== Baseline Results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
