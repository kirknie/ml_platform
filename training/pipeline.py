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

import hashlib
import logging

import mlflow
import pandas as pd
import xgboost as xgb

from features.baseline import FEATURE_COLUMNS
from features.definitions import ENGINEERED_V2
from features.store import FeatureStore
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

    # Raw OHLCV baseline: no feature engineering, just select the columns.
    # Select all needed columns from df directly — avoids any index join issues
    # since features_df is derived from the same DataFrame.
    logger.info("Selecting raw OHLCV baseline features")
    df = df[["ticker", "label"] + FEATURE_COLUMNS].copy()

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


def _fingerprint(df: pd.DataFrame) -> str:
    """
    MD5 hash of a DataFrame's values for reproducibility auditing.

    Two runs that produce the same fingerprint used identical training data.
    Logged as a param so you can verify reproducibility from the MLflow UI.
    """
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def build_dataset_from_store(store: FeatureStore) -> pd.DataFrame:
    """
    Load labels from raw data and join with precomputed features from the store.

    Labels are derived from raw close prices (same logic as Week 5).
    Features come from the offline feature store (Week 6).
    Joined on the date index — only dates present in both are kept (inner join).

    Returns:
        DataFrame with label + feature columns, NaN rows dropped, ready for split.
    """
    logger.info("Loading raw data for labels: %s", TICKERS)
    raw_df = load_all(TICKERS)

    logger.info("Adding labels (5-day forward direction)")
    labeled_df = add_labels_all_tickers(raw_df)
    labeled_df = drop_unlabeled(labeled_df)
    labels = labeled_df[["ticker", "label"]].reset_index()  # columns: date, ticker, label

    logger.info(
        "Loading engineered features from store: %s v%d",
        store.feature_set.name,
        store.feature_set.version,
    )
    features_df = store.load_offline_all(TICKERS).reset_index()  # columns: date, features..., ticker

    # Merge on (ticker, date) — joining on date alone causes a many-to-many
    # cartesian join because the date index has one row per ticker per date.
    df = labels.merge(features_df, on=["ticker", "date"], how="inner")
    df = df.set_index("date").dropna()
    logger.info("Dataset: %d rows, %d feature columns", len(df), len(features_df.columns) - 2)
    return df


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
    store = FeatureStore(ENGINEERED_V2)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="engineered_xgboost") as active_run:
        df = build_dataset_from_store(store)
        train_df, test_df = time_split(df, test_split_date)

        feature_cols = store.feature_set.feature_names

        # Save training distribution for drift detection at serving time
        store.save_training_stats(train_df[feature_cols])

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
            "training_stats_path": str(store.store_dir / "training_stats.json"),
        })
        if run_date is not None:
            mlflow.log_param("run_date", run_date)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")

        run_id = active_run.info.run_id

    logger.info("Engineered metrics: %s", metrics)
    return run_id, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- Baseline (raw OHLCV) ---")
    run_pipeline()
    print("\n--- Engineered features ---")
    run_id, metrics = run_engineered_pipeline()
    print(f"\n  run_id: {run_id}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
