---
  What the Project Is

  A ML infrastructure platform — not a model-building exercise. The emphasis is on the engineering that
  makes ML reliable and reproducible: how you get data in, how you define and serve features consistently,
  how you track experiments, how you deploy and monitor models. The model itself (XGBoost price direction
  classifier) is a vehicle to demonstrate the infrastructure.

  This is the ML equivalent of Project 1's "production-minded" framing: the interesting work is the
  plumbing, not the accuracy number.

  ---
  The Four Layers and What They Mean

  Layer 1 — Data Ingestion (ingestion/)

  Download historical OHLCV (Open/High/Low/Close/Volume) data for the same 5 tickers. The plan calls for
  both batch (historical backfill) and simulated streaming (replay historical data as if it were arriving
  live).

  - Batch: yfinance or a CSV download — fetch N years of daily/minute bars, write to Parquet files
  partitioned by ticker and date
  - Simulated streaming: a generator that replays those Parquet files in time order, emitting one row at a
  time with a configurable delay — used to test the online serving path without a real data feed

  Layer 2 — Feature Store (features/)

  This is the core engineering challenge of the project. The problem it solves: features computed at
  training time must be computed identically at serving time — if you compute a 20-day moving average
  differently in training vs serving, your model silently degrades.

  - Feature definitions as code: each feature (e.g. returns_5d, volatility_20d, rsi_14) is a pure function
  that takes a price series and returns a value. Defined once, used everywhere.
  - Offline store: applies feature functions to the full historical dataset, writes results to Parquet.
  Used for training.
  - Online store: applies the same feature functions to the latest N rows of incoming data. Used for
  inference.
  - Versioning: a feature set is a named, immutable list of feature functions + a version tag. Training a
  model records which feature set version was used; serving uses the same version.

  The non-trivial part is point-in-time correctness: when computing features for a training label at date
  T, you must only use data available before T. Using data after T is look-ahead bias (leakage), which
  causes unrealistically good training metrics that collapse in production.

  Layer 3 — Training Pipeline (training/)

  A script (or Dagster pipeline) that:
  1. Loads the offline feature store for a date range
  2. Defines labels (e.g. "did price go up in the next 5 days?")
  3. Splits train/validation/test by time (not random — random split causes leakage)
  4. Trains XGBoost
  5. Evaluates (accuracy, precision, recall, AUC — on test set only)
  6. Logs everything to MLflow: parameters, metrics, the model artifact, and the feature set version used

  Reproducibility: given the same feature set version + date range + random seed, the pipeline must produce
   the same model. This means no non-deterministic data sources and pinned library versions.

  Layer 4 — Serving + Monitoring (serving/, monitoring/)

  - FastAPI inference endpoint: POST /predict takes a ticker, fetches latest features from the online
  store, runs the model, returns a prediction + confidence score
  - Hot model reload: the server can swap to a new model version without restarting (watches a
  models/current symlink or polls MLflow for the latest registered model)
  - Prediction latency: Prometheus histogram, same pattern as Project 1
  - Data drift detection: compare the distribution of incoming feature values against the training
  distribution. Simplest approach: track mean and stddev at training time; flag if inference-time values
  are more than N standard deviations outside that range
  - Model performance tracking: log predictions + actuals (once the future price is known) to measure
  real-world accuracy over time

  ---
  What Makes This Senior-Level

  The same framing as Project 1 — not the ML itself but the engineering decisions:

  Decision: Point-in-time feature computation
  What to articulate: Prevents look-ahead bias; same mechanism used by professional feature stores (Feast,
    Tecton)
  ────────────────────────────────────────
  Decision: Training/serving feature parity
  What to articulate: Single feature definition used for both paths — no dual maintenance
  ────────────────────────────────────────
  Decision: Time-based train/test split
  What to articulate: Why random split is wrong for time series
  ────────────────────────────────────────
  Decision: Feature set versioning
  What to articulate: Model registry records which feature version it was trained on; serving uses the same
  ────────────────────────────────────────
  Decision: Simulated streaming
  What to articulate: Lets you test the online path without a live data feed
  ────────────────────────────────────────
  Decision: Drift detection
  What to articulate: Statistical comparison of inference vs training distributions; early warning before
    accuracy degrades

  ---
  Suggested Concrete Plan

  Week 5 — Data + Baseline

  - Set up ml_platform/ repo (separate from the trading platform, or a sibling directory)
  - Download 3 years of daily OHLCV for 5 tickers via yfinance, store as Parquet
  - Define labels: 5-day forward return > 0 → binary classification
  - Implement point-in-time split
  - Train a minimal XGBoost on raw OHLCV (no feature engineering yet) as the baseline
  - Log to MLflow: parameters, AUC, feature list

  Week 6 — Feature Store

  - Define 8–12 features as pure functions: returns (1d, 5d, 20d), volatility (20d), RSI, MACD signal,
  volume ratio
  - Implement offline store: apply all features across the full history, write to Parquet with feature set
  version tag
  - Implement online store: given the last N rows of prices, compute the same features
  - Write property tests: offline and online must return identical values for the same input window
  - Retrain on feature store output; compare AUC to baseline

  Week 7 — Training Pipeline

  - Wrap training in a reproducible pipeline (Dagster job or a plain Python script with CLI args)
  - MLflow experiment tracking: log feature set version, date range, hyperparameters, metrics, model
  artifact
  - Implement model registry: register_model(run_id, name, version) writes metadata to a JSON file or
  SQLite; load_model(version) returns the trained model
  - Reproducibility test: run the pipeline twice with the same inputs, assert identical predictions on the
  test set

  Week 8 — Serving + Monitoring

  - FastAPI POST /predict: ticker → online feature computation → model inference → {"direction": "UP",
  "confidence": 0.72}
  - Hot reload: background thread polls MLflow or checks a models/current symlink; swaps model atomically
  - Prometheus metrics: prediction_latency_seconds histogram, predictions_total counter
  - Drift detection: at training time, compute and save per-feature mean/stddev; at inference time, flag if
   any feature is beyond 3σ
  - GET /health and GET /ready (same pattern as Project 1)
  - Integration tests for the full pipeline: ingest → features → train → serve → predict

  ---
  Key Risks to Plan For

  1. Data quality: yfinance occasionally returns NaN or adjusted-price artifacts. Need explicit data
  validation at ingestion.
  2. Feature computation correctness: the offline/online parity property test is the most important test to
   write. It is easy to accidentally use a rolling window that peeks ahead.
  3. MLflow complexity: MLflow's tracking server and artifact store have several deployment modes. For this
   project, use local file-based tracking (mlflow.set_tracking_uri("file:./mlruns")) — simple and
  zero-dependency.
  4. Label leakage: the most common mistake. Forward returns must be shifted correctly — label for day T
  uses price at T+5, but features for day T use only data up to and including day T.