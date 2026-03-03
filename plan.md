# Senior SWE Project Plan (8 Weeks, Python)

## Goal
Build **two production-minded projects** suitable for a **Senior Software Engineer** resume:
1. **Trading Systems Backend** (systems / finance)
2. **ML Data & Tooling Platform** (ML infra, not modeling-heavy)

Focus areas:
- System design & tradeoffs
- Correctness under concurrency
- Recoverability & observability
- Production ML workflows

---

# Project 1: Trading Backend System (Weeks 1–4)

## 1. Functional Requirements

### Core Trading
- Support **limit orders and market orders**
  - Limit orders: rest in book with price protection
  - Market orders: immediate execution (IOC - Immediate-or-Cancel)
- Multiple instruments (3-5 US equities, e.g., AAPL, MSFT, GOOGL, TSLA, NVDA)
- Price–time priority matching per ticker
- Partial fills supported
- Order lifecycle:
  - NEW → PARTIALLY_FILLED → FILLED / CANCELED / REJECTED

### APIs
- REST:
  - `POST /orders` (with ticker, side, order_type, quantity, price)
  - `POST /orders/{id}/cancel`
  - `GET /book/{ticker}` (order book for specific ticker)
  - `GET /tickers` (list available tickers)
- WebSocket:
  - Top-of-book updates per ticker
  - Trade events per ticker
  - Order status updates
  - Ticker subscription filtering

### Risk
- Pre-trade checks:
  - Max position per account per ticker
  - Max notional exposure per ticker
  - Total portfolio exposure (across all tickers)
- Reject invalid orders deterministically
- Market order price protection (reject if spread too wide)

### Persistence & Recovery
- Append-only event log:
  - Orders (with ticker)
  - Trades (with ticker)
  - Cancels
- Snapshot order books periodically (all tickers)
- Replay log to restore identical state for all books

---

## 2. Non-Functional Requirements

- Deterministic replay
- Idempotent order submission
- Graceful handling of duplicate messages
- Throughput benchmarked and documented
- Observability:
  - Latency
  - Orders/sec
  - Error rates

---

## 3. Architecture Overview

```
Client
  |
FastAPI (Async)
  |
Order Queue (asyncio.Queue)
  |
Order Book Manager --> Order Book (AAPL)
                   --> Order Book (MSFT)
                   --> Order Book (GOOGL)
                   --> Order Book (TSLA)
                   --> Order Book (NVDA)
  |
Matching Engine (Single-threaded, deterministic per book)
  |
Event Log --> Snapshot Store
  |
WebSocket Broadcaster (per-ticker subscriptions)
```

### Design Notes
- **Multiple tickers:** Each ticker has its own order book, managed by a central router
- **Order types:** Limit orders rest in book; market orders execute immediately or reject
- **Determinism:** Each order book processes sequentially; multi-ticker parallelism via async
- **Scope control:** Fixed set of 3-5 tickers (no dynamic ticker addition)

---

## 4. Tech Stack

| Component | Technology |
|--------|-----------|
| Language | Python 3.13 |
| API | FastAPI |
| Async | asyncio |
| Persistence | PostgreSQL (events, snapshots) |
| Cache | Redis (optional) |
| Testing | pytest |
| Load Test | locust / custom asyncio benchmark |
| Metrics | prometheus_client |
| Logging | structlog |

---

## 5. Module Breakdown

```text
trading/
├── api/
│ ├── routes.py
│ └── websocket.py
├── engine/
│ ├── order_book.py         # Single order book for one ticker
│ ├── order_book_manager.py # Manages multiple order books
│ ├── matcher.py             # Matching logic for limit/market orders
│ └── risk.py
├── events/
│ ├── models.py
│ └── store.py
├── persistence/
│ ├── snapshots.py
│ └── replay.py
├── metrics/
├── tests/
└── main.py
```

---

## 6. Weekly Milestones

### Week 1 – Matching Engine
- Order book implementation (single ticker)
- Order book manager (routing for multiple tickers)
- Price–time priority matching for both limit and market orders
- Market order execution logic (immediate-or-cancel)
- Property tests for invariants (no negative positions, correct fills)
- Deterministic behavior validated
- Support for 3-5 tickers (AAPL, MSFT, GOOGL, TSLA, NVDA)

### Week 2 – API & Concurrency
- FastAPI REST endpoints with ticker routing
- Async ingestion pipeline (asyncio.Queue)
- WebSocket market data with per-ticker subscriptions
- Basic throughput benchmark (orders/sec per ticker and total)

### Week 3 – Risk & Persistence
- Pre-trade risk checks (per-ticker and portfolio-wide)
- Market order price protection (spread limits)
- Event-sourced persistence (multi-ticker aware)
- Snapshot + replay recovery for all order books
- Idempotency handling

### Week 4 – Observability & Polish
- Metrics & logging
- Failure injection tests
- README + architecture diagram
- Resume bullets finalized

---

# Project 2: ML Data & Tooling Platform (Weeks 5–8)

## 1. Functional Requirements

### Data
- Historical stock market data ingestion
- Batch + simulated streaming mode
- Feature backfill supported

### Features
- Feature definitions as code
- Offline (training) + online (serving) parity
- Versioned feature sets

### Training
- Reproducible training pipeline
- Experiment tracking
- Model versioning

### Serving
- Online inference via REST
- Hot model reload
- Low-latency predictions

### Monitoring
- Prediction latency
- Data drift detection
- Model performance tracking

---

## 2. Non-Functional Requirements

- Training/serving skew prevention
- Schema evolution support
- Re-runnable pipelines
- Clear failure modes documented

---

## 3. Architecture Overview

Raw Data
|
Feature Pipeline
|
Offline Store (Parquet)
|
Training Pipeline ----> MLflow
|
Model Registry
|
FastAPI Inference Service
|
Monitoring & Drift Detection

---

## 4. Tech Stack

| Component | Technology |
|--------|-----------|
| Language | Python 3.13 |
| Data | pandas, DuckDB |
| Features | Custom feature store (Feast optional) |
| ML | XGBoost / scikit-learn |
| Tracking | MLflow |
| Orchestration | Dagster / simple scheduler |
| Serving | FastAPI |
| Monitoring | Prometheus |

---

## 5. Module Breakdown

```text
ml_platform/
├── ingestion/
├── features/
│ ├── definitions.py
│ └── store.py
├── training/
│ ├── pipeline.py
│ └── evaluation.py
├── registry/
├── serving/
├── monitoring/
├── experiments/
└── tests/
```

---

## 6. Weekly Milestones

### Week 5 – Data & Baseline
- Dataset ingestion
- Label definition (no leakage)
- Baseline model
- Evaluation metrics

### Week 6 – Feature Store
- Feature definitions
- Offline/online consistency
- Feature backfills
- Versioning

### Week 7 – Training Pipeline
- Automated training pipeline
- Experiment tracking
- Model registry
- Reproducibility guarantees

### Week 8 – Serving & Monitoring
- Inference API
- Drift detection
- Performance metrics
- Final documentation + resume bullets

---

# Deliverables Checklist (Both Projects)

- [ ] Clean repo structure
- [ ] Architecture diagram
- [ ] Benchmarks
- [ ] Failure modes documented
- [ ] Limitations & future work
- [ ] Resume bullets ready

---

## Resume Framing (Key)
Do NOT emphasize:
- Model accuracy
- Toy features
- "Hello world" implementations

DO emphasize:
- **Multi-ticker routing** and order book management
- **Limit and market order** execution logic
- **Determinism** and replay guarantees
- **Recovery** from failures
- **Tradeoffs** (e.g., per-ticker vs global locking)
- **Scale thinking** (orders/sec, latency p99)
- **Portfolio-wide risk** management