# Sentinel AI — Real-Time Fraud Detection with MLOps Pipeline

An end-to-end, production-grade fraud detection system combining **Apache Kafka**, **Apache Spark**, **XGBoost**, **MLflow**, **Apache Airflow**, and a **Gemini-powered LLM agent** for autonomous triage.

---

## Architecture

```
                       ┌─────────────────────────────────────────────┐
                       │              ML PIPELINE (NEW)               │
                       │                                              │
  creditcard.csv       │  Airflow DAG (Weekly / On-Demand)           │
       │               │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
       │               │  │ Validate │→ │ Feature  │→ │ Retrain  │  │
       │               │  │  Data    │  │  Eng +   │  │ XGBoost  │  │
       │               │  └──────────┘  │  SMOTE   │  └────┬─────┘  │
       │               │                └──────────┘       │         │
       │               │  ┌──────────┐  ┌──────────┐  ┌───▼──────┐  │
       │               │  │  Drift   │← │ Evidently│  │  MLflow  │  │
       │               │  │ Detected?│  │  Report  │  │ Registry │  │
       │               │  └────┬─────┘  └──────────┘  └──────────┘  │
       │               │       │                                      │
       │               │  ┌────▼─────────────────┐                   │
       │               │  │ Promote / Reject      │                   │
       │               │  │ (Champion/Challenger) │                   │
       │               │  └──────────────────────┘                   │
       │               └──────────────┬──────────────────────────────┘
       │                              │  Hot-swap fraud_model.joblib
       ▼                              ▼
  producer.py ──Kafka──► processor.py (Spark + XGBoost UDF)
                                      │
                              flagged_alerts.txt
                                      │
                              drift_monitor.py ──► (auto-trigger DAG)
                                      │
                              agent.py (Gemini LLM)
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼             ▼
                    Freeze Card   SOC Ticket   SMS Challenge
```

---

## Services

| Service | URL | Description |
|---|---|---|
| **Airflow UI** | http://localhost:8081 | Trigger / monitor retraining DAG |
| **MLflow UI** | http://localhost:5000 | Experiment runs, model registry |
| **Spark UI** | http://localhost:8080 | Live streaming job status |
| **Kafka** | localhost:9092 | Transaction event stream |
| **Redis** | localhost:6379 | Feature cache (future use) |

---

## Quick Start

### 1. Environment Setup

```bash
# Optional: Create and activate conda environment
conda create --name sentinel python=3.10 -y
conda activate sentinel

# Install Python dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and insert your GOOGLE_API_KEY
```

### 2. Dataset Setup
Download `creditcard.csv` from the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root directory. *(Note: If `creditcard.csv` is not present, `producer.py` will automatically fall back to simulating live transactions adhering to the Kaggle schema).*

### 3. Start all services

```bash
docker compose up -d
```

### 4. Run the live pipeline

Open **three terminals**:

```bash
# Terminal 1 — Kafka producer (streams creditcard.csv → Kafka)
python producer.py

# Terminal 2 — Spark processor (scores stream with XGBoost)
python processor.py

# Terminal 3 — LLM agent (triages flagged alerts)
python agent.py
```

### 5. (Optional) Run the drift monitor

```bash
# Monitors distributions and can auto-trigger retraining
AUTO_TRIGGER_DAG=true python drift_monitor.py
```

### 6. Trigger a manual retrain

Via Airflow UI → DAGs → `sentinel_fraud_retrain` → Trigger DAG

Or via REST API:
```bash
curl -X POST http://localhost:8081/api/v1/dags/sentinel_fraud_retrain/dagRuns \
  -H "Content-Type: application/json" \
  -u admin:admin \
  -d '{"conf": {"triggered_by": "manual"}}'
```

---

## ML Pipeline — Airflow DAG

**File**: [`dags/sentinel_retrain_dag.py`](dags/sentinel_retrain_dag.py)

**Schedule**: Every Sunday at 02:00 UTC (`0 2 * * 0`)

| Task | Description |
|---|---|
| `validate_data` | Schema check, row count, class balance |
| `engineer_features` | StandardScaler + SMOTE oversampling |
| `train_model` | XGBoost (300 trees) + full metrics logged to MLflow |
| `detect_drift` | Evidently HTML report → logged as MLflow artifact |
| `evaluate_vs_champion` | AUC comparison: challenger vs. current Production model |
| `promote_model` | Transitions model to Production + hot-swaps `fraud_model.joblib` |
| `reject_model` | Tags challenger as rejected; champion is retained |

---

## Drift Monitor

**File**: [`drift_monitor.py`](drift_monitor.py)

- Samples live alerts every 60 seconds
- Computes **Jensen-Shannon Divergence** between live and reference distributions
- Triggers Airflow retraining DAG automatically after **3 consecutive drift signals**

```bash
# Enable auto-triggering
AUTO_TRIGGER_DAG=true python drift_monitor.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API key for the LLM agent |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `AIRFLOW_HOST` | `http://localhost:8081` | Airflow webserver URL |
| `AIRFLOW_USER` | `admin` | Airflow username |
| `AIRFLOW_PASSWORD` | `admin` | Airflow password |
| `AUTO_TRIGGER_DAG` | `false` | Set to `true` to auto-trigger retrain on drift |

---

## Stack

| Layer | Technology |
|---|---|
| Event Streaming | Apache Kafka (KRaft) |
| Stream Processing | Apache Spark 3.5 + PySpark |
| ML Model | XGBoost + scikit-learn + SMOTE |
| Experiment Tracking | MLflow |
| Orchestration | Apache Airflow 2.9 |
| Drift Detection | Evidently AI |
| Feature Cache | Redis |
| LLM Agent | Google Gemini via LangChain |
| Containerization | Docker Compose |
