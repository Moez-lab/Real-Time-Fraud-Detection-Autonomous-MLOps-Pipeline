"""
Sentinel AI — Airflow Retraining DAG
=====================================
Orchestrates the full ML lifecycle:
  1. Data validation & ingestion
  2. Feature engineering
  3. Model retraining (XGBoost)
  4. Evaluation vs. champion model in MLflow registry
  5. Drift detection (Evidently)
  6. Conditional model promotion (challenger → champion)
  7. Hot-swap the live fraud_model.joblib used by processor.py

Schedule: Weekly (every Sunday at 02:00 UTC)
Can also be triggered manually from the Airflow UI.
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# ── Project root is one level above /dags (or /opt/airflow/project in Docker) ──
PROJECT_ROOT = Path("/opt/airflow/project") if Path("/opt/airflow/project").exists() else Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "fraud_model.joblib"
DATA_PATH    = PROJECT_ROOT / "creditcard.csv"
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT   = "sentinel-fraud-detection"

log = logging.getLogger(__name__)


def get_retrain_strategy(dag_run, params=None) -> str:
    """
    Determine retraining strategy:
    - 'A': Monthly full retrain (from scratch, 90-day window)
    - 'B': Incremental retrain (+5 trees, 30-day window)
    """
    if dag_run and dag_run.conf:
        strategy = dag_run.conf.get("strategy")
        if strategy in ("A", "B"):
            return strategy

    if params and "strategy" in params:
        strategy = params["strategy"]
        if strategy in ("A", "B"):
            return strategy

    if dag_run:
        run_date = dag_run.logical_date or getattr(dag_run, "execution_date", datetime.utcnow())
        
        # Monthly scheduled run -> Strategy A
        if run_date.day == 1:
            return "A"
            
        # Triggered by drift monitor
        triggered_by = dag_run.conf.get("triggered_by")
        if triggered_by == "drift_monitor":
            # If triggered on Tuesday (weekday 1)
            if run_date.weekday() == 1:
                return "B"
            return "B"  # Default drift monitor triggers to incremental hotfix
            
    return "A"  # Default fallback

# ── Default DAG args ────────────────────────────────────────────────────────
default_args = {
    "owner": "sentinel-ai",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ============================================================
# TASK 1 — Data Validation
# ============================================================
def validate_data(**ctx):
    """
    Sanity-check the CSV before spending compute on retraining.
    Raises if data is missing, too small, or has wrong schema.
    """
    import pandas as pd

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    expected_cols = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")

    n_fraud   = df["Class"].sum()
    n_legit   = len(df) - n_fraud
    fraud_pct = n_fraud / len(df) * 100

    log.info("Data validation passed ✅")
    log.info(f"  Rows      : {len(df):,}")
    log.info(f"  Fraud     : {n_fraud:,}  ({fraud_pct:.2f}%)")
    log.info(f"  Legit     : {n_legit:,}")

    if len(df) < 1000:
        raise ValueError("Dataset too small for meaningful retraining (< 1 000 rows).")

    # Push summary to XCom for downstream tasks
    ctx["ti"].xcom_push(key="data_stats", value={
        "rows": len(df),
        "n_fraud": int(n_fraud),
        "fraud_pct": round(fraud_pct, 4),
    })


# ============================================================
# TASK 2 — Feature Engineering
# ============================================================
def engineer_features(**ctx):
    """
    Build the feature matrix used for training.
    Applies SMOTE oversampling to handle class imbalance.
    Saves artefacts to /tmp for downstream tasks.
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    log.info("Loading dataset ...")
    df = pd.read_csv(DATA_PATH)

    # Resolve strategy
    dag_run = ctx.get("dag_run")
    strategy = get_retrain_strategy(dag_run, ctx.get("params"))
    log.info(f"Feature engineering running with strategy: {strategy}")

    # Map relative Time to simulated timestamps spanning the last 120 days
    # (since the original Kaggle Time column spans only 2 days)
    now = datetime.utcnow()
    max_time = df["Time"].max()
    df["simulated_days"] = (df["Time"] / max_time) * 120
    df["timestamp"] = now - timedelta(days=120) + pd.to_timedelta(df["simulated_days"], unit="D")

    # Filter data based on Strategy
    if strategy == "A":
        # Monthly Full Retrain: 90-day window
        cutoff = now - timedelta(days=90)
        df_filtered = df[df["timestamp"] >= cutoff]
        log.info(f"Strategy A (Full): filtered for 90-day window. Rows: {len(df_filtered):,} (out of {len(df):,})")
    else:
        # Strategy B (Incremental): 30-day window
        cutoff = now - timedelta(days=30)
        df_filtered = df[df["timestamp"] >= cutoff]
        log.info(f"Strategy B (Incremental): filtered for 30-day window. Rows: {len(df_filtered):,} (out of {len(df):,})")

    # Drop simulated columns to maintain original schema
    df = df_filtered.drop(columns=["simulated_days", "timestamp"])

    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[feature_cols].copy()
    y = df["Class"].copy()

    # Scale Time & Amount (V-features are already PCA'd)
    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE oversampling on training split only
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        log.info(f"SMOTE applied -> training set: {len(X_train):,} rows")
    except ImportError:
        log.warning("imbalanced-learn not installed — skipping SMOTE.")

    import joblib, tempfile
    tmp = tempfile.gettempdir()
    joblib.dump((X_train, X_test, y_train, y_test), f"{tmp}/sentinel_features.joblib")
    joblib.dump(scaler, f"{tmp}/sentinel_scaler.joblib")

    log.info(f"Feature engineering complete ✅  Train: {len(X_train):,} | Test: {len(X_test):,}")
    ctx["ti"].xcom_push(key="feature_path", value=f"{tmp}/sentinel_features.joblib")


# ============================================================
# TASK 3 — Model Training
# ============================================================
def train_model(**ctx):
    """
    Train an XGBoost classifier on the prepared features.
    Logs every metric and the model artifact to MLflow.
    """
    import joblib, tempfile
    import mlflow
    import mlflow.xgboost
    from xgboost import XGBClassifier
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score,
        confusion_matrix,
    )

    tmp = tempfile.gettempdir()
    X_train, X_test, y_train, y_test = joblib.load(f"{tmp}/sentinel_features.joblib")

    # Resolve strategy
    dag_run = ctx.get("dag_run")
    strategy = get_retrain_strategy(dag_run, ctx.get("params"))
    log.info(f"Model training running with strategy: {strategy}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=f"{strategy}_retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M')}") as run:
        mlflow.log_param("retrain_strategy", strategy)

        if strategy == "A":
            params = {
                "n_estimators"    : 300,
                "max_depth"       : 6,
                "learning_rate"   : 0.05,
                "subsample"       : 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 1,
                "eval_metric"     : "aucpr",
                "use_label_encoder": False,
                "random_state"    : 42,
                "n_jobs"          : -1,
            }
            mlflow.log_params(params)
            clf = XGBClassifier(**params)
            clf.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=50,
            )
        else:  # Strategy B (Incremental)
            log.info(f"Loading champion model from {MODEL_PATH} for incremental training ...")
            if not MODEL_PATH.exists():
                log.warning("No champion model found at MODEL_PATH. Initializing new booster for Strategy B.")
                xgb_booster = None
            else:
                champion_model = joblib.load(MODEL_PATH)
                xgb_booster = champion_model.get_booster()

            params = {
                "n_estimators"    : 5,  # Add 5 trees
                "max_depth"       : 6,
                "learning_rate"   : 0.05,
                "subsample"       : 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 1,
                "eval_metric"     : "aucpr",
                "use_label_encoder": False,
                "random_state"    : 42,
                "n_jobs"          : -1,
            }
            mlflow.log_params(params)
            clf = XGBClassifier(**params)
            clf.fit(
                X_train, y_train,
                xgb_model=xgb_booster,
                eval_set=[(X_test, y_test)],
                verbose=50,
            )

        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred       = (y_pred_proba > 0.5).astype(int)

        metrics = {
            "roc_auc"  : roc_auc_score(y_test, y_pred_proba),
            "pr_auc"   : average_precision_score(y_test, y_pred_proba),
            "f1"       : f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall"   : recall_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)
        log.info(f"Challenger metrics: {json.dumps(metrics, indent=2)}")

        # Log confusion matrix as a JSON artefact
        cm = confusion_matrix(y_test, y_pred).tolist()
        mlflow.log_dict({"confusion_matrix": cm}, "confusion_matrix.json")

        # Save challenger model locally & log to MLflow
        challenger_path = f"{tmp}/sentinel_challenger.joblib"
        joblib.dump(clf, challenger_path)
        mlflow.log_artifact(challenger_path, artifact_path="model")

        # Register version in MLflow Model Registry
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        try:
            client.create_registered_model("SentinelFraudModel")
        except Exception:
            pass
        try:
            artifact_uri = client.get_run(run.info.run_id).info.artifact_uri
            client.create_model_version(
                name="SentinelFraudModel",
                source=f"{artifact_uri}/model",
                run_id=run.info.run_id,
                description=f"Retrained via Strategy {strategy}",
            )
        except Exception as e:
            log.warning(f"Could not create model version in registry: {e}")

        ctx["ti"].xcom_push(key="run_id",           value=run.info.run_id)
        ctx["ti"].xcom_push(key="challenger_auc",   value=metrics["roc_auc"])
        ctx["ti"].xcom_push(key="challenger_prauc", value=metrics["pr_auc"])
        ctx["ti"].xcom_push(key="challenger_path",  value=challenger_path)

    log.info(f"Training complete ✅  Run ID: {run.info.run_id}")


# ============================================================
# TASK 4 — Drift Detection (Evidently)
# ============================================================
def detect_drift(**ctx):
    """
    Compare the distribution of the current training data against
    a reference snapshot. Logs a drift report to MLflow as an HTML file.
    If evidently is not installed, this task is skipped gracefully.
    """
    try:
        import pandas as pd
        import mlflow
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        log.info("Running Evidently drift detection ...")
        df = pd.read_csv(DATA_PATH)
        feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

        # Use first 70% as reference, last 30% as current
        split     = int(len(df) * 0.7)
        reference = df[feature_cols].iloc[:split]
        current   = df[feature_cols].iloc[split:]

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference, current_data=current)

        import tempfile
        tmp         = tempfile.gettempdir()
        report_path = f"{tmp}/drift_report.html"
        report.save_html(report_path)

        run_id = ctx["ti"].xcom_pull(task_ids="train_model", key="run_id")
        mlflow.set_tracking_uri(MLFLOW_URI)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(report_path, artifact_path="drift")

        log.info(f"Drift report logged to MLflow run {run_id} ✅")

    except ImportError:
        log.warning("evidently not installed — skipping drift detection. Run: pip install evidently")


# ============================================================
# TASK 5 — Champion vs Challenger evaluation (Branch)
# ============================================================
def evaluate_vs_champion(**ctx):
    """
    Compare challenger AUC against the champion currently in MLflow registry.
    Returns the task ID to branch to (promote or reject).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    # Resolve strategy
    dag_run = ctx.get("dag_run")
    strategy = get_retrain_strategy(dag_run, ctx.get("params"))
    if strategy == "B":
        log.info("Strategy B (Incremental) detected — promoting by default to stop the bleeding.")
        return "promote_model"

    challenger_auc = ctx["ti"].xcom_pull(task_ids="train_model", key="challenger_auc")
    log.info(f"Challenger ROC-AUC: {challenger_auc:.4f}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    champion_auc = None
    try:
        versions = client.get_latest_versions("SentinelFraudModel", stages=["Production"])
        if versions:
            champ_run    = client.get_run(versions[0].run_id)
            champion_auc = float(champ_run.data.metrics.get("roc_auc", 0))
            log.info(f"Champion ROC-AUC : {champion_auc:.4f}")
    except Exception as e:
        log.warning(f"No champion found in registry ({e}) — will promote challenger by default.")

    if champion_auc is None or challenger_auc > champion_auc:
        log.info("Challenger wins — promoting to production.")
        return "promote_model"
    else:
        log.info("Champion holds — rejecting challenger.")
        return "reject_model"


# ============================================================
# TASK 6a — Promote challenger to champion
# ============================================================
def promote_model(**ctx):
    """
    Transition the new model version to Production in MLflow registry
    and hot-swap the fraud_model.joblib on disk so processor.py picks
    it up on the next Spark micro-batch restart.
    """
    import joblib
    import shutil
    import mlflow
    from mlflow.tracking import MlflowClient

    challenger_path = ctx["ti"].xcom_pull(task_ids="train_model", key="challenger_path")
    run_id          = ctx["ti"].xcom_pull(task_ids="train_model", key="run_id")

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Demote old Production to Archived
    try:
        old_versions = client.get_latest_versions("SentinelFraudModel", stages=["Production"])
        for v in old_versions:
            client.transition_model_version_stage(
                name="SentinelFraudModel", version=v.version, stage="Archived"
            )
            log.info(f"Archived old champion v{v.version}")
    except Exception as e:
        log.warning(f"Could not archive old champion: {e}")

    # Promote challenger to Production
    new_versions = client.get_latest_versions("SentinelFraudModel", stages=["None"])
    if new_versions:
        latest = max(new_versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(
            name="SentinelFraudModel", version=latest.version, stage="Production"
        )
        log.info(f"Promoted v{latest.version} to Production ✅")

    # Hot-swap the on-disk model used by processor.py
    # Save a timestamped versioned backup so old models are NEVER lost
    versions_dir = PROJECT_ROOT / "model_versions"
    versions_dir.mkdir(exist_ok=True)

    if MODEL_PATH.exists():
        ts          = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = versions_dir / f"fraud_model_{ts}.joblib"
        shutil.copyfile(MODEL_PATH, backup_path)
        log.info(f"Versioned backup saved → {backup_path}")

        # Keep only the 5 most recent backups to save disk space
        backups = sorted(versions_dir.glob("fraud_model_*.joblib"))
        for old in backups[:-5]:
            old.unlink()
            log.info(f"Pruned old backup: {old.name}")

    shutil.copyfile(challenger_path, MODEL_PATH)
    log.info(f"Hot-swapped fraud_model.joblib ✅  ({MODEL_PATH})")


# ============================================================
# TASK 6b — Reject challenger
# ============================================================
def reject_model(**ctx):
    """Champion holds — log the rejection and archive the challenger run."""
    import mlflow
    from mlflow.tracking import MlflowClient

    run_id = ctx["ti"].xcom_pull(task_ids="train_model", key="run_id")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    try:
        client.set_tag(run_id, "promotion_status", "rejected")
        log.info(f"Challenger run {run_id} marked as rejected in MLflow.")
    except Exception as e:
        log.warning(f"Could not tag run: {e}")

    log.info("Champion model retained. No changes to production model. ✅")


# ============================================================
# DAG DEFINITION
# ============================================================
with DAG(
    dag_id="sentinel_fraud_retrain",
    description="Monthly ML retraining pipeline for Sentinel AI fraud detection",
    default_args=default_args,
    schedule_interval="0 2 1 * *",   # 1st of every month at 02:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "fraud-detection", "sentinel-ai"],
    params={
        "strategy": Param(
            "A",
            type="string",
            title="Retraining Strategy",
            description="Select Strategy: 'A' (Full retrain from scratch) or 'B' (Incremental +5 trees hotfix)",
            enum=["A", "B"],
        ),
    },
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    t_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    t_features = PythonOperator(
        task_id="engineer_features",
        python_callable=engineer_features,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift,
    )

    t_evaluate = BranchPythonOperator(
        task_id="evaluate_vs_champion",
        python_callable=evaluate_vs_champion,
    )

    t_promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    t_reject = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model,
    )

    # DAG wiring
    start >> t_validate >> t_features >> t_train >> t_drift >> t_evaluate
    t_evaluate >> [t_promote, t_reject]
    [t_promote, t_reject] >> end
