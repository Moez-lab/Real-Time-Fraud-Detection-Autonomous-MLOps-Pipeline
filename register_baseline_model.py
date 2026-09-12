"""
Register the existing fraud_model.joblib into MLflow as the baseline champion.
Run this ONCE before the first Airflow retrain so MLflow has a starting point.

Usage:
    python register_baseline_model.py
"""

import os
import sys

# Fix Windows CP1252 emoji crash from MLflow's own print statements
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import shutil
import tempfile
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH   = PROJECT_ROOT / "fraud_model.joblib"
DATA_PATH    = PROJECT_ROOT / "creditcard.csv"
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

print("=" * 60)
print("Sentinel AI - Registering baseline model into MLflow")
print("=" * 60)

# 1. Load existing model
print(f"\n[1/4] Loading model from {MODEL_PATH} ...")
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load(MODEL_PATH)
print(f"      Model type: {type(model).__name__}")

# 2. Evaluate against held-out test split
print(f"\n[2/4] Loading data and evaluating ...")
df = pd.read_csv(DATA_PATH)
feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
X, y = df[feature_cols], df["Class"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# .values bypasses XGBoost feature-name validation (version mismatch in pickle)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    y_pred_proba = model.predict_proba(X_test.values)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

metrics = {
    "roc_auc"  : roc_auc_score(y_test, y_pred_proba),
    "pr_auc"   : average_precision_score(y_test, y_pred_proba),
    "f1"       : f1_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall"   : recall_score(y_test, y_pred),
}
print("\n      Baseline metrics:")
for k, v in metrics.items():
    print(f"        {k:<12}: {v:.4f}")

# 3. Log run to MLflow using low-level MlflowClient
#    (avoids mlflow.xgboost.log_model which needs the newer /logged-models API)
print(f"\n[3/4] Logging to MLflow at {MLFLOW_URI} ...")
mlflow.set_tracking_uri(MLFLOW_URI)
client = mlflow.tracking.MlflowClient()

# Get or create experiment
exp = client.get_experiment_by_name("sentinel-fraud-detection")
if exp is None:
    exp_id = client.create_experiment("sentinel-fraud-detection")
    print("      Created experiment 'sentinel-fraud-detection'")
else:
    exp_id = exp.experiment_id
    print(f"      Using existing experiment (id={exp_id})")

run = client.create_run(exp_id, run_name="baseline_model_v0",
                        tags={"source": "kaggle_notebook"})
run_id = run.info.run_id

client.log_param(run_id, "model_type", type(model).__name__)
client.log_param(run_id, "dataset",    "creditcard.csv")
client.log_param(run_id, "source",     "manually_trained")
for k, v in metrics.items():
    client.log_metric(run_id, k, v)

# Copy model to a temp dir and log as artifact
tmp_dir    = Path(tempfile.mkdtemp())
model_copy = tmp_dir / "fraud_model.joblib"
shutil.copy2(MODEL_PATH, model_copy)
client.log_artifact(run_id, str(model_copy), artifact_path="model")
client.set_terminated(run_id, status="FINISHED")
shutil.rmtree(tmp_dir)
print(f"      Run logged. Run ID: {run_id}")

# 4. Register model version & promote to Production
print(f"\n[4/4] Registering model in MLflow registry ...")
try:
    client.create_registered_model("SentinelFraudModel")
    print("      Created registered model 'SentinelFraudModel'")
except Exception:
    print("      Model already registered - adding new version")

artifact_uri  = client.get_run(run_id).info.artifact_uri
model_version = client.create_model_version(
    name="SentinelFraudModel",
    source=f"{artifact_uri}/model",
    run_id=run_id,
    description="Baseline from Kaggle notebook",
)
client.transition_model_version_stage(
    name="SentinelFraudModel",
    version=model_version.version,
    stage="Production",
)
print(f"      v{model_version.version} -> Production [OK]")

print("\n" + "=" * 60)
print("Done! Open http://localhost:5000 to see your model.")
print("  Experiment : sentinel-fraud-detection")
print("  Registry   : SentinelFraudModel  (Stage: Production)")
print("=" * 60)
