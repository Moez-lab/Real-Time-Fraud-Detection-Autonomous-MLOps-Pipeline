"""
Sentinel AI — Real-Time Drift Monitor
=======================================
Standalone script that runs alongside the live Spark pipeline.
Periodically samples the Kafka stream (via flagged_alerts.txt) and
compares the live distribution against the training baseline to
detect data drift. Raises an alert if drift is detected and can
optionally trigger the Airflow retraining DAG via its REST API.

Usage:
    python drift_monitor.py

Env vars (optional):
    AIRFLOW_HOST      — Airflow webserver URL  (default: http://localhost:8081)
    AIRFLOW_USER      — Airflow username        (default: admin)
    AIRFLOW_PASSWORD  — Airflow password        (default: admin)
    AUTO_TRIGGER_DAG  — Set to "true" to auto-trigger retrain on drift
"""

import os
import time
import json
import logging
import requests
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [DRIFT-MONITOR]  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent
DATA_PATH       = PROJECT_ROOT / "creditcard.csv"
ALERTS_PATH     = PROJECT_ROOT / "flagged_alerts.txt"
CHECK_INTERVAL  = 60          # seconds between drift checks
DRIFT_THRESHOLD = 0.10        # Jensen-Shannon divergence threshold (0-1)

AIRFLOW_HOST    = os.getenv("AIRFLOW_HOST",     "http://localhost:8081")
AIRFLOW_USER    = os.getenv("AIRFLOW_USER",     "admin")
AIRFLOW_PASS    = os.getenv("AIRFLOW_PASSWORD", "admin")
AUTO_TRIGGER    = os.getenv("AUTO_TRIGGER_DAG", "false").lower() == "true"
COOLDOWN_PERIOD = int(os.getenv("DRIFT_COOLDOWN_PERIOD", "3600"))  # in seconds (default: 1 hour)

FEATURE_NAMES   = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def build_reference_stats() -> dict:
    """
    Compute per-feature mean and std from the training dataset.
    These become our baseline distribution fingerprint.
    """
    log.info("Computing reference statistics from training data ...")
    df = pd.read_csv(DATA_PATH, usecols=FEATURE_NAMES)
    stats = {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
        for col in FEATURE_NAMES
    }
    log.info(f"Reference stats built for {len(stats)} features ✅")
    return stats

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two histograms.
    Returns a value in [0, 1] — higher means more drift.
    """
    from scipy.spatial.distance import jensenshannon
    bins   = np.linspace(min(p.min(), q.min()), max(p.max(), q.max()), 30)
    p_hist = np.histogram(p, bins=bins, density=True)[0] + 1e-10
    q_hist = np.histogram(q, bins=bins, density=True)[0] + 1e-10
    return float(jensenshannon(p_hist, q_hist))


def check_drift(reference_stats: dict, live_data: pd.DataFrame) -> dict:
    """
    Compare live feature distributions against reference using JSD.
    Returns a drift report dict.
    """
    results     = {}
    drifted     = []

    for col in FEATURE_NAMES:
        if col not in live_data.columns:
            continue
        live_vals = live_data[col].dropna().values
        if len(live_vals) < 10:
            continue

        ref_mean = reference_stats[col]["mean"]
        ref_std  = reference_stats[col]["std"]
        # Synthesise a reference sample from Gaussian for JSD comparison
        ref_sample = np.random.normal(ref_mean, ref_std, size=len(live_vals))

        jsd = jensen_shannon_divergence(live_vals, ref_sample)
        results[col] = round(jsd, 4)

        if jsd > DRIFT_THRESHOLD:
            drifted.append(col)

    avg_jsd     = float(np.mean(list(results.values()))) if results else 0.0
    drift_score = round(avg_jsd, 4)
    is_drifted  = len(drifted) > 0

    return {
        "timestamp"   : datetime.utcnow().isoformat(),
        "is_drifted"  : is_drifted,
        "drift_score" : drift_score,
        "drifted_feats": drifted,
        "per_feature" : results,
    }


def trigger_airflow_retrain(strategy: str = "A"):
    """
    Hit the Airflow REST API to manually trigger the retraining DAG with a strategy.
    """
    url     = f"{AIRFLOW_HOST}/api/v1/dags/sentinel_fraud_retrain/dagRuns"
    payload = {"conf": {"triggered_by": "drift_monitor", "strategy": strategy}}
    try:
        resp = requests.post(
            url,
            json=payload,
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=10,
        )
        if resp.status_code in (200, 201):
            run_id = resp.json().get("dag_run_id", "unknown")
            log.info(f"Airflow DAG triggered successfully! Run ID: {run_id}")
        else:
            log.warning(f"Failed to trigger DAG: {resp.status_code} — {resp.text}")
    except requests.exceptions.RequestException as e:
        log.error(f"Could not reach Airflow: {e}")


def parse_live_alerts() -> pd.DataFrame:
    """
    Read the flagged_alerts.txt to extract live transaction features.
    Returns a minimal DataFrame with Amount and fraud_probability.
    """
    if not ALERTS_PATH.exists():
        return pd.DataFrame()

    records = []
    with open(ALERTS_PATH, "r") as f:
        for line in f:
            # FRAUD ALERT | Time ID: 54 | Amount: $1200.00 | AI Score: 0.9123
            try:
                parts  = dict(p.strip().split(": ", 1) for p in line.split("|")[1:])
                amount = float(parts.get("Amount", "$0").replace("$", ""))
                score  = float(parts.get("AI Score", 0))
                records.append({"Amount": amount, "fraud_probability": score})
            except Exception:
                continue

    return pd.DataFrame(records)


def process_drift_check(
    live_df: pd.DataFrame, 
    reference_stats: dict, 
    consecutive_drifts: int, 
    last_trigger_time: float
) -> tuple[int, float]:
    """
    Process a single iteration of the drift check.
    Computes drift, increments counter, checks cooldown, and triggers Airflow if necessary.
    Returns the updated (consecutive_drifts, last_trigger_time).
    """
    if "Amount" in live_df.columns and len(live_df) >= 10:
        report = check_drift(
            {"Amount": reference_stats["Amount"]},
            live_df[["Amount"]]
        )

        drift_emoji = "🔴" if report["is_drifted"] else "🟢"
        log.info(
            f"{drift_emoji} Drift check | Score: {report['drift_score']:.4f} | "
            f"Drifted features: {report['drifted_feats'] or 'none'} | "
            f"Alerts sampled: {len(live_df)}"
        )

        if report["is_drifted"]:
            consecutive_drifts += 1
            log.warning(
                f"DRIFT DETECTED! ({consecutive_drifts} consecutive) "
                f"Threshold: {DRIFT_THRESHOLD}"
            )
            # Only auto-trigger after 3 consecutive drift signals
            if consecutive_drifts >= 3 and AUTO_TRIGGER:
                current_time = time.time()
                if current_time - last_trigger_time >= COOLDOWN_PERIOD:
                    # Decide strategy based on weekday (1 is Tuesday)
                    is_tuesday = datetime.now().weekday() == 1
                    strategy = "B" if is_tuesday else "A"
                    
                    log.warning(f"Triggering Airflow retraining DAG with Strategy {strategy} ...")
                    trigger_airflow_retrain(strategy=strategy)
                    last_trigger_time = current_time
                    consecutive_drifts = 0   # Reset after trigger
                else:
                    remaining = int(COOLDOWN_PERIOD - (current_time - last_trigger_time))
                    log.info(
                        f"Drift detected but Airflow trigger is on cooldown. "
                        f"Cooldown remaining: {remaining}s. Skipping trigger."
                    )
                    consecutive_drifts = 0   # Reset counter to avoid immediate check next loop
        else:
            consecutive_drifts = 0

    else:
        log.info(f"Insufficient alert data ({len(live_df)} rows) — need at least 10.")

    return consecutive_drifts, last_trigger_time


def main():
    log.info("=" * 60)
    log.info("Sentinel AI Drift Monitor started")
    log.info(f"  Check interval  : {CHECK_INTERVAL}s")
    log.info(f"  Drift threshold : {DRIFT_THRESHOLD} (JSD)")
    log.info(f"  Auto-trigger DAG: {AUTO_TRIGGER}")
    log.info(f"  Cooldown period : {COOLDOWN_PERIOD}s")
    log.info("=" * 60)

    # Build reference stats once at startup
    reference_stats = build_reference_stats()
    consecutive_drifts = 0
    last_trigger_time = 0.0

    while True:
        time.sleep(CHECK_INTERVAL)

        live_df = parse_live_alerts()
        if live_df.empty:
            log.info("No live alerts yet — waiting ...")
            continue

        consecutive_drifts, last_trigger_time = process_drift_check(
            live_df, reference_stats, consecutive_drifts, last_trigger_time
        )


if __name__ == "__main__":
    main()
