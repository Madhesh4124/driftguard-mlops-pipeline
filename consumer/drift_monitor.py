import os
import json
import time
import sys
import pandas as pd
from collections import deque
import mlflow
from evidently import Report
from evidently.metrics import DataDriftTable
from evidently.presets import DataSummaryPreset

from utils.config_loader import load_config
from utils.kafka_utils import get_consumer, get_producer

# Disable evidently telemetry warning / telemetry if needed
import warnings
warnings.filterwarnings("ignore")

def monitor_drift(config_path=None):
    config = load_config(config_path)
    
    # MLflow Setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Kafka configuration
    valid_topic = config["kafka"]["topics"]["valid_data"]
    retrain_topic = config["kafka"]["topics"]["retrain_events"]
    
    consumer = get_consumer(config, group_id="drift-monitor-group")
    consumer.subscribe([valid_topic])
    
    producer = get_producer(config)

    # Window sizes and thresholds
    ref_size = config["drift"]["reference_window_size"]
    cur_size = config["drift"]["current_window_size"]
    interval = config["drift"]["detection_interval"]
    min_drifted = config["drift"]["min_features_drifted"]
    monitored_features = config["drift"]["monitored_features"]

    reference_buffer = []
    current_buffer = deque(maxlen=cur_size)
    
    ref_df = None
    total_records_seen = 0
    cycle = 0
    retrain_cooldown_cycles = 0
    COOLDOWN = 3  # don't retrigger for N cycles after a retrain event

    print(f"Drift monitor started. Consuming valid records from {valid_topic}...")
    print(f"Waiting to fill reference window ({ref_size} rows)...")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}", file=sys.stderr)
                continue

            record = json.loads(msg.value().decode("utf-8"))

            # Persist valid records to validated history file for retraining
            os.makedirs("data", exist_ok=True)
            with open("data/validated_history.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

            if ref_df is None:
                # Still filling reference window
                reference_buffer.append(record)
                if len(reference_buffer) == ref_size:
                    ref_df = pd.DataFrame(reference_buffer)
                    print(f"Reference window populated with {ref_size} rows. Now filling sliding current window ({cur_size} rows)...")
            else:
                # Filling sliding current window
                current_buffer.append(record)
                total_records_seen += 1
                
                # Only run drift check when buffer is full AND interval has passed
                if len(current_buffer) >= cur_size and total_records_seen % interval == 0:
                    cycle += 1
                    print(f"\n--- Starting Drift Detection Cycle {cycle} ---")
                    
                    # Create DataFrame from current window
                    cur_df = pd.DataFrame(list(current_buffer))
                    
                    # 1. Run Evidently DataDriftTable Report
                    print("Running Evidently Data Drift Analysis...")
                    drift_report = Report(metrics=[DataDriftTable()])
                    drift_result = drift_report.run(
                        reference_data=ref_df[monitored_features], 
                        current_data=cur_df[monitored_features]
                    )
                    summary = drift_result.as_dict()
                    
                    # Extract drift results from Evidently report
                    drift_by_columns = summary["metrics"][0]["result"]["drift_by_columns"]
                    
                    ks_metrics = {}
                    drifted_features = []
                    
                    for feature in monitored_features:
                        info = drift_by_columns[feature]
                        p_val = info["drift_score"]  # drift_score is the p-value for KS test
                        is_drifted = info["drift_detected"]
                        
                        ks_metrics[f"p_value_{feature}"] = p_val
                        ks_metrics[f"drift_detected_{feature}"] = 1.0 if is_drifted else 0.0
                        
                        print(f"Feature: {feature} | p-value: {p_val:.4e} | Drifted: {is_drifted}")
                        if is_drifted:
                            drifted_features.append(feature)
                            
                    drift_detected = len(drifted_features) >= min_drifted
                    print(f"Drift status: {drift_detected} (Drifted features: {drifted_features}, count: {len(drifted_features)}/{len(monitored_features)})")

                    # Generate HTML reports
                    os.makedirs("reports", exist_ok=True)
                    drift_report_path = "reports/drift_report.html"
                    drift_result.save_html(drift_report_path)
                    
                    # Data Quality Report
                    quality_report = Report(metrics=[DataSummaryPreset()])
                    quality_result = quality_report.run(
                        reference_data=ref_df[monitored_features], 
                        current_data=cur_df[monitored_features]
                    )
                    quality_report_path = "reports/quality_report.html"
                    quality_result.save_html(quality_report_path)
                    
                    # 2. Log to MLflow
                    print("Logging metrics and reports to MLflow...")
                    with mlflow.start_run(run_name=f"drift_detection_cycle_{cycle}") as run:
                        mlflow.log_metrics(ks_metrics)
                        mlflow.set_tags({
                            "drift_detected": str(drift_detected),
                            "drifted_features": ",".join(drifted_features),
                            "cycle": str(cycle)
                        })
                        mlflow.log_artifact(drift_report_path)
                        mlflow.log_artifact(quality_report_path)
                    
                    # Decrement cooldown if active
                    if retrain_cooldown_cycles > 0:
                        retrain_cooldown_cycles -= 1
                        print(f"[DriftMonitor] Retraining cooldown active: {retrain_cooldown_cycles} cycles remaining.")

                    # 3. Trigger Retraining if drift detected and not in cooldown
                    if drift_detected:
                        if retrain_cooldown_cycles == 0:
                            print(f"Drift threshold crossed! Publishing retrain event to {retrain_topic}...")
                            
                            # Construct retraining event payload (includes data snapshot)
                            retrain_payload = {
                                "cycle": cycle,
                                "timestamp": time.time(),
                                "drifted_features": drifted_features,
                                "data_snapshot": cur_df.to_dict(orient="records")
                            }
                            
                            producer.produce(
                                retrain_topic,
                                value=json.dumps(retrain_payload).encode("utf-8")
                            )
                            producer.flush()
                            print("Retrain event successfully published.")
                            retrain_cooldown_cycles = COOLDOWN
                        else:
                            print(f"[DriftMonitor] Drift detected but skipped due to retraining cooldown ({retrain_cooldown_cycles} cycles remaining).")
                    else:
                        print("No retraining triggered (drift within acceptable bounds).")

    except KeyboardInterrupt:
        print("\nDrift monitor stopped by user.")
    finally:
        consumer.close()
        producer.flush()
        print("Drift monitor stopped.")

if __name__ == "__main__":
    monitor_drift()
