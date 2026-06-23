import os
import json
import time
import sys
import pandas as pd
from collections import deque
import mlflow
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from utils.config_loader import load_config
from utils.kafka_utils import get_consumer, get_producer
from utils.drift_utils import calculate_ks_test

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
    ks_threshold = config["drift"]["ks_threshold"]
    min_drifted = config["drift"]["min_features_drifted"]
    monitored_features = config["drift"]["monitored_features"]

    reference_buffer = []
    current_buffer = deque(maxlen=cur_size)
    
    ref_df = None
    records_since_last_check = 0
    cycle = 0

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

            if ref_df is None:
                # Still filling reference window
                reference_buffer.append(record)
                if len(reference_buffer) == ref_size:
                    ref_df = pd.DataFrame(reference_buffer)
                    print(f"Reference window populated with {ref_size} rows. Now filling sliding current window ({cur_size} rows)...")
            else:
                # Filling sliding current window
                current_buffer.append(record)
                if len(current_buffer) == cur_size:
                    records_since_last_check += 1
                    
                    if records_since_last_check >= interval:
                        cycle += 1
                        print(f"\n--- Starting Drift Detection Cycle {cycle} ---")
                        records_since_last_check = 0
                        
                        # Create DataFrame from current window
                        cur_df = pd.DataFrame(list(current_buffer))
                        
                        # 1. Calculate KS-test for monitored features
                        ks_metrics = {}
                        drifted_features = []
                        
                        for feature in monitored_features:
                            if feature not in ref_df.columns or feature not in cur_df.columns:
                                print(f"Warning: Monitored feature '{feature}' not present in window columns.")
                                continue
                            
                            stat, p_val = calculate_ks_test(ref_df[feature], cur_df[feature])
                            ks_metrics[f"ks_stat_{feature}"] = stat
                            ks_metrics[f"p_value_{feature}"] = p_val
                            
                            is_drifted = p_val < ks_threshold
                            print(f"Feature: {feature} | KS Stat: {stat:.4f} | p-value: {p_val:.4e} | Drifted: {is_drifted}")
                            if is_drifted:
                                drifted_features.append(feature)
                                
                        drift_detected = len(drifted_features) >= min_drifted
                        print(f"Drift status: {drift_detected} (Drifted features: {drifted_features}, count: {len(drifted_features)}/{len(monitored_features)})")

                        # 2. Generate Evidently HTML reports
                        print("Generating Evidently HTML reports...")
                        os.makedirs("reports", exist_ok=True)
                        
                        # Data Drift Report
                        drift_report = Report(metrics=[DataDriftPreset()])
                        drift_result = drift_report.run(reference_data=ref_df[monitored_features], current_data=cur_df[monitored_features])
                        drift_report_path = "reports/drift_report.html"
                        drift_result.save_html(drift_report_path)
                        
                        # Data Quality Report
                        quality_report = Report(metrics=[DataSummaryPreset()])
                        quality_result = quality_report.run(reference_data=ref_df[monitored_features], current_data=cur_df[monitored_features])
                        quality_report_path = "reports/quality_report.html"
                        quality_result.save_html(quality_report_path)
                        
                        # 3. Log to MLflow
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
                        
                        # 4. Trigger Retraining if drift detected
                        if drift_detected:
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
