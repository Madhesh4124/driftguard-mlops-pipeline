import os
import json
import sys
import time
import mlflow
from utils.config_loader import load_config
from utils.kafka_utils import get_consumer

def consume_dlq(config_path=None):
    config = load_config(config_path)
    dlq_topic = config["kafka"]["topics"]["dead_letter"]
    
    consumer = get_consumer(config, group_id="dlq-monitor-group")
    consumer.subscribe([dlq_topic])
    
    # MLflow Setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    print(f"DLQ Consumer started. Listening for rejections on: {dlq_topic}...")
    
    rejection_count = 0
    log_path = "data/dlq_rejections.jsonl"
    
    with mlflow.start_run(run_name="dlq-monitor-service") as run:
        run_id = run.info.run_id
        try:
            while True:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    print(f"DLQ Consumer error: {msg.error()}", file=sys.stderr)
                    continue
                    
                raw_payload = msg.value().decode("utf-8")
                
                try:
                    envelope = json.loads(raw_payload)
                except json.JSONDecodeError:
                    envelope = {
                        "original_payload": raw_payload,
                        "validation_error": "Payload was not valid JSON",
                        "timestamp": time.time()
                    }
                
                rejection_count += 1
                error_msg = envelope.get("validation_error", "Unknown error")
                print(f"[DLQ Alert] Rejection #{rejection_count}: {error_msg}")
                
                # Log to local file
                os.makedirs("data", exist_ok=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(envelope) + "\n")
                    
                # Log to MLflow
                try:
                    mlflow.log_metric("total_dlq_rejections", rejection_count)
                except Exception as e:
                    pass
                    
        except KeyboardInterrupt:
            print("\nDLQ Consumer stopped by user.")
        finally:
            consumer.close()
            print("DLQ Consumer stopped.")

if __name__ == "__main__":
    consume_dlq()
