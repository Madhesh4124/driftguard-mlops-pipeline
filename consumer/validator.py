import json
import time
import sys
import mlflow
from utils.config_loader import load_config
from utils.kafka_utils import get_consumer, get_producer
from utils.schema import validate_batch

def delivery_report(err, msg):
    if err:
        print(f"[Validator] Delivery FAILED: {err}", file=sys.stderr)

def _process_batch(records, producer, valid_topic, dlq_topic, valid_count, dlq_count, run_id=None):
    valid_records, invalid_records = validate_batch(records)
    
    # Route valid records
    for record in valid_records:
        producer.produce(
            valid_topic,
            value=json.dumps(record).encode("utf-8"),
            callback=delivery_report
        )
        producer.poll(0)
        valid_count += 1
        
    # Route invalid records to DLQ
    for record, err_msg in invalid_records:
        dlq_envelope = {
            "original_payload": record,
            "validation_error": f"Schema ValidationError: {err_msg}",
            "timestamp": time.time()
        }
        producer.produce(
            dlq_topic,
            value=json.dumps(dlq_envelope).encode("utf-8"),
            callback=delivery_report
        )
        producer.poll(0)
        dlq_count += 1
        print(f"[Validator] Invalid record: Routed to DLQ. Reason: {err_msg}")
        
    if len(valid_records) > 0:
        print(f"[Validator] ✓ Batch processed. Routed {len(valid_records)} valid records. Total valid: {valid_count}")
        
    # Log metrics to MLflow
    if run_id:
        try:
            mlflow.log_metrics({
                "valid_records_routed": valid_count,
                "dlq_rejections": dlq_count
            })
        except Exception as e:
            print(f"[Validator] MLflow logging failed: {e}")
            
    return valid_count, dlq_count

def route_messages(config_path=None):
    config = load_config(config_path)
    
    raw_topic = config["kafka"]["topics"]["raw_data"]
    valid_topic = config["kafka"]["topics"]["valid_data"]
    dlq_topic = config["kafka"]["topics"]["dead_letter"]
    
    consumer = get_consumer(config, group_id="data-validator-group")
    consumer.subscribe([raw_topic])
    
    producer = get_producer(config)
    valid_count = 0
    dlq_count = 0
    batch_buffer = []
    MICRO_BATCH_SIZE = 50

    # MLflow Setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    print(f"Validator started. Consuming from {raw_topic}...")
    
    with mlflow.start_run(run_name="data-validator-service") as run:
        run_id = run.info.run_id
        try:
            while True:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    # Flush partial batch on idle
                    if batch_buffer:
                        valid_count, dlq_count = _process_batch(
                            batch_buffer, producer, valid_topic, dlq_topic,
                            valid_count, dlq_count, run_id
                        )
                        batch_buffer = []
                    continue
                if msg.error():
                    print(f"Consumer error: {msg.error()}", file=sys.stderr)
                    continue
                    
                raw_payload = msg.value().decode("utf-8")
                
                # 1. Parse JSON
                try:
                    record = json.loads(raw_payload)
                    batch_buffer.append(record)
                except json.JSONDecodeError as e:
                    # Malformed JSON, route to DLQ immediately
                    dlq_envelope = {
                        "original_payload": raw_payload,
                        "validation_error": f"JSONDecodeError: {str(e)}",
                        "timestamp": time.time()
                    }
                    producer.produce(dlq_topic, value=json.dumps(dlq_envelope).encode("utf-8"), callback=delivery_report)
                    producer.poll(0)
                    dlq_count += 1
                    print(f"[Validator] Malformed JSON: Routed to DLQ")
                    
                    try:
                        mlflow.log_metrics({
                            "valid_records_routed": valid_count,
                            "dlq_rejections": dlq_count
                        })
                    except Exception:
                        pass
                    continue
                    
                if len(batch_buffer) >= MICRO_BATCH_SIZE:
                    valid_count, dlq_count = _process_batch(
                        batch_buffer, producer, valid_topic, dlq_topic,
                        valid_count, dlq_count, run_id
                    )
                    batch_buffer = []
                    
        except KeyboardInterrupt:
            print("\nValidator consumer stopped by user.")
        finally:
            # Final flush on exit
            if batch_buffer:
                _process_batch(
                    batch_buffer, producer, valid_topic, dlq_topic,
                    valid_count, dlq_count, run_id
                )
            consumer.close()
            producer.flush()
            print("Validator stopped.")

if __name__ == "__main__":
    route_messages()
