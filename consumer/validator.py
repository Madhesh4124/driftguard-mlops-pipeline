import json
import time
import sys
from utils.config_loader import load_config
from utils.kafka_utils import get_consumer, get_producer
from utils.schema import validate_record

def route_messages(config_path=None):
    config = load_config(config_path)
    
    raw_topic = config["kafka"]["topics"]["raw_data"]
    valid_topic = config["kafka"]["topics"]["valid_data"]
    dlq_topic = config["kafka"]["topics"]["dead_letter"]
    
    consumer = get_consumer(config, group_id="data-validator-group")
    consumer.subscribe([raw_topic])
    
    producer = get_producer(config)
    
    print(f"Validator started. Consuming from {raw_topic}...")
    
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}", file=sys.stderr)
                continue
                
            raw_payload = msg.value().decode("utf-8")
            
            # 1. Parse JSON
            try:
                record = json.loads(raw_payload)
            except json.JSONDecodeError as e:
                # Malformed JSON, route to DLQ
                dlq_envelope = {
                    "original_payload": raw_payload,
                    "validation_error": f"JSONDecodeError: {str(e)}",
                    "timestamp": time.time()
                }
                producer.produce(dlq_topic, value=json.dumps(dlq_envelope).encode("utf-8"))
                producer.poll(0)
                print(f"[Validator] Malformed JSON: Routed to DLQ")
                continue
                
            # 2. Validate using Pandera
            is_valid, err_msg = validate_record(record)
            if is_valid:
                # Route to valid-data-topic
                producer.produce(valid_topic, value=json.dumps(record).encode("utf-8"))
                producer.poll(0)
            else:
                # Route to DLQ
                dlq_envelope = {
                    "original_payload": record,
                    "validation_error": f"Schema ValidationError: {err_msg}",
                    "timestamp": time.time()
                }
                producer.produce(dlq_topic, value=json.dumps(dlq_envelope).encode("utf-8"))
                producer.poll(0)
                print(f"[Validator] Invalid record: Routed to DLQ. Reason: {err_msg}")
                
    except KeyboardInterrupt:
        print("\nValidator consumer stopped by user.")
    finally:
        consumer.close()
        producer.flush()
        print("Validator stopped.")

if __name__ == "__main__":
    route_messages()
