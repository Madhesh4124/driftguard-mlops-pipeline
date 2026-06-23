import time
import json
import pandas as pd
import sys
from utils.config_loader import load_config
from utils.kafka_utils import get_producer

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}", file=sys.stderr)
    # Silent on success to avoid stdout flooding, or print occasionally

def stream_data(config_path=None):
    config = load_config(config_path)
    topic = config["kafka"]["topics"]["raw_data"]
    delay_s = config["kafka"]["producer_delay_ms"] / 1000.0
    phase1_rows = config["data"]["phase1_rows"]

    print("Loading two-phase dataset...")
    try:
        df = pd.read_csv("data/credit_default_two_phase.csv")
    except FileNotFoundError:
        print("Dataset not found at data/credit_default_two_phase.csv. Please run data/generator.py first.")
        sys.exit(1)

    print(f"Initializing Kafka producer and streaming to topic: {topic}...")
    producer = get_producer(config)

    total_rows = len(df)
    print(f"Streaming {total_rows} rows...")

    try:
        for idx, row in df.iterrows():
            record = row.to_dict()
            
            # Determine active phase for logging
            phase = 1 if idx < phase1_rows else 2
            
            # Print status update every 500 rows
            if idx % 500 == 0:
                print(f"[Producer] Streaming index: {idx}/{total_rows} | Current Phase: {phase}")

            # Serialize to JSON and encode to bytes
            payload = json.dumps(record).encode("utf-8")
            
            # Asynchronously send record to raw-data-topic
            producer.produce(
                topic,
                value=payload,
                callback=delivery_report
            )
            
            # Serve delivery callback reports
            producer.poll(0)
            
            # Inject config delay
            time.sleep(delay_s)
            
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user. Flushing buffered messages...")
    finally:
        # Wait for outstanding messages to be delivered
        producer.flush()
        print("Producer stopped.")

if __name__ == "__main__":
    stream_data()
