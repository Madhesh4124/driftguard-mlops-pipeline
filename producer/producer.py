import time
import json
import pandas as pd
import sys
import random
from utils.config_loader import load_config
from utils.kafka_utils import get_producer

# Global counters for summary statistics
produced_count = 0
failed_count = 0

def delivery_report(err, msg):
    global produced_count, failed_count
    if err is not None:
        print(f"[Producer] Message delivery failed: {err}", file=sys.stderr)
        failed_count += 1
    else:
        produced_count += 1

def stream_data(config_path=None, corrupt_rate=0.0):
    global produced_count, failed_count
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
    if corrupt_rate > 0:
        print(f"[Producer] WARNING: Corruption injection active at rate {corrupt_rate:.2%}")
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

            # Corrupt record optionally to test DLQ
            corrupt_record = False
            corrupt_json = False
            
            if corrupt_rate > 0 and random.random() < corrupt_rate:
                if random.random() < 0.5:
                    corrupt_record = True  # Value corruption (fails schema)
                else:
                    corrupt_json = True    # JSON structure corruption
            
            if corrupt_record:
                # Corrupt age and limit balance to violate constraints
                record["AGE"] = -5.0
                record["LIMIT_BAL"] = -100.0
                payload = json.dumps(record).encode("utf-8")
                print(f"[Producer] Injecting VALUE corruption at index {idx}")
            elif corrupt_json:
                # Append malformed string to make it invalid JSON
                payload = (json.dumps(record) + "invalid_json_payload}").encode("utf-8")
                print(f"[Producer] Injecting JSON corruption at index {idx}")
            else:
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
        print("\n--- Producer Summary ---")
        print(f"Total Messages successfully delivered: {produced_count}")
        print(f"Total Messages failed:                 {failed_count}")
        print("Producer stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kafka Data Stream Producer")
    parser.add_argument(
        "--corrupt-rate",
        type=float,
        default=0.0,
        help="Fraction of records to corrupt (e.g. 0.05 for 5%)"
    )
    args = parser.parse_args()
    stream_data(corrupt_rate=args.corrupt_rate)
