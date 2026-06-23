from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import Producer, Consumer

def get_admin_client(config):
    return AdminClient({"bootstrap.servers": config["kafka"]["bootstrap_servers"]})

def create_topics_if_not_exist(config):
    """
    Creates the raw-data, retrain-events, and dead-letter Kafka topics if they do not exist.
    """
    admin = get_admin_client(config)
    
    topics = [
        config["kafka"]["topics"]["raw_data"],
        config["kafka"]["topics"]["valid_data"],
        config["kafka"]["topics"]["retrain_events"],
        config["kafka"]["topics"]["dead_letter"]
    ]
    
    # Retrieve metadata with timeout
    try:
        metadata = admin.list_topics(timeout=10)
        existing_topics = set(metadata.topics.keys())
    except Exception as e:
        print(f"Error fetching Kafka metadata: {e}")
        return False
    
    new_topics = []
    for topic in topics:
        if topic not in existing_topics:
            print(f"Queueing topic creation: {topic}")
            new_topics.append(NewTopic(topic, num_partitions=1, replication_factor=1))
            
    if new_topics:
        futures = admin.create_topics(new_topics)
        for topic, future in futures.items():
            try:
                future.result() # Wait for topic creation to complete
                print(f"Topic '{topic}' created successfully.")
            except Exception as e:
                print(f"Failed to create topic '{topic}': {e}")
                return False
    else:
        print("All required topics already exist.")
    return True

def get_producer(config):
    """
    Returns a configured confluent-kafka Producer client.
    """
    conf = {
        "bootstrap.servers": config["kafka"]["bootstrap_servers"],
        "client.id": "drift-mlops-producer"
    }
    return Producer(conf)

def get_consumer(config, group_id):
    """
    Returns a configured confluent-kafka Consumer client.
    """
    conf = {
        "bootstrap.servers": config["kafka"]["bootstrap_servers"],
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True
    }
    return Consumer(conf)

if __name__ == "__main__":
    from utils.config_loader import load_config
    import sys
    
    # Allow running standalone to initialize topics
    config = load_config()
    print("Initializing topics...")
    success = create_topics_if_not_exist(config)
    if success:
        print("Kafka initialization complete.")
    else:
        print("Kafka initialization failed.")
        sys.exit(1)
