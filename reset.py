import os
import shutil
import time
import sys
import subprocess
from utils.config_loader import load_config
from utils.kafka_utils import create_topics_if_not_exist
from confluent_kafka.admin import AdminClient

def kill_locking_processes():
    print("[Reset] Scanning for running processes that might lock mlflow.db...")
    try:
        output = subprocess.check_output(
            'wmic process where "name=\'python.exe\'" get CommandLine,ProcessId /format:csv',
            shell=True,
            text=True
        )
        lines = output.strip().split('\n')
        current_pid = os.getpid()
        
        processes = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('Node'):
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                pid_str = parts[-1]
                cmdline = ",".join(parts[1:-1])
                try:
                    pid = int(pid_str)
                    processes.append((pid, cmdline))
                except ValueError:
                    continue

        pids_to_kill = set()
        
        # Pass 1: Find main processes
        for pid, cmdline in processes:
            if pid == current_pid:
                continue
            is_mlflow = 'mlflow' in cmdline.lower()
            is_api = 'api.app' in cmdline.lower()
            is_uvicorn = 'uvicorn' in cmdline.lower() and ('mlflow' in cmdline or 'api.app' in cmdline)
            
            if is_mlflow or is_api or is_uvicorn:
                pids_to_kill.add(pid)
                
        # Pass 2: Find child spawn processes whose parent PID is in our list or is orphaned (not running)
        all_pids = {p for p, _ in processes}
        for pid, cmdline in processes:
            if pid == current_pid or pid in pids_to_kill:
                continue
            if 'parent_pid=' in cmdline:
                try:
                    sub = cmdline.split('parent_pid=')[1]
                    parent_pid_str = sub.split(',')[0].split(')')[0].strip()
                    parent_pid = int(parent_pid_str)
                    if parent_pid in pids_to_kill or parent_pid not in all_pids:
                        if '.venv' in cmdline.lower() or 'mlflow' in cmdline.lower():
                            pids_to_kill.add(pid)
                except Exception:
                    pass

                    
        # Kill all identified processes
        for pid in pids_to_kill:
            cmd = next((c for p, c in processes if p == pid), "")
            print(f"[Reset] Killing locking process PID {pid}: {cmd[:80]}...")
            try:
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[Reset] Warning: Could not kill process {pid}: {e}")
                
        time.sleep(1.5)
    except Exception as e:
        print(f"[Reset] Warning: Error checking/killing locking processes: {e}")


def reset_project():
    print("==================================================")
    print("              RESETTING PROJECT STATE             ")
    print("==================================================")
    
    kill_locking_processes()
    config = load_config()

    
    # 1. Delete SQLite MLflow database
    db_path = "mlflow.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"[Reset] Deleted MLflow database: {db_path}")
        except Exception as e:
            print(f"[Reset] Warning: Could not delete {db_path} (it might be locked): {e}")
    else:
        print("[Reset] No MLflow database found to delete.")

    # 2. Delete mlruns/ directory
    mlruns_dir = "mlruns"
    if os.path.exists(mlruns_dir) and os.path.isdir(mlruns_dir):
        try:
            shutil.rmtree(mlruns_dir)
            print(f"[Reset] Deleted mlruns directory: {mlruns_dir}")
        except Exception as e:
            print(f"[Reset] Warning: Could not delete {mlruns_dir}: {e}")
    else:
        print("[Reset] No mlruns directory found to delete.")

    # 3. Delete holdout.parquet file and other generated data/logs
    files_to_delete = [
        "data/holdout.parquet",
        "data/validated_history.jsonl",
        "data/prediction_log.jsonl",
        "data/dlq_rejections.jsonl"
    ]
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"[Reset] Deleted: {file_path}")
            except Exception as e:
                print(f"[Reset] Warning: Could not delete {file_path}: {e}")
        else:
            print(f"[Reset] No {file_path} found to delete.")

    # 4. Reset Kafka topics
    print("[Reset] Resetting Kafka topics...")
    bootstrap_servers = config["kafka"]["bootstrap_servers"]
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    
    topics_to_reset = [
        config["kafka"]["topics"]["raw_data"],
        config["kafka"]["topics"]["valid_data"],
        config["kafka"]["topics"]["retrain_events"],
        config["kafka"]["topics"]["dead_letter"]
    ]
    
    try:
        metadata = admin.list_topics(timeout=5)
        existing_topics = set(metadata.topics.keys())
        
        # Filter existing topics to delete
        topics_to_delete = [t for t in topics_to_reset if t in existing_topics]
        
        if topics_to_delete:
            print(f"[Reset] Deleting existing topics: {topics_to_delete}")
            futures = admin.delete_topics(topics_to_delete)
            for topic, future in futures.items():
                try:
                    future.result() # wait for deletion
                    print(f"[Reset] Topic '{topic}' deleted.")
                except Exception as e:
                    print(f"[Reset] Failed to delete topic '{topic}': {e}")
            
            # Wait a moment for Kafka to finalize deletion
            time.sleep(2)
        else:
            print("[Reset] No active topics to delete.")
            
        # Recreate topics
        print("[Reset] Recreating clean Kafka topics...")
        success = create_topics_if_not_exist(config)
        if success:
            print("[Reset] Kafka topics successfully recreated.")
        else:
            print("[Reset] Error: Failed to recreate Kafka topics.")
            
    except Exception as e:
        print(f"[Reset] Error communicating with Kafka broker: {e}")
        print("[Reset] Make sure your Docker container is up and running (`docker compose up -d`).")
        sys.exit(1)

    print("==================================================")
    print("Project successfully reset to a clean state!")
    print("==================================================")
    print("\nNext steps to restart:")
    print("1. Start Docker containers (if not running):  docker compose up -d")
    print("2. Run baseline training:                    python -m model.train_baseline")
    print("3. Start your consumers (validator, drift_monitor, retraining_engine)")
    print("4. Run uvicorn serving API:                  python -m uvicorn api.app:app --host 127.0.0.1 --port 8000")
    print("5. Start streaming producer:                 python -m producer.producer")

if __name__ == "__main__":
    reset_project()
