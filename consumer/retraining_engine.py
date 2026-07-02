import json
import os
import sys
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from utils.config_loader import load_config
from utils.kafka_utils import get_consumer
from model.train_optuna import run_hpo
from model.promoter import evaluate_and_promote

def listen_for_retrain_events(config_path=None):
    config = load_config(config_path)
    
    # MLflow Setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Kafka Setup
    retrain_topic = config["kafka"]["topics"]["retrain_events"]
    model_name = config["mlflow"]["model_name"]
    n_trials = config["retraining"]["n_trials"]
    val_split = config["retraining"]["val_split"]
    holdout_split = config["retraining"]["holdout_split"]

    consumer = get_consumer(config, group_id="retraining-engine-group")
    consumer.subscribe([retrain_topic])

    print(f"Retraining engine started. Listening for alerts on topic: {retrain_topic}...")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}", file=sys.stderr)
                continue

            event = json.loads(msg.value().decode("utf-8"))
            cycle = event.get("cycle", 0)
            drifted_features = event.get("drifted_features", [])
            snapshot_data = event.get("data_snapshot", [])

            print(f"\n==================================================")
            print(f"Received Retraining Event for Cycle {cycle}!")
            print(f"Drifted features: {drifted_features}")
            print(f"Snapshot size: {len(snapshot_data)} rows")
            print(f"==================================================")

            # 1. Load accumulated valid records if available, else fallback to snapshot
            if os.path.exists("data/validated_history.jsonl"):
                print("[Retrainer] Loading accumulated historical valid records from data/validated_history.jsonl...")
                df_snapshot = pd.read_json("data/validated_history.jsonl", lines=True)
            else:
                print("[Retrainer] validated_history.jsonl not found. Falling back to event snapshot data...")
                df_snapshot = pd.DataFrame(snapshot_data)
            
            # Minimum data size guard
            if len(df_snapshot) < 1000:
                print(f"[Retrainer] Too few samples ({len(df_snapshot)}). Skipping retraining.")
                continue

            # Clean ID column if it exists
            if "ID" in df_snapshot.columns:
                df_snapshot = df_snapshot.drop(columns=["ID"])
                
            target_col = "default.payment.next.month"
            X_snap = df_snapshot.drop(columns=[target_col])
            y_snap = df_snapshot[target_col]

            # 2. Perform train/validation/holdout splits with stratification crash guard
            try:
                X_train_val, X_eval, y_train_val, y_eval = train_test_split(
                    X_snap, y_snap, test_size=holdout_split, random_state=42, stratify=y_snap
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=val_split, random_state=42, stratify=y_train_val
                )
            except ValueError as e:
                print(f"[Retrainer] Stratified split failed ({e}). Falling back to random split.")
                X_train_val, X_eval, y_train_val, y_eval = train_test_split(
                    X_snap, y_snap, test_size=holdout_split, random_state=42
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=val_split, random_state=42
                )

            parent_run_name = f"retraining_cycle_{cycle}"
            print(f"Starting MLflow parent run: {parent_run_name}...")
            
            # Dynamically reduce Optuna trials if dataset is relatively small
            current_trials = n_trials
            if len(df_snapshot) < 2000:
                current_trials = min(10, n_trials)
                print(f"[Retrainer] Small dataset size ({len(df_snapshot)}). Reducing HPO trials to {current_trials}.")

            with mlflow.start_run(run_name=parent_run_name) as run:
                mlflow.set_tags({
                    "cycle": str(cycle),
                    "drifted_features": ",".join(drifted_features)
                })
                mlflow.log_params({
                    "snapshot_size": len(df_snapshot),
                    "training_rows": len(X_train),
                    "validation_rows": len(X_val),
                    "eval_rows": len(X_eval)
                })

                # 3. Trigger Optuna HPO
                print(f"Running HPO search ({current_trials} trials)...")
                final_model, best_params = run_hpo(X_train, y_train, X_val, y_val, n_trials=current_trials)
                
                # Evaluate final model on the held-out eval set
                from sklearn.metrics import f1_score
                y_eval_pred = final_model.predict(X_eval)
                eval_f1 = f1_score(y_eval, y_eval_pred)
                mlflow.log_metric("retraining_eval_f1", eval_f1)
                print(f"[Retrainer] Final model evaluated on retraining holdout. F1: {eval_f1:.4f}")

                # 4. Log and register final model
                print("Registering final model in Registry...")
                from mlflow.models.signature import infer_signature
                signature = infer_signature(X_train, final_model.predict(X_train))
                
                model_info = mlflow.xgboost.log_model(
                    final_model,
                    artifact_path="model",
                    registered_model_name=model_name,
                    signature=signature
                )
                
                # Get the registered version number
                client = mlflow.tracking.MlflowClient()
                latest_version = max(
                    int(v.version)
                    for v in client.search_model_versions(f"name='{model_name}'")
                )
                print(f"Registered model version: {latest_version}")

            # 5. Call promoter.py (outside of the parent run block to prevent nested/concurrent run conflicts)
            print(f"Triggering Champion-Challenger evaluation for version {latest_version}...")
            decision = evaluate_and_promote(latest_version)
            print(f"Promotion decision completed: {decision}")
            print(f"Retraining Cycle {cycle} successfully completed!")
            print("==================================================\n")

    except KeyboardInterrupt:
        print("\nRetraining engine stopped by user.")
    finally:
        consumer.close()
        print("Retraining engine stopped.")

if __name__ == "__main__":
    listen_for_retrain_events()
