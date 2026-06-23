import mlflow
import pandas as pd
from sklearn.metrics import f1_score
from mlflow.tracking import MlflowClient
from model.registry import (
    get_production_model,
    transition_stage,
    log_promotion_decision
)
from utils.config_loader import load_config

def evaluate_and_promote(challenger_version, config_path=None):
    """
    Evaluates challenger model against current Production champion on holdout dataset.
    Promotes challenger to Production if it beats the champion by threshold.
    """
    config = load_config(config_path)
    model_name = config["mlflow"]["model_name"]
    promotion_threshold = config["retraining"]["promotion_threshold"]
    holdout_path = "data/holdout.parquet"

    # 1. Load holdout dataset
    df_holdout = pd.read_parquet(holdout_path)
    target_col = "default.payment.next.month"
    
    # Clean ID column if it exists
    if "ID" in df_holdout.columns:
        df_holdout = df_holdout.drop(columns=["ID"])
        
    X_holdout = df_holdout.drop(columns=[target_col])
    y_holdout = df_holdout[target_col]

    # 2. Load challenger model
    challenger_uri = f"models:/{model_name}/{challenger_version}"
    challenger_model = mlflow.pyfunc.load_model(challenger_uri)
    challenger_pred = challenger_model.predict(X_holdout)
    challenger_f1 = f1_score(y_holdout, challenger_pred)
    
    client = MlflowClient()
    version_details = client.get_model_version(model_name, str(challenger_version))
    challenger_run_id = version_details.run_id

    # 3. Get current active Production model
    champion_version_obj = get_production_model(model_name)
    
    if champion_version_obj is not None:
        champion_version = champion_version_obj.version
        champion_uri = f"models:/{model_name}/Production"
        champion_model = mlflow.pyfunc.load_model(champion_uri)
        champion_pred = champion_model.predict(X_holdout)
        champion_f1 = f1_score(y_holdout, champion_pred)
        
        metric_gap = challenger_f1 - champion_f1
        print(f"Holdout Evaluation:")
        print(f"Champion (v{champion_version}) F1: {champion_f1:.4f}")
        print(f"Challenger (v{challenger_version}) F1: {challenger_f1:.4f}")
        print(f"Margin: {metric_gap:.4f} (Required: {promotion_threshold:.4f})")

        if challenger_f1 > champion_f1 + promotion_threshold:
            print("Challenger beats Champion by threshold. Promoting Challenger!")
            # Promote challenger to Production and automatically archive old champion
            transition_stage(
                model_name=model_name,
                version=challenger_version,
                stage="Production",
                archive_existing=True
            )
            decision = "challenger_promoted"
        else:
            print("Challenger fails to beat Champion by threshold. Archiving Challenger.")
            transition_stage(
                model_name=model_name,
                version=challenger_version,
                stage="Archived"
            )
            decision = "champion_retained"
    else:
        print("No existing Production champion found. Automatically promoting Challenger!")
        transition_stage(
            model_name=model_name,
            version=challenger_version,
            stage="Production"
        )
        metric_gap = 0.0
        decision = "challenger_promoted"

    # Log metrics & tag to the challenger run
    log_promotion_decision(challenger_run_id, decision, metric_gap)
    print(f"Evaluation finished. Decision logged: {decision}")
    return decision

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python model/promoter.py <challenger_version_number>")
        sys.exit(1)
        
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    evaluate_and_promote(int(sys.argv[1]))
