import pandas as pd
from sklearn.model_selection import train_test_split
from model.trainer import train_model
import mlflow
from mlflow.tracking import MlflowClient

# Set tracking URI to local SQLite database and Experiment
EXPERIMENT_NAME = "Drift-Guard-MLOPS-Pipeline"
MODEL_NAME = "credit-default-classifier"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

df = pd.read_csv("data/credit_default_two_phase.csv")
data = df.iloc[:15000].copy()
data.drop(columns=['ID'], inplace=True)
X = data.drop(columns=['default.payment.next.month'])
y = data['default.payment.next.month']

# splitting data into train,validation,holdout (stratify makes sure there is class balance because dataset is imbalanced)
# Train      ≈ 70%
# Validation ≈ 15%
# Holdout    ≈ 15%
X_temp, X_holdout, y_temp, y_holdout = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.1765,
    random_state=42,
    stratify=y_temp
)

# Save holdout split to disk as reference set
holdout_df = X_holdout.copy()
holdout_df['default.payment.next.month'] = y_holdout
holdout_df.to_parquet("data/holdout.parquet", index=False)

# Compute Class Imbalance
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
)


params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "n_jobs": -1
}

with mlflow.start_run(run_name="baseline-training"):
    model, metrics, Confusion_Matrix = train_model(
        X_train,
        X_val,
        y_train,
        y_val,
        params
    )

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    
    # Log and register model in registry
    model_info = mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )

    # Get latest registered version
    client = MlflowClient()

    latest_version = max(
        int(v.version)
        for v in client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )
    )

# Promote latest version to Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )

print(metrics)
print(Confusion_Matrix)

