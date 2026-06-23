from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def train_model(
    X_train,
    X_val,
    y_train,
    y_val,
    params=None
):

    if params is None:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1
        }

    model = XGBClassifier(**params)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        
    }
    Confusion_Matrix=confusion_matrix(y_val, y_pred)
    # Convert confusion matrix to a labeled DataFrame
    confusion_df = pd.DataFrame(
    Confusion_Matrix,
    index=["Actual: No Default", "Actual: Default"],
    columns=["Predicted: No Default", "Predicted: Default"]
    )



    return model, metrics, confusion_df