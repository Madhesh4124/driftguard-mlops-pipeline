import os
import optuna
import mlflow
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from utils.config_loader import load_config

# Disable optuna logs in stdout to keep things neat, unless needed
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_hpo(X_train, y_train, X_val, y_val, n_trials=30):
    # Derive scale_pos_weight
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }

        # Start nested MLflow run
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            model = XGBClassifier(early_stopping_rounds=10, **params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            val_f1 = f1_score(y_val, y_pred)
            val_acc = accuracy_score(y_val, y_pred)
            val_prec = precision_score(y_val, y_pred, zero_division=0)
            val_rec = recall_score(y_val, y_pred, zero_division=0)

            # Log metrics and params for the trial
            mlflow.log_params(params)
            mlflow.log_metrics({
                "f1": val_f1,
                "accuracy": val_acc,
                "precision": val_prec,
                "recall": val_rec
            })
            
            return val_f1

    # Setup study with MedianPruner
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials)

    # Save and log Optuna summary plots
    os.makedirs("reports", exist_ok=True)
    
    try:
        import optuna.visualization as vis
        
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html("reports/optimization_history.html")
        mlflow.log_artifact("reports/optimization_history.html")
        
        fig2 = vis.plot_param_importances(study)
        fig2.write_html("reports/param_importances.html")
        mlflow.log_artifact("reports/param_importances.html")
        
        fig3 = vis.plot_contour(study)
        fig3.write_html("reports/contour_plot.html")
        mlflow.log_artifact("reports/contour_plot.html")
    except Exception as e:
        print(f"Failed to log optuna visualization plots: {e}")

    # Log best trial details to parent run
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_f1", study.best_value)

    # Train final model on the best parameters
    best_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
        **study.best_params
    }
    
    # Concat train and validation to retrain final model
    X_full_train = pd.concat([X_train, X_val], ignore_index=True)
    y_full_train = pd.concat([y_train, y_val], ignore_index=True)
    
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_full_train, y_full_train)
    
    # Re-evaluate final model on validation set to get final model metrics
    y_pred_final = final_model.predict(X_val)
    final_f1 = f1_score(y_val, y_pred_final)
    final_acc = accuracy_score(y_val, y_pred_final)
    
    mlflow.log_metrics({
        "final_model_f1": final_f1,
        "final_model_accuracy": final_acc,
        "n_optuna_trials": n_trials,
        "training_rows": len(X_full_train),
    })
    print(f"[Optuna HPO] Final model trained on {len(X_full_train)} rows. Val F1: {final_f1:.4f}")
    
    return final_model, best_params

if __name__ == "__main__":
    # Dry run mock HPO search
    config = load_config()
    
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    df = pd.read_csv("data/credit_default_two_phase.csv")
    data = df.iloc[:15000].copy()
    data.drop(columns=['ID'], inplace=True)
    X = data.drop(columns=['default.payment.next.month'])
    y = data['default.payment.next.month']
    
    X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
        X, y, test_size=config["retraining"]["holdout_split"], random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=config["retraining"]["val_split"], random_state=42, stratify=y_train_val
    )
    
    print("Starting mock HPO search run (5 trials)...")
    with mlflow.start_run(run_name="mock-hpo-run") as run:
        model, best_params = run_hpo(X_train, y_train, X_val, y_val, n_trials=5)
        print(f"Finished mock HPO search. Best params: {best_params}")
