import mlflow
from mlflow.tracking import MlflowClient

def get_production_model(model_name):
    """
    Returns the latest model version object with the 'champion' alias, or None.
    """
    client = MlflowClient()
    try:
        return client.get_model_version_by_alias(model_name, "champion")
    except Exception:
        return None

def transition_stage(model_name, version, stage, archive_existing=False):
    """
    Transition a model version using aliases: 'champion' for Production,
    and 'challenger_archived' for Archived.
    """
    client = MlflowClient()
    if stage == "Production":
        # Setting the alias 'champion' automatically removes it from any other version
        return client.set_registered_model_alias(model_name, "champion", str(version))
    elif stage == "Archived":
        try:
            client.delete_registered_model_alias(model_name, "champion")
        except Exception:
            pass
        return client.set_registered_model_alias(model_name, "challenger_archived", str(version))

def get_model_version_metric(run_id, metric_key):
    """
    Get a metric value from a specific run ID.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics.get(metric_key, 0.0)

def log_promotion_decision(run_id, decision, metric_gap):
    """
    Set tag and log promotion decision metrics to the run.
    """
    client = MlflowClient()
    client.set_tag(run_id, "promotion_decision", decision)
    client.log_metric(run_id, "metric_gap", metric_gap)
