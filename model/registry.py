import mlflow
from mlflow.tracking import MlflowClient

def get_production_model(model_name):
    """
    Returns the latest model version object in the 'Production' stage, or None.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if v.current_stage == "Production":
            return v
    return None

def transition_stage(model_name, version, stage, archive_existing=False):
    """
    Transition a model version's stage.
    """
    client = MlflowClient()
    return client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing
    )

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
