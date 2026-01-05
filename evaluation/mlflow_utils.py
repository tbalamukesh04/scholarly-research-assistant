from datetime import datetime, timezone

import mlflow

def start_run(name: str, dataset_hash: str):
    mlflow.set_experiment(name)
    mlflow.start_run(run_name=f"{name}_{datetime.now(timezone.utc)}")
    mlflow.log_param("dataset_hash", dataset_hash)


def log_metrics(metrics:dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))


def log_artifact(path: str):
    mlflow.log_artifact(path)


def end_run():
    mlflow.end_run()
