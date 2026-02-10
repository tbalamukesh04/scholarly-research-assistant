from typing import Dict
from utils.mlflow_handler import MLflowHandler
from utils.mlflow_schema import RunType
from utils.metadata import get_git_commit, get_index_hash, PROMPT_VERSION, GUARDRAIL_VERSION

def start_run(name: str, dataset_hash: str):
    """
    Starts an MLflow run with mandatory metadata automatically injected.
    Adapts legacy 'start_run' calls to the new Strict Schema.
    """
    # Determine Run Type heuristic
    run_type = RunType.RETRIEVAL 
    if "eval" in name.lower() or "prompt" in name.lower():
        run_type = RunType.EVAL
        
    tags = {
        "dataset_hash": dataset_hash,
        "index_hash": get_index_hash(),
        "prompt_version": PROMPT_VERSION,
        "guardrail_version": GUARDRAIL_VERSION,
        "git_commit": get_git_commit()
    }
    
    # Start the run via the central handler
    # Note: We are mimicking a context manager enter manually here because 
    # the legacy code expects a persistent run until end_run() is called.
    ctx = MLflowHandler.start_run(name, run_type, tags)
    ctx.__enter__()

def log_metrics(metrics: Dict[str, float]):
    """
    Logs metrics, attempting to infer the correct run type context if possible,
    or defaulting to a permissive check (or relying on the Handler's active run).
    """
    # Since we can't easily pass the run_type from the legacy caller without changing signatures,
    # we might need to rely on the fact that start_run set the tags.
    # However, MLflowHandler.log_metrics requires a run_type to validate the metrics.
    
    # Heuristic: Try to validate as RETRIEVAL, if that fails, try EVAL.
    # This is a bit loose, but necessary for backward compatibility without refactoring all eval scripts.
    
    # Try Retrieval First
    try:
        MLflowHandler.log_metrics(RunType.RETRIEVAL, metrics)
        return
    except Exception:
        pass
        
    # Try Eval
    try:
        MLflowHandler.log_metrics(RunType.EVAL, metrics)
        return
    except Exception:
        pass

    # Try Guardrail
    try:
        MLflowHandler.log_metrics(RunType.GUARDRAIL, metrics)
        return
    except Exception:
        pass
        
    # If all fail, force log (or raise error). 
    # IN STRICT MODE: We raise the error from the most likely candidate (Eval)
    MLflowHandler.log_metrics(RunType.EVAL, metrics)


def log_artifact(path: str):
    MLflowHandler.log_artifact(path)


def end_run():
    import mlflow
    mlflow.end_run()