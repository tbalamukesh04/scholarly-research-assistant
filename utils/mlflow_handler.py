import mlflow
from typing import Dict, Any, Optional
from contextlib import contextmanager
from utils.mlflow_schema import validate_run_structure, RunType

class MLflowHandler:
    """
    Central entry point for all MLflow interactions.
    Enforces schema validation before logging.
    """

    @staticmethod
    @contextmanager
    def start_run(
        run_name: str, 
        run_type: RunType, 
        tags: Dict[str, Any], 
        nested: bool = False
    ):
        """
        Starts an MLflow run with strict schema validation.
        
        Args:
            run_name: Name of the run.
            run_type: Type of run (RunType.RETRIEVAL, EVAL, GUARDRAIL).
            tags: Dictionary of lineage tags. Must include all REQUIRED_TAGS.
            nested: Whether to nest this run under an active run.
        """
        # 1. Prepare Tags
        all_tags = tags.copy()
        all_tags["run_type"] = run_type.value
        
        # 2. Pre-Validation (Tags)
        # We can't validate metrics yet, but we MUST validate lineage tags.
        # Passing empty metrics dict to validate_run_structure just to check tags.
        validate_run_structure(all_tags, {}) 

        # 3. Start Run
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            mlflow.set_tags(all_tags)
            try:
                yield run
            finally:
                pass # mlflow.start_run handles end_run automatically

    @staticmethod
    def log_metrics(run_type: RunType, metrics: Dict[str, float]):
        """
        Logs metrics after validating they are allowed for the RunType.
        """
        # We re-validate the metrics against the run_type
        # We can't easily validate tags here without refetching them, 
        # so we assume tags were validated at start_run.
        
        # Construct a dummy tag dict with just run_type to pass to validator
        dummy_tags = {
            "run_type": run_type.value,
            "dataset_hash": "skip", # partial mock to bypass tag check
            "index_hash": "skip",
            "prompt_version": "skip",
            "guardrail_version": "skip",
            "git_commit": "skip"
        }
        
        # This call checks if 'metrics' contains any forbidden keys for 'run_type'
        validate_run_structure(dummy_tags, metrics)
        
        mlflow.log_metrics(metrics)

    @staticmethod
    def log_artifact(local_path: str):
        mlflow.log_artifact(local_path)