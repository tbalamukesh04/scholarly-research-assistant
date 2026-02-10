import mlflow
import pandas as pd
from typing import Dict, Any, List
from utils.mlflow_schema import REQUIRED_TAGS, ALLOWED_METRICS, RunType

def check_run_validity(run: pd.Series) -> Dict[str, Any]:
    """
    Validates a single run against the Strict Schema.
    Returns a dict with status and violation details.
    """
    status = {"valid": True, "reasons": [], "type": "Unknown"}
    
    # 1. Check Tags (Lineage)
    run_tags = run.get("tags", {})
    # mlflow.search_runs returns tags as keys prefixed with 'tags.' in pandas usually, 
    # but let's assume we handle the dictionary if iterating or row access.
    # Actually, mlflow.search_runs() returns a big DF. 
    
    # Let's verify presence of required keys in the columns
    missing_tags = []
    for tag in REQUIRED_TAGS:
        # In the DF, tags are 'tags.tag_name'
        if f"tags.{tag}" not in run or pd.isna(run[f"tags.{tag}"]):
            missing_tags.append(tag)
            
    if missing_tags:
        status["valid"] = False
        status["reasons"].append(f"Missing Tags: {missing_tags}")
        status["type"] = "Legacy/Incomplete"
        return status

    # 2. Determine Run Type
    try:
        run_type_str = run[f"tags.run_type"]
        run_type = RunType(run_type_str)
        status["type"] = run_type.value
    except ValueError:
        status["valid"] = False
        status["reasons"].append(f"Invalid RunType: {run.get(f'tags.run_type')}")
        return status

    # 3. Check Metrics (Ad-Hoc Restriction)
    # This is harder on a flat DF because columns are sparse.
    # We check which metric columns have values for this row.
    allowed = ALLOWED_METRICS[run_type]
    
    # Identify metric columns (usually prefixed with 'metrics.')
    metric_cols = [c for c in run.index if c.startswith("metrics.") and not pd.isna(run[c])]
    cleaned_metrics = {c.replace("metrics.", "") for c in metric_cols}
    
    forbidden = cleaned_metrics - allowed
    if forbidden:
        status["valid"] = False
        status["reasons"].append(f"Forbidden Metrics: {forbidden}")

    # 4. Artifact Check (Heuristic)
    # We can't easily ls the artifact path without API calls per run, 
    # but we can check if the artifact_uri exists.
    if not run.get("artifact_uri"):
        status["valid"] = False
        status["reasons"].append("No Artifact URI")

    return status

def audit_history(experiment_names: List[str] = None):
    print(f"{'Run ID':<34} | {'Type':<10} | {'Status':<10} | {'Details'}")
    print("-" * 100)
    
    if not experiment_names:
        experiments = mlflow.search_experiments()
        experiment_ids = [e.experiment_id for e in experiments]
    else:
        experiment_ids = [] # logic to resolve names to IDs would go here
        
    # Search all runs
    try:
        runs = mlflow.search_runs(experiment_ids=experiment_ids)
    except Exception:
        print("No runs found.")
        return

    if runs.empty:
        print("No runs recorded.")
        return

    stats = {"Valid": 0, "Legacy": 0}

    for _, run in runs.iterrows():
        result = check_run_validity(run)
        
        status_str = "VALID" if result["valid"] else "INVALID"
        if result["valid"]:
            stats["Valid"] += 1
            color_start = "\033[92m" # Green
        else:
            stats["Legacy"] += 1
            color_start = "\033[91m" # Red
        color_end = "\033[0m"
        
        # Truncate details
        details = "; ".join(result["reasons"])
        if len(details) > 40:
            details = details[:37] + "..."
            
        print(f"{run['run_id']:<34} | {result['type']:<10} | {color_start}{status_str:<10}{color_end} | {details}")

    print("-" * 100)
    print(f"AUDIT COMPLETE: {stats['Valid']} Canonical Runs | {stats['Legacy']} Legacy/Invalid Runs")
    print("Optimization: Only 'VALID' runs are structurally comparable.")

if __name__ == "__main__":
    # Ensure we are pointing to the right tracking URI if needed
    # mlflow.set_tracking_uri("...") 
    audit_history()