"""
MLflow Run Schema & Validation Contract
---------------------------------------
Enforces the "Single Source of Truth" policy.
- All runs must have mandatory lineage tags.
- Metrics are restricted by RunType.
- Ad-hoc logging is prohibited.
"""

from enum import Enum
from typing import Dict, Any, Set, Optional

class RunType(Enum):
    RETRIEVAL = "retrieval"
    EVAL = "eval"
    GUARDRAIL = "guardrail"

# 1. Mandatory Metadata (Tags)
REQUIRED_TAGS: Set[str] = {
    "run_type",
    "dataset_hash",
    "index_hash",
    "prompt_version",
    "guardrail_version",
    "git_commit",
}

# 2. Standardized Metrics by Run Type
ALLOWED_METRICS: Dict[RunType, Set[str]] = {
    RunType.RETRIEVAL: {
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "num_chunks_retrieved",
        "retrieval_latency",
    },
    RunType.EVAL: {
        "answer_accuracy",
        "citation_precision",
        "citation_recall",
        "avg_confidence",
        "refusal_rate",
        "llm_latency",
        "refusal_accuracy", # <--- Added to support the new evaluation metric
    },
    RunType.GUARDRAIL: {
        "refusal_triggered",
        "confidence_score",
        "num_supported_sentences",
        "num_total_sentences",
        "alignment_score",
        "recall_score",
        "retrieval_latency",
        "llm_latency",
        "citation_precision", # <--- Added this allowed metric
    }
}

# 3. Context-Specific Params
ALLOWED_PARAMS: Dict[RunType, Set[str]] = {
    RunType.RETRIEVAL: {"query"},
    RunType.EVAL: {"query", "answer"},
    RunType.GUARDRAIL: {"query", "refusal_reason"},
}

class SchemaViolationError(Exception):
    """Raised when a run violates the strict logging contract."""
    pass

def validate_run_structure(tags: Dict[str, Any], metrics: Dict[str, float]) -> None:
    """
    Validates a run's metadata and metrics against the canonical schema.
    """
    # 1. Validate Required Tags
    missing_tags = REQUIRED_TAGS - tags.keys()
    if missing_tags:
        raise SchemaViolationError(f"Missing mandatory lineage tags: {missing_tags}")

    # 2. Validate Run Type
    try:
        run_type_str = tags["run_type"]
        run_type = RunType(run_type_str)
    except ValueError:
        raise SchemaViolationError(
            f"Invalid run_type: '{tags.get('run_type')}'. Must be one of {[t.value for t in RunType]}"
        )

    # 3. Validate Metrics (No Ad-Hoc Allowed)
    allowed_metrics = ALLOWED_METRICS[run_type]
    unknown_metrics = set(metrics.keys()) - allowed_metrics
    
    if unknown_metrics:
        raise SchemaViolationError(
            f"RunType '{run_type.value}' logged forbidden metrics: {unknown_metrics}. "
            f"Allowed: {allowed_metrics}"
        )