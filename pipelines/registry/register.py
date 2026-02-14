import sys
import yaml
import json
import mlflow
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

from utils.metadata import PROMPT_VERSION, GUARDRAIL_VERSION, get_index_hash, get_git_commit
from utils.mlflow_handler import MLflowHandler

MODEL_CARD_PATH = Path("registry/model_card.yaml")
DATASET_MANIFEST_PATH = Path("data/versions/dataset_manifest.json")

def load_model_card() -> Dict[str, Any]:
    if not MODEL_CARD_PATH.exists():
        raise FileNotFoundError(f"Model Card not found at {MODEL_CARD_PATH}")
    with open(MODEL_CARD_PATH, "r") as f:
        return yaml.safe_load(f)

def get_current_dataset_hash() -> str:
    if not DATASET_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Dataset manifest not found at {DATASET_MANIFEST_PATH}")
    with open(DATASET_MANIFEST_PATH, "r") as f:
        data = json.load(f)
    return data.get("dataset_hash")

def resolve_current_identity(card_config: Dict) -> Dict[str, str]:
    """Resolves the dynamic identity of the CURRENT codebase/environment."""
    print("🔍 Resolving System Identity...")
    identity = {
        "base_llm": card_config['identity']['base_llm']['value'],
        "prompt_version": PROMPT_VERSION,
        "guardrail_version": GUARDRAIL_VERSION,
        "dataset_hash": get_current_dataset_hash(),
        "index_hash": get_index_hash()
    }
    print(json.dumps(identity, indent=2))
    return identity

def find_qualifying_run(identity: Dict[str, str]) -> mlflow.entities.Run:
    """Searches MLflow for a run that matches the current system identity."""
    print("\n🔍 Searching for Matching MLflow Run...")
    
    # Construct filter: tags.X = 'Y' AND status = 'FINISHED'
    filter_conditions = [f"tags.{k} = '{v}'" for k, v in identity.items()]
    filter_conditions.append("status = 'FINISHED'")
    filter_string = " AND ".join(filter_conditions)

    runs = mlflow.search_runs(
        experiment_names=["Scholarly-RAG-Evaluation"], 
        filter_string=filter_string,
        order_by=["attribute.start_time DESC"],
        max_results=1
    )
    
    if runs.empty:
        raise ValueError(
            "❌ No matching MLflow run found for this configuration.\n"
            "   Run 'evaluation/run_mlf_eval.py' first to generate proof."
        )
    
    run_id = runs.iloc[0].run_id
    print(f"✅ Found Run ID: {run_id}")
    return mlflow.get_run(run_id)

def validate_gates(run: mlflow.entities.Run, card_config: Dict):
    """Enforces the Quality Contract. FAILS FAST if thresholds are not met."""
    print("\n🛡️ Verifying Quality Gates...")
    
    metrics = run.data.metrics
    requirements = card_config['evaluation_requirements']['metrics']
    
    failures = []
    
    for req in requirements:
        name = req['name']
        threshold = req['threshold']
        actual = metrics.get(name, 0.0) 
        
        if actual < threshold:
            failures.append(f"   ❌ {name}: {actual:.4f} < {threshold}")
        else:
            print(f"   ✅ {name}: {actual:.4f} >= {threshold}")
            
    if failures:
        print("\n⛔ PROMOTION DENIED. The following gates failed:")
        for f in failures:
            print(f)
        sys.exit(1) 

def upload_contract_to_run(run_id: str, card_path: Path):
    """
    Uploads the Model Card to the run artifacts so it becomes part of the registered model.
    """
    print(f"\n📎 Attaching Contract ({card_path}) to Run {run_id}...")
    client = mlflow.MlflowClient()
    client.log_artifact(run_id, str(card_path), artifact_path="registry_context")

def register_and_promote(run_id: str, identity: Dict, card_config: Dict):
    """Registers the model, attaches artifacts, and promotes to Production."""
    print("\n📝 Registering & Promoting Model...")
    
    model_name = card_config['identity']['model_name']
    client = mlflow.MlflowClient()

    # 1. Attach the Contract to the Run
    upload_contract_to_run(run_id, MODEL_CARD_PATH)

    # 2. Ensure Registered Model Exists (The Fix)
    try:
        client.create_registered_model(model_name)
        print(f"   Created new registered model container: '{model_name}'")
    except Exception:
        # It already exists, which is fine
        pass

    # 3. Create Model Version
    mv = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/artifacts",
        run_id=run_id
    )

    # 4. Tag with Identity (Traceability)
    for k, v in identity.items():
        client.set_model_version_tag(
            name=model_name,
            version=mv.version,
            key=k,
            value=v
        )
    
    # 5. Tag with Git Commit
    client.set_model_version_tag(
        name=model_name,
        version=mv.version,
        key="git_commit",
        value=get_git_commit()
    )

    # 6. PROMOTE TO PRODUCTION
    print(f"🚀 Promoting Version {mv.version} to PRODUCTION...")
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True 
    )

    print(f"\n🎉 SUCCESS: Model '{model_name}' Version {mv.version} is now LIVE in Production.")
    print(f"   Identity: {identity}")
    print(f"   Proof: Run {run_id}")
    print(f"   Contract: registry_context/model_card.yaml")

if __name__ == "__main__":
    try:
        # 1. Load Contract
        card = load_model_card()
        
        # 2. Get System Identity
        identity = resolve_current_identity(card)
        
        # 3. Find Evidence
        run = find_qualifying_run(identity)
        
        # 4. Verify Quality (The Gate)
        validate_gates(run, card)
        
        # 5. Execute Promotion
        register_and_promote(run.info.run_id, identity, card)
        
    except Exception as e:
        print(f"\n⛔ SYSTEM ERROR: {e}")
        sys.exit(1)