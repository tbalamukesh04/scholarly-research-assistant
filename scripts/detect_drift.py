import sys
import json
import mlflow
from pathlib import Path
from typing import Dict

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from pipelines.registry.register import load_model_card, resolve_current_identity

def check_drift():
    """
    Compares the local system identity against the MLflow Model Registry.
    """
    print("🔍 CALCULATING LOCAL SYSTEM IDENTITY...")
    
    # 1. Get Local Identity
    try:
        card = load_model_card()
        local_identity = resolve_current_identity(card)
        model_name = card['identity']['model_name']
    except Exception as e:
        print(f"⛔ FATAL: Could not resolve local identity. {e}")
        sys.exit(1)

    print(f"\n📋 Local Configuration:")
    print(json.dumps(local_identity, indent=2))

    print(f"\n🔍 CHECKING REGISTRY FOR '{model_name}'...")
    
    client = mlflow.MlflowClient()
    
    # 2. Get all versions of the registered model
    try:
        registered_versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        print(f"⚠️ Model '{model_name}' does not exist in registry.")
        print("❌ STATUS: UNREGISTERED")
        sys.exit(1)

    # 3. Compare Tags
    match_found = False
    matched_version = None
    
    for mv in registered_versions:
        # We only care about the Identity tags
        tags = mv.tags
        
        # Check if all local identity keys exist and match in the registry tags
        is_match = True
        for key, local_val in local_identity.items():
            reg_val = tags.get(key)
            if reg_val != local_val:
                is_match = False
                break
        
        if is_match:
            match_found = True
            matched_version = mv
            break

    # 4. Report Status
    print("-" * 30)
    if match_found:
        print(f"✅ STATUS: VALID (SYNCED)")
        print(f"   Matches Model Version: {matched_version.version}")
        print(f"   Stage: {matched_version.current_stage}")
        
        if matched_version.current_stage != "Production":
            print("   ⚠️ WARNING: Local code matches a registered version, but it is NOT Production.")
    else:
        print(f"❌ STATUS: DRIFT DETECTED (INVALID)")
        print("   The current configuration (Code/Data/Config) is NOT registered.")
        print("   Action Required: Run 'python pipelines/registry/register.py' to legitimize this state.")
        sys.exit(1)
    print("-" * 30)

if __name__ == "__main__":
    check_drift()