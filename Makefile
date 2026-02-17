# ==============================================================================
# Scholarly Research Assistant - Automation & Integrity
# ==============================================================================

.PHONY: all help repro evaluate register verify audit clean

all: help

help:
@echo 'Available commands:'
@echo '  make repro		- Rebuild the DVC pipeline'
@echo '  make evaluate	- Run the evaluation harness'
@echo '  make register	- Register the model'
@echo '  make verify	  - Verify reproducibility'
@echo '  make audit		- Audit MLflow runs'
@echo '  make clean		- PURGE all artifacts'

repro:
dvc repro

evaluate:
python -m evaluation.evaluate

register:
python -m pipelines.registry.register

verify:
bash scripts/verify_repro.sh

audit:
python scripts/audit_mlflow_runs.py

clean:
@echo 'WARNING: Deleting all generated data...'
rm -rf data/raw data/chunks data/indexes
rm -rf evaluation/results evaluation/metrics.json
rm -rf mlruns/ mlflow.db
@echo 'System purged.'
