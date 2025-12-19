Write-Host "Starting Day 1 setup..."

# 1. Create virtual environment if missing
if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

# 2. Activate virtual environment
Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
Write-Host "Installing dependencies..."
pip install -r requirements.txt

# 5. Config sanity check
Write-Host "Running config check..."
python -c "import yaml; from pathlib import Path; 
cfgs=['configs/project.yaml','configs/ingestion.yaml']; 
[ (Path(c).exists() or (_ for _ in ()).throw(FileNotFoundError(c))) for c in cfgs ];
[ yaml.safe_load(Path(c).read_text()) for c in cfgs ];
print('Config check passed.')"

# 6. Logging sanity check
Write-Host "Running logging check..."
python -c "import yaml, logging; 
from utils.logging import setup_logger, log_event;
cfg=yaml.safe_load(open('configs/project.yaml'));
log_dir=cfg['paths']['log_root'];
logger=setup_logger('day1_setup', log_dir);
log_event(logger, logging.INFO, 'Day 1 setup logging check');
print('Logging check passed.')"

Write-Host "Day 1 setup complete."
