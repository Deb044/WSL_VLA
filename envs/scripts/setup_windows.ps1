# ==============================================================================
# `legacy-smoke` profile on Windows (PowerShell): PyTorch smoke-test environment
# ==============================================================================
# Not a research environment. The Octo-Small research profiles are created
# under Linux/WSL2 with envs/scripts/setup_research.sh.
$ErrorActionPreference = "Stop"
$ENV_NAME = if ($env:ENV_NAME) { $env:ENV_NAME } else { "vla_legacy_smoke" }

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Step 1: Checking Conda Environment ($ENV_NAME)" -ForegroundColor Cyan
Write-Host "======================================================================"

# Check if environment already exists
$envList = conda env list
if ($envList -match "\b$ENV_NAME\b") {
    Write-Host "  -> Conda environment '$ENV_NAME' already exists." -ForegroundColor Green
} else {
    Write-Host "  -> Creating Conda environment '$ENV_NAME' with Python 3.10..." -ForegroundColor Yellow
    conda create -n $ENV_NAME python=3.10 -y
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Step 2: Installing PyTorch & Dependencies" -ForegroundColor Cyan
Write-Host "======================================================================"

# Run pip within the target environment
conda run -n $ENV_NAME python -m pip install --upgrade pip setuptools wheel

Write-Host "  -> Installing PyTorch..." -ForegroundColor Yellow
conda run -n $ENV_NAME python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host "  -> Installing requirements from envs/requirements/legacy-smoke.txt..." -ForegroundColor Yellow
conda run -n $ENV_NAME python -m pip install -r "$PSScriptRoot/../requirements/legacy-smoke.txt"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Step 3: Running Pre-Flight Dry Run Verification" -ForegroundColor Cyan
Write-Host "======================================================================"
conda run -n $ENV_NAME python tests/unit/test_smoke/test_local_dryrun.py

Write-Host "======================================================================" -ForegroundColor Green
Write-Host " Setup Completed Successfully!" -ForegroundColor Green
Write-Host " To activate the environment, run:" -ForegroundColor Green
Write-Host "   conda activate $ENV_NAME" -ForegroundColor Yellow
Write-Host "======================================================================"
