# ==============================================================================
# WSL_VLA: Windows PowerShell Setup Script
# ==============================================================================
$ErrorActionPreference = "Stop"
$ENV_NAME = "vla_zoo"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Step 1: Checking Conda Environment ($ENV_NAME)" -ForegroundColor Cyan
Write-Host "======================================================================"

$envList = conda env list
if ($envList -match "\b$ENV_NAME\b") {
    Write-Host "  -> Conda environment '$ENV_NAME' already exists." -ForegroundColor Green
} else {
    Write-Host "  -> Creating Conda environment '$ENV_NAME' with Python 3.10..." -ForegroundColor Yellow
    conda create -n $ENV_NAME python=3.10 -y
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Step 2: Installing JAX & Dependencies" -ForegroundColor Cyan
Write-Host "======================================================================"

conda run -n $ENV_NAME python -m pip install --upgrade pip setuptools wheel
conda run -n $ENV_NAME python -m pip install jax jaxlib
conda run -n $ENV_NAME python -m pip install -r requirements.txt
conda run -n $ENV_NAME python -m pip install -e .

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Step 3: Running Test Suite Verification" -ForegroundColor Cyan
Write-Host "======================================================================"
conda run -n $ENV_NAME pytest tests/

Write-Host "======================================================================" -ForegroundColor Green
Write-Host " Setup Completed Successfully!" -ForegroundColor Green
Write-Host " To activate the environment, run:" -ForegroundColor Green
Write-Host "   conda activate $ENV_NAME" -ForegroundColor Yellow
Write-Host "======================================================================"
