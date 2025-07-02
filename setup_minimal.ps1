# PowerShell script to set up minimal CleoAI API
Write-Host "Setting up CleoAI Minimal API..." -ForegroundColor Green

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Virtual environment not activated. Activating..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
}

# Install minimal requirements
Write-Host "`nInstalling minimal requirements..." -ForegroundColor Green
pip install -r requirements-minimal.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nInstallation successful!" -ForegroundColor Green
    Write-Host "`nStarting minimal API server..." -ForegroundColor Green
    
    # Start the API server
    python main_api_minimal.py --debug
} else {
    Write-Host "`nInstallation failed. Please check the error messages above." -ForegroundColor Red
}