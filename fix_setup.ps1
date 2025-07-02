# Fix setup issues and install minimal requirements

Write-Host "Fixing setup issues..." -ForegroundColor Green

# Upgrade pip and setuptools first
Write-Host "`nUpgrading pip and setuptools..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install packages one by one to avoid build issues
Write-Host "`nInstalling core packages..." -ForegroundColor Green

# Core packages that should install without issues
$core_packages = @(
    "fastapi==0.109.0",
    "uvicorn[standard]==0.25.0",
    "python-dotenv==1.0.0",
    "pydantic==2.5.0",
    "psutil==5.9.6",
    "ariadne==0.21.0",
    "graphql-core==3.2.3",
    "python-multipart==0.0.6",
    "loguru==0.7.0"
)

foreach ($package in $core_packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    pip install $package
}

# Try to install optional packages (may fail on some systems)
Write-Host "`nInstalling optional packages (errors are OK)..." -ForegroundColor Yellow
pip install redis==5.0.1 2>$null
pip install pymongo==4.6.1 2>$null

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "`nStarting minimal API server..." -ForegroundColor Green

# Start the API
python main_api_minimal.py --debug