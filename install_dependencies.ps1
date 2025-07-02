# Gradual dependency installation script for CleoAI
# This script installs dependencies in stages to avoid conflicts

param(
    [Parameter(Position=0)]
    [ValidateSet("minimal", "core", "ml", "full", "check")]
    [string]$Stage = "check"
)

Write-Host "`nCleoAI Dependency Installer" -ForegroundColor Cyan
Write-Host "===========================" -ForegroundColor Cyan

# Function to check if a package is installed
function Test-PackageInstalled {
    param([string]$PackageName)
    $installed = pip list 2>$null | Select-String -Pattern "^$PackageName\s+"
    return $null -ne $installed
}

# Function to install packages with error handling
function Install-Package {
    param(
        [string]$Package,
        [string]$Description = ""
    )
    
    Write-Host "`nInstalling $Package..." -ForegroundColor Yellow
    if ($Description) {
        Write-Host "  $Description" -ForegroundColor Gray
    }
    
    try {
        pip install $Package
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Success" -ForegroundColor Green
            return $true
        }
    } catch {
        Write-Host "  ✗ Failed" -ForegroundColor Red
        return $false
    }
    Write-Host "  ✗ Failed" -ForegroundColor Red
    return $false
}

# Check current installation status
if ($Stage -eq "check") {
    Write-Host "`nChecking installed packages..." -ForegroundColor Cyan
    
    $packages = @{
        "Core" = @("fastapi", "uvicorn", "python-dotenv", "pydantic")
        "API" = @("ariadne", "graphql-core", "python-multipart")
        "Storage" = @("redis", "pymongo", "supabase", "pinecone-client")
        "ML" = @("torch", "transformers", "sentence-transformers")
        "Vector" = @("chromadb", "faiss-cpu")
    }
    
    foreach ($category in $packages.Keys) {
        Write-Host "`n$category packages:" -ForegroundColor Yellow
        foreach ($pkg in $packages[$category]) {
            if (Test-PackageInstalled $pkg) {
                Write-Host "  ✓ $pkg" -ForegroundColor Green
            } else {
                Write-Host "  ✗ $pkg" -ForegroundColor Red
            }
        }
    }
    
    Write-Host "`nUse one of these commands to install:" -ForegroundColor Cyan
    Write-Host "  .\install_dependencies.ps1 minimal   # Just API framework" -ForegroundColor White
    Write-Host "  .\install_dependencies.ps1 core      # API + basic storage" -ForegroundColor White
    Write-Host "  .\install_dependencies.ps1 ml        # Add ML capabilities" -ForegroundColor White
    Write-Host "  .\install_dependencies.ps1 full      # Everything" -ForegroundColor White
    exit 0
}

# Ensure pip is up to date
Write-Host "`nUpgrading pip and core tools..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install based on stage
switch ($Stage) {
    "minimal" {
        Write-Host "`nInstalling minimal dependencies (API only)..." -ForegroundColor Cyan
        
        $packages = @(
            @("fastapi==0.109.0", "Web framework"),
            @("uvicorn[standard]==0.25.0", "ASGI server"),
            @("python-dotenv==1.0.0", "Environment variables"),
            @("pydantic==2.5.0", "Data validation"),
            @("psutil==5.9.6", "System monitoring"),
            @("loguru==0.7.0", "Logging")
        )
        
        foreach ($pkg in $packages) {
            Install-Package -Package $pkg[0] -Description $pkg[1]
        }
    }
    
    "core" {
        Write-Host "`nInstalling core dependencies (API + Storage)..." -ForegroundColor Cyan
        
        # First install minimal
        & $PSCommandPath minimal
        
        # Then add storage and API packages
        $packages = @(
            @("ariadne==0.21.0", "GraphQL library"),
            @("graphql-core==3.2.3", "GraphQL core"),
            @("python-multipart==0.0.6", "File uploads"),
            @("redis==5.0.1", "Redis client"),
            @("sqlite3-api==0.1.0", "SQLite wrapper"),
            @("pandas==2.1.0", "Data processing"),
            @("tqdm==4.66.1", "Progress bars")
        )
        
        foreach ($pkg in $packages) {
            Install-Package -Package $pkg[0] -Description $pkg[1]
        }
        
        # Try MongoDB (may need Visual C++)
        Write-Host "`nAttempting MongoDB installation..." -ForegroundColor Yellow
        Write-Host "  Note: This may fail if Visual C++ is not installed" -ForegroundColor Gray
        Install-Package -Package "pymongo==4.6.1" -Description "MongoDB client"
    }
    
    "ml" {
        Write-Host "`nInstalling ML dependencies..." -ForegroundColor Cyan
        Write-Host "This will take several minutes and requires ~5GB disk space" -ForegroundColor Yellow
        
        # First ensure core is installed
        if (-not (Test-PackageInstalled "fastapi")) {
            & $PSCommandPath core
        }
        
        # Install PyTorch CPU version (smaller and faster to install)
        Write-Host "`nInstalling PyTorch (CPU version)..." -ForegroundColor Yellow
        $torchInstalled = Install-Package -Package "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu" -Description "PyTorch CPU version"
        
        if ($torchInstalled) {
            # Install ML packages
            $mlPackages = @(
                @("transformers==4.38.0", "Hugging Face Transformers"),
                @("sentence-transformers==2.2.2", "Sentence embeddings"),
                @("accelerate==0.23.0", "Training acceleration"),
                @("datasets==2.15.0", "Dataset loading"),
                @("einops==0.7.0", "Tensor operations"),
                @("safetensors==0.4.0", "Safe model storage")
            )
            
            foreach ($pkg in $mlPackages) {
                Install-Package -Package $pkg[0] -Description $pkg[1]
            }
            
            # Try vector databases
            Write-Host "`nInstalling vector databases..." -ForegroundColor Yellow
            Install-Package -Package "chromadb==0.4.18" -Description "Local vector database"
            Install-Package -Package "faiss-cpu==1.7.4" -Description "Facebook AI Similarity Search"
        }
    }
    
    "full" {
        Write-Host "`nInstalling all dependencies..." -ForegroundColor Cyan
        
        # Install ML first (includes core)
        & $PSCommandPath ml
        
        # Add remaining packages
        $additionalPackages = @(
            @("pinecone-client==3.0.0", "Pinecone vector database"),
            @("supabase==2.3.0", "Supabase client"),
            @("deepspeed==0.12.3", "Distributed training"),
            @("wandb==0.16.0", "Experiment tracking"),
            @("mlflow==2.9.0", "ML lifecycle"),
            @("tensorboard==2.15.0", "Training visualization"),
            @("evaluate==0.4.1", "Model evaluation"),
            @("pre-commit==3.5.0", "Git hooks"),
            @("black==23.12.0", "Code formatter"),
            @("flake8==6.1.0", "Linter"),
            @("mypy==1.7.1", "Type checker"),
            @("pytest==7.4.3", "Testing framework"),
            @("pytest-cov==4.1.0", "Coverage reporting"),
            @("pytest-asyncio==0.21.1", "Async testing")
        )
        
        foreach ($pkg in $additionalPackages) {
            Install-Package -Package $pkg[0] -Description $pkg[1]
        }
    }
}

Write-Host "`n✓ Installation complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan

switch ($Stage) {
    "minimal" {
        Write-Host "  1. Run the minimal API: python main_api_minimal.py" -ForegroundColor White
        Write-Host "  2. Install core dependencies: .\install_dependencies.ps1 core" -ForegroundColor White
    }
    "core" {
        Write-Host "  1. Configure .env file with your settings" -ForegroundColor White
        Write-Host "  2. Run the full API: python main.py api" -ForegroundColor White
        Write-Host "  3. Install ML dependencies: .\install_dependencies.ps1 ml" -ForegroundColor White
    }
    "ml" {
        Write-Host "  1. Download a model: python download_model.py" -ForegroundColor White
        Write-Host "  2. Test inference: python main.py infer --model <path>" -ForegroundColor White
    }
    "full" {
        Write-Host "  1. All dependencies installed!" -ForegroundColor White
        Write-Host "  2. Configure .env for production backends" -ForegroundColor White
        Write-Host "  3. Run tests: pytest" -ForegroundColor White
    }
}

Write-Host "`nFor GPU support, see: https://pytorch.org/get-started/locally/" -ForegroundColor Gray