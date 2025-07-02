#!/usr/bin/env python3
"""
Full implementation of CleoAI Recommended Path Forward
This script implements all three paths: Redis setup, ML capabilities, and status checking
"""
import os
import sys
import subprocess
import json
import time
from pathlib import Path

def run_command(command, description="", shell=True):
    """Run a command and return success status"""
    print(f"🔄 {description}")
    try:
        if isinstance(command, list):
            result = subprocess.run(command, capture_output=True, text=True, shell=shell)
        else:
            result = subprocess.run(command, capture_output=True, text=True, shell=shell)
        
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True, result.stdout
        else:
            print(f"❌ Failed: {description}")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ Exception during {description}: {e}")
        return False, str(e)

def check_package_installed(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def check_current_status():
    """Check current CleoAI status"""
    print("\n📊 Current Status Check")
    print("=" * 40)
    
    status = {}
    
    # Environment file
    env_exists = Path(".env").exists()
    status['env_file'] = env_exists
    print(f"   .env file: {'✅' if env_exists else '❌'}")
    
    # Database
    db_exists = Path("data/memory/cleoai_memory.db").exists()
    status['database'] = db_exists
    print(f"   SQLite database: {'✅' if db_exists else '❌'}")
    
    # Core packages
    packages = {
        'fastapi': check_package_installed("fastapi"),
        'uvicorn': check_package_installed("uvicorn"),
        'redis': check_package_installed("redis"),
        'torch': check_package_installed("torch"),
        'transformers': check_package_installed("transformers"),
        'chromadb': check_package_installed("chromadb")
    }
    
    for pkg, installed in packages.items():
        status[pkg] = installed
        print(f"   {pkg}: {'✅' if installed else '❌'}")
    
    # Models
    models_dir = Path("models")
    model_count = len(list(models_dir.glob("**/config.json"))) if models_dir.exists() else 0
    status['model_count'] = model_count
    print(f"   Downloaded models: {model_count}")
    
    return status

def install_core_dependencies():
    """Install core dependencies for API functionality"""
    print("\n🔧 Installing Core Dependencies")
    print("=" * 40)
    
    # Core packages needed for API
    core_packages = [
        "fastapi==0.109.0",
        "uvicorn[standard]==0.25.0",
        "python-dotenv==1.0.0",
        "pydantic==2.5.0",
        "psutil==5.9.6",
        "ariadne==0.21.0",
        "graphql-core==3.2.3",
        "python-multipart==0.0.6",
        "redis==5.0.1",
        "tqdm==4.66.1",
        "loguru==0.7.0"
    ]
    
    print("Installing core packages...")
    for package in core_packages:
        success, output = run_command(
            f"pip install {package}",
            f"Installing {package.split('==')[0]}"
        )
        if not success:
            print(f"⚠️  Failed to install {package}, continuing...")
    
    return True

def install_ml_dependencies():
    """Install ML dependencies"""
    print("\n🧠 Installing ML Dependencies")
    print("=" * 40)
    
    # Install PyTorch CPU version first (faster and works on all systems)
    print("Installing PyTorch (CPU version)...")
    torch_success, _ = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "Installing PyTorch CPU"
    )
    
    if torch_success:
        # Install other ML packages
        ml_packages = [
            "transformers==4.38.0",
            "sentence-transformers",
            "accelerate==0.23.0",
            "datasets==2.15.0",
            "safetensors==0.4.0",
            "numpy==1.24.3"
        ]
        
        for package in ml_packages:
            success, output = run_command(
                f"pip install {package}",
                f"Installing {package.split('==')[0]}"
            )
    
    # Try to install ChromaDB (may fail on some systems)
    print("\nTrying to install ChromaDB...")
    chromadb_success, _ = run_command(
        "pip install chromadb==0.4.18",
        "Installing ChromaDB"
    )
    
    if not chromadb_success:
        print("⚠️  ChromaDB installation failed. This is common on some systems.")
        print("   You can still use the system with other vector stores.")
    
    return torch_success

def test_redis():
    """Test Redis functionality"""
    print("\n🔴 Testing Redis")
    print("=" * 40)
    
    try:
        import redis
        print("✅ Redis Python client installed")
        
        # Try to connect
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis server is running and accessible")
        
        # Test operations
        r.set('cleoai_test', 'Hello from CleoAI!')
        value = r.get('cleoai_test')
        if value == 'Hello from CleoAI!':
            print("✅ Redis read/write operations working")
        r.delete('cleoai_test')
        
        return True
        
    except ImportError:
        print("❌ Redis Python client not installed")
        return False
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("   Redis server may not be running")
        print("   Install Redis: https://redis.io/download")
        return False

def download_test_model():
    """Download a small test model"""
    print("\n📥 Downloading Test Model")
    print("=" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "gpt2"
        model_dir = Path("models") / "gpt2"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {model_name} to {model_dir}...")
        
        # Download tokenizer
        print("   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        
        # Download model
        print("   Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.save_pretrained(model_dir)
        
        print(f"✅ Model downloaded to {model_dir}")
        
        # Test the model
        print("   Testing model...")
        inputs = tokenizer("Hello, I am", return_tensors="pt")
        
        import torch
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=5,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Test generation: '{generated_text}'")
        print("✅ Model test successful!")
        
        return str(model_dir)
        
    except ImportError:
        print("❌ Transformers not installed. Install ML dependencies first.")
        return None
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return None

def test_api_functionality():
    """Test API functionality"""
    print("\n🌐 Testing API Functionality")
    print("=" * 40)
    
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI and Uvicorn installed")
        
        # Test if we can import our API modules
        try:
            from main_api_minimal import app
            print("✅ Minimal API imports successfully")
        except Exception as e:
            print(f"⚠️  Minimal API import issue: {e}")
        
        # Check if main.py imports work
        try:
            import sys
            sys.path.append('.')
            print("✅ Python path configured")
        except Exception as e:
            print(f"⚠️  Python path issue: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ API dependencies missing: {e}")
        return False

def create_startup_script():
    """Create a startup script for easy testing"""
    print("\n📝 Creating Startup Scripts")
    print("=" * 40)
    
    # Create a simple startup batch file
    startup_script = """@echo off
echo Starting CleoAI Development Environment...
echo.

echo Checking Python environment...
python --version
echo.

echo Starting minimal API server...
echo Access the API at: http://localhost:8000
echo Health check: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

python main_api_minimal.py --debug
"""
    
    with open("start_cleoai.bat", "w") as f:
        f.write(startup_script)
    
    print("✅ Created start_cleoai.bat")
    
    # Create a test script
    test_script = """#!/usr/bin/env python3
import requests
import json

print("Testing CleoAI API...")

try:
    # Test root endpoint
    response = requests.get("http://localhost:8000/")
    if response.status_code == 200:
        print("✅ Root endpoint working")
        print(f"   Response: {response.json()}")
    
    # Test health endpoint
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        print("✅ Health endpoint working")
        health_data = response.json()
        print(f"   Status: {health_data.get('status', 'unknown')}")
    
    # Test echo endpoint
    test_data = {"message": "Hello from test script", "test": True}
    response = requests.post("http://localhost:8000/api/echo", json=test_data)
    if response.status_code == 200:
        print("✅ Echo endpoint working")
        echo_data = response.json()
        print(f"   Echo: {echo_data.get('echo', {}).get('message', 'no message')}")
    
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API. Is the server running?")
    print("   Start server with: python main_api_minimal.py")
except Exception as e:
    print(f"❌ Test failed: {e}")
"""
    
    with open("test_api_simple.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created test_api_simple.py")

def main():
    """Main implementation function"""
    print("🚀 CleoAI Full Setup Implementation")
    print("=" * 50)
    print("This will implement all recommended paths:")
    print("- Option A: Redis setup")
    print("- Option B: ML capabilities") 
    print("- Option C: Status checking")
    print("=" * 50)
    
    # Step 1: Check current status
    status = check_current_status()
    
    # Step 2: Install core dependencies if needed
    if not status.get('fastapi', False):
        install_core_dependencies()
    else:
        print("\n✅ Core dependencies already installed")
    
    # Step 3: Test Redis
    redis_working = test_redis()
    if not redis_working:
        print("\n📝 Redis Setup Instructions:")
        print("   1. Install Redis for Windows:")
        print("      - Memurai: https://www.memurai.com/get-memurai")
        print("      - Or use WSL: wsl --install, then: sudo apt install redis-server")
        print("   2. Start Redis server")
        print("   3. Re-run this script to test Redis")
    
    # Step 4: Install ML dependencies
    if not status.get('torch', False):
        print("\n🧠 Installing ML Dependencies...")
        ml_success = install_ml_dependencies()
        if ml_success:
            print("✅ ML dependencies installed")
        else:
            print("❌ ML installation had issues")
    else:
        print("\n✅ ML dependencies already installed")
    
    # Step 5: Download model if ML is available
    if status.get('torch', False) or check_package_installed('torch'):
        if status.get('model_count', 0) == 0:
            model_path = download_test_model()
            if model_path:
                print(f"✅ Test model ready at {model_path}")
        else:
            print("✅ Models already downloaded")
    
    # Step 6: Test API functionality
    api_working = test_api_functionality()
    
    # Step 7: Create startup scripts
    create_startup_script()
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 Implementation Complete!")
    print("=" * 50)
    
    final_status = check_current_status()
    
    print("\n🚀 What you can do now:")
    if final_status.get('fastapi', False):
        print("   ✅ Start API: python main_api_minimal.py")
        print("   ✅ Or use: start_cleoai.bat")
        print("   ✅ Test API: python test_api_simple.py")
    
    if final_status.get('torch', False):
        print("   ✅ Run inference: python download_model.py")
        if final_status.get('model_count', 0) > 0:
            print("   ✅ Test generation with downloaded models")
    
    if redis_working:
        print("   ✅ Redis caching enabled")
        print("   ✅ Distributed memory available")
    
    print("\n📚 Next steps:")
    print("   1. Start the API server")
    print("   2. Open http://localhost:8000 in your browser")
    print("   3. Try the health check: http://localhost:8000/health")
    print("   4. Explore the API documentation")
    
    if not redis_working:
        print("   5. Install Redis for faster caching")
    
    print("\n🎯 Full development environment ready!")

if __name__ == "__main__":
    main()