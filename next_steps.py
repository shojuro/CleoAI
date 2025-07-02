#!/usr/bin/env python3
"""
Next steps guide for CleoAI development
Shows what you can do next based on current setup
"""
import os
import sys
from pathlib import Path

def check_file_exists(path):
    """Check if a file exists"""
    return Path(path).exists()

def check_package_installed(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def check_env_setting(key, expected_value="true"):
    """Check environment variable setting"""
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv(key, "false").lower() == expected_value.lower()

def main():
    print("🚀 CleoAI Next Steps Guide")
    print("=" * 40)
    
    # Check current status
    print("\n📊 Current Status:")
    
    # Environment
    env_exists = check_file_exists(".env")
    print(f"   .env file: {'✅' if env_exists else '❌'}")
    
    # Database
    db_exists = check_file_exists("data/memory/cleoai_memory.db")
    print(f"   SQLite database: {'✅' if db_exists else '❌'}")
    
    # Core packages
    fastapi_installed = check_package_installed("fastapi")
    print(f"   FastAPI: {'✅' if fastapi_installed else '❌'}")
    
    # ML packages
    torch_installed = check_package_installed("torch")
    transformers_installed = check_package_installed("transformers")
    print(f"   PyTorch: {'✅' if torch_installed else '❌'}")
    print(f"   Transformers: {'✅' if transformers_installed else '❌'}")
    
    # Redis
    redis_installed = check_package_installed("redis")
    print(f"   Redis client: {'✅' if redis_installed else '❌'}")
    
    # Models
    models_dir = Path("models")
    model_count = len(list(models_dir.glob("**/config.json"))) if models_dir.exists() else 0
    print(f"   Downloaded models: {model_count}")
    
    # Determine next steps
    print("\n🎯 Recommended Next Steps:")
    
    if not env_exists or not db_exists:
        print("\n1️⃣ SETUP ENVIRONMENT")
        print("   python setup_dev_environment.py")
        print("   python create_test_data.py")
        return
    
    if not fastapi_installed:
        print("\n1️⃣ INSTALL CORE DEPENDENCIES")
        print("   .\\install_dependencies.ps1 core")
        print("   This will enable the full API with GraphQL")
        return
    
    if fastapi_installed and not torch_installed:
        print("\n1️⃣ START THE API SERVER")
        print("   Current options:")
        print("   a) Minimal API: python main_api_minimal.py")
        print("   b) Full API (if core deps installed): python main.py api")
        print("")
        print("2️⃣ INSTALL ML DEPENDENCIES")
        print("   .\\install_dependencies.ps1 ml")
        print("   This adds PyTorch, Transformers, and ChromaDB")
        
    elif torch_installed and model_count == 0:
        print("\n1️⃣ DOWNLOAD A MODEL")
        print("   python download_model.py")
        print("   Choose a small model like GPT-2 for testing")
        
    elif model_count > 0:
        print("\n1️⃣ TEST INFERENCE")
        model_dirs = list(models_dir.glob("*/"))
        if model_dirs:
            model_path = model_dirs[0]
            print(f"   python main.py infer --model {model_path}")
            print("   Try: 'Hello, can you tell me about yourself?'")
        
        print("\n2️⃣ ENABLE DISTRIBUTED BACKENDS")
        print("   Setup Redis:")
        print("   .\\setup_redis_windows.ps1 check")
        print("   .\\setup_redis_windows.ps1 install  # if needed")
        
    if redis_installed:
        print("\n3️⃣ EXPLORE ADVANCED FEATURES")
        print("   a) GraphQL API: http://localhost:8000/graphql")
        print("   b) Health monitoring: http://localhost:8000/health/detailed")
        print("   c) Memory migration: python scripts/migrate_memory.py --help")
        print("   d) Run tests: python -m pytest tests/")
    
    # Current capabilities
    print("\n🔧 What You Can Do Right Now:")
    
    if fastapi_installed:
        print("   ✅ Run API server")
        print("   ✅ Test endpoints")
        print("   ✅ Store/retrieve memories in SQLite")
    
    if torch_installed:
        print("   ✅ Download and run ML models")
        print("   ✅ Generate text with transformers")
        print("   ✅ Create embeddings with sentence-transformers")
    
    if model_count > 0:
        print("   ✅ Run inference with downloaded models")
        print("   ✅ Test conversational AI")
    
    if redis_installed:
        print("   ✅ Use Redis for fast memory caching")
        print("   ✅ Scale to distributed architecture")
    
    # Quick commands
    print("\n⚡ Quick Commands:")
    print("   Check dependencies: .\\install_dependencies.ps1 check")
    print("   Test API: .\\test_api.ps1")
    print("   Test memory: python test_memory_simple.py")
    print("   View health: curl http://localhost:8000/health")
    
    # Learning resources
    print("\n📚 Documentation:")
    print("   Backend guide: docs/backend_selection_guide.md")
    print("   Docker setup: docs/docker_setup_guide.md")
    print("   Project README: README.md")
    
    print("\n🎉 You're making great progress!")

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    main()