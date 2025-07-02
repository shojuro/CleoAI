#!/usr/bin/env python3
"""
Model download and setup script for CleoAI
Downloads and configures small models for development and testing
"""
import os
import sys
from pathlib import Path
import json

# Try to import transformers, fallback to manual download if not available
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def check_dependencies():
    """Check if required dependencies are installed"""
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers library not installed")
        print("\nTo install ML dependencies:")
        print("  .\\install_dependencies.ps1 ml")
        print("  OR")
        print("  pip install transformers torch")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        from transformers import __version__
        print(f"✅ Transformers version: {__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    return True

def download_model(model_name, output_dir):
    """Download a model and tokenizer"""
    print(f"\n📥 Downloading {model_name}...")
    
    try:
        # Create output directory
        model_path = Path(output_dir) / model_name.replace("/", "_")
        model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"   Saving to: {model_path}")
        
        # Download tokenizer
        print("   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)
        
        # Download model
        print("   Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None
        )
        model.save_pretrained(model_path)
        
        # Download config
        print("   Downloading config...")
        config = AutoConfig.from_pretrained(model_name)
        config.save_pretrained(model_path)
        
        # Create model info file
        model_info = {
            "model_name": model_name,
            "model_path": str(model_path),
            "downloaded_at": str(datetime.now()),
            "model_type": config.model_type if hasattr(config, 'model_type') else "unknown",
            "vocab_size": config.vocab_size if hasattr(config, 'vocab_size') else "unknown",
            "max_position_embeddings": getattr(config, 'max_position_embeddings', "unknown"),
            "hidden_size": getattr(config, 'hidden_size', "unknown"),
            "num_attention_heads": getattr(config, 'num_attention_heads', "unknown"),
            "num_hidden_layers": getattr(config, 'num_hidden_layers', "unknown")
        }
        
        with open(model_path / "cleoai_model_info.json", "w") as f:
            json.dump(model_info, f, indent=2, default=str)
        
        print(f"✅ Successfully downloaded {model_name}")
        print(f"   Model path: {model_path}")
        
        # Test the model
        print("\n🧪 Testing model...")
        test_input = "Hello, I am"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Test input: '{test_input}'")
        print(f"   Generated: '{generated_text}'")
        print("✅ Model test successful!")
        
        return str(model_path)
        
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return None

def main():
    print("CleoAI Model Downloader")
    print("=" * 30)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Available models (small ones suitable for development)
    models = {
        "1": {
            "name": "gpt2",
            "description": "GPT-2 (117M parameters) - Fast, good for testing",
            "size": "~500MB"
        },
        "2": {
            "name": "microsoft/DialoGPT-small",
            "description": "DialoGPT Small (117M) - Conversational",
            "size": "~500MB"
        },
        "3": {
            "name": "distilgpt2",
            "description": "DistilGPT-2 (82M) - Smaller, faster",
            "size": "~350MB"
        },
        "4": {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "description": "TinyLlama (1.1B) - Modern small model",
            "size": "~2.2GB"
        }
    }
    
    print("\nAvailable models:")
    for key, model in models.items():
        print(f"  {key}. {model['name']}")
        print(f"     {model['description']}")
        print(f"     Size: {model['size']}")
        print()
    
    # Get user choice
    choice = input("Select model to download (1-4, or 'all' for all): ").strip()
    
    if choice.lower() == 'all':
        selected_models = list(models.values())
    elif choice in models:
        selected_models = [models[choice]]
    else:
        print("❌ Invalid choice")
        return
    
    # Set output directory
    output_dir = os.getenv("MODEL_CACHE_DIR", "models")
    print(f"\nDownloading to: {output_dir}")
    
    # Download selected models
    downloaded_models = []
    for model_info in selected_models:
        model_path = download_model(model_info["name"], output_dir)
        if model_path:
            downloaded_models.append({
                "name": model_info["name"],
                "path": model_path
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    
    if downloaded_models:
        print(f"✅ Successfully downloaded {len(downloaded_models)} model(s):")
        for model in downloaded_models:
            print(f"   - {model['name']}")
            print(f"     Path: {model['path']}")
        
        print("\n📝 Next steps:")
        print("   1. Test inference:")
        print(f"      python -c \"from transformers import pipeline; print(pipeline('text-generation', model='{downloaded_models[0]['path']}')('Hello'))\"")
        print("   2. Start CleoAI with model:")
        print(f"      python main.py infer --model {downloaded_models[0]['path']}")
        print("   3. Or update .env:")
        print(f"      DEFAULT_MODEL={downloaded_models[0]['path']}")
    else:
        print("❌ No models were downloaded successfully")

if __name__ == "__main__":
    import torch
    from datetime import datetime
    main()