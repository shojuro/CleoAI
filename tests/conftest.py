"""Pytest configuration and shared fixtures."""
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model_path(temp_dir: Path) -> Path:
    """Create a mock model directory."""
    model_dir = temp_dir / "mock_model"
    model_dir.mkdir()
    
    # Create minimal config files
    config_file = model_dir / "config.json"
    config_file.write_text('{"model_type": "gpt2", "vocab_size": 50257}')
    
    return model_dir


@pytest.fixture
def device() -> str:
    """Get the appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration for testing."""
    return {
        "model_name": "gpt2",
        "num_experts": 4,
        "num_experts_per_token": 2,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "max_length": 1024,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "warmup_steps": 100,
        "logging_steps": 50,
        "save_steps": 500,
        "eval_steps": 500,
        "gradient_accumulation_steps": 1,
        "fp16": False,
        "device": "cpu",
        "use_moe": True,
        "use_deepspeed": False,
    }


@pytest.fixture(autouse=True)
def setup_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("WANDB_MODE", "disabled")