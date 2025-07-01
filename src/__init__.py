"""
Source code for the AI Autonomous Agent.
Provides model architecture, memory systems, training pipeline, and inference capabilities.
"""
# Version of the package
__version__ = "0.1.0"

# Import main components
from src.model import MoEModel, load_pretrained_model_with_moe
from src.memory import MemoryManager
from src.training import ModelTrainer
from src.inference import InferenceEngine, InteractiveSession

__all__ = [
    "MoEModel",
    "load_pretrained_model_with_moe",
    "MemoryManager",
    "ModelTrainer",
    "InferenceEngine",
    "InteractiveSession"
]
