"""
Configuration file for the AI Autonomous Agent for Dating App.

This module contains all hyperparameters and settings for model architecture, 
training, and evaluation. It provides centralized configuration management
for the entire CleoAI system.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
from dataclasses import dataclass
import torch

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)


class ModelConfig:
    """
    Model architecture configuration.
    
    This class defines all parameters related to the model architecture,
    including MoE (Mixture of Experts) settings, dimensions, and optimization flags.
    
    Attributes:
        model_name: Base model identifier from HuggingFace
        model_type: Type of model architecture ('moe' or 'dense')
        num_experts: Total number of experts in MoE architecture
        num_experts_per_token: Number of experts activated per token
        expert_dropout: Dropout rate for expert selection
        hidden_size: Hidden dimension size of the model
        intermediate_size: Intermediate dimension for feed-forward layers
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        max_seq_length: Maximum sequence length in tokens
        bf16: Whether to use BF16 precision
        fp16: Whether to use FP16 precision
    """
    # Base model specifications
    model_name: str = "mistralai/Mistral-7B-v0.1"
    model_type: Literal["moe", "dense"] = "moe"
    
    # MoE specifications
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_dropout: float = 0.1
    
    # Dimensions and sizes
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    
    # Context length
    max_seq_length: int = 32768
    
    # Optimization
    bf16: bool = True
    fp16: bool = False
    
    # Additional model settings
    vocab_size: int = 32000
    use_cache: bool = True
    gradient_checkpointing: bool = False
    
    # Model behavior
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 512
    max_length: int = 2048
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    
class MemoryConfig:
    """
    Memory systems configuration.
    
    This class configures the various memory components of the system,
    including short-term conversation memory, long-term user preferences,
    episodic memory, and procedural memory.
    
    Attributes:
        short_term_memory_type: Type of short-term memory implementation
        short_term_max_tokens: Maximum tokens to retain in short-term memory
        recency_weight_decay: Decay factor for recency-based weighting
        long_term_storage_type: Storage backend for long-term memory
        embedding_model: Model used for generating embeddings
        vector_db: Vector database for semantic search
        episodic_memory_enabled: Whether to use episodic memory
        episodic_memory_type: Implementation type for episodic memory
        procedural_memory_enabled: Whether to use procedural memory
        procedural_memory_format: Storage format for procedures
    """
    # Short-term memory (conversation history)
    short_term_memory_type: Literal["buffer", "summary", "window"] = "buffer"
    short_term_max_tokens: int = 16384
    recency_weight_decay: float = 0.98
    
    # Long-term memory (user preferences)
    long_term_storage_type: str = "vector"  # Options: "vector", "relational", "hybrid"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"  # For embeddings
    vector_db: str = "chromadb"  # Vector DB for semantic storage
    
    # Episodic memory
    episodic_memory_enabled: bool = True
    episodic_memory_type: str = "hybrid"  # Options: "vector", "graph", "hybrid"
    
    # Procedural memory
    procedural_memory_enabled: bool = True
    procedural_memory_format: str = "json"  # Format for storing task execution protocols
    
    # Persistent storage
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 2  # In hours
    max_checkpoints: int = 10  # Maximum number of checkpoints to keep

class TrainingConfig:
    """Training configuration"""
    # General training settings
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # Training phases
    phase1_steps: int = 10000  # Foundation phase
    phase2_steps: int = 8000   # Emotional & Safety phase
    phase3_steps: int = 8000   # Relationship & Dating phase
    phase4_steps: int = 5000   # Integration & Refinement phase
    
    # Optimization
    optimizer: str = "adamw_torch"
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    
    # Distributed training
    deepspeed_config: str = "configs/deepspeed_config.json"
    
    # Checkpointing and logging
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    
    # Datasets (to be filled in during dataset preparation)
    train_datasets: Dict[str, str] = {
        "phase1": ["redpajama_subset", "openassistant_subset"],
        "phase2": ["empathetic_dialogues", "go_emotions", "meld"],
        "phase3": ["reddit_relationship_advice", "dating_preferences"],
        "phase4": ["multimodal_conversation", "task_execution"]
    }
    
    # Evaluation metrics
    eval_metrics: List[str] = [
        "perplexity", 
        "accuracy", 
        "emotion_detection_f1",
        "safety_compliance",
        "conversation_quality"
    ]

class EvaluationConfig:
    """Evaluation configuration"""
    eval_batch_size: int = 8
    test_size: float = 0.1  # Percentage of data to use for testing
    human_eval_samples: int = 100  # Number of samples for human evaluation
    
    # Benchmark datasets
    benchmark_datasets: List[str] = [
        "dating_app_benchmark",
        "emotional_support_benchmark",
        "safety_protocol_benchmark"
    ]
    
    # Evaluation frequency (in steps)
    daily_benchmarking: bool = True
    adversarial_testing: bool = True

@dataclass
class MemoryBackendConfig:
    """
    Configuration for distributed memory storage backends.
    
    This class configures the new memory services for production deployment.
    """
    # Redis configuration
    use_redis: bool = True
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    redis_db: int = 0
    redis_max_connections: int = 50
    
    # Supabase configuration
    use_supabase: bool = True
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_anon_key: str = os.getenv("SUPABASE_ANON_KEY", "")
    
    # Pinecone configuration
    use_pinecone: bool = True
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    pinecone_index_name: str = "cleoai-memories"
    pinecone_dimension: int = 1536  # OpenAI embedding dimension
    pinecone_metric: str = "cosine"
    
    # MongoDB configuration
    use_mongodb: bool = True
    mongodb_connection_string: str = os.getenv(
        "MONGODB_CONNECTION_STRING", 
        "mongodb://localhost:27017/"
    )
    mongodb_database: str = "cleoai_memory"
    
    # Legacy backend settings
    use_sqlite: bool = True  # Keep for backward compatibility
    use_chromadb: bool = False  # Replaced by Pinecone
    
    # Migration settings
    enable_migration: bool = True
    migration_batch_size: int = 100


@dataclass
class APIConfig:
    """
    Configuration for API server.
    """
    # API server settings
    enable_api: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # GraphQL settings
    graphql_debug: bool = True
    graphql_playground: bool = True
    
    # CORS settings
    cors_origins: List[str] = None  # None means allow all
    cors_credentials: bool = True
    
    # Authentication (future)
    enable_auth: bool = False
    jwt_secret: str = os.getenv("JWT_SECRET", "")
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]  # Allow all in development


class ProjectConfig:
    """Project directory structure and file paths"""
    # Base paths
    base_dir: Path = Path(os.getcwd())
    data_dir: Path = base_dir / "data"
    model_dir: Path = base_dir / "models"
    output_dir: Path = base_dir / "outputs"
    log_dir: Path = base_dir / "logs"
    
    # Ensure directories exist
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        try:
            for dir_path in [
                self.data_dir, 
                self.model_dir, 
                self.output_dir, 
                self.log_dir,
                self.data_dir / "raw",
                self.data_dir / "processed",
                self.model_dir / "checkpoints",
                self.output_dir / "evaluations",
                self.log_dir / "training"
            ]:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Directory created or already exists: {dir_path}")
                except PermissionError:
                    logger.error(f"Permission denied creating directory: {dir_path}")
                    raise
                except OSError as e:
                    logger.error(f"OS error creating directory {dir_path}: {e}")
                    raise
        except Exception as e:
            logger.error(f"Failed to create project directories: {e}")
            raise RuntimeError(f"Failed to create project directories: {e}") from e

# Create a singleton instance of each config
model_config = ModelConfig()
memory_config = MemoryConfig()
memory_backend_config = MemoryBackendConfig()
api_config = APIConfig()
training_config = TrainingConfig()
evaluation_config = EvaluationConfig()
project_config = ProjectConfig()

# Create necessary directories
project_config.create_directories()

# Export key paths for compatibility
MODELS_DIR = project_config.model_dir
DATA_DIR = project_config.data_dir
