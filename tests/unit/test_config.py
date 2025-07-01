"""Unit tests for config.py module."""
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from config import (
    ModelConfig,
    MemoryConfig,
    TrainingConfig,
    EvaluationConfig,
    ProjectConfig,
    model_config,
    memory_config,
    training_config,
    evaluation_config,
    project_config,
)


class TestModelConfig:
    """Test cases for ModelConfig class."""
    
    def test_model_config_defaults(self):
        """Test default values for ModelConfig."""
        config = ModelConfig()
        
        assert config.model_name == "mistralai/Mistral-7B-v0.1"
        assert config.model_type == "moe"
        assert config.num_experts == 8
        assert config.num_experts_per_token == 2
        assert config.expert_dropout == 0.1
        assert config.hidden_size == 4096
        assert config.intermediate_size == 11008
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32
        assert config.max_seq_length == 32768
        assert config.bf16 is True
        assert config.fp16 is False
    
    def test_model_config_types(self):
        """Test data types of ModelConfig attributes."""
        config = ModelConfig()
        
        assert isinstance(config.model_name, str)
        assert isinstance(config.num_experts, int)
        assert isinstance(config.expert_dropout, float)
        assert isinstance(config.bf16, bool)
    
    def test_model_config_value_ranges(self):
        """Test that config values are within reasonable ranges."""
        config = ModelConfig()
        
        assert config.num_experts > 0
        assert config.num_experts_per_token > 0
        assert config.num_experts_per_token <= config.num_experts
        assert 0 <= config.expert_dropout <= 1
        assert config.hidden_size > 0
        assert config.num_hidden_layers > 0
        assert config.num_attention_heads > 0
        assert config.max_seq_length > 0


class TestMemoryConfig:
    """Test cases for MemoryConfig class."""
    
    def test_memory_config_defaults(self):
        """Test default values for MemoryConfig."""
        config = MemoryConfig()
        
        assert config.short_term_memory_type == "buffer"
        assert config.short_term_max_tokens == 16384
        assert config.recency_weight_decay == 0.98
        assert config.long_term_storage_type == "vector"
        assert config.embedding_model == "sentence-transformers/all-mpnet-base-v2"
        assert config.vector_db == "chromadb"
        assert config.episodic_memory_enabled is True
        assert config.episodic_memory_type == "hybrid"
        assert config.procedural_memory_enabled is True
        assert config.procedural_memory_format == "json"
        assert config.checkpoint_dir == "checkpoints"
        assert config.checkpoint_interval == 2
        assert config.max_checkpoints == 10
    
    def test_memory_config_valid_options(self):
        """Test that memory config options are valid."""
        config = MemoryConfig()
        
        assert config.short_term_memory_type in ["buffer", "summary", "window"]
        assert config.long_term_storage_type in ["vector", "relational", "hybrid"]
        assert config.episodic_memory_type in ["vector", "graph", "hybrid"]
        assert config.procedural_memory_format in ["json", "yaml", "pickle"]
    
    def test_memory_config_value_ranges(self):
        """Test that memory config values are within reasonable ranges."""
        config = MemoryConfig()
        
        assert config.short_term_max_tokens > 0
        assert 0 < config.recency_weight_decay < 1
        assert config.checkpoint_interval > 0
        assert config.max_checkpoints > 0


class TestTrainingConfig:
    """Test cases for TrainingConfig class."""
    
    def test_training_config_defaults(self):
        """Test default values for TrainingConfig."""
        config = TrainingConfig()
        
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.gradient_accumulation_steps == 8
        assert config.learning_rate == 2e-5
        assert config.weight_decay == 0.01
        assert config.warmup_steps == 500
        assert config.optimizer == "adamw_torch"
        assert config.scheduler == "cosine"
        assert config.max_grad_norm == 1.0
    
    def test_training_phases(self):
        """Test training phase configuration."""
        config = TrainingConfig()
        
        assert config.phase1_steps == 10000
        assert config.phase2_steps == 8000
        assert config.phase3_steps == 8000
        assert config.phase4_steps == 5000
        
        # Total steps should be reasonable
        total_steps = (
            config.phase1_steps + 
            config.phase2_steps + 
            config.phase3_steps + 
            config.phase4_steps
        )
        assert total_steps == 31000
    
    def test_training_datasets(self):
        """Test training dataset configuration."""
        config = TrainingConfig()
        
        assert isinstance(config.train_datasets, dict)
        assert "phase1" in config.train_datasets
        assert "phase2" in config.train_datasets
        assert "phase3" in config.train_datasets
        assert "phase4" in config.train_datasets
        
        # Check that each phase has datasets assigned
        for phase, datasets in config.train_datasets.items():
            assert isinstance(datasets, list)
            assert len(datasets) > 0
    
    def test_training_config_value_ranges(self):
        """Test that training config values are within reasonable ranges."""
        config = TrainingConfig()
        
        assert config.epochs > 0
        assert config.batch_size > 0
        assert config.gradient_accumulation_steps > 0
        assert config.learning_rate > 0
        assert 0 <= config.weight_decay <= 1
        assert config.warmup_steps >= 0
        assert config.max_grad_norm > 0
        assert config.save_steps > 0
        assert config.eval_steps > 0
        assert config.logging_steps > 0


class TestEvaluationConfig:
    """Test cases for EvaluationConfig class."""
    
    def test_evaluation_config_defaults(self):
        """Test default values for EvaluationConfig."""
        config = EvaluationConfig()
        
        assert config.eval_batch_size == 8
        assert config.test_size == 0.1
        assert config.human_eval_samples == 100
        assert config.daily_benchmarking is True
        assert config.adversarial_testing is True
    
    def test_benchmark_datasets(self):
        """Test benchmark dataset configuration."""
        config = EvaluationConfig()
        
        assert isinstance(config.benchmark_datasets, list)
        assert len(config.benchmark_datasets) == 3
        assert "dating_app_benchmark" in config.benchmark_datasets
        assert "emotional_support_benchmark" in config.benchmark_datasets
        assert "safety_protocol_benchmark" in config.benchmark_datasets
    
    def test_evaluation_config_value_ranges(self):
        """Test that evaluation config values are within reasonable ranges."""
        config = EvaluationConfig()
        
        assert config.eval_batch_size > 0
        assert 0 < config.test_size < 1
        assert config.human_eval_samples > 0


class TestProjectConfig:
    """Test cases for ProjectConfig class."""
    
    def test_project_config_paths(self):
        """Test that project paths are correctly set."""
        config = ProjectConfig()
        
        assert isinstance(config.base_dir, Path)
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.model_dir, Path)
        assert isinstance(config.output_dir, Path)
        assert isinstance(config.log_dir, Path)
        
        # Check relative paths
        assert config.data_dir == config.base_dir / "data"
        assert config.model_dir == config.base_dir / "models"
        assert config.output_dir == config.base_dir / "outputs"
        assert config.log_dir == config.base_dir / "logs"
    
    @patch('pathlib.Path.mkdir')
    def test_create_directories(self, mock_mkdir):
        """Test directory creation method."""
        config = ProjectConfig()
        config.create_directories()
        
        # Check that mkdir was called for each directory
        expected_dirs = [
            config.data_dir,
            config.model_dir,
            config.output_dir,
            config.log_dir,
            config.data_dir / "raw",
            config.data_dir / "processed",
            config.model_dir / "checkpoints",
            config.output_dir / "evaluations",
            config.log_dir / "training"
        ]
        
        assert mock_mkdir.call_count == len(expected_dirs)
        
        # Verify mkdir was called with correct arguments
        for call in mock_mkdir.call_args_list:
            args, kwargs = call
            assert kwargs.get('parents') is True
            assert kwargs.get('exist_ok') is True


class TestSingletonInstances:
    """Test the singleton instances created in config.py."""
    
    def test_singleton_instances_exist(self):
        """Test that singleton instances are created."""
        assert isinstance(model_config, ModelConfig)
        assert isinstance(memory_config, MemoryConfig)
        assert isinstance(training_config, TrainingConfig)
        assert isinstance(evaluation_config, EvaluationConfig)
        assert isinstance(project_config, ProjectConfig)
    
    def test_singleton_instances_are_unique(self):
        """Test that singleton instances are unique objects."""
        # Create new instances
        new_model_config = ModelConfig()
        new_memory_config = MemoryConfig()
        
        # They should be different objects
        assert model_config is not new_model_config
        assert memory_config is not new_memory_config
    
    def test_singleton_instances_have_correct_defaults(self):
        """Test that singleton instances have expected default values."""
        assert model_config.model_name == "mistralai/Mistral-7B-v0.1"
        assert memory_config.vector_db == "chromadb"
        assert training_config.epochs == 3
        assert evaluation_config.eval_batch_size == 8
        assert project_config.data_dir.name == "data"


@pytest.mark.parametrize("config_class,attribute,expected_type", [
    (ModelConfig, "model_name", str),
    (ModelConfig, "num_experts", int),
    (ModelConfig, "expert_dropout", float),
    (ModelConfig, "bf16", bool),
    (MemoryConfig, "short_term_memory_type", str),
    (MemoryConfig, "short_term_max_tokens", int),
    (TrainingConfig, "epochs", int),
    (TrainingConfig, "learning_rate", float),
    (EvaluationConfig, "eval_batch_size", int),
    (EvaluationConfig, "test_size", float),
])
def test_config_attribute_types(config_class, attribute, expected_type):
    """Parametrized test for config attribute types."""
    config = config_class()
    assert hasattr(config, attribute)
    assert isinstance(getattr(config, attribute), expected_type)


@pytest.mark.parametrize("config_class,attribute,min_value,max_value", [
    (ModelConfig, "num_experts", 1, None),
    (ModelConfig, "expert_dropout", 0, 1),
    (MemoryConfig, "recency_weight_decay", 0, 1),
    (TrainingConfig, "learning_rate", 0, 1),
    (TrainingConfig, "weight_decay", 0, 1),
    (EvaluationConfig, "test_size", 0, 1),
])
def test_config_value_ranges(config_class, attribute, min_value, max_value):
    """Parametrized test for config value ranges."""
    config = config_class()
    value = getattr(config, attribute)
    
    if min_value is not None:
        assert value >= min_value
    if max_value is not None:
        assert value <= max_value