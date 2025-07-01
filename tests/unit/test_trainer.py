"""Unit tests for training module."""
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, call
from typing import Dict, List, Any

import pytest
import torch
import numpy as np
from transformers import AutoTokenizer

from src.training.trainer import (
    TrainingPhase,
    CustomDataset,
    ModelTrainer,
    create_training_phases,
    prepare_dataset,
    compute_metrics,
)


class TestTrainingPhase:
    """Test cases for TrainingPhase class."""
    
    def test_training_phase_initialization(self):
        """Test TrainingPhase initialization with default values."""
        phase = TrainingPhase(
            phase_name="test_phase",
            datasets=["dataset1", "dataset2"],
            steps=1000,
            description="Test phase description"
        )
        
        assert phase.phase_name == "test_phase"
        assert phase.datasets == ["dataset1", "dataset2"]
        assert phase.steps == 1000
        assert phase.description == "Test phase description"
        assert phase.learning_rate == 2e-5
        assert phase.eval_datasets == []
        assert phase.custom_metrics == []
    
    def test_training_phase_with_custom_values(self):
        """Test TrainingPhase with custom parameters."""
        phase = TrainingPhase(
            phase_name="custom_phase",
            datasets=["data1"],
            steps=500,
            description="Custom phase",
            learning_rate=1e-4,
            eval_datasets=["eval1", "eval2"],
            custom_metrics=["bleu", "rouge"]
        )
        
        assert phase.learning_rate == 1e-4
        assert phase.eval_datasets == ["eval1", "eval2"]
        assert phase.custom_metrics == ["bleu", "rouge"]
    
    def test_training_phase_repr(self):
        """Test string representation of TrainingPhase."""
        phase = TrainingPhase("test", ["data"], 100, "desc")
        assert repr(phase) == "TrainingPhase(name=test, steps=100)"


class TestCustomDataset:
    """Test cases for CustomDataset class."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (100,)),
            "attention_mask": torch.ones(100),
        }
        return tokenizer
    
    def test_custom_dataset_initialization(self, mock_tokenizer):
        """Test CustomDataset initialization."""
        data = [
            {"input": "Hello", "output": "Hi there!"},
            {"input": "How are you?", "output": "I'm doing well!"}
        ]
        
        dataset = CustomDataset(
            data=data,
            tokenizer=mock_tokenizer,
            max_length=512
        )
        
        assert len(dataset) == 2
        assert dataset.max_length == 512
        assert dataset.input_key == "input"
        assert dataset.output_key == "output"
    
    def test_custom_dataset_getitem(self, mock_tokenizer):
        """Test getting items from CustomDataset."""
        data = [{"input": "Test input", "output": "Test output"}]
        dataset = CustomDataset(data, mock_tokenizer)
        
        item = dataset[0]
        
        # Check tokenizer was called
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args[0][0]
        assert "Test input" in call_args
        assert "Test output" in call_args
        
        # Check returned item structure
        assert "input_ids" in item
        assert "attention_mask" in item
    
    def test_custom_dataset_with_custom_keys(self, mock_tokenizer):
        """Test CustomDataset with custom input/output keys."""
        data = [{"question": "What's 2+2?", "answer": "4"}]
        dataset = CustomDataset(
            data,
            mock_tokenizer,
            input_key="question",
            output_key="answer"
        )
        
        item = dataset[0]
        call_args = mock_tokenizer.call_args[0][0]
        assert "What's 2+2?" in call_args
        assert "4" in call_args


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    @pytest.fixture
    def trainer_config(self, temp_dir):
        """Create trainer configuration."""
        return {
            "model_name": "test-model",
            "output_dir": str(temp_dir / "output"),
            "config": MagicMock(
                epochs=3,
                batch_size=4,
                gradient_accumulation_steps=1,
                learning_rate=2e-5,
                warmup_steps=100,
                save_steps=500,
                eval_steps=500,
                logging_steps=50,
            ),
            "use_moe": True,
            "num_experts": 4,
            "num_experts_per_token": 2,
            "device": "cpu",
            "fp16": False,
            "bf16": False,
            "use_deepspeed": False,
            "log_to_wandb": False,
        }
    
    @patch('src.training.trainer.load_pretrained_model_with_moe')
    @patch('src.training.trainer.AutoTokenizer.from_pretrained')
    def test_trainer_initialization(self, mock_tokenizer, mock_model, trainer_config):
        """Test ModelTrainer initialization."""
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        trainer = ModelTrainer(**trainer_config)
        
        assert trainer.model_name == "test-model"
        assert trainer.output_dir == trainer_config["output_dir"]
        assert trainer.use_moe is True
        assert trainer.num_experts == 4
        assert trainer.device == "cpu"
        
        # Check model and tokenizer were loaded
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
    
    @patch('src.training.trainer.load_pretrained_model_with_moe')
    @patch('src.training.trainer.AutoTokenizer.from_pretrained')
    def test_create_training_phases(self, mock_tokenizer, mock_model, trainer_config):
        """Test creation of training phases."""
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        trainer = ModelTrainer(**trainer_config)
        phases = trainer.create_training_phases()
        
        assert len(phases) >= 4  # Should have at least 4 phases
        assert all(isinstance(phase, TrainingPhase) for phase in phases)
        
        # Check phase names
        phase_names = [p.phase_name for p in phases]
        assert "foundation" in phase_names
        assert "emotional_safety" in phase_names
        assert "relationship_dating" in phase_names
        assert "integration" in phase_names
    
    @patch('src.training.trainer.load_dataset')
    @patch('src.training.trainer.load_pretrained_model_with_moe')
    @patch('src.training.trainer.AutoTokenizer.from_pretrained')
    def test_prepare_datasets(self, mock_tokenizer, mock_model, mock_load_dataset, trainer_config):
        """Test dataset preparation."""
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock(pad_token_id=0)
        mock_dataset = MagicMock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        trainer = ModelTrainer(**trainer_config)
        datasets = trainer.prepare_datasets(["test_dataset"])
        
        assert "train" in datasets
        assert "validation" in datasets
        mock_load_dataset.assert_called()
    
    @patch('src.training.trainer.Trainer')
    @patch('src.training.trainer.load_pretrained_model_with_moe')
    @patch('src.training.trainer.AutoTokenizer.from_pretrained')
    def test_train_phase(self, mock_tokenizer, mock_model, mock_trainer_class, trainer_config):
        """Test training a single phase."""
        # Setup mocks
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock(pad_token_id=0)
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = None
        mock_trainer.evaluate.return_value = {"eval_loss": 0.5}
        mock_trainer_class.return_value = mock_trainer
        
        trainer = ModelTrainer(**trainer_config)
        
        # Mock datasets
        with patch.object(trainer, 'prepare_datasets') as mock_prep:
            mock_prep.return_value = {
                "train": MagicMock(spec=['__len__']),
                "validation": MagicMock(spec=['__len__'])
            }
            
            phase = TrainingPhase("test", ["dataset"], 100, "Test phase")
            result = trainer.train_phase(phase)
            
            assert "checkpoint_path" in result
            assert mock_trainer.train.called
            assert mock_trainer.evaluate.called
    
    @patch('src.training.trainer.load_pretrained_model_with_moe')
    @patch('src.training.trainer.AutoTokenizer.from_pretrained')
    def test_evaluate_model(self, mock_tokenizer, mock_model, trainer_config):
        """Test model evaluation."""
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock(pad_token_id=0)
        
        trainer = ModelTrainer(**trainer_config)
        
        with patch.object(trainer, 'compute_metrics') as mock_metrics:
            mock_metrics.return_value = {"accuracy": 0.95}
            
            with patch('src.training.trainer.load_dataset') as mock_load:
                mock_dataset = MagicMock()
                mock_load.return_value = mock_dataset
                
                results = trainer.evaluate_model(["test_dataset"])
                
                assert isinstance(results, dict)
                assert len(results) > 0
    
    @patch('src.training.trainer.load_pretrained_model_with_moe')
    @patch('src.training.trainer.AutoTokenizer.from_pretrained')
    def test_save_and_load_checkpoint(self, mock_tokenizer, mock_model, trainer_config, temp_dir):
        """Test checkpoint saving and loading."""
        mock_model_instance = MagicMock()
        mock_model_instance.state_dict.return_value = {"param": torch.tensor([1.0])}
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = MagicMock()
        
        trainer = ModelTrainer(**trainer_config)
        
        # Test save
        checkpoint_path = trainer.save_checkpoint(
            epoch=1,
            step=100,
            best_loss=0.5,
            phase_name="test"
        )
        
        assert Path(checkpoint_path).exists()
        assert "checkpoint" in str(checkpoint_path)
        
        # Test load
        trainer.load_checkpoint(checkpoint_path)
        mock_model_instance.load_state_dict.assert_called()


class TestTrainingUtilities:
    """Test training utility functions."""
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        # Mock evaluation predictions
        eval_pred = MagicMock()
        eval_pred.predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
        eval_pred.label_ids = np.array([1, 0])
        
        metrics = compute_metrics(eval_pred)
        
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    @patch('src.training.trainer.load_dataset')
    def test_prepare_dataset_function(self, mock_load_dataset):
        """Test standalone dataset preparation function."""
        mock_dataset = MagicMock()
        mock_dataset.train_test_split.return_value = {
            "train": MagicMock(),
            "test": MagicMock()
        }
        mock_load_dataset.return_value = mock_dataset
        
        tokenizer = MagicMock()
        result = prepare_dataset("test_dataset", tokenizer, max_length=512)
        
        assert "train" in result
        assert "validation" in result
        mock_load_dataset.assert_called_once()
    
    def test_create_training_phases_function(self):
        """Test standalone training phases creation."""
        config = MagicMock(
            phase1_steps=1000,
            phase2_steps=2000,
            phase3_steps=3000,
            phase4_steps=4000,
        )
        
        phases = create_training_phases(config)
        
        assert len(phases) == 4
        assert sum(p.steps for p in phases) == 10000
        assert phases[0].phase_name == "foundation"
        assert phases[-1].phase_name == "integration"


@pytest.mark.integration
class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    @patch('wandb.init')
    @patch('wandb.log')
    def test_wandb_integration(self, mock_wandb_log, mock_wandb_init):
        """Test Weights & Biases integration."""
        with patch('src.training.trainer.load_pretrained_model_with_moe'):
            with patch('src.training.trainer.AutoTokenizer.from_pretrained'):
                trainer = ModelTrainer(
                    model_name="test",
                    output_dir="./test",
                    log_to_wandb=True,
                    wandb_project="test_project"
                )
                
                # Check wandb was initialized
                mock_wandb_init.assert_called_once()
                
                # Test logging
                trainer.log_metrics({"loss": 0.5}, step=100)
                mock_wandb_log.assert_called_with({"loss": 0.5}, step=100)
    
    def test_memory_efficient_training(self):
        """Test memory-efficient training configurations."""
        config = MagicMock(
            gradient_accumulation_steps=8,
            batch_size=1,
            gradient_checkpointing=True,
        )
        
        with patch('src.training.trainer.load_pretrained_model_with_moe') as mock_model:
            with patch('src.training.trainer.AutoTokenizer.from_pretrained'):
                model_instance = MagicMock()
                mock_model.return_value = model_instance
                
                trainer = ModelTrainer(
                    model_name="test",
                    output_dir="./test",
                    config=config,
                    gradient_checkpointing=True
                )
                
                # Verify gradient checkpointing was enabled
                model_instance.gradient_checkpointing_enable.assert_called_once()
    
    @pytest.mark.parametrize("scheduler_type", ["linear", "cosine", "constant"])
    def test_scheduler_types(self, scheduler_type):
        """Test different learning rate schedulers."""
        config = MagicMock(scheduler=scheduler_type)
        
        with patch('src.training.trainer.load_pretrained_model_with_moe'):
            with patch('src.training.trainer.AutoTokenizer.from_pretrained'):
                trainer = ModelTrainer(
                    model_name="test",
                    output_dir="./test",
                    config=config
                )
                
                scheduler = trainer.get_scheduler(
                    scheduler_type,
                    MagicMock(),  # optimizer
                    num_training_steps=1000
                )
                
                assert scheduler is not None


def test_error_handling_in_training():
    """Test error handling during training."""
    with patch('src.training.trainer.load_pretrained_model_with_moe', side_effect=Exception("Model load failed")):
        with pytest.raises(Exception) as exc_info:
            ModelTrainer(
                model_name="nonexistent",
                output_dir="./test"
            )
        assert "Model load failed" in str(exc_info.value)


def test_distributed_training_setup():
    """Test distributed training setup."""
    with patch('src.training.trainer.load_pretrained_model_with_moe'):
        with patch('src.training.trainer.AutoTokenizer.from_pretrained'):
            with patch('accelerate.Accelerator') as mock_accelerator:
                trainer = ModelTrainer(
                    model_name="test",
                    output_dir="./test",
                    use_accelerate=True
                )
                
                mock_accelerator.assert_called_once()
                assert hasattr(trainer, 'accelerator')