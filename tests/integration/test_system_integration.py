"""Integration tests for the complete CleoAI system."""
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from config import model_config, training_config, memory_config
from src.model.moe_model import MoEModel, MoEConfig
from src.memory.memory_manager import MemoryManager, Conversation
from src.training.trainer import ModelTrainer
from src.inference.inference_engine import InferenceEngine


@pytest.mark.integration
class TestSystemIntegration:
    """Integration tests for end-to-end system functionality."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectories
            base_path = Path(tmpdir)
            (base_path / "models").mkdir()
            (base_path / "data").mkdir()
            (base_path / "outputs").mkdir()
            (base_path / "logs").mkdir()
            yield base_path
    
    @pytest.fixture
    def mock_model(self):
        """Create a small mock model for testing."""
        config = MoEConfig(
            num_experts=4,
            num_experts_per_token=2,
            hidden_size=128,
            intermediate_size=512
        )
        return MoEModel(config)
    
    @pytest.mark.slow
    def test_training_to_inference_pipeline(self, temp_project_dir, mock_model):
        """Test the complete pipeline from training to inference."""
        # Mock the model loading
        with patch('src.model.moe_model.load_pretrained_model_with_moe') as mock_load:
            mock_load.return_value = mock_model
            
            # 1. Initialize trainer
            trainer = ModelTrainer(
                model_name="mock-model",
                output_dir=str(temp_project_dir / "models"),
                config=training_config,
                use_moe=True,
                num_experts=4,
                device="cpu",
                use_deepspeed=False,
                log_to_wandb=False
            )
            
            # 2. Mock training data
            with patch.object(trainer, 'prepare_datasets') as mock_datasets:
                mock_datasets.return_value = {
                    "train": MagicMock(spec=['__len__', '__getitem__']),
                    "validation": MagicMock(spec=['__len__', '__getitem__'])
                }
                
                # 3. Run a minimal training step
                with patch.object(trainer, 'train_phase') as mock_train:
                    mock_train.return_value = {
                        "train_loss": 0.5,
                        "eval_loss": 0.6,
                        "checkpoint_path": str(temp_project_dir / "models" / "checkpoint")
                    }
                    
                    # Create mock checkpoint
                    checkpoint_dir = temp_project_dir / "models" / "checkpoint"
                    checkpoint_dir.mkdir(parents=True)
                    
                    # Save mock model state
                    torch.save({
                        "model_state_dict": mock_model.state_dict(),
                        "config": {"num_experts": 4}
                    }, checkpoint_dir / "pytorch_model.bin")
                    
                    # Save config
                    with open(checkpoint_dir / "config.json", "w") as f:
                        json.dump({"model_type": "moe", "num_experts": 4}, f)
                    
                    # 4. Initialize inference engine
                    with patch('src.inference.inference_engine.load_model') as mock_inf_load:
                        mock_inf_load.return_value = (mock_model, MagicMock())
                        
                        engine = InferenceEngine(
                            model_path=str(checkpoint_dir),
                            use_moe=True,
                            device="cpu"
                        )
                        
                        # 5. Test inference
                        with patch.object(engine, 'generate_response') as mock_generate:
                            mock_generate.return_value = "Hello! How can I help you?"
                            
                            response = engine.generate_response(
                                user_id="test_user",
                                conversation_id="test_conv",
                                query="Hello"
                            )
                            
                            assert response == "Hello! How can I help you?"
                            mock_generate.assert_called_once()
    
    def test_memory_persistence_across_sessions(self, temp_project_dir):
        """Test that memory persists across different sessions."""
        memory_path = temp_project_dir / "memory"
        
        # Session 1: Create and save memory
        with patch('chromadb.PersistentClient'):
            manager1 = MemoryManager(base_path=memory_path)
            
            # Add conversation
            conv = Conversation("conv1", "user1")
            conv.add_message("user", "Remember that I like pizza")
            conv.add_message("assistant", "I'll remember that you like pizza!")
            
            manager1.short_term_memory.add_conversation(conv)
            
            # Save state
            manager1.save_state()
        
        # Session 2: Load memory in new instance
        with patch('chromadb.PersistentClient'):
            manager2 = MemoryManager(base_path=memory_path)
            manager2.load_state()
            
            # Verify conversation was loaded
            loaded_conv = manager2.short_term_memory.get_conversation("conv1")
            assert loaded_conv is not None
            assert len(loaded_conv.messages) == 2
            assert "pizza" in loaded_conv.messages[0]["content"]
    
    def test_error_recovery_mechanisms(self, temp_project_dir):
        """Test system's ability to recover from errors."""
        # Test config error handling
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("No permission")):
            from config import ProjectConfig
            config = ProjectConfig()
            
            with pytest.raises(RuntimeError) as exc_info:
                config.create_directories()
            
            assert "Failed to create project directories" in str(exc_info.value)
    
    @pytest.mark.parametrize("component", ["model", "memory", "inference"])
    def test_component_initialization_errors(self, component, temp_project_dir):
        """Test error handling during component initialization."""
        if component == "model":
            with patch('transformers.AutoModel.from_pretrained', side_effect=Exception("Model not found")):
                with pytest.raises(Exception) as exc_info:
                    from src.model.moe_model import load_pretrained_model_with_moe
                    load_pretrained_model_with_moe("nonexistent-model")
                assert "Model not found" in str(exc_info.value)
        
        elif component == "memory":
            with patch('chromadb.PersistentClient', side_effect=Exception("DB connection failed")):
                with pytest.raises(Exception) as exc_info:
                    MemoryManager(base_path=temp_project_dir)
                assert "DB connection failed" in str(exc_info.value)
        
        elif component == "inference":
            with patch('src.inference.inference_engine.load_model', side_effect=Exception("Load failed")):
                with pytest.raises(Exception) as exc_info:
                    InferenceEngine(model_path="fake_path", device="cpu")
                assert "Load failed" in str(exc_info.value)
    
    def test_concurrent_access(self, temp_project_dir):
        """Test system behavior under concurrent access."""
        import threading
        import time
        
        results = []
        errors = []
        
        def access_memory(user_id, num_messages):
            try:
                with patch('chromadb.PersistentClient'):
                    manager = MemoryManager(base_path=temp_project_dir)
                    
                    for i in range(num_messages):
                        conv = Conversation(f"conv_{user_id}_{i}", user_id)
                        conv.add_message("user", f"Message {i}")
                        manager.short_term_memory.add_conversation(conv)
                    
                    results.append(f"User {user_id} completed")
            except Exception as e:
                errors.append(f"User {user_id} error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=access_memory, args=(f"user{i}", 10))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join(timeout=10)
        
        # Verify results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 5, "Not all threads completed"
    
    def test_resource_cleanup(self, temp_project_dir):
        """Test that resources are properly cleaned up."""
        import gc
        import weakref
        
        # Create components
        with patch('chromadb.PersistentClient'):
            manager = MemoryManager(base_path=temp_project_dir)
            weak_ref = weakref.ref(manager)
            
            # Use the manager
            conv = Conversation("test", "user1")
            manager.short_term_memory.add_conversation(conv)
            
            # Delete reference
            del manager
            
            # Force garbage collection
            gc.collect()
            
            # Check that object was cleaned up
            assert weak_ref() is None, "Memory manager not properly cleaned up"


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndScenarios:
    """Test complete user scenarios."""
    
    def test_conversation_flow_with_memory(self, temp_project_dir):
        """Test a complete conversation flow with memory retention."""
        with patch('chromadb.PersistentClient'):
            # Initialize system components
            memory_manager = MemoryManager(base_path=temp_project_dir)
            
            # Simulate conversation
            user_id = "test_user"
            conv_id = "test_conversation"
            
            # Message 1
            memory_manager.process_interaction(
                user_id=user_id,
                conversation_id=conv_id,
                user_message="Hi! My name is John.",
                assistant_response="Hello John! Nice to meet you."
            )
            
            # Message 2
            memory_manager.process_interaction(
                user_id=user_id,
                conversation_id=conv_id,
                user_message="What's my name?",
                assistant_response="Your name is John."
            )
            
            # Get context for next message
            context = memory_manager.get_context(user_id, conv_id)
            
            # Verify context contains conversation history
            assert "conversation_history" in context
            assert len(context["conversation_history"]) >= 4  # 2 exchanges
            
            # Verify name is in history
            history_text = " ".join([
                msg["content"] for msg in context["conversation_history"]
            ])
            assert "John" in history_text
    
    def test_model_switching(self, temp_project_dir):
        """Test switching between different models."""
        model_paths = []
        
        # Create multiple mock models
        for i in range(2):
            model_dir = temp_project_dir / f"model_{i}"
            model_dir.mkdir()
            
            # Save mock config
            with open(model_dir / "config.json", "w") as f:
                json.dump({
                    "model_type": "moe",
                    "num_experts": 4 + i * 2,
                    "model_id": f"model_{i}"
                }, f)
            
            model_paths.append(str(model_dir))
        
        # Test loading different models
        for path in model_paths:
            with patch('src.inference.inference_engine.load_model') as mock_load:
                mock_load.return_value = (MagicMock(), MagicMock())
                
                engine = InferenceEngine(
                    model_path=path,
                    device="cpu"
                )
                
                # Verify model was loaded
                mock_load.assert_called_once()
                assert engine.model_path == path