"""Unit tests for inference engine module."""
import os
import json
import torch
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, call
from typing import Dict, List, Any

import pytest

from src.inference.inference_engine import (
    InferenceEngine,
    StreamingCallback,
    ConversationContext,
    ResponseGenerator,
    create_inference_engine,
    create_interactive_session,
)


class TestInferenceEngine:
    """Test cases for InferenceEngine class."""
    
    @pytest.fixture
    def engine_config(self, temp_dir):
        """Create inference engine configuration."""
        return {
            "model_path": str(temp_dir / "model"),
            "use_moe": True,
            "device": "cpu",
            "precision": "fp32",
            "max_length": 2048,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
    
    @pytest.fixture
    def mock_model_files(self, temp_dir):
        """Create mock model files."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        # Create config files
        config = {
            "model_type": "gpt2",
            "vocab_size": 50257,
            "n_positions": 1024,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        # Create MoE config
        moe_config = {
            "num_experts": 4,
            "num_experts_per_token": 2,
        }
        with open(model_dir / "moe_config.json", "w") as f:
            json.dump(moe_config, f)
        
        return model_dir
    
    @patch('src.inference.inference_engine.AutoTokenizer.from_pretrained')
    @patch('src.inference.inference_engine.MoEModel.from_pretrained')
    @patch('src.inference.inference_engine.MemoryManager')
    def test_initialization(self, mock_memory, mock_model, mock_tokenizer, engine_config):
        """Test InferenceEngine initialization."""
        # Setup mocks
        mock_tokenizer.return_value = MagicMock(pad_token=None, eos_token="</s>")
        mock_model.return_value = MagicMock()
        mock_memory.return_value = MagicMock()
        
        # Create engine
        engine = InferenceEngine(**engine_config)
        
        # Verify initialization
        assert engine.model_path == engine_config["model_path"]
        assert engine.use_moe is True
        assert engine.device == "cpu"
        assert engine.temperature == 0.7
        assert engine.active_conversations == {}
        
        # Verify model and tokenizer loaded
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
        mock_memory.assert_called_once()
    
    @patch('src.inference.inference_engine.AutoTokenizer.from_pretrained')
    @patch('src.inference.inference_engine.AutoModelForCausalLM.from_pretrained')
    def test_initialization_without_moe(self, mock_model, mock_tokenizer, engine_config):
        """Test initialization without MoE."""
        engine_config["use_moe"] = False
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        with patch('src.inference.inference_engine.MemoryManager'):
            engine = InferenceEngine(**engine_config)
        
        assert engine.use_moe is False
        mock_model.assert_called_once()  # Regular model loaded
    
    @patch('src.inference.inference_engine.AutoTokenizer.from_pretrained')
    @patch('src.inference.inference_engine.MoEModel.from_pretrained')
    @patch('src.inference.inference_engine.MemoryManager')
    def test_generate_response(self, mock_memory, mock_model, mock_tokenizer, engine_config):
        """Test response generation."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.encode.return_value = torch.tensor([1, 2, 3])
        mock_tokenizer_instance.decode.return_value = "Generated response"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.return_value = mock_model_instance
        
        mock_memory.return_value = MagicMock()
        
        # Create engine and generate response
        engine = InferenceEngine(**engine_config)
        response = engine.generate_response(
            user_id="test_user",
            conversation_id="test_conv",
            query="Hello"
        )
        
        # Verify response
        assert response == "Generated response"
        mock_model_instance.generate.assert_called_once()
        
        # Verify conversation tracking
        assert "test_conv" in engine.active_conversations
    
    @patch('src.inference.inference_engine.AutoTokenizer.from_pretrained')
    @patch('src.inference.inference_engine.MoEModel.from_pretrained')
    @patch('src.inference.inference_engine.MemoryManager')
    def test_stream_response(self, mock_memory, mock_model, mock_tokenizer, engine_config):
        """Test streaming response generation."""
        # Setup mocks
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_memory.return_value = MagicMock()
        
        # Create engine
        engine = InferenceEngine(**engine_config)
        
        # Mock streaming
        tokens = ["Hello", " ", "world", "!"]
        collected_tokens = []
        
        def callback(token):
            collected_tokens.append(token)
        
        with patch.object(engine, '_generate_streaming') as mock_stream:
            mock_stream.return_value = tokens
            
            result = engine.stream_response(
                user_id="test_user",
                conversation_id="test_conv",
                query="Hi",
                callback=callback
            )
            
            assert result == "Hello world!"
            assert collected_tokens == tokens
    
    @patch('src.inference.inference_engine.AutoTokenizer.from_pretrained')
    @patch('src.inference.inference_engine.MoEModel.from_pretrained')
    @patch('src.inference.inference_engine.MemoryManager')
    def test_build_context(self, mock_memory, mock_model, mock_tokenizer, engine_config):
        """Test context building."""
        # Setup mocks
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        mock_memory_instance = MagicMock()
        mock_memory_instance.get_context.return_value = {
            "conversation_history": [
                {"role": "user", "content": "Previous message"},
                {"role": "assistant", "content": "Previous response"}
            ],
            "user_preferences": {"style": "casual"},
            "related_episodes": []
        }
        mock_memory.return_value = mock_memory_instance
        
        # Create engine and build context
        engine = InferenceEngine(**engine_config)
        context = engine.build_context("test_user", "test_conv", "New message")
        
        # Verify context structure
        assert "system_prompt" in context
        assert "conversation_history" in context
        assert "current_query" in context
        assert context["current_query"] == "New message"
        
        # Verify memory was queried
        mock_memory_instance.get_context.assert_called_once_with("test_user", "test_conv")
    
    def test_format_prompt(self):
        """Test prompt formatting."""
        engine = InferenceEngine.__new__(InferenceEngine)
        
        context = {
            "system_prompt": "You are a helpful assistant.",
            "conversation_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "current_query": "How are you?"
        }
        
        prompt = engine.format_prompt(context)
        
        assert "You are a helpful assistant" in prompt
        assert "Hello" in prompt
        assert "Hi there!" in prompt
        assert "How are you?" in prompt
    
    @patch('src.inference.inference_engine.AutoTokenizer.from_pretrained')
    @patch('src.inference.inference_engine.MoEModel.from_pretrained')
    @patch('src.inference.inference_engine.MemoryManager')
    def test_clear_conversation(self, mock_memory, mock_model, mock_tokenizer, engine_config):
        """Test clearing conversation history."""
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_memory.return_value = MagicMock()
        
        engine = InferenceEngine(**engine_config)
        
        # Add a conversation
        engine.active_conversations["test_conv"] = {"messages": []}
        
        # Clear it
        engine.clear_conversation("test_conv")
        
        assert "test_conv" not in engine.active_conversations
    
    @patch('src.inference.inference_engine.AutoTokenizer.from_pretrained')
    @patch('src.inference.inference_engine.MoEModel.from_pretrained')
    @patch('src.inference.inference_engine.MemoryManager')
    def test_save_and_load_state(self, mock_memory, mock_model, mock_tokenizer, engine_config, temp_dir):
        """Test saving and loading engine state."""
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        
        engine = InferenceEngine(**engine_config)
        
        # Add some state
        engine.active_conversations["conv1"] = {"messages": ["test"]}
        
        # Save state
        state_path = temp_dir / "engine_state"
        engine.save_state(str(state_path))
        
        # Verify memory manager save was called
        mock_memory_instance.save_state.assert_called_once()
        
        # Load state in new engine
        engine2 = InferenceEngine(**engine_config)
        engine2.load_state(str(state_path))
        
        # Verify memory manager load was called
        mock_memory_instance.load_state.assert_called()


class TestStreamingCallback:
    """Test cases for StreamingCallback class."""
    
    def test_streaming_callback_initialization(self):
        """Test StreamingCallback initialization."""
        callback = StreamingCallback()
        assert callback.tokens == []
        assert callback.finished is False
    
    def test_streaming_callback_on_token(self):
        """Test token collection."""
        callback = StreamingCallback()
        
        callback.on_token("Hello")
        callback.on_token(" ")
        callback.on_token("world")
        
        assert callback.tokens == ["Hello", " ", "world"]
        assert callback.get_text() == "Hello world"
    
    def test_streaming_callback_finish(self):
        """Test finishing callback."""
        callback = StreamingCallback()
        callback.on_token("Test")
        callback.finish()
        
        assert callback.finished is True
        assert callback.get_text() == "Test"


class TestConversationContext:
    """Test cases for ConversationContext class."""
    
    def test_conversation_context_initialization(self):
        """Test ConversationContext initialization."""
        context = ConversationContext(
            user_id="user1",
            conversation_id="conv1",
            system_prompt="Be helpful"
        )
        
        assert context.user_id == "user1"
        assert context.conversation_id == "conv1"
        assert context.system_prompt == "Be helpful"
        assert context.messages == []
        assert context.metadata == {}
    
    def test_add_message(self):
        """Test adding messages to context."""
        context = ConversationContext("user1", "conv1")
        
        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi there!")
        
        assert len(context.messages) == 2
        assert context.messages[0]["role"] == "user"
        assert context.messages[1]["content"] == "Hi there!"
    
    def test_get_formatted_history(self):
        """Test getting formatted conversation history."""
        context = ConversationContext("user1", "conv1")
        context.add_message("user", "Question?")
        context.add_message("assistant", "Answer!")
        
        history = context.get_formatted_history()
        
        assert "User: Question?" in history
        assert "Assistant: Answer!" in history
    
    def test_truncate_history(self):
        """Test conversation history truncation."""
        context = ConversationContext("user1", "conv1")
        
        # Add many messages
        for i in range(20):
            context.add_message("user", f"Message {i}")
        
        context.truncate_history(max_messages=10)
        
        assert len(context.messages) == 10
        assert context.messages[0]["content"] == "Message 10"
        assert context.messages[-1]["content"] == "Message 19"


class TestResponseGenerator:
    """Test cases for ResponseGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a ResponseGenerator instance."""
        model = MagicMock()
        tokenizer = MagicMock()
        return ResponseGenerator(model, tokenizer, device="cpu")
    
    def test_response_generator_initialization(self, generator):
        """Test ResponseGenerator initialization."""
        assert generator.device == "cpu"
        assert generator.model is not None
        assert generator.tokenizer is not None
    
    def test_preprocess_input(self, generator):
        """Test input preprocessing."""
        generator.tokenizer.encode.return_value = [1, 2, 3]
        generator.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        inputs = generator.preprocess_input("Test text", max_length=100)
        
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"].shape[1] <= 100
    
    def test_postprocess_output(self, generator):
        """Test output postprocessing."""
        generator.tokenizer.decode.return_value = "Generated text"
        
        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        text = generator.postprocess_output(output_ids)
        
        assert text == "Generated text"
        generator.tokenizer.decode.assert_called_once()
    
    def test_generate_with_params(self, generator):
        """Test generation with custom parameters."""
        generator.model.generate.return_value = torch.tensor([[1, 2, 3]])
        
        input_ids = torch.tensor([[1, 2]])
        params = {
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 100
        }
        
        output = generator.generate(input_ids, **params)
        
        generator.model.generate.assert_called_once()
        call_kwargs = generator.model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.95


class TestInferenceUtilities:
    """Test inference utility functions."""
    
    @patch('src.inference.inference_engine.InferenceEngine')
    def test_create_inference_engine(self, mock_engine_class):
        """Test inference engine factory function."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        
        config = {
            "temperature": 0.5,
            "max_length": 1024
        }
        
        engine = create_inference_engine(
            model_path="/path/to/model",
            device="cuda",
            config=config
        )
        
        assert engine == mock_engine
        mock_engine_class.assert_called_once()
        
        # Check config was passed
        call_kwargs = mock_engine_class.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_length"] == 1024
    
    @patch('src.inference.inference_engine.InteractiveSession')
    def test_create_interactive_session(self, mock_session_class):
        """Test interactive session creation."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        engine = MagicMock()
        session = create_interactive_session(
            engine=engine,
            user_id="test_user",
            system_prompt="Be helpful"
        )
        
        assert session == mock_session
        mock_session_class.assert_called_once_with(
            engine=engine,
            user_id="test_user",
            system_prompt="Be helpful"
        )


@pytest.mark.integration
class TestInferenceIntegration:
    """Integration tests for inference engine."""
    
    def test_end_to_end_inference(self, temp_dir):
        """Test complete inference pipeline."""
        # Create mock model files
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        with open(model_dir / "config.json", "w") as f:
            json.dump({"model_type": "gpt2"}, f)
        
        # Mock all dependencies
        with patch('src.inference.inference_engine.AutoTokenizer.from_pretrained') as mock_tok:
            with patch('src.inference.inference_engine.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('src.inference.inference_engine.MemoryManager'):
                    # Setup mocks
                    tokenizer = MagicMock()
                    tokenizer.encode.return_value = torch.tensor([1, 2, 3])
                    tokenizer.decode.return_value = "Hello, I can help you!"
                    mock_tok.return_value = tokenizer
                    
                    model = MagicMock()
                    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
                    mock_model.return_value = model
                    
                    # Create engine and run inference
                    engine = InferenceEngine(
                        model_path=str(model_dir),
                        use_moe=False,
                        device="cpu"
                    )
                    
                    response = engine.generate_response(
                        user_id="user1",
                        conversation_id="conv1",
                        query="Hello!"
                    )
                    
                    assert response == "Hello, I can help you!"
    
    def test_error_recovery(self):
        """Test error handling and recovery."""
        with patch('src.inference.inference_engine.AutoTokenizer.from_pretrained') as mock_tok:
            mock_tok.side_effect = Exception("Tokenizer load failed")
            
            with pytest.raises(Exception) as exc_info:
                InferenceEngine(model_path="/fake/path")
            
            assert "Tokenizer load failed" in str(exc_info.value)
    
    def test_memory_integration(self, temp_dir):
        """Test integration with memory system."""
        with patch('src.inference.inference_engine.AutoTokenizer.from_pretrained'):
            with patch('src.inference.inference_engine.AutoModelForCausalLM.from_pretrained'):
                # Create engine with memory
                engine = InferenceEngine(
                    model_path=str(temp_dir),
                    memory_config={"base_path": temp_dir / "memory"}
                )
                
                # Verify memory manager was created
                assert engine.memory_manager is not None
                
                # Test memory persistence
                with patch.object(engine.memory_manager, 'process_interaction') as mock_process:
                    engine.generate_response(
                        user_id="user1",
                        conversation_id="conv1",
                        query="Remember this"
                    )
                    
                    # Verify interaction was processed
                    mock_process.assert_called_once()