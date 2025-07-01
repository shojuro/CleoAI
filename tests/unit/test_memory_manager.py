"""Unit tests for memory manager module."""
import os
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import uuid

import pytest
import numpy as np
import torch

from src.memory.memory_manager import (
    Conversation,
    UserPreference,
    MemoryManager,
    ShortTermMemory,
    LongTermMemory,
    EpisodicMemory,
    ProceduralMemory,
)


class TestConversation:
    """Test cases for Conversation dataclass."""
    
    def test_conversation_creation(self):
        """Test creating a conversation."""
        conv_id = "test_conv_123"
        user_id = "test_user_456"
        
        conv = Conversation(
            conversation_id=conv_id,
            user_id=user_id
        )
        
        assert conv.conversation_id == conv_id
        assert conv.user_id == user_id
        assert isinstance(conv.created_at, float)
        assert isinstance(conv.updated_at, float)
        assert conv.messages == []
        assert conv.metadata == {}
    
    def test_add_message(self):
        """Test adding messages to conversation."""
        conv = Conversation("conv1", "user1")
        
        # Add a message
        conv.add_message("user", "Hello!")
        
        assert len(conv.messages) == 1
        assert conv.messages[0]["role"] == "user"
        assert conv.messages[0]["content"] == "Hello!"
        assert "message_id" in conv.messages[0]
        assert "timestamp" in conv.messages[0]
        
        # Add another message with custom timestamp and id
        custom_time = time.time()
        custom_id = "custom_msg_id"
        conv.add_message("assistant", "Hi there!", timestamp=custom_time, message_id=custom_id)
        
        assert len(conv.messages) == 2
        assert conv.messages[1]["message_id"] == custom_id
        assert conv.messages[1]["timestamp"] == custom_time
    
    def test_get_messages(self):
        """Test retrieving messages from conversation."""
        conv = Conversation("conv1", "user1")
        
        # Add multiple messages
        for i in range(5):
            conv.add_message("user", f"Message {i}")
        
        # Get all messages
        all_messages = conv.get_messages()
        assert len(all_messages) == 5
        
        # Get limited messages
        limited_messages = conv.get_messages(limit=3)
        assert len(limited_messages) == 3
        assert limited_messages[0]["content"] == "Message 2"
        assert limited_messages[-1]["content"] == "Message 4"
    
    def test_conversation_serialization(self):
        """Test converting conversation to/from dict."""
        conv = Conversation("conv1", "user1")
        conv.add_message("user", "Test message")
        conv.metadata["key"] = "value"
        
        # Convert to dict
        conv_dict = conv.to_dict()
        assert conv_dict["conversation_id"] == "conv1"
        assert conv_dict["user_id"] == "user1"
        assert len(conv_dict["messages"]) == 1
        assert conv_dict["metadata"]["key"] == "value"
        
        # Create from dict
        conv2 = Conversation.from_dict(conv_dict)
        assert conv2.conversation_id == conv.conversation_id
        assert conv2.user_id == conv.user_id
        assert len(conv2.messages) == len(conv.messages)
        assert conv2.metadata == conv.metadata


class TestUserPreference:
    """Test cases for UserPreference dataclass."""
    
    def test_user_preference_creation(self):
        """Test creating a user preference."""
        pref = UserPreference(
            user_id="user1",
            preference_id="pref1",
            category="communication",
            key="response_style",
            value="friendly",
            confidence=0.8
        )
        
        assert pref.user_id == "user1"
        assert pref.preference_id == "pref1"
        assert pref.category == "communication"
        assert pref.key == "response_style"
        assert pref.value == "friendly"
        assert pref.confidence == 0.8
        assert isinstance(pref.created_at, float)
        assert isinstance(pref.updated_at, float)
    
    def test_preference_serialization(self):
        """Test preference serialization."""
        pref = UserPreference(
            user_id="user1",
            preference_id="pref1",
            category="test",
            key="test_key",
            value={"nested": "value"},
            confidence=0.9
        )
        
        pref_dict = pref.to_dict()
        assert pref_dict["user_id"] == "user1"
        assert pref_dict["value"]["nested"] == "value"
        
        pref2 = UserPreference.from_dict(pref_dict)
        assert pref2.user_id == pref.user_id
        assert pref2.value == pref.value


class TestShortTermMemory:
    """Test cases for ShortTermMemory."""
    
    @pytest.fixture
    def memory(self, temp_dir):
        """Create a ShortTermMemory instance."""
        return ShortTermMemory(
            max_tokens=100,
            recency_weight_decay=0.95,
            storage_path=temp_dir / "short_term"
        )
    
    def test_initialization(self, memory):
        """Test ShortTermMemory initialization."""
        assert memory.max_tokens == 100
        assert memory.recency_weight_decay == 0.95
        assert memory.conversations == {}
        assert memory.storage_path.exists()
    
    def test_add_conversation(self, memory):
        """Test adding a conversation to short-term memory."""
        conv = Conversation("conv1", "user1")
        conv.add_message("user", "Hello")
        
        memory.add_conversation(conv)
        
        assert "conv1" in memory.conversations
        assert memory.conversations["conv1"].conversation_id == "conv1"
    
    def test_get_conversation(self, memory):
        """Test retrieving a conversation."""
        conv = Conversation("conv1", "user1")
        memory.add_conversation(conv)
        
        retrieved = memory.get_conversation("conv1")
        assert retrieved is not None
        assert retrieved.conversation_id == "conv1"
        
        # Test non-existent conversation
        assert memory.get_conversation("nonexistent") is None
    
    def test_get_recent_context(self, memory):
        """Test getting recent context with token limit."""
        conv = Conversation("conv1", "user1")
        
        # Add multiple messages
        for i in range(10):
            conv.add_message("user", f"Message {i} " * 10)  # Each message ~10 tokens
        
        memory.add_conversation(conv)
        
        # Mock tokenizer
        with patch.object(memory, 'tokenizer') as mock_tokenizer:
            mock_tokenizer.encode.return_value = list(range(10))  # 10 tokens per message
            
            context = memory.get_recent_context("conv1", max_tokens=50)
            
            # Should return recent messages that fit in token limit
            assert len(context) <= 5  # 50 tokens / 10 tokens per message
    
    def test_clear_old_conversations(self, memory):
        """Test clearing old conversations."""
        # Add conversations with different timestamps
        old_conv = Conversation("old", "user1")
        old_conv.updated_at = time.time() - 86400 * 8  # 8 days old
        
        recent_conv = Conversation("recent", "user1")
        recent_conv.updated_at = time.time() - 3600  # 1 hour old
        
        memory.add_conversation(old_conv)
        memory.add_conversation(recent_conv)
        
        # Clear conversations older than 7 days
        memory.clear_old_conversations(max_age_days=7)
        
        assert "old" not in memory.conversations
        assert "recent" in memory.conversations
    
    def test_save_and_load(self, memory, temp_dir):
        """Test saving and loading memory state."""
        conv = Conversation("conv1", "user1")
        conv.add_message("user", "Test message")
        memory.add_conversation(conv)
        
        # Save state
        memory.save_state()
        
        # Create new memory instance and load state
        new_memory = ShortTermMemory(
            max_tokens=100,
            storage_path=temp_dir / "short_term"
        )
        new_memory.load_state()
        
        assert "conv1" in new_memory.conversations
        assert len(new_memory.conversations["conv1"].messages) == 1


class TestLongTermMemory:
    """Test cases for LongTermMemory."""
    
    @pytest.fixture
    def memory(self, temp_dir):
        """Create a LongTermMemory instance."""
        with patch('chromadb.PersistentClient'):
            return LongTermMemory(
                storage_type="vector",
                storage_path=temp_dir / "long_term"
            )
    
    def test_initialization(self, memory):
        """Test LongTermMemory initialization."""
        assert memory.storage_type == "vector"
        assert memory.storage_path.exists()
        assert memory.user_preferences == {}
    
    def test_store_user_preference(self, memory):
        """Test storing user preferences."""
        pref = UserPreference(
            user_id="user1",
            preference_id="pref1",
            category="style",
            key="tone",
            value="casual",
            confidence=0.85
        )
        
        memory.store_user_preference(pref)
        
        assert "user1" in memory.user_preferences
        assert "style" in memory.user_preferences["user1"]
        assert "tone" in memory.user_preferences["user1"]["style"]
    
    def test_get_user_preferences(self, memory):
        """Test retrieving user preferences."""
        # Store multiple preferences
        prefs = [
            UserPreference("user1", "p1", "style", "tone", "casual", 0.8),
            UserPreference("user1", "p2", "style", "length", "brief", 0.7),
            UserPreference("user1", "p3", "topics", "interests", ["tech", "sports"], 0.9),
        ]
        
        for pref in prefs:
            memory.store_user_preference(pref)
        
        # Get all preferences
        all_prefs = memory.get_user_preferences("user1")
        assert len(all_prefs) == 2  # 2 categories
        
        # Get preferences by category
        style_prefs = memory.get_user_preferences("user1", category="style")
        assert len(style_prefs) == 1
        assert "tone" in style_prefs["style"]
        assert "length" in style_prefs["style"]
    
    @patch('chromadb.Collection')
    def test_semantic_search(self, mock_collection, memory):
        """Test semantic search in vector database."""
        memory.vector_collection = mock_collection
        
        # Mock query results
        mock_results = {
            'ids': [['id1', 'id2']],
            'distances': [[0.1, 0.2]],
            'metadatas': [[
                {'user_id': 'user1', 'type': 'preference'},
                {'user_id': 'user1', 'type': 'memory'}
            ]],
            'documents': [['doc1', 'doc2']]
        }
        mock_collection.query.return_value = mock_results
        
        results = memory.semantic_search("test query", user_id="user1", top_k=2)
        
        assert len(results) == 2
        mock_collection.query.assert_called_once()


class TestEpisodicMemory:
    """Test cases for EpisodicMemory."""
    
    @pytest.fixture
    def memory(self, temp_dir):
        """Create an EpisodicMemory instance."""
        return EpisodicMemory(
            memory_type="hybrid",
            storage_path=temp_dir / "episodic"
        )
    
    def test_store_episode(self, memory):
        """Test storing an episode."""
        episode = {
            "episode_id": "ep1",
            "user_id": "user1",
            "type": "conversation",
            "content": "Had a great conversation about hobbies",
            "timestamp": time.time(),
            "metadata": {"topic": "hobbies", "sentiment": "positive"}
        }
        
        memory.store_episode(episode)
        
        assert "user1" in memory.episodes
        assert len(memory.episodes["user1"]) == 1
        assert memory.episodes["user1"][0]["episode_id"] == "ep1"
    
    def test_get_related_episodes(self, memory):
        """Test retrieving related episodes."""
        # Store multiple episodes
        episodes = [
            {
                "episode_id": "ep1",
                "user_id": "user1",
                "type": "conversation",
                "content": "Discussed favorite movies",
                "timestamp": time.time(),
                "metadata": {"topic": "movies"}
            },
            {
                "episode_id": "ep2",
                "user_id": "user1",
                "type": "conversation",
                "content": "Talked about recent films",
                "timestamp": time.time(),
                "metadata": {"topic": "movies"}
            },
            {
                "episode_id": "ep3",
                "user_id": "user1",
                "type": "conversation",
                "content": "Discussed cooking recipes",
                "timestamp": time.time(),
                "metadata": {"topic": "food"}
            }
        ]
        
        for ep in episodes:
            memory.store_episode(ep)
        
        # Get related episodes
        related = memory.get_related_episodes(
            "user1",
            current_context="What movies do you like?",
            top_k=2
        )
        
        # Should prioritize movie-related episodes
        assert len(related) <= 2
    
    def test_consolidate_episodes(self, memory):
        """Test episode consolidation."""
        # Add many episodes
        for i in range(20):
            episode = {
                "episode_id": f"ep{i}",
                "user_id": "user1",
                "type": "conversation",
                "content": f"Episode {i}",
                "timestamp": time.time() - i * 3600,  # Each hour older
                "metadata": {}
            }
            memory.store_episode(episode)
        
        # Consolidate old episodes
        memory.consolidate_episodes("user1", max_episodes=10)
        
        assert len(memory.episodes["user1"]) <= 10
        # Should keep recent episodes
        assert memory.episodes["user1"][0]["episode_id"] == "ep0"


class TestProceduralMemory:
    """Test cases for ProceduralMemory."""
    
    @pytest.fixture
    def memory(self, temp_dir):
        """Create a ProceduralMemory instance."""
        return ProceduralMemory(
            storage_path=temp_dir / "procedural"
        )
    
    def test_store_procedure(self, memory):
        """Test storing a procedure."""
        procedure = {
            "procedure_id": "proc1",
            "name": "schedule_meeting",
            "steps": [
                {"action": "check_calendar", "params": {"user_id": "user1"}},
                {"action": "find_available_slot", "params": {"duration": 60}},
                {"action": "send_invitation", "params": {}}
            ],
            "success_rate": 0.95,
            "usage_count": 10
        }
        
        memory.store_procedure(procedure)
        
        assert "schedule_meeting" in memory.procedures
        assert memory.procedures["schedule_meeting"]["procedure_id"] == "proc1"
    
    def test_get_procedure(self, memory):
        """Test retrieving a procedure."""
        procedure = {
            "procedure_id": "proc1",
            "name": "test_procedure",
            "steps": [{"action": "test"}],
            "success_rate": 1.0
        }
        
        memory.store_procedure(procedure)
        
        retrieved = memory.get_procedure("test_procedure")
        assert retrieved is not None
        assert retrieved["name"] == "test_procedure"
        
        # Test non-existent procedure
        assert memory.get_procedure("nonexistent") is None
    
    def test_update_procedure_stats(self, memory):
        """Test updating procedure statistics."""
        procedure = {
            "procedure_id": "proc1",
            "name": "test_proc",
            "steps": [],
            "success_rate": 0.8,
            "usage_count": 10
        }
        
        memory.store_procedure(procedure)
        
        # Update with success
        memory.update_procedure_stats("test_proc", success=True)
        proc = memory.get_procedure("test_proc")
        
        assert proc["usage_count"] == 11
        assert proc["success_rate"] > 0.8  # Should increase
        
        # Update with failure
        memory.update_procedure_stats("test_proc", success=False)
        proc = memory.get_procedure("test_proc")
        
        assert proc["usage_count"] == 12
        assert proc["success_rate"] < memory.procedures["test_proc"]["success_rate"]


class TestMemoryManager:
    """Test cases for the main MemoryManager class."""
    
    @pytest.fixture
    def manager(self, temp_dir):
        """Create a MemoryManager instance."""
        with patch('chromadb.PersistentClient'):
            return MemoryManager(base_path=temp_dir)
    
    def test_initialization(self, manager):
        """Test MemoryManager initialization."""
        assert manager.short_term_memory is not None
        assert manager.long_term_memory is not None
        assert manager.episodic_memory is not None
        assert manager.procedural_memory is not None
    
    def test_process_interaction(self, manager):
        """Test processing a user interaction."""
        user_id = "user1"
        conversation_id = "conv1"
        user_message = "I love watching sci-fi movies"
        assistant_response = "That's great! What's your favorite sci-fi movie?"
        
        # Process the interaction
        manager.process_interaction(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_response=assistant_response
        )
        
        # Check that conversation was stored
        conv = manager.short_term_memory.get_conversation(conversation_id)
        assert conv is not None
        assert len(conv.messages) == 2
        
        # Check that preferences might be extracted
        # (actual extraction depends on implementation)
    
    def test_get_context(self, manager):
        """Test getting context for a conversation."""
        user_id = "user1"
        conversation_id = "conv1"
        
        # Add some conversation history
        conv = Conversation(conversation_id, user_id)
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there!")
        manager.short_term_memory.add_conversation(conv)
        
        # Get context
        context = manager.get_context(user_id, conversation_id)
        
        assert "conversation_history" in context
        assert "user_preferences" in context
        assert "related_episodes" in context
    
    def test_save_and_load_state(self, manager, temp_dir):
        """Test saving and loading complete memory state."""
        # Add some data
        conv = Conversation("conv1", "user1")
        conv.add_message("user", "Test")
        manager.short_term_memory.add_conversation(conv)
        
        # Save state
        manager.save_state()
        
        # Create new manager and load state
        with patch('chromadb.PersistentClient'):
            new_manager = MemoryManager(base_path=temp_dir)
            new_manager.load_state()
        
        # Verify data was loaded
        loaded_conv = new_manager.short_term_memory.get_conversation("conv1")
        assert loaded_conv is not None
        assert len(loaded_conv.messages) == 1
    
    def test_clear_user_data(self, manager):
        """Test clearing all data for a user."""
        user_id = "user1"
        
        # Add data for the user
        conv = Conversation("conv1", user_id)
        manager.short_term_memory.add_conversation(conv)
        
        pref = UserPreference(
            user_id=user_id,
            preference_id="pref1",
            category="test",
            key="test",
            value="test"
        )
        manager.long_term_memory.store_user_preference(pref)
        
        # Clear user data
        manager.clear_user_data(user_id)
        
        # Verify data was cleared
        assert manager.long_term_memory.get_user_preferences(user_id) == {}


@pytest.mark.integration
class TestMemoryIntegration:
    """Integration tests for memory components."""
    
    def test_full_conversation_flow(self, temp_dir):
        """Test a complete conversation flow with all memory types."""
        with patch('chromadb.PersistentClient'):
            manager = MemoryManager(base_path=temp_dir)
            
            user_id = "test_user"
            conv_id = "test_conv"
            
            # Simulate a conversation
            messages = [
                ("I prefer casual conversations", "I'll keep that in mind! I'll maintain a casual tone."),
                ("What movies do you recommend?", "Based on our casual vibe, how about some fun action movies?"),
                ("I actually prefer sci-fi", "Oh, sci-fi is great! Let me adjust my recommendations."),
            ]
            
            for user_msg, assistant_msg in messages:
                manager.process_interaction(
                    user_id=user_id,
                    conversation_id=conv_id,
                    user_message=user_msg,
                    assistant_response=assistant_msg
                )
            
            # Get context for next interaction
            context = manager.get_context(user_id, conv_id)
            
            # Verify context contains relevant information
            assert len(context["conversation_history"]) > 0
            assert context["conversation_history"][-1]["content"] == "Oh, sci-fi is great! Let me adjust my recommendations."