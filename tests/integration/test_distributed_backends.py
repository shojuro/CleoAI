"""
Integration tests for distributed memory backends.
"""
import os
import time
import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

from config import memory_backend_config
from src.memory.adapters.memory_adapter import HybridMemoryAdapter
from src.utils.health_check import HealthChecker, perform_health_check


@pytest.fixture
async def memory_adapter():
    """Create a memory adapter for testing."""
    adapter = HybridMemoryAdapter(
        legacy_manager=None,
        redis_config={
            "host": memory_backend_config.redis_host,
            "port": memory_backend_config.redis_port,
            "password": memory_backend_config.redis_password
        } if memory_backend_config.use_redis else None,
        supabase_config={
            "url": memory_backend_config.supabase_url,
            "key": memory_backend_config.supabase_anon_key
        } if memory_backend_config.use_supabase else None,
        pinecone_config={
            "api_key": memory_backend_config.pinecone_api_key,
            "environment": memory_backend_config.pinecone_environment,
            "index_name": memory_backend_config.pinecone_index_name,
            "dimension": memory_backend_config.pinecone_dimension
        } if memory_backend_config.use_pinecone else None,
        mongodb_config={
            "connection_string": memory_backend_config.mongodb_connection_string,
            "database": memory_backend_config.mongodb_database
        } if memory_backend_config.use_mongodb else None,
        use_legacy=False,
        use_distributed=True
    )
    
    yield adapter
    
    # Cleanup after tests
    # Note: Add cleanup logic here if needed


@pytest.fixture
def test_user_id():
    """Generate a unique test user ID."""
    return f"test_user_{int(time.time())}"


@pytest.fixture
def test_conversation_id():
    """Generate a unique test conversation ID."""
    return f"test_conv_{int(time.time())}"


class TestHealthChecks:
    """Test health check functionality for all backends."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self):
        """Test that health check runs without errors."""
        health_result = await perform_health_check(memory_backend_config)
        
        assert "status" in health_result
        assert "timestamp" in health_result
        assert "services" in health_result
        assert health_result["status"] in ["healthy", "degraded", "unhealthy"]
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not memory_backend_config.use_redis, reason="Redis not enabled")
    async def test_redis_health_check(self):
        """Test Redis-specific health check."""
        checker = HealthChecker(memory_backend_config)
        health = await checker._check_redis()
        
        assert health.name == "redis"
        if health.status == "healthy":
            assert health.latency_ms > 0
            assert health.details is not None
            assert "version" in health.details
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not memory_backend_config.use_mongodb, reason="MongoDB not enabled")
    async def test_mongodb_health_check(self):
        """Test MongoDB-specific health check."""
        checker = HealthChecker(memory_backend_config)
        health = await checker._check_mongodb()
        
        assert health.name == "mongodb"
        if health.status == "healthy":
            assert health.latency_ms > 0
            assert health.details is not None
            assert "version" in health.details


class TestConversationManagement:
    """Test conversation storage across backends."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not (memory_backend_config.use_redis or memory_backend_config.use_mongodb),
        reason="No distributed backends enabled"
    )
    async def test_create_and_retrieve_conversation(self, memory_adapter, test_user_id):
        """Test creating and retrieving a conversation."""
        # Create conversation
        conversation = memory_adapter.create_conversation(
            user_id=test_user_id,
            metadata={"test": True, "timestamp": time.time()}
        )
        
        assert conversation is not None
        assert conversation.user_id == test_user_id
        assert conversation.metadata["test"] is True
        
        # Add messages
        memory_adapter.add_message(
            conversation_id=conversation.conversation_id,
            user_id=test_user_id,
            role="user",
            content="Hello, this is a test message"
        )
        
        memory_adapter.add_message(
            conversation_id=conversation.conversation_id,
            user_id=test_user_id,
            role="assistant",
            content="Hello! I received your test message."
        )
        
        # Retrieve conversation
        retrieved = memory_adapter.get_conversation(conversation.conversation_id)
        
        assert retrieved is not None
        assert len(retrieved.messages) == 2
        assert retrieved.messages[0]["role"] == "user"
        assert retrieved.messages[1]["role"] == "assistant"
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not memory_backend_config.use_redis, reason="Redis not enabled")
    async def test_conversation_ttl_redis(self, memory_adapter, test_user_id):
        """Test that conversations expire in Redis according to TTL."""
        # Create conversation with short TTL
        conversation = memory_adapter.create_conversation(
            user_id=test_user_id,
            metadata={"ttl_test": True}
        )
        
        # Should exist immediately
        assert memory_adapter.get_conversation(conversation.conversation_id) is not None
        
        # Note: In real tests, you'd wait for TTL and verify expiration
        # For now, just verify it was stored in Redis
        if hasattr(memory_adapter, 'redis_service'):
            key = f"conversation:{conversation.conversation_id}"
            ttl = memory_adapter.redis_service.client.ttl(key)
            assert ttl > 0  # Has TTL set


class TestUserPreferences:
    """Test user preference storage across backends."""
    
    @pytest.mark.asyncio
    async def test_set_and_get_preferences(self, memory_adapter, test_user_id):
        """Test setting and retrieving user preferences."""
        # Set preferences
        pref1 = memory_adapter.set_user_preference(
            user_id=test_user_id,
            category="communication",
            key="language",
            value="English",
            confidence=0.9,
            source="explicit"
        )
        
        pref2 = memory_adapter.set_user_preference(
            user_id=test_user_id,
            category="behavior",
            key="response_style",
            value="concise",
            confidence=0.7,
            source="inferred"
        )
        
        # Get preferences
        language_pref = memory_adapter.get_user_preference(
            user_id=test_user_id,
            category="communication",
            key="language"
        )
        
        assert language_pref is not None
        assert language_pref.preference_value == "English"
        assert language_pref.confidence == 0.9
        
        # Get all preferences for user
        all_prefs = memory_adapter.get_user_preferences(test_user_id)
        assert len(all_prefs) >= 2
    
    @pytest.mark.asyncio
    async def test_update_preference_confidence(self, memory_adapter, test_user_id):
        """Test updating preference confidence scores."""
        # Set initial preference
        memory_adapter.set_user_preference(
            user_id=test_user_id,
            category="interests",
            key="topic",
            value="technology",
            confidence=0.5
        )
        
        # Update with higher confidence
        memory_adapter.set_user_preference(
            user_id=test_user_id,
            category="interests",
            key="topic",
            value="technology",
            confidence=0.8,
            source="repeated_interaction"
        )
        
        # Verify update
        pref = memory_adapter.get_user_preference(
            user_id=test_user_id,
            category="interests",
            key="topic"
        )
        
        assert pref.confidence == 0.8
        assert pref.source == "repeated_interaction"


class TestEpisodicMemory:
    """Test episodic memory storage with vector embeddings."""
    
    @pytest.mark.asyncio
    async def test_create_episodic_memory(self, memory_adapter, test_user_id):
        """Test creating episodic memories."""
        memory = memory_adapter.create_episodic_memory(
            user_id=test_user_id,
            title="First meeting",
            content="User expressed interest in learning Python programming",
            importance=0.8,
            emotion="excited",
            tags=["programming", "python", "learning"]
        )
        
        assert memory is not None
        assert memory.user_id == test_user_id
        assert memory.importance == 0.8
        assert "python" in memory.tags
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not (memory_backend_config.use_pinecone or memory_backend_config.use_chromadb),
        reason="No vector backend enabled"
    )
    async def test_semantic_memory_search(self, memory_adapter, test_user_id):
        """Test semantic search across memories."""
        # Create multiple memories
        memories = [
            ("Learning Python", "User wants to learn Python for data science", ["python", "learning"]),
            ("Favorite foods", "User mentioned they love Italian cuisine", ["food", "preferences"]),
            ("Work experience", "User has 5 years of software development experience", ["work", "experience"]),
        ]
        
        for title, content, tags in memories:
            memory_adapter.create_episodic_memory(
                user_id=test_user_id,
                title=title,
                content=content,
                importance=0.7,
                tags=tags
            )
        
        # Allow time for indexing
        await asyncio.sleep(2)
        
        # Search for related memories
        results = memory_adapter.search_memories(
            user_id=test_user_id,
            query_text="programming experience",
            limit=2
        )
        
        assert len(results) > 0
        # Should find work experience and possibly Python learning


class TestMemoryRouter:
    """Test memory routing and tiered storage."""
    
    @pytest.mark.asyncio
    async def test_tiered_storage_flow(self, memory_adapter, test_user_id):
        """Test that data flows through storage tiers correctly."""
        # Create a conversation (should go to Redis if enabled)
        conversation = memory_adapter.create_conversation(
            user_id=test_user_id,
            metadata={"tier_test": True}
        )
        
        # Add many messages to trigger potential archival
        for i in range(10):
            memory_adapter.add_message(
                conversation_id=conversation.conversation_id,
                user_id=test_user_id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}"
            )
        
        # Verify retrieval works
        retrieved = memory_adapter.get_conversation(conversation.conversation_id)
        assert len(retrieved.messages) == 10
    
    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, memory_adapter, test_user_id):
        """Test that system falls back gracefully when a backend fails."""
        # This test would require mocking backend failures
        # For now, just verify the adapter handles missing data gracefully
        
        non_existent_conv = memory_adapter.get_conversation("non_existent_id")
        # Should return None or empty, not crash
        assert non_existent_conv is None or len(non_existent_conv.messages) == 0


class TestPerformance:
    """Performance benchmarks for distributed backends."""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_redis_performance(self, memory_adapter, test_user_id, benchmark):
        """Benchmark Redis operations."""
        if not memory_backend_config.use_redis:
            pytest.skip("Redis not enabled")
        
        conversation = memory_adapter.create_conversation(user_id=test_user_id)
        
        def add_and_retrieve():
            memory_adapter.add_message(
                conversation_id=conversation.conversation_id,
                user_id=test_user_id,
                role="user",
                content="Test message"
            )
            memory_adapter.get_conversation(conversation.conversation_id)
        
        # Benchmark should complete in under 10ms for Redis
        result = benchmark(add_and_retrieve)
        assert result.avg < 0.01  # 10ms
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, memory_adapter, test_user_id):
        """Test concurrent access to memory systems."""
        conversation = memory_adapter.create_conversation(user_id=test_user_id)
        
        async def add_message(idx):
            memory_adapter.add_message(
                conversation_id=conversation.conversation_id,
                user_id=test_user_id,
                role="user",
                content=f"Concurrent message {idx}"
            )
        
        # Run 10 concurrent operations
        await asyncio.gather(*[add_message(i) for i in range(10)])
        
        # Verify all messages were stored
        retrieved = memory_adapter.get_conversation(conversation.conversation_id)
        assert len(retrieved.messages) == 10


class TestDataIntegrity:
    """Test data integrity across backends."""
    
    @pytest.mark.asyncio
    async def test_data_consistency(self, memory_adapter, test_user_id):
        """Test that data remains consistent across operations."""
        # Create preference
        memory_adapter.set_user_preference(
            user_id=test_user_id,
            category="test",
            key="consistency",
            value={"nested": "data", "number": 42},
            confidence=1.0
        )
        
        # Retrieve multiple times
        for _ in range(5):
            pref = memory_adapter.get_user_preference(
                user_id=test_user_id,
                category="test",
                key="consistency"
            )
            assert pref.preference_value["nested"] == "data"
            assert pref.preference_value["number"] == 42
            assert pref.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_unicode_handling(self, memory_adapter, test_user_id):
        """Test handling of Unicode and special characters."""
        test_strings = [
            "Hello 世界",
            "Emoji test: 🚀 🤖 🧠",
            "Special chars: <>&\"'",
            "Newlines\nand\ttabs",
        ]
        
        for i, test_str in enumerate(test_strings):
            memory_adapter.set_user_preference(
                user_id=test_user_id,
                category="unicode",
                key=f"test_{i}",
                value=test_str
            )
        
        # Retrieve and verify
        for i, expected in enumerate(test_strings):
            pref = memory_adapter.get_user_preference(
                user_id=test_user_id,
                category="unicode",
                key=f"test_{i}"
            )
            assert pref.preference_value == expected


# Cleanup fixture
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_data():
    """Clean up test data after all tests."""
    yield
    # Add cleanup logic here if needed
    # This could include removing test users, conversations, etc.