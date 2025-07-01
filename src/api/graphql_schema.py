"""
GraphQL schema for CleoAI API.
"""
from ariadne import QueryType, MutationType, make_executable_schema, gql
from typing import Dict, Any, Optional
import logging

from ..inference.inference_engine import create_inference_engine
from ..memory.adapters.memory_adapter import HybridMemoryAdapter

logger = logging.getLogger(__name__)

# GraphQL type definitions
type_defs = gql("""
    type Query {
        # AI interaction
        ask(question: String!, userId: String!, conversationId: String): AskResponse!
        
        # Memory queries
        getConversation(conversationId: String!, userId: String!): Conversation
        getUserPreferences(userId: String!): [UserPreference!]!
        searchMemories(userId: String!, query: String!, limit: Int = 10): [Memory!]!
        
        # System status
        healthCheck: HealthStatus!
        memoryStats(userId: String!): MemoryStats!
    }
    
    type Mutation {
        # Conversation management
        createConversation(userId: String!, metadata: JSON): CreateConversationResponse!
        
        # Preference management
        setUserPreference(
            userId: String!,
            category: String!,
            key: String!,
            value: JSON!,
            confidence: Float = 0.5
        ): UserPreference!
        
        # Memory management
        createMemory(
            userId: String!,
            title: String!,
            content: String!,
            importance: Float = 0.5,
            emotion: String
        ): Memory!
    }
    
    type AskResponse {
        response: String!
        conversationId: String!
        responseTime: Float!
        metadata: JSON
    }
    
    type Conversation {
        conversationId: String!
        userId: String!
        messages: [Message!]!
        createdAt: String!
        metadata: JSON
    }
    
    type Message {
        role: String!
        content: String!
        timestamp: String!
    }
    
    type CreateConversationResponse {
        conversationId: String!
        success: Boolean!
    }
    
    type UserPreference {
        preferenceId: String!
        userId: String!
        category: String!
        key: String!
        value: JSON!
        confidence: Float!
        source: String!
        updatedAt: String!
    }
    
    type Memory {
        memoryId: String!
        userId: String!
        title: String!
        content: String!
        importance: Float!
        emotion: String
        createdAt: String!
        score: Float
    }
    
    type HealthStatus {
        status: String!
        services: JSON!
        timestamp: String!
    }
    
    type MemoryStats {
        userId: String!
        conversationCount: Int!
        preferenceCount: Int!
        memoryCount: Int!
        lastActivity: String
    }
    
    scalar JSON
""")

# Initialize query and mutation types
query = QueryType()
mutation = MutationType()

# Global instances (initialized by the application)
inference_engine = None
memory_adapter = None


def initialize_api(engine, adapter):
    """Initialize API with required services."""
    global inference_engine, memory_adapter
    inference_engine = engine
    memory_adapter = adapter


# Query resolvers
@query.field("ask")
async def resolve_ask(_, info, question: str, userId: str, conversationId: Optional[str] = None):
    """Resolve AI question."""
    try:
        if not inference_engine:
            raise Exception("Inference engine not initialized")
        
        # Create conversation if not provided
        if not conversationId:
            conv = memory_adapter.create_conversation(userId)
            conversationId = conv.conversation_id if conv else None
        
        # Get response from inference engine
        response = await inference_engine.respond_async(
            user_message=question,
            user_id=userId,
            conversation_id=conversationId
        )
        
        return {
            "response": response["response"],
            "conversationId": conversationId or "",
            "responseTime": response.get("response_time", 0.0),
            "metadata": response.get("metadata", {})
        }
        
    except Exception as e:
        logger.error(f"Error in ask resolver: {e}")
        return {
            "response": f"Error: {str(e)}",
            "conversationId": conversationId or "",
            "responseTime": 0.0,
            "metadata": {"error": True}
        }


@query.field("getConversation")
def resolve_get_conversation(_, info, conversationId: str, userId: str):
    """Get conversation details."""
    try:
        if not memory_adapter:
            raise Exception("Memory adapter not initialized")
        
        conversation = memory_adapter.get_conversation(conversationId)
        
        if not conversation or conversation.user_id != userId:
            return None
        
        return {
            "conversationId": conversation.conversation_id,
            "userId": conversation.user_id,
            "messages": conversation.messages,
            "createdAt": conversation.created_at,
            "metadata": conversation.metadata
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        return None


@query.field("getUserPreferences")
def resolve_get_user_preferences(_, info, userId: str):
    """Get all user preferences."""
    try:
        if not memory_adapter:
            raise Exception("Memory adapter not initialized")
        
        # Get from legacy system (comprehensive)
        if memory_adapter.use_legacy:
            preferences = memory_adapter.legacy_manager.long_term.get_user_preferences(userId)
            return [
                {
                    "preferenceId": pref.preference_id,
                    "userId": pref.user_id,
                    "category": pref.category,
                    "key": pref.preference_key,
                    "value": pref.preference_value,
                    "confidence": pref.confidence,
                    "source": pref.source,
                    "updatedAt": pref.updated_at
                }
                for pref in preferences
            ]
        
        return []
        
    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        return []


@query.field("searchMemories")
def resolve_search_memories(_, info, userId: str, query: str, limit: int = 10):
    """Search user memories."""
    try:
        if not memory_adapter:
            raise Exception("Memory adapter not initialized")
        
        memories = memory_adapter.search_memories(
            user_id=userId,
            query_text=query,
            limit=limit
        )
        
        return [
            {
                "memoryId": mem.memory_id,
                "userId": mem.user_id,
                "title": mem.title,
                "content": mem.content,
                "importance": mem.importance,
                "emotion": mem.emotion,
                "createdAt": mem.created_at,
                "score": getattr(mem, 'score', 0.0)
            }
            for mem in memories
        ]
        
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        return []


@query.field("healthCheck")
async def resolve_health_check(_, info):
    """System health check."""
    try:
        from datetime import datetime
        import asyncio
        from ..utils.health_check import perform_health_check
        from config import memory_backend_config
        
        # Run comprehensive health check
        health_result = await perform_health_check(memory_backend_config)
        
        # Add inference engine status
        if inference_engine:
            health_result["services"]["inference"] = {
                "name": "inference",
                "status": "healthy",
                "latency_ms": 0.0,
                "details": {"model_loaded": True}
            }
        else:
            health_result["services"]["inference"] = {
                "name": "inference",
                "status": "unhealthy",
                "latency_ms": 0.0,
                "message": "No model loaded"
            }
        
        return health_result
        
    except Exception as e:
        from datetime import datetime
        return {
            "status": "unhealthy",
            "services": {"error": str(e)},
            "timestamp": datetime.utcnow().isoformat(),
            "warnings": [f"Health check failed: {str(e)}"]
        }


@query.field("memoryStats")
def resolve_memory_stats(_, info, userId: str):
    """Get memory statistics for a user."""
    try:
        if not memory_adapter:
            raise Exception("Memory adapter not initialized")
        
        stats = {
            "userId": userId,
            "conversationCount": 0,
            "preferenceCount": 0,
            "memoryCount": 0,
            "lastActivity": None
        }
        
        # Get counts from legacy system
        if memory_adapter.use_legacy:
            # Count conversations
            conversations = memory_adapter.legacy_manager.short_term.get_user_conversations(
                userId, limit=100
            )
            stats["conversationCount"] = len(conversations)
            
            # Count preferences
            preferences = memory_adapter.legacy_manager.long_term.get_user_preferences(userId)
            stats["preferenceCount"] = len(preferences)
            
            # Count memories
            memories = memory_adapter.legacy_manager.episodic.get_user_memories(
                userId, limit=100
            )
            stats["memoryCount"] = len(memories)
            
            # Get last activity
            if conversations:
                stats["lastActivity"] = conversations[0].created_at
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {
            "userId": userId,
            "conversationCount": 0,
            "preferenceCount": 0,
            "memoryCount": 0,
            "lastActivity": None
        }


# Mutation resolvers
@mutation.field("createConversation")
def resolve_create_conversation(_, info, userId: str, metadata: Optional[Dict[str, Any]] = None):
    """Create a new conversation."""
    try:
        if not memory_adapter:
            raise Exception("Memory adapter not initialized")
        
        conversation = memory_adapter.create_conversation(
            user_id=userId,
            metadata=metadata
        )
        
        if conversation:
            return {
                "conversationId": conversation.conversation_id,
                "success": True
            }
        
        return {
            "conversationId": "",
            "success": False
        }
        
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        return {
            "conversationId": "",
            "success": False
        }


@mutation.field("setUserPreference")
def resolve_set_user_preference(_, info, userId: str, category: str, key: str, 
                               value: Any, confidence: float = 0.5):
    """Set a user preference."""
    try:
        if not memory_adapter:
            raise Exception("Memory adapter not initialized")
        
        preference = memory_adapter.set_user_preference(
            user_id=userId,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            source="api"
        )
        
        if preference:
            return {
                "preferenceId": preference.preference_id,
                "userId": preference.user_id,
                "category": preference.category,
                "key": preference.preference_key,
                "value": preference.preference_value,
                "confidence": preference.confidence,
                "source": preference.source,
                "updatedAt": preference.updated_at
            }
        
        # Return a default if creation failed
        return {
            "preferenceId": f"{category}:{key}",
            "userId": userId,
            "category": category,
            "key": key,
            "value": value,
            "confidence": confidence,
            "source": "api",
            "updatedAt": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error setting preference: {e}")
        raise


@mutation.field("createMemory")
def resolve_create_memory(_, info, userId: str, title: str, content: str,
                         importance: float = 0.5, emotion: Optional[str] = None):
    """Create an episodic memory."""
    try:
        if not memory_adapter:
            raise Exception("Memory adapter not initialized")
        
        memory = memory_adapter.create_episodic_memory(
            user_id=userId,
            title=title,
            content=content,
            importance=importance,
            emotion=emotion
        )
        
        if memory:
            return {
                "memoryId": memory.memory_id,
                "userId": memory.user_id,
                "title": memory.title,
                "content": memory.content,
                "importance": memory.importance,
                "emotion": memory.emotion,
                "createdAt": memory.created_at,
                "score": 0.0
            }
        
        raise Exception("Failed to create memory")
        
    except Exception as e:
        logger.error(f"Error creating memory: {e}")
        raise


# Create executable schema
schema = make_executable_schema(type_defs, query, mutation)
