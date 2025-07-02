"""
GraphQL API implementation with authentication for CleoAI.

This module provides a secure GraphQL API using Strawberry with
JWT authentication, role-based access control, and rate limiting.
"""
import os
import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

import strawberry
from strawberry.permission import BasePermission
from strawberry.types import Info
from strawberry.fastapi import GraphQLRouter
from strawberry.schema.config import StrawberryConfig

from .auth import (
    IsAuthenticated, HasRole, RateLimited, UserRole,
    create_access_token, hash_password, verify_password
)
from ..inference.inference_engine import create_inference_engine
from ..memory.memory_manager import MemoryManager
from ..utils.error_handling import handle_errors
from ..utils.secrets_manager import get_secrets_manager

logger = logging.getLogger(__name__)

# Get environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
secrets_manager = get_secrets_manager()


# GraphQL Types
@strawberry.type
class Message:
    role: str
    content: str
    timestamp: datetime


@strawberry.type
class Conversation:
    conversation_id: str
    user_id: str
    messages: List[Message]
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None


@strawberry.type
class UserPreference:
    preference_id: str
    user_id: str
    category: str
    key: str
    value: Dict[str, Any]
    confidence: float
    source: str
    updated_at: datetime


@strawberry.type
class Memory:
    memory_id: str
    user_id: str
    title: str
    content: str
    importance: float
    emotion: Optional[str] = None
    created_at: datetime
    score: Optional[float] = None


@strawberry.type
class AskResponse:
    response: str
    conversation_id: str
    response_time: float
    metadata: Optional[Dict[str, Any]] = None


@strawberry.type
class AuthResponse:
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 hours


@strawberry.type
class HealthStatus:
    status: str
    timestamp: datetime
    memory_health: Dict[str, Any]
    inference_ready: bool


@strawberry.type
class MemoryStats:
    total_memories: int
    total_conversations: int
    total_preferences: int
    storage_backends: List[str]


# Input Types
@strawberry.input
class LoginInput:
    username: str
    password: str


@strawberry.input
class CreateMemoryInput:
    title: str
    content: str
    importance: float = 0.5
    emotion: Optional[str] = None


@strawberry.input
class SetPreferenceInput:
    category: str
    key: str
    value: Dict[str, Any]
    confidence: float = 0.5


# Queries
@strawberry.type
class Query:
    @strawberry.field(
        permission_classes=[IsAuthenticated, RateLimited(60, 1000)]
    )
    async def ask(
        self,
        info: Info,
        question: str,
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> AskResponse:
        """Ask a question to the AI with memory context."""
        start_time = time.time()
        
        # Verify user can access this user_id
        request_user = info.context["request"].user
        if request_user.user_id != user_id and UserRole.ADMIN not in request_user.roles:
            raise Exception("Unauthorized to access this user's data")
        
        # Get inference engine
        engine = create_inference_engine()
        
        # Generate response
        response = await engine.generate_response(
            user_id=user_id,
            conversation_id=conversation_id,
            query=question
        )
        
        return AskResponse(
            response=response["response"],
            conversation_id=response["conversation_id"],
            response_time=time.time() - start_time,
            metadata=response.get("metadata")
        )
    
    @strawberry.field(
        permission_classes=[IsAuthenticated]
    )
    async def get_conversation(
        self,
        info: Info,
        conversation_id: str,
        user_id: str
    ) -> Optional[Conversation]:
        """Get a specific conversation."""
        # Verify user access
        request_user = info.context["request"].user
        if request_user.user_id != user_id and UserRole.ADMIN not in request_user.roles:
            raise Exception("Unauthorized to access this user's data")
        
        memory_manager = MemoryManager()
        conv_data = memory_manager.get_conversation(conversation_id, user_id)
        
        if not conv_data:
            return None
        
        return Conversation(
            conversation_id=conv_data["conversation_id"],
            user_id=conv_data["user_id"],
            messages=[
                Message(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=datetime.fromisoformat(msg["timestamp"])
                )
                for msg in conv_data.get("messages", [])
            ],
            created_at=datetime.fromisoformat(conv_data["created_at"]),
            metadata=conv_data.get("metadata")
        )
    
    @strawberry.field(
        permission_classes=[IsAuthenticated]
    )
    async def get_user_preferences(
        self,
        info: Info,
        user_id: str
    ) -> List[UserPreference]:
        """Get all preferences for a user."""
        # Verify user access
        request_user = info.context["request"].user
        if request_user.user_id != user_id and UserRole.ADMIN not in request_user.roles:
            raise Exception("Unauthorized to access this user's data")
        
        memory_manager = MemoryManager()
        preferences = memory_manager.get_all_user_preferences(user_id)
        
        return [
            UserPreference(
                preference_id=pref.get("preference_id", f"{pref['category']}:{pref['key']}"),
                user_id=pref["user_id"],
                category=pref["category"],
                key=pref["key"],
                value=pref["value"],
                confidence=pref["confidence"],
                source=pref.get("source", "inferred"),
                updated_at=datetime.fromisoformat(pref.get("updated_at", datetime.now().isoformat()))
            )
            for pref in preferences
        ]
    
    @strawberry.field(
        permission_classes=[IsAuthenticated]
    )
    async def search_memories(
        self,
        info: Info,
        user_id: str,
        query: str,
        limit: int = 10
    ) -> List[Memory]:
        """Search memories for a user."""
        # Verify user access
        request_user = info.context["request"].user
        if request_user.user_id != user_id and UserRole.ADMIN not in request_user.roles:
            raise Exception("Unauthorized to access this user's data")
        
        memory_manager = MemoryManager()
        memories = memory_manager.search_memories(
            user_id=user_id,
            query=query,
            limit=limit
        )
        
        return [
            Memory(
                memory_id=mem["memory_id"],
                user_id=mem["user_id"],
                title=mem["title"],
                content=mem["content"],
                importance=mem["importance"],
                emotion=mem.get("emotion"),
                created_at=datetime.fromisoformat(mem["created_at"]),
                score=mem.get("score")
            )
            for mem in memories
        ]
    
    @strawberry.field
    async def health_check(self, info: Info) -> HealthStatus:
        """Get system health status."""
        memory_manager = MemoryManager()
        memory_health = memory_manager.get_health_status()
        
        # Check inference engine
        try:
            engine = create_inference_engine()
            inference_ready = engine is not None
        except:
            inference_ready = False
        
        return HealthStatus(
            status="healthy" if memory_health.get("healthy", False) else "degraded",
            timestamp=datetime.now(),
            memory_health=memory_health,
            inference_ready=inference_ready
        )
    
    @strawberry.field(
        permission_classes=[IsAuthenticated]
    )
    async def memory_stats(
        self,
        info: Info,
        user_id: str
    ) -> MemoryStats:
        """Get memory statistics for a user."""
        # Verify user access
        request_user = info.context["request"].user
        if request_user.user_id != user_id and UserRole.ADMIN not in request_user.roles:
            raise Exception("Unauthorized to access this user's data")
        
        memory_manager = MemoryManager()
        stats = memory_manager.get_user_stats(user_id)
        
        return MemoryStats(
            total_memories=stats.get("total_memories", 0),
            total_conversations=stats.get("total_conversations", 0),
            total_preferences=stats.get("total_preferences", 0),
            storage_backends=list(stats.get("backends", {}).keys())
        )


# Mutations
@strawberry.type
class Mutation:
    @strawberry.mutation
    async def login(
        self,
        info: Info,
        input: LoginInput
    ) -> AuthResponse:
        """Authenticate and get access token."""
        # In production, verify against user database
        # For now, use demo credentials
        demo_password_hash = hash_password("demo123")
        
        if input.username == "demo" and verify_password(input.password, demo_password_hash):
            token = create_access_token(
                user_id="demo-user",
                roles=[UserRole.USER]
            )
            return AuthResponse(access_token=token)
        
        raise Exception("Invalid credentials")
    
    @strawberry.mutation(
        permission_classes=[IsAuthenticated]
    )
    async def create_conversation(
        self,
        info: Info,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """Create a new conversation."""
        # Verify user access
        request_user = info.context["request"].user
        if request_user.user_id != user_id and UserRole.ADMIN not in request_user.roles:
            raise Exception("Unauthorized to create conversation for this user")
        
        memory_manager = MemoryManager()
        conversation_id = memory_manager.create_conversation(
            user_id=user_id,
            metadata=metadata
        )
        
        return Conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            messages=[],
            created_at=datetime.now(),
            metadata=metadata
        )
    
    @strawberry.mutation(
        permission_classes=[IsAuthenticated]
    )
    async def set_user_preference(
        self,
        info: Info,
        user_id: str,
        input: SetPreferenceInput
    ) -> UserPreference:
        """Set a user preference."""
        # Verify user access
        request_user = info.context["request"].user
        if request_user.user_id != user_id and UserRole.ADMIN not in request_user.roles:
            raise Exception("Unauthorized to set preferences for this user")
        
        memory_manager = MemoryManager()
        memory_manager.store_user_preference(
            user_id=user_id,
            preference_type=input.category,
            preference_key=input.key,
            preference_value=input.value,
            confidence=input.confidence,
            source="explicit"
        )
        
        return UserPreference(
            preference_id=f"{input.category}:{input.key}",
            user_id=user_id,
            category=input.category,
            key=input.key,
            value=input.value,
            confidence=input.confidence,
            source="explicit",
            updated_at=datetime.now()
        )
    
    @strawberry.mutation(
        permission_classes=[IsAuthenticated]
    )
    async def create_memory(
        self,
        info: Info,
        user_id: str,
        input: CreateMemoryInput
    ) -> Memory:
        """Create a new memory."""
        # Verify user access
        request_user = info.context["request"].user
        if request_user.user_id != user_id and UserRole.ADMIN not in request_user.roles:
            raise Exception("Unauthorized to create memories for this user")
        
        memory_manager = MemoryManager()
        memory_id = memory_manager.store_episodic_memory(
            user_id=user_id,
            title=input.title,
            content=input.content,
            importance=input.importance,
            emotion=input.emotion
        )
        
        return Memory(
            memory_id=memory_id,
            user_id=user_id,
            title=input.title,
            content=input.content,
            importance=input.importance,
            emotion=input.emotion,
            created_at=datetime.now()
        )


# Create schema with security config
def create_schema():
    """Create GraphQL schema with security configuration."""
    config = StrawberryConfig(
        auto_camel_case=True,
        relay_style_pagination=True
    )
    
    schema = strawberry.Schema(
        query=Query,
        mutation=Mutation,
        config=config,
        # Disable introspection in production
        introspection=ENVIRONMENT != "production"
    )
    
    return schema


# Create GraphQL router
def create_graphql_router():
    """Create FastAPI GraphQL router with security."""
    schema = create_schema()
    
    graphql_app = GraphQLRouter(
        schema,
        debug=ENVIRONMENT == "development",
        # Disable GraphiQL in production
        graphiql=ENVIRONMENT != "production"
    )
    
    return graphql_app