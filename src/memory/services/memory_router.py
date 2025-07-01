"""
Memory router for coordinating between different memory services.
"""
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from .redis_service import RedisMemoryService
from .mongodb_service import MongoDBMemoryService
from .supabase_service import SupabaseMemoryService
from .pinecone_service import PineconeMemoryService
from src.utils.error_handling import handle_errors, MemoryError

logger = logging.getLogger(__name__)


class MemoryRouter:
    """
    Central router for coordinating between different memory storage services.
    
    This class implements a tiered memory architecture:
    - Redis: Hot cache for active conversations
    - Supabase: Relational data and user preferences
    - Pinecone: Vector embeddings for similarity search
    - MongoDB: Cold storage for archival
    """
    
    def __init__(self,
                 redis_config: Optional[Dict[str, Any]] = None,
                 supabase_config: Optional[Dict[str, Any]] = None,
                 pinecone_config: Optional[Dict[str, Any]] = None,
                 mongodb_config: Optional[Dict[str, Any]] = None,
                 enable_all: bool = True):
        """
        Initialize memory router with service configurations.
        
        Args:
            redis_config: Redis service configuration
            supabase_config: Supabase service configuration
            pinecone_config: Pinecone service configuration
            mongodb_config: MongoDB service configuration
            enable_all: Enable all services (for testing, some can be disabled)
        """
        self.services = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize Redis (hot cache)
        if enable_all or redis_config:
            try:
                self.services['redis'] = RedisMemoryService(
                    **(redis_config or {})
                )
                logger.info("Redis service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
        
        # Initialize Supabase (relational storage)
        if enable_all or supabase_config:
            try:
                self.services['supabase'] = SupabaseMemoryService(
                    **(supabase_config or {})
                )
                logger.info("Supabase service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase: {e}")
        
        # Initialize Pinecone (vector storage)
        if enable_all or pinecone_config:
            try:
                self.services['pinecone'] = PineconeMemoryService(
                    **(pinecone_config or {})
                )
                logger.info("Pinecone service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Pinecone: {e}")
        
        # Initialize MongoDB (archival storage)
        if enable_all or mongodb_config:
            try:
                self.services['mongodb'] = MongoDBMemoryService(
                    **(mongodb_config or {})
                )
                logger.info("MongoDB service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MongoDB: {e}")
        
        if not self.services:
            raise MemoryError("No memory services could be initialized")
        
        logger.info(f"Memory router initialized with services: {list(self.services.keys())}")
    
    @handle_errors(default_return={})
    async def store_conversation_turn(self,
                                    conversation_id: str,
                                    user_id: str,
                                    user_message: str,
                                    assistant_response: str,
                                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a conversation turn across appropriate services.
        
        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            user_message: User's message
            assistant_response: Assistant's response
            metadata: Additional metadata
            
        Returns:
            Storage results from each service
        """
        results = {}
        timestamp = datetime.utcnow().isoformat()
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": user_message,
                "timestamp": timestamp
            },
            {
                "role": "assistant",
                "content": assistant_response,
                "timestamp": timestamp
            }
        ]
        
        # Store in Redis (hot cache)
        if 'redis' in self.services:
            try:
                for msg in messages:
                    self.services['redis'].store_conversation_message(
                        conversation_id, msg, ttl=3600
                    )
                results['redis'] = "success"
            except Exception as e:
                logger.error(f"Redis storage failed: {e}")
                results['redis'] = f"error: {e}"
        
        # Update Supabase metadata
        if 'supabase' in self.services:
            try:
                self.services['supabase'].store_conversation_metadata(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    metadata=metadata
                )
                results['supabase'] = "success"
            except Exception as e:
                logger.error(f"Supabase storage failed: {e}")
                results['supabase'] = f"error: {e}"
        
        return results
    
    @handle_errors(default_return=None)
    def get_conversation_context(self,
                               conversation_id: str,
                               user_id: str,
                               message_limit: int = 50) -> Optional[Dict[str, Any]]:
        """
        Retrieve conversation context from the fastest available source.
        
        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            message_limit: Maximum messages to retrieve
            
        Returns:
            Conversation context or None
        """
        # Try Redis first (fastest)
        if 'redis' in self.services:
            messages = self.services['redis'].get_conversation_messages(
                conversation_id, limit=message_limit
            )
            if messages:
                return {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "messages": messages,
                    "source": "redis"
                }
        
        # Try MongoDB if Redis miss
        if 'mongodb' in self.services:
            archived = self.services['mongodb'].get_archived_conversation(
                conversation_id
            )
            if archived:
                return {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "messages": archived.get("messages", [])[-message_limit:],
                    "source": "mongodb"
                }
        
        return None
    
    @handle_errors(default_return={})
    async def store_memory_with_embedding(self,
                                        memory_id: str,
                                        user_id: str,
                                        title: str,
                                        content: str,
                                        embedding: List[float],
                                        memory_type: str = "episodic",
                                        importance: float = 0.5,
                                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a memory with its embedding across services.
        
        Args:
            memory_id: Unique memory identifier
            user_id: User identifier
            title: Memory title
            content: Memory content
            embedding: Vector embedding
            memory_type: Type of memory
            importance: Importance score (0-1)
            metadata: Additional metadata
            
        Returns:
            Storage results
        """
        results = {}
        full_metadata = {
            "title": title,
            "type": memory_type,
            "importance": importance,
            **(metadata or {})
        }
        
        # Store in Pinecone (vector storage)
        if 'pinecone' in self.services:
            try:
                self.services['pinecone'].upsert_memory(
                    memory_id=memory_id,
                    embedding=embedding,
                    user_id=user_id,
                    metadata=full_metadata
                )
                results['pinecone'] = "success"
            except Exception as e:
                logger.error(f"Pinecone storage failed: {e}")
                results['pinecone'] = f"error: {e}"
        
        # Store in Supabase (relational storage)
        if 'supabase' in self.services:
            try:
                self.services['supabase'].store_episodic_memory(
                    memory_id=memory_id,
                    user_id=user_id,
                    title=title,
                    content=content,
                    importance=importance,
                    metadata=metadata
                )
                results['supabase'] = "success"
            except Exception as e:
                logger.error(f"Supabase storage failed: {e}")
                results['supabase'] = f"error: {e}"
        
        # Store in MongoDB for archival
        if 'mongodb' in self.services:
            try:
                self.services['mongodb'].store_memory_snapshot(
                    user_id=user_id,
                    memory_type=memory_type,
                    data={
                        "memory_id": memory_id,
                        "title": title,
                        "content": content,
                        "metadata": metadata
                    },
                    importance=importance
                )
                results['mongodb'] = "success"
            except Exception as e:
                logger.error(f"MongoDB storage failed: {e}")
                results['mongodb'] = f"error: {e}"
        
        # Cache high-importance memories in Redis
        if importance > 0.7 and 'redis' in self.services:
            try:
                self.services['redis'].cache_user_context(
                    user_id=user_id,
                    context={
                        f"memory_{memory_id}": {
                            "title": title,
                            "content": content[:500],  # Truncate for cache
                            "importance": importance
                        }
                    },
                    ttl=7200  # 2 hours for important memories
                )
            except Exception as e:
                logger.warning(f"Redis caching failed: {e}")
        
        return results
    
    @handle_errors(default_return=[])
    def search_memories(self,
                       user_id: str,
                       query_embedding: List[float],
                       memory_type: Optional[str] = None,
                       min_importance: float = 0.0,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar memories using vector similarity.
        
        Args:
            user_id: User identifier
            query_embedding: Query vector
            memory_type: Filter by memory type
            min_importance: Minimum importance threshold
            limit: Maximum results
            
        Returns:
            List of similar memories
        """
        # Use Pinecone for vector search
        if 'pinecone' in self.services:
            filter_dict = {}
            if memory_type:
                filter_dict["type"] = {"$eq": memory_type}
            if min_importance > 0:
                filter_dict["importance"] = {"$gte": min_importance}
            
            results = self.services['pinecone'].query_memories(
                query_embedding=query_embedding,
                user_id=user_id,
                top_k=limit,
                filter_dict=filter_dict if filter_dict else None
            )
            
            # Enrich with full content from Supabase if available
            if results and 'supabase' in self.services:
                memory_ids = [r['id'] for r in results]
                
                # Batch fetch from Supabase
                for i, result in enumerate(results):
                    try:
                        memories = self.services['supabase'].search_episodic_memories(
                            user_id=user_id,
                            limit=1
                        )
                        if memories:
                            result['content'] = memories[0].get('content', '')
                    except:
                        pass
            
            return results
        
        # Fallback to text search in Supabase
        if 'supabase' in self.services:
            memories = self.services['supabase'].search_episodic_memories(
                user_id=user_id,
                min_importance=min_importance,
                limit=limit
            )
            return [
                {
                    "id": m["memory_id"],
                    "score": m.get("importance", 0.5),
                    "metadata": m
                }
                for m in memories
            ]
        
        return []
    
    @handle_errors(default_return={})
    def store_user_preference(self,
                            user_id: str,
                            preference_type: str,
                            preference_key: str,
                            preference_value: Any,
                            confidence: float = 0.5,
                            source: str = "inferred") -> Dict[str, Any]:
        """
        Store a user preference.
        
        Args:
            user_id: User identifier
            preference_type: Type of preference
            preference_key: Preference key
            preference_value: Preference value
            confidence: Confidence score
            source: Source of preference
            
        Returns:
            Storage results
        """
        results = {}
        
        # Store in Supabase (primary storage for preferences)
        if 'supabase' in self.services:
            try:
                pref = self.services['supabase'].store_user_preference(
                    user_id=user_id,
                    preference_type=preference_type,
                    preference_key=preference_key,
                    preference_value=preference_value,
                    confidence=confidence,
                    source=source
                )
                results['supabase'] = "success"
                
                # Cache in Redis for fast access
                if 'redis' in self.services:
                    cache_key = f"pref:{preference_type}:{preference_key}"
                    self.services['redis'].cache_user_context(
                        user_id=user_id,
                        context={cache_key: preference_value},
                        ttl=3600
                    )
                    
            except Exception as e:
                logger.error(f"Preference storage failed: {e}")
                results['supabase'] = f"error: {e}"
        
        return results
    
    @handle_errors(default_return=None)
    def get_user_preference(self,
                          user_id: str,
                          preference_type: str,
                          preference_key: str) -> Optional[Any]:
        """
        Get a user preference.
        
        Args:
            user_id: User identifier
            preference_type: Type of preference
            preference_key: Preference key
            
        Returns:
            Preference value or None
        """
        # Check Redis cache first
        if 'redis' in self.services:
            context = self.services['redis'].get_user_context(user_id)
            if context:
                cache_key = f"pref:{preference_type}:{preference_key}"
                if cache_key in context:
                    return context[cache_key]
        
        # Get from Supabase
        if 'supabase' in self.services:
            pref = self.services['supabase'].get_user_preference(
                user_id=user_id,
                preference_type=preference_type,
                preference_key=preference_key
            )
            if pref:
                return pref.get("preference_value")
        
        return None
    
    async def archive_conversation(self,
                                 conversation_id: str,
                                 user_id: str,
                                 retention_days: Optional[int] = None) -> bool:
        """
        Archive a conversation from hot to cold storage.
        
        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            retention_days: Days to retain
            
        Returns:
            True if successful
        """
        try:
            # Get messages from Redis
            messages = []
            if 'redis' in self.services:
                messages = self.services['redis'].get_conversation_messages(
                    conversation_id, limit=1000
                )
            
            if not messages:
                logger.warning(f"No messages found for conversation {conversation_id}")
                return False
            
            # Archive to MongoDB
            if 'mongodb' in self.services:
                self.services['mongodb'].archive_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    messages=messages,
                    retention_days=retention_days
                )
            
            # Clear from Redis
            if 'redis' in self.services:
                self.services['redis'].invalidate_cache(
                    f"conversation:{conversation_id}:*"
                )
            
            logger.info(f"Archived conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive conversation: {e}")
            return False
    
    def get_service_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all services."""
        stats = {}
        
        for name, service in self.services.items():
            try:
                stats[name] = service.get_stats()
            except Exception as e:
                stats[name] = {"error": str(e)}
        
        return stats
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all services."""
        health = {}
        
        for name, service in self.services.items():
            try:
                health[name] = service.health_check()
            except:
                health[name] = False
        
        return health
    
    def close(self):
        """Close all service connections."""
        self.executor.shutdown(wait=True)
        
        for name, service in self.services.items():
            try:
                if hasattr(service, 'close'):
                    service.close()
                elif hasattr(service, 'client') and hasattr(service.client, 'close'):
                    service.client.close()
            except Exception as e:
                logger.warning(f"Error closing {name}: {e}")
