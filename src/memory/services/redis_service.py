"""
Redis service for high-speed short-term memory caching.
"""
import json
import redis
import logging
from typing import Optional, Dict, Any, List
from datetime import timedelta
import pickle

from src.utils.error_handling import retry_on_error, handle_errors, MemoryError

logger = logging.getLogger(__name__)


class RedisMemoryService:
    """
    Redis-based memory service for short-term storage and caching.
    
    This service provides high-speed access to recent conversations
    and frequently accessed data with automatic expiration.
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 decode_responses: bool = False,
                 max_connections: int = 50):
        """
        Initialize Redis memory service.
        
        Args:
            host: Redis host address
            port: Redis port
            db: Redis database number
            password: Redis password (if required)
            decode_responses: Whether to decode responses to strings
            max_connections: Maximum number of connections in the pool
        """
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses,
            max_connections=max_connections
        )
        self.client = redis.Redis(connection_pool=self.pool)
        
        # Test connection
        try:
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise MemoryError(f"Redis connection failed: {e}")
    
    @retry_on_error(max_attempts=3, delay=0.1)
    def store_conversation_message(self, 
                                  conversation_id: str,
                                  message: Dict[str, Any],
                                  ttl: int = 3600) -> bool:
        """
        Store a conversation message in Redis.
        
        Args:
            conversation_id: Unique conversation identifier
            message: Message data to store
            ttl: Time to live in seconds (default: 1 hour)
            
        Returns:
            True if successful
        """
        key = f"conversation:{conversation_id}:messages"
        
        try:
            # Serialize message
            serialized = json.dumps(message)
            
            # Add to list
            self.client.lpush(key, serialized)
            
            # Set expiration
            self.client.expire(key, ttl)
            
            # Update last activity
            self.client.set(
                f"conversation:{conversation_id}:last_activity",
                message.get('timestamp', ''),
                ex=ttl
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store conversation message: {e}")
            raise MemoryError(f"Failed to store in Redis: {e}")
    
    @handle_errors(default_return=[])
    def get_conversation_messages(self, 
                                 conversation_id: str,
                                 limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve conversation messages from Redis.
        
        Args:
            conversation_id: Unique conversation identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of messages (newest first)
        """
        key = f"conversation:{conversation_id}:messages"
        
        try:
            # Get messages
            raw_messages = self.client.lrange(key, 0, limit - 1)
            
            # Deserialize
            messages = []
            for raw in raw_messages:
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8')
                messages.append(json.loads(raw))
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to retrieve conversation messages: {e}")
            return []
    
    @retry_on_error(max_attempts=3)
    def cache_user_context(self,
                          user_id: str,
                          context: Dict[str, Any],
                          ttl: int = 1800) -> bool:
        """
        Cache user context for quick retrieval.
        
        Args:
            user_id: User identifier
            context: Context data to cache
            ttl: Time to live in seconds (default: 30 minutes)
            
        Returns:
            True if successful
        """
        key = f"user:{user_id}:context"
        
        try:
            # Use pickle for complex objects
            serialized = pickle.dumps(context)
            return bool(self.client.set(key, serialized, ex=ttl))
            
        except Exception as e:
            logger.error(f"Failed to cache user context: {e}")
            return False
    
    @handle_errors(default_return=None)
    def get_user_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached user context.
        
        Args:
            user_id: User identifier
            
        Returns:
            Context data or None if not found
        """
        key = f"user:{user_id}:context"
        
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve user context: {e}")
            return None
    
    def store_memory_index(self,
                          user_id: str,
                          memory_type: str,
                          memory_ids: List[str],
                          ttl: int = 7200) -> bool:
        """
        Store an index of memory IDs for quick lookup.
        
        Args:
            user_id: User identifier
            memory_type: Type of memory (episodic, semantic, etc.)
            memory_ids: List of memory IDs
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        key = f"user:{user_id}:memory_index:{memory_type}"
        
        try:
            # Store as Redis set
            pipeline = self.client.pipeline()
            pipeline.delete(key)  # Clear existing
            
            if memory_ids:
                pipeline.sadd(key, *memory_ids)
                pipeline.expire(key, ttl)
            
            pipeline.execute()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory index: {e}")
            return False
    
    def get_memory_index(self,
                        user_id: str,
                        memory_type: str) -> List[str]:
        """
        Retrieve memory index.
        
        Args:
            user_id: User identifier
            memory_type: Type of memory
            
        Returns:
            List of memory IDs
        """
        key = f"user:{user_id}:memory_index:{memory_type}"
        
        try:
            members = self.client.smembers(key)
            return [m.decode('utf-8') if isinstance(m, bytes) else m 
                   for m in members]
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory index: {e}")
            return []
    
    def invalidate_cache(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "user:123:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics."""
        try:
            info = self.client.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', 'N/A'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': (
                    info.get('keyspace_hits', 0) / 
                    (info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1))
                    * 100
                )
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check Redis health."""
        try:
            return self.client.ping()
        except:
            return False
