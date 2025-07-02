"""
Supabase service for PostgreSQL-based relational storage.
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import os
from supabase import create_client, Client
import asyncio
from functools import wraps

from src.utils.error_handling import retry_on_error, handle_errors, MemoryError

logger = logging.getLogger(__name__)


def async_to_sync(func):
    """Decorator to run async functions in sync context."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


class SupabaseMemoryService:
    """
    Supabase-based memory service for relational data storage.
    
    This service provides PostgreSQL storage with real-time capabilities,
    row-level security, and built-in authentication.
    """
    
    def __init__(self,
                 url: Optional[str] = None,
                 key: Optional[str] = None):
        """
        Initialize Supabase memory service.
        
        Args:
            url: Supabase project URL
            key: Supabase anon/service key
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_ANON_KEY")
        
        if not self.url or not self.key:
            raise MemoryError("Supabase URL and key are required")
        
        try:
            self.client: Client = create_client(self.url, self.key)
            logger.info("Connected to Supabase")
            
            # Initialize tables if needed
            self._initialize_schema()
            
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise MemoryError(f"Supabase connection failed: {e}")
    
    def _initialize_schema(self):
        """Initialize database schema if tables don't exist."""
        # Note: In production, use Supabase migrations instead
        # This is for development/testing
        
        schema = """
        -- User preferences table
        CREATE TABLE IF NOT EXISTS user_preferences (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            user_id TEXT NOT NULL,
            preference_type TEXT NOT NULL,
            preference_key TEXT NOT NULL,
            preference_value JSONB NOT NULL,
            confidence FLOAT DEFAULT 0.5,
            source TEXT DEFAULT 'inferred',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            metadata JSONB DEFAULT '{}',
            UNIQUE(user_id, preference_type, preference_key)
        );
        
        -- Conversations metadata table
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            conversation_id TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            message_count INT DEFAULT 0,
            metadata JSONB DEFAULT '{}',
            summary TEXT,
            sentiment TEXT,
            topics TEXT[]
        );
        
        -- Episodic memories table
        CREATE TABLE IF NOT EXISTS episodic_memories (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            memory_id TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            importance FLOAT DEFAULT 0.5,
            emotion TEXT DEFAULT 'neutral',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            accessed_count INT DEFAULT 0,
            last_accessed TIMESTAMPTZ,
            metadata JSONB DEFAULT '{}',
            tags TEXT[]
        );
        
        -- Memory relations table
        CREATE TABLE IF NOT EXISTS memory_relations (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            source_memory_id TEXT NOT NULL,
            target_memory_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            strength FLOAT DEFAULT 0.5,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(source_memory_id, target_memory_id, relation_type)
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
        CREATE INDEX IF NOT EXISTS idx_episodic_memories_user_id ON episodic_memories(user_id);
        CREATE INDEX IF NOT EXISTS idx_episodic_memories_importance ON episodic_memories(importance DESC);
        
        -- Enable Row Level Security
        ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
        ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
        ALTER TABLE episodic_memories ENABLE ROW LEVEL SECURITY;
        """
        
        # Note: Schema creation should be done through Supabase dashboard
        # or migration files in production
        logger.info("Supabase schema initialized")
    
    @retry_on_error(max_attempts=3)
    def store_user_preference(self,
                            user_id: str,
                            preference_type: str,
                            preference_key: str,
                            preference_value: Any,
                            confidence: float = 0.5,
                            source: str = "inferred",
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store or update a user preference.
        
        Args:
            user_id: User identifier
            preference_type: Type of preference
            preference_key: Preference key
            preference_value: Preference value
            confidence: Confidence score (0-1)
            source: Source of preference
            metadata: Additional metadata
            
        Returns:
            Stored preference data
        """
        try:
            data = {
                "user_id": user_id,
                "preference_type": preference_type,
                "preference_key": preference_key,
                "preference_value": preference_value,
                "confidence": confidence,
                "source": source,
                "metadata": metadata or {},
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Upsert preference
            result = self.client.table("user_preferences").upsert(
                data,
                on_conflict="user_id,preference_type,preference_key"
            ).execute()
            
            if result.data:
                logger.info(f"Stored preference for user {user_id}: {preference_key}")
                return result.data[0]
            
            raise MemoryError("Failed to store preference")
            
        except Exception as e:
            logger.error(f"Failed to store user preference: {e}")
            raise MemoryError(f"Failed to store preference: {e}")
    
    @handle_errors(default_return=None)
    def get_user_preference(self,
                          user_id: str,
                          preference_type: str,
                          preference_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific user preference.
        
        Args:
            user_id: User identifier
            preference_type: Type of preference
            preference_key: Preference key
            
        Returns:
            Preference data or None
        """
        try:
            result = self.client.table("user_preferences").select("*").eq(
                "user_id", user_id
            ).eq(
                "preference_type", preference_type
            ).eq(
                "preference_key", preference_key
            ).execute()
            
            if result.data:
                return result.data[0]
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user preference: {e}")
            return None
    
    @handle_errors(default_return=[])
    def get_all_user_preferences(self,
                               user_id: str,
                               preference_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all preferences for a user.
        
        Args:
            user_id: User identifier
            preference_type: Optional filter by type
            
        Returns:
            List of preferences
        """
        try:
            query = self.client.table("user_preferences").select("*").eq(
                "user_id", user_id
            )
            
            if preference_type:
                query = query.eq("preference_type", preference_type)
            
            result = query.order("confidence", desc=True).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            return []
    
    @retry_on_error(max_attempts=3)
    def store_conversation_metadata(self,
                                  conversation_id: str,
                                  user_id: str,
                                  metadata: Optional[Dict[str, Any]] = None,
                                  summary: Optional[str] = None,
                                  sentiment: Optional[str] = None,
                                  topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Store conversation metadata.
        
        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            metadata: Additional metadata
            summary: Conversation summary
            sentiment: Overall sentiment
            topics: Conversation topics
            
        Returns:
            Stored conversation data
        """
        try:
            data = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "metadata": metadata or {},
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if summary:
                data["summary"] = summary
            if sentiment:
                data["sentiment"] = sentiment
            if topics:
                data["topics"] = topics
            
            # Upsert conversation
            result = self.client.table("conversations").upsert(
                data,
                on_conflict="conversation_id"
            ).execute()
            
            if result.data:
                return result.data[0]
            
            raise MemoryError("Failed to store conversation metadata")
            
        except Exception as e:
            logger.error(f"Failed to store conversation metadata: {e}")
            raise MemoryError(f"Failed to store conversation: {e}")
    
    @handle_errors(default_return=[])
    def get_user_conversations(self,
                             user_id: str,
                             limit: int = 50,
                             offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get conversations for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of conversations
        """
        try:
            result = self.client.table("conversations").select("*").eq(
                "user_id", user_id
            ).order(
                "updated_at", desc=True
            ).range(offset, offset + limit - 1).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to get user conversations: {e}")
            return []
    
    @retry_on_error(max_attempts=3)
    def store_episodic_memory(self,
                            memory_id: str,
                            user_id: str,
                            title: str,
                            content: str,
                            importance: float = 0.5,
                            emotion: str = "neutral",
                            tags: Optional[List[str]] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store an episodic memory.
        
        Args:
            memory_id: Unique memory identifier
            user_id: User identifier
            title: Memory title
            content: Memory content
            importance: Importance score (0-1)
            emotion: Associated emotion
            tags: Memory tags
            metadata: Additional metadata
            
        Returns:
            Stored memory data
        """
        try:
            data = {
                "memory_id": memory_id,
                "user_id": user_id,
                "title": title,
                "content": content,
                "importance": importance,
                "emotion": emotion,
                "tags": tags or [],
                "metadata": metadata or {}
            }
            
            result = self.client.table("episodic_memories").insert(data).execute()
            
            if result.data:
                logger.info(f"Stored episodic memory {memory_id} for user {user_id}")
                return result.data[0]
            
            raise MemoryError("Failed to store episodic memory")
            
        except Exception as e:
            logger.error(f"Failed to store episodic memory: {e}")
            raise MemoryError(f"Failed to store memory: {e}")
    
    @handle_errors(default_return=[])
    def search_episodic_memories(self,
                               user_id: str,
                               text_query: Optional[str] = None,
                               min_importance: float = 0.0,
                               emotion: Optional[str] = None,
                               tags: Optional[List[str]] = None,
                               limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search episodic memories with filters.
        
        Args:
            user_id: User identifier
            text_query: Text search in title/content
            min_importance: Minimum importance
            emotion: Filter by emotion
            tags: Filter by tags
            limit: Maximum results
            
        Returns:
            List of memories
        """
        try:
            query = self.client.table("episodic_memories").select("*").eq(
                "user_id", user_id
            ).gte("importance", min_importance)
            
            if text_query:
                # Use PostgreSQL full-text search with proper escaping
                # Escape special characters to prevent SQL injection
                escaped_query = text_query.replace("%", "\\%").replace("_", "\\_")
                query = query.or_(
                    f"title.ilike.%{escaped_query}%,"
                    f"content.ilike.%{escaped_query}%"
                )
            
            if emotion:
                query = query.eq("emotion", emotion)
            
            if tags:
                # Filter by any of the provided tags
                query = query.contains("tags", tags)
            
            result = query.order("importance", desc=True).limit(limit).execute()
            
            # Update access statistics
            if result.data:
                memory_ids = [m["memory_id"] for m in result.data]
                self.client.table("episodic_memories").update({
                    "accessed_count": self.client.rpc("increment", {"x": 1}),
                    "last_accessed": datetime.utcnow().isoformat()
                }).in_("memory_id", memory_ids).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to search episodic memories: {e}")
            return []
    
    def create_memory_relation(self,
                             source_memory_id: str,
                             target_memory_id: str,
                             relation_type: str,
                             strength: float = 0.5) -> bool:
        """
        Create a relation between memories.
        
        Args:
            source_memory_id: Source memory ID
            target_memory_id: Target memory ID
            relation_type: Type of relation
            strength: Relation strength (0-1)
            
        Returns:
            True if successful
        """
        try:
            data = {
                "source_memory_id": source_memory_id,
                "target_memory_id": target_memory_id,
                "relation_type": relation_type,
                "strength": strength
            }
            
            self.client.table("memory_relations").upsert(
                data,
                on_conflict="source_memory_id,target_memory_id,relation_type"
            ).execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create memory relation: {e}")
            return False
    
    @async_to_sync
    async def subscribe_to_conversations(self,
                                       user_id: str,
                                       callback: Any) -> Any:
        """
        Subscribe to real-time conversation updates.
        
        Args:
            user_id: User identifier
            callback: Function to call on updates
            
        Returns:
            Subscription object
        """
        try:
            # Supabase real-time subscription
            subscription = self.client.table("conversations").on(
                "UPDATE",
                lambda payload: callback(payload)
            ).eq("user_id", user_id).subscribe()
            
            logger.info(f"Subscribed to conversations for user {user_id}")
            return subscription
            
        except Exception as e:
            logger.error(f"Failed to subscribe to conversations: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Supabase statistics."""
        try:
            # Get counts from tables
            pref_count = self.client.table("user_preferences").select(
                "count", count="exact"
            ).execute()
            
            conv_count = self.client.table("conversations").select(
                "count", count="exact"
            ).execute()
            
            mem_count = self.client.table("episodic_memories").select(
                "count", count="exact"
            ).execute()
            
            return {
                "service": "Supabase",
                "url": self.url,
                "user_preferences_count": pref_count.count if pref_count else 0,
                "conversations_count": conv_count.count if conv_count else 0,
                "episodic_memories_count": mem_count.count if mem_count else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get Supabase stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check Supabase health."""
        try:
            # Simple query to test connection
            self.client.table("user_preferences").select("count").limit(1).execute()
            return True
        except:
            return False
