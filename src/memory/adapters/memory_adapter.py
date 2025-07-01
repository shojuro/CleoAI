"""
Hybrid memory adapter for bridging new microservice memory with existing MemoryManager.
"""
import logging
from typing import Optional, Dict, Any, List
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..memory_manager import MemoryManager, Conversation, UserPreference, EpisodicMemory
from ..services.memory_router import MemoryRouter
from src.utils.error_handling import handle_errors, MemoryError

logger = logging.getLogger(__name__)


class HybridMemoryAdapter:
    """
    Adapter that bridges the new distributed memory services with the existing
    MemoryManager, providing backward compatibility while adding new capabilities.
    """
    
    def __init__(self,
                 legacy_manager: Optional[MemoryManager] = None,
                 memory_router: Optional[MemoryRouter] = None,
                 redis_config: Optional[Dict[str, Any]] = None,
                 supabase_config: Optional[Dict[str, Any]] = None,
                 pinecone_config: Optional[Dict[str, Any]] = None,
                 mongodb_config: Optional[Dict[str, Any]] = None,
                 use_legacy: bool = True,
                 use_distributed: bool = True):
        """
        Initialize hybrid memory adapter.
        
        Args:
            legacy_manager: Existing MemoryManager instance
            memory_router: New MemoryRouter instance
            redis_config: Redis configuration
            supabase_config: Supabase configuration
            pinecone_config: Pinecone configuration
            mongodb_config: MongoDB configuration
            use_legacy: Whether to use legacy storage
            use_distributed: Whether to use distributed storage
        """
        self.use_legacy = use_legacy
        self.use_distributed = use_distributed
        
        # Initialize legacy manager if needed
        if use_legacy and not legacy_manager:
            self.legacy_manager = MemoryManager()
        else:
            self.legacy_manager = legacy_manager
        
        # Initialize memory router if needed
        if use_distributed and not memory_router:
            self.memory_router = MemoryRouter(
                redis_config=redis_config,
                supabase_config=supabase_config,
                pinecone_config=pinecone_config,
                mongodb_config=mongodb_config
            )
        else:
            self.memory_router = memory_router
        
        # Async executor for distributed operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(
            f"Initialized HybridMemoryAdapter "
            f"(legacy={use_legacy}, distributed={use_distributed})"
        )
    
    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    # ========== Conversation Management ==========
    
    @handle_errors(default_return=None)
    def create_conversation(self,
                          user_id: str,
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[Conversation]:
        """
        Create a new conversation.
        
        Args:
            user_id: User identifier
            metadata: Additional metadata
            
        Returns:
            Conversation object or None
        """
        conversation = None
        
        # Create in legacy system
        if self.use_legacy:
            conversation = self.legacy_manager.short_term.create_conversation(
                user_id=user_id,
                metadata=metadata
            )
        
        # Store metadata in distributed system
        if self.use_distributed and conversation:
            try:
                self.memory_router.store_conversation_metadata(
                    conversation_id=conversation.conversation_id,
                    user_id=user_id,
                    metadata=metadata
                )
            except Exception as e:
                logger.warning(f"Failed to store in distributed system: {e}")
        
        return conversation
    
    @handle_errors(default_return=False)
    def add_message(self,
                   conversation_id: str,
                   role: str,
                   content: str,
                   user_id: Optional[str] = None) -> bool:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation identifier
            role: Message role (user/assistant)
            content: Message content
            user_id: User identifier (for distributed storage)
            
        Returns:
            True if successful
        """
        success = True
        
        # Add to legacy system
        if self.use_legacy:
            success = self.legacy_manager.short_term.add_message(
                conversation_id=conversation_id,
                role=role,
                content=content
            )
        
        # Add to distributed system
        if self.use_distributed and user_id:
            try:
                if role == "user":
                    # Store after getting assistant response
                    self._pending_user_message = content
                elif role == "assistant" and hasattr(self, '_pending_user_message'):
                    # Store complete turn
                    self._run_async(
                        self.memory_router.store_conversation_turn(
                            conversation_id=conversation_id,
                            user_id=user_id,
                            user_message=self._pending_user_message,
                            assistant_response=content
                        )
                    )
                    delattr(self, '_pending_user_message')
            except Exception as e:
                logger.warning(f"Failed to store in distributed system: {e}")
        
        return success
    
    @handle_errors(default_return=None)
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Conversation object or None
        """
        # Try distributed system first (Redis cache)
        if self.use_distributed:
            try:
                context = self.memory_router.get_conversation_context(
                    conversation_id=conversation_id,
                    user_id="",  # Not needed for retrieval
                    message_limit=100
                )
                if context:
                    # Convert to Conversation object
                    conv = Conversation(
                        conversation_id=conversation_id,
                        user_id=context.get("user_id", ""),
                        messages=context.get("messages", [])
                    )
                    return conv
            except Exception as e:
                logger.warning(f"Failed to get from distributed system: {e}")
        
        # Fall back to legacy system
        if self.use_legacy:
            return self.legacy_manager.short_term.get_conversation(conversation_id)
        
        return None
    
    # ========== User Preferences ==========
    
    @handle_errors(default_return=None)
    def set_user_preference(self,
                          user_id: str,
                          category: str,
                          key: str,
                          value: Any,
                          confidence: float = 0.5,
                          source: str = "inference") -> Optional[UserPreference]:
        """
        Set a user preference.
        
        Args:
            user_id: User identifier
            category: Preference category
            key: Preference key
            value: Preference value
            confidence: Confidence score
            source: Source of preference
            
        Returns:
            UserPreference object or None
        """
        preference = None
        
        # Store in legacy system
        if self.use_legacy:
            preference = self.legacy_manager.long_term.set_preference(
                user_id=user_id,
                category=category,
                key=key,
                value=value,
                confidence=confidence,
                source=source
            )
        
        # Store in distributed system
        if self.use_distributed:
            try:
                self.memory_router.store_user_preference(
                    user_id=user_id,
                    preference_type=category,
                    preference_key=key,
                    preference_value=value,
                    confidence=confidence,
                    source=source
                )
            except Exception as e:
                logger.warning(f"Failed to store preference in distributed system: {e}")
        
        return preference
    
    @handle_errors(default_return=None)
    def get_user_preference(self,
                          user_id: str,
                          category: str,
                          key: str) -> Optional[UserPreference]:
        """
        Get a user preference.
        
        Args:
            user_id: User identifier
            category: Preference category
            key: Preference key
            
        Returns:
            UserPreference object or None
        """
        # Try distributed system first (faster)
        if self.use_distributed:
            try:
                value = self.memory_router.get_user_preference(
                    user_id=user_id,
                    preference_type=category,
                    preference_key=key
                )
                if value is not None:
                    return UserPreference(
                        user_id=user_id,
                        preference_id=f"{category}:{key}",
                        category=category,
                        preference_key=key,
                        preference_value=value,
                        confidence=0.5,
                        source="distributed"
                    )
            except Exception as e:
                logger.warning(f"Failed to get from distributed system: {e}")
        
        # Fall back to legacy system
        if self.use_legacy:
            return self.legacy_manager.long_term.get_preference(
                user_id=user_id,
                category=category,
                key=key
            )
        
        return None
    
    # ========== Episodic Memory ==========
    
    @handle_errors(default_return=None)
    def create_episodic_memory(self,
                             user_id: str,
                             title: str,
                             content: str,
                             embedding: Optional[List[float]] = None,
                             importance: float = 0.5,
                             emotion: Optional[str] = None) -> Optional[EpisodicMemory]:
        """
        Create an episodic memory.
        
        Args:
            user_id: User identifier
            title: Memory title
            content: Memory content
            embedding: Vector embedding (optional)
            importance: Importance score
            emotion: Associated emotion
            
        Returns:
            EpisodicMemory object or None
        """
        memory = None
        
        # Create in legacy system
        if self.use_legacy:
            memory = self.legacy_manager.episodic.create_memory(
                user_id=user_id,
                title=title,
                content=content,
                importance=importance,
                emotion=emotion
            )
        
        # Store in distributed system with embedding
        if self.use_distributed and memory and embedding:
            try:
                self._run_async(
                    self.memory_router.store_memory_with_embedding(
                        memory_id=memory.memory_id,
                        user_id=user_id,
                        title=title,
                        content=content,
                        embedding=embedding,
                        memory_type="episodic",
                        importance=importance,
                        metadata={"emotion": emotion} if emotion else None
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to store memory in distributed system: {e}")
        
        return memory
    
    @handle_errors(default_return=[])
    def search_memories(self,
                       user_id: str,
                       query_embedding: Optional[List[float]] = None,
                       query_text: Optional[str] = None,
                       limit: int = 10) -> List[EpisodicMemory]:
        """
        Search for memories.
        
        Args:
            user_id: User identifier
            query_embedding: Query vector (for similarity search)
            query_text: Query text (for text search)
            limit: Maximum results
            
        Returns:
            List of EpisodicMemory objects
        """
        memories = []
        
        # Vector search in distributed system
        if self.use_distributed and query_embedding:
            try:
                results = self.memory_router.search_memories(
                    user_id=user_id,
                    query_embedding=query_embedding,
                    limit=limit
                )
                
                # Convert to EpisodicMemory objects
                for result in results:
                    metadata = result.get("metadata", {})
                    memory = EpisodicMemory(
                        memory_id=result["id"],
                        user_id=user_id,
                        title=metadata.get("title", ""),
                        content=result.get("content", ""),
                        importance=metadata.get("importance", 0.5),
                        emotion=metadata.get("emotion")
                    )
                    memories.append(memory)
                    
                return memories
            except Exception as e:
                logger.warning(f"Failed to search in distributed system: {e}")
        
        # Text search in legacy system
        if self.use_legacy and query_text:
            return self.legacy_manager.episodic.search_memories(
                query=query_text,
                user_id=user_id,
                limit=limit
            )
        
        return memories
    
    # ========== Utility Methods ==========
    
    def migrate_to_distributed(self, user_id: str) -> Dict[str, int]:
        """
        Migrate a user's data from legacy to distributed storage.
        
        Args:
            user_id: User identifier
            
        Returns:
            Migration statistics
        """
        stats = {
            "preferences": 0,
            "memories": 0,
            "conversations": 0
        }
        
        if not self.use_legacy or not self.use_distributed:
            logger.warning("Both legacy and distributed systems must be enabled for migration")
            return stats
        
        # Migrate preferences
        try:
            preferences = self.legacy_manager.long_term.get_user_preferences(user_id)
            for pref in preferences:
                self.memory_router.store_user_preference(
                    user_id=user_id,
                    preference_type=pref.category,
                    preference_key=pref.preference_key,
                    preference_value=pref.preference_value,
                    confidence=pref.confidence,
                    source=pref.source
                )
                stats["preferences"] += 1
        except Exception as e:
            logger.error(f"Failed to migrate preferences: {e}")
        
        # Migrate memories
        try:
            memories = self.legacy_manager.episodic.get_user_memories(user_id, limit=1000)
            for memory in memories:
                # Note: Would need embeddings for full migration
                stats["memories"] += 1
        except Exception as e:
            logger.error(f"Failed to migrate memories: {e}")
        
        logger.info(f"Migration stats for user {user_id}: {stats}")
        return stats
    
    def get_unified_user_state(self, user_id: str) -> Dict[str, Any]:
        """
        Get unified user state from all storage systems.
        
        Args:
            user_id: User identifier
            
        Returns:
            Combined user state
        """
        state = {
            "user_id": user_id,
            "sources": []
        }
        
        # Get from legacy system
        if self.use_legacy:
            try:
                legacy_state = self.legacy_manager.get_user_state(user_id)
                state["legacy"] = legacy_state
                state["sources"].append("legacy")
            except Exception as e:
                logger.warning(f"Failed to get legacy state: {e}")
        
        # Get from distributed system
        if self.use_distributed:
            try:
                # Get service statistics
                stats = self.memory_router.get_service_stats()
                state["distributed_stats"] = stats
                state["sources"].append("distributed")
            except Exception as e:
                logger.warning(f"Failed to get distributed stats: {e}")
        
        return state
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all systems."""
        health = {
            "adapter": "healthy",
            "systems": {}
        }
        
        if self.use_legacy:
            health["systems"]["legacy"] = "healthy"  # Legacy doesn't have health check
        
        if self.use_distributed:
            health["systems"]["distributed"] = self.memory_router.health_check()
        
        return health
    
    def close(self):
        """Close all connections."""
        self.executor.shutdown(wait=True)
        
        if self.use_distributed:
            self.memory_router.close()
