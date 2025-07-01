"""
MongoDB service for long-term archival storage.
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId

from src.utils.error_handling import retry_on_error, handle_errors, MemoryError

logger = logging.getLogger(__name__)


class MongoDBMemoryService:
    """
    MongoDB-based memory service for long-term archival storage.
    
    This service provides document-based storage for conversations,
    memories, and other data that needs to be preserved long-term.
    """
    
    def __init__(self,
                 connection_string: str = "mongodb://localhost:27017/",
                 database: str = "cleoai_memory",
                 max_pool_size: int = 100):
        """
        Initialize MongoDB memory service.
        
        Args:
            connection_string: MongoDB connection string
            database: Database name
            max_pool_size: Maximum connection pool size
        """
        try:
            self.client = MongoClient(
                connection_string,
                maxPoolSize=max_pool_size,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client[database]
            
            # Initialize collections
            self.conversations = self.db.conversations
            self.memories = self.db.memories
            self.user_states = self.db.user_states
            
            # Create indexes
            self._create_indexes()
            
            # Test connection
            self.client.server_info()
            logger.info(f"Connected to MongoDB database: {database}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise MemoryError(f"MongoDB connection failed: {e}")
    
    def _create_indexes(self):
        """Create necessary indexes for optimal performance."""
        try:
            # Conversation indexes
            self.conversations.create_index([("user_id", ASCENDING)])
            self.conversations.create_index([("created_at", DESCENDING)])
            self.conversations.create_index([
                ("user_id", ASCENDING),
                ("created_at", DESCENDING)
            ])
            
            # Memory indexes
            self.memories.create_index([("user_id", ASCENDING)])
            self.memories.create_index([("type", ASCENDING)])
            self.memories.create_index([("importance", DESCENDING)])
            self.memories.create_index([
                ("user_id", ASCENDING),
                ("type", ASCENDING),
                ("created_at", DESCENDING)
            ])
            
            # TTL index for automatic cleanup
            self.conversations.create_index(
                "expires_at",
                expireAfterSeconds=0,
                sparse=True
            )
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create some indexes: {e}")
    
    @retry_on_error(max_attempts=3)
    def archive_conversation(self,
                           conversation_id: str,
                           user_id: str,
                           messages: List[Dict[str, Any]],
                           metadata: Optional[Dict[str, Any]] = None,
                           retention_days: Optional[int] = None) -> str:
        """
        Archive a complete conversation to MongoDB.
        
        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
            messages: List of conversation messages
            metadata: Additional metadata
            retention_days: Days to retain (None = permanent)
            
        Returns:
            MongoDB document ID
        """
        try:
            document = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "messages": messages,
                "message_count": len(messages),
                "created_at": datetime.utcnow(),
                "metadata": metadata or {},
                "archived": True
            }
            
            # Add expiration if specified
            if retention_days:
                document["expires_at"] = (
                    datetime.utcnow() + timedelta(days=retention_days)
                )
            
            # Calculate conversation statistics
            if messages:
                document["first_message_at"] = messages[0].get("timestamp")
                document["last_message_at"] = messages[-1].get("timestamp")
                document["user_message_count"] = sum(
                    1 for m in messages if m.get("role") == "user"
                )
                document["assistant_message_count"] = sum(
                    1 for m in messages if m.get("role") == "assistant"
                )
            
            result = self.conversations.insert_one(document)
            logger.info(f"Archived conversation {conversation_id} with ID: {result.inserted_id}")
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to archive conversation: {e}")
            raise MemoryError(f"Failed to archive conversation: {e}")
    
    @handle_errors(default_return=None)
    def get_archived_conversation(self, 
                                 conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an archived conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Conversation data or None
        """
        try:
            # Try by conversation_id first
            doc = self.conversations.find_one({"conversation_id": conversation_id})
            
            # Try by ObjectId if not found
            if not doc and ObjectId.is_valid(conversation_id):
                doc = self.conversations.find_one({"_id": ObjectId(conversation_id)})
            
            if doc:
                doc["_id"] = str(doc["_id"])
                return doc
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve archived conversation: {e}")
            return None
    
    @handle_errors(default_return=[])
    def get_user_conversations(self,
                             user_id: str,
                             limit: int = 50,
                             skip: int = 0,
                             include_expired: bool = False) -> List[Dict[str, Any]]:
        """
        Get archived conversations for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number to return
            skip: Number to skip (for pagination)
            include_expired: Include expired conversations
            
        Returns:
            List of conversation documents
        """
        try:
            query = {"user_id": user_id}
            
            # Exclude expired unless requested
            if not include_expired:
                query["$or"] = [
                    {"expires_at": {"$exists": False}},
                    {"expires_at": {"$gt": datetime.utcnow()}}
                ]
            
            cursor = self.conversations.find(query).sort(
                "created_at", DESCENDING
            ).skip(skip).limit(limit)
            
            conversations = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                conversations.append(doc)
            
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to get user conversations: {e}")
            return []
    
    @retry_on_error(max_attempts=3)
    def store_memory_snapshot(self,
                            user_id: str,
                            memory_type: str,
                            data: Dict[str, Any],
                            importance: float = 0.5) -> str:
        """
        Store a memory snapshot.
        
        Args:
            user_id: User identifier
            memory_type: Type of memory (episodic, semantic, etc.)
            data: Memory data
            importance: Importance score (0-1)
            
        Returns:
            Document ID
        """
        try:
            document = {
                "user_id": user_id,
                "type": memory_type,
                "data": data,
                "importance": importance,
                "created_at": datetime.utcnow(),
                "accessed_count": 0,
                "last_accessed": None
            }
            
            result = self.memories.insert_one(document)
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to store memory snapshot: {e}")
            raise MemoryError(f"Failed to store memory: {e}")
    
    @handle_errors(default_return=[])
    def search_memories(self,
                       user_id: str,
                       memory_type: Optional[str] = None,
                       min_importance: float = 0.0,
                       limit: int = 100,
                       text_search: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for memories with filters.
        
        Args:
            user_id: User identifier
            memory_type: Filter by type
            min_importance: Minimum importance threshold
            limit: Maximum results
            text_search: Text to search in memory data
            
        Returns:
            List of memory documents
        """
        try:
            query = {
                "user_id": user_id,
                "importance": {"$gte": min_importance}
            }
            
            if memory_type:
                query["type"] = memory_type
            
            if text_search:
                query["$text"] = {"$search": text_search}
            
            cursor = self.memories.find(query).sort(
                "importance", DESCENDING
            ).limit(limit)
            
            memories = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                # Update access statistics
                self.memories.update_one(
                    {"_id": doc["_id"]},
                    {
                        "$inc": {"accessed_count": 1},
                        "$set": {"last_accessed": datetime.utcnow()}
                    }
                )
                memories.append(doc)
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    def save_user_state(self,
                       user_id: str,
                       state: Dict[str, Any]) -> bool:
        """
        Save complete user state for backup/restore.
        
        Args:
            user_id: User identifier
            state: Complete user state data
            
        Returns:
            True if successful
        """
        try:
            document = {
                "user_id": user_id,
                "state": state,
                "created_at": datetime.utcnow(),
                "version": 1
            }
            
            # Update or insert
            self.user_states.replace_one(
                {"user_id": user_id},
                document,
                upsert=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user state: {e}")
            return False
    
    def get_user_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve saved user state.
        
        Args:
            user_id: User identifier
            
        Returns:
            User state or None
        """
        try:
            doc = self.user_states.find_one(
                {"user_id": user_id},
                sort=[("created_at", DESCENDING)]
            )
            
            if doc:
                return doc.get("state")
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve user state: {e}")
            return None
    
    def cleanup_old_data(self, days: int = 90) -> Dict[str, int]:
        """
        Clean up old data.
        
        Args:
            days: Delete data older than this many days
            
        Returns:
            Deletion counts by collection
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            results = {}
            
            # Clean old conversations (without explicit retention)
            result = self.conversations.delete_many({
                "created_at": {"$lt": cutoff_date},
                "expires_at": {"$exists": False}
            })
            results["conversations"] = result.deleted_count
            
            # Clean old low-importance memories
            result = self.memories.delete_many({
                "created_at": {"$lt": cutoff_date},
                "importance": {"$lt": 0.3},
                "accessed_count": {"$lt": 2}
            })
            results["memories"] = result.deleted_count
            
            logger.info(f"Cleanup completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MongoDB statistics."""
        try:
            stats = self.db.command("dbstats")
            return {
                "database": self.db.name,
                "collections": stats.get("collections", 0),
                "data_size": stats.get("dataSize", 0),
                "storage_size": stats.get("storageSize", 0),
                "conversation_count": self.conversations.count_documents({}),
                "memory_count": self.memories.count_documents({}),
                "user_state_count": self.user_states.count_documents({})
            }
        except Exception as e:
            logger.error(f"Failed to get MongoDB stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check MongoDB health."""
        try:
            self.client.admin.command('ping')
            return True
        except:
            return False
