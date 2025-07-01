
"""
Memory management systems for the AI Autonomous Agent.
This module implements various memory types: short-term, long-term, episodic, and procedural.
"""
import os
import json
import time
import datetime
import logging
import sqlite3
import redis
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import uuid
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.utils.error_handling import (
    MemoryError,
    ErrorSeverity,
    retry_on_error,
    handle_errors,
    ErrorContext,
    validate_input,
    ResourceManager,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Conversation:
    """Data class for storing conversation information"""
    conversation_id: str
    user_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(
        self, 
        role: str, 
        content: str, 
        timestamp: Optional[float] = None,
        message_id: Optional[str] = None
    ):
        """Add a message to the conversation"""
        if timestamp is None:
            timestamp = time.time()
            
        if message_id is None:
            message_id = str(uuid.uuid4())
            
        message = {
            "message_id": message_id,
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        
        self.messages.append(message)
        self.updated_at = timestamp
        
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from the conversation"""
        if limit is None:
            return self.messages
        return self.messages[-limit:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create a conversation from a dictionary"""
        return cls(
            conversation_id=data["conversation_id"],
            user_id=data["user_id"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            messages=data.get("messages", []),
            metadata=data.get("metadata", {})
        )

@dataclass
class UserPreference:
    """Data class for storing user preferences"""
    user_id: str
    preference_id: str
    preference_type: str  # e.g., "food", "travel", "personality"
    value: str
    confidence: float  # 0-1 confidence score
    source: str  # e.g., "explicit", "inferred", "default"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "preference_id": self.preference_id,
            "preference_type": self.preference_type,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreference':
        """Create a user preference from a dictionary"""
        return cls(
            user_id=data["user_id"],
            preference_id=data["preference_id"],
            preference_type=data["preference_type"],
            value=data["value"],
            confidence=data["confidence"],
            source=data["source"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {})
        )

@dataclass
class EpisodicMemory:
    """Data class for storing episodic memories"""
    memory_id: str
    user_id: str
    title: str
    content: str
    created_at: float = field(default_factory=time.time)
    importance: float = 0.5  # 0-1 importance score
    emotion: str = "neutral"  # emotional tone of the memory
    embedding: Optional[List[float]] = None
    related_memories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "created_at": self.created_at,
            "importance": self.importance,
            "emotion": self.emotion,
            "embedding": self.embedding,
            "related_memories": self.related_memories,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        """Create an episodic memory from a dictionary"""
        return cls(
            memory_id=data["memory_id"],
            user_id=data["user_id"],
            title=data["title"],
            content=data["content"],
            created_at=data.get("created_at", time.time()),
            importance=data.get("importance", 0.5),
            emotion=data.get("emotion", "neutral"),
            embedding=data.get("embedding"),
            related_memories=data.get("related_memories", []),
            metadata=data.get("metadata", {})
        )

@dataclass
class ProceduralMemory:
    """Data class for storing procedural memories (e.g., how to perform tasks)"""
    protocol_id: str
    name: str
    steps: List[Dict[str, Any]]
    description: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "protocol_id": self.protocol_id,
            "name": self.name,
            "steps": self.steps,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProceduralMemory':
        """Create a procedural memory from a dictionary"""
        return cls(
            protocol_id=data["protocol_id"],
            name=data["name"],
            steps=data["steps"],
            description=data.get("description", ""),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )

class ShortTermMemory:
    """
    Short-term memory system for storing recent conversation history.
    """
    def __init__(
        self,
        max_tokens: int = 16384,
        recency_weight_decay: float = 0.98,
        tokenizer_name: str = "mistralai/Mistral-7B-v0.1"
    ):
        self.max_tokens = max_tokens
        self.recency_weight_decay = recency_weight_decay
        self.conversations = {}  # conversation_id -> Conversation
        
        # Load tokenizer for token counting
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        logger.info(f"Initialized ShortTermMemory with max_tokens={max_tokens}")
    
    def add_message(
        self, 
        conversation_id: str, 
        user_id: str, 
        role: str, 
        content: str,
        timestamp: Optional[float] = None,
        message_id: Optional[str] = None
    ):
        """Add a message to a conversation"""
        # Create conversation if it doesn't exist
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = Conversation(
                conversation_id=conversation_id,
                user_id=user_id
            )
            
        # Add message to conversation
        self.conversations[conversation_id].add_message(
            role=role,
            content=content,
            timestamp=timestamp,
            message_id=message_id
        )
        
        # Truncate if needed
        self._truncate_conversation(conversation_id)
    
    def _truncate_conversation(self, conversation_id: str):
        """Truncate a conversation if it exceeds max_tokens"""
        if conversation_id not in self.conversations:
            return
            
        conversation = self.conversations[conversation_id]
        messages = conversation.get_messages()
        
        # Count tokens
        all_text = " ".join([msg["content"] for msg in messages])
        tokens = self.tokenizer.encode(all_text)
        
        # If within limit, no truncation needed
        if len(tokens) <= self.max_tokens:
            return
            
        # Truncation needed - remove oldest messages until under limit
        while len(tokens) > self.max_tokens and len(messages) > 1:
            # Always keep at least the most recent message
            messages.pop(0)  # Remove oldest message
            
            # Recount tokens
            all_text = " ".join([msg["content"] for msg in messages])
            tokens = self.tokenizer.encode(all_text)
            
        # Update conversation
        conversation.messages = messages
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        return self.conversations.get(conversation_id)
    
    def get_conversation_history(
        self, 
        conversation_id: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
            
        return conversation.get_messages(limit)
    
    def get_weighted_conversation(
        self, 
        conversation_id: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history with recency weights"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
            
        messages = conversation.get_messages(limit)
        
        # Calculate recency weights
        weights = []
        for i in range(len(messages)):
            weight = self.recency_weight_decay ** (len(messages) - i - 1)
            weights.append(weight)
            
        # Add weights to messages
        for i, message in enumerate(messages):
            message["weight"] = weights[i]
            
        return messages
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def save_to_disk(self, path: str):
        """Save conversations to disk"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save each conversation to a separate file
        for conversation_id, conversation in self.conversations.items():
            file_path = save_path / f"{conversation_id}.json"
            with open(file_path, "w") as f:
                json.dump(conversation.to_dict(), f, indent=2)
                
        logger.info(f"Saved {len(self.conversations)} conversations to {path}")
    
    def load_from_disk(self, path: str):
        """Load conversations from disk"""
        load_path = Path(path)
        if not load_path.exists():
            logger.warning(f"Path {path} does not exist, no conversations loaded")
            return
            
        # Load each conversation from its file
        for file_path in load_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    conversation = Conversation.from_dict(data)
                    self.conversations[conversation.conversation_id] = conversation
            except Exception as e:
                logger.error(f"Error loading conversation from {file_path}: {e}")
                
        logger.info(f"Loaded {len(self.conversations)} conversations from {path}")

class LongTermMemory:
    """
    Long-term memory system for storing user preferences and semantic information.
    Uses both vector storage (for semantic search) and relational storage (for structured data).
    """
    def __init__(
        self,
        vector_db_path: str = "data/vector_db",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        sqlite_path: str = "data/preferences.db",
        vector_dimension: int = 768
    ):
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model
        self.sqlite_path = sqlite_path
        self.vector_dimension = vector_dimension
        
        # Initialize vector database
        self._init_vector_db()
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Initialize SQLite database
        self._init_sqlite_db()
        
        logger.info(f"Initialized LongTermMemory with vector DB at {vector_db_path}")
    
    def _init_vector_db(self):
        """Initialize vector database"""
        # Create directory if it doesn't exist
        Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.vector_db = chromadb.PersistentClient(path=self.vector_db_path)
        
        # Create collection if it doesn't exist
        try:
            self.preference_collection = self.vector_db.get_collection("preferences")
            logger.info("Using existing preferences collection in vector DB")
        except Exception:
            self.preference_collection = self.vector_db.create_collection(
                name="preferences",
                metadata={"description": "User preferences"}
            )
            logger.info("Created new preferences collection in vector DB")
    
    def _init_embedding_model(self):
        """Initialize embedding model"""
        # Load model
        self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.to("cuda")
            
        # Set to evaluation mode
        self.embedding_model.eval()
        
        logger.info(f"Loaded embedding model {self.embedding_model_name}")
    
    def _init_sqlite_db(self):
        """Initialize SQLite database"""
        # Create directory if it doesn't exist
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(self.sqlite_path)
        cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            preference_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            preference_type TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            metadata TEXT
        )
        ''')
        
        # Create indexes
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id 
        ON user_preferences(user_id)
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_preferences_type 
        ON user_preferences(preference_type)
        ''')
        
        # Commit changes
        self.conn.commit()
        
        logger.info(f"Initialized SQLite database at {self.sqlite_path}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        # Tokenize and encode
        tokens = self.embedding_model.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            tokens = {k: v.to("cuda") for k, v in tokens.items()}
            
        # Get embedding
        with torch.no_grad():
            model_output = self.embedding_model(**tokens)
            
        # Use mean pooling
        embedding = torch.mean(model_output.last_hidden_state, dim=1).cpu().numpy()[0]
        
        return embedding.tolist()
    
    def store_preference(self, preference: UserPreference):
        """Store a user preference"""
        # Create embedding for the preference
        text_to_embed = f"{preference.preference_type}: {preference.value}"
        embedding = self._get_embedding(text_to_embed)
        
        # Update vector database
        self.preference_collection.upsert(
            ids=[preference.preference_id],
            embeddings=[embedding],
            metadatas=[{
                "user_id": preference.user_id,
                "preference_type": preference.preference_type,
                "value": preference.value,
                "confidence": preference.confidence
            }],
            documents=[text_to_embed]
        )
        
        # Update SQLite database
        cursor = self.conn.cursor()
        
        # Check if preference already exists
        cursor.execute(
            "SELECT preference_id FROM user_preferences WHERE preference_id = ?",
            (preference.preference_id,)
        )
        
        if cursor.fetchone():
            # Update existing preference
            cursor.execute('''
            UPDATE user_preferences
            SET value = ?, confidence = ?, source = ?, updated_at = ?, metadata = ?
            WHERE preference_id = ?
            ''', (
                preference.value,
                preference.confidence,
                preference.source,
                preference.updated_at,
                json.dumps(preference.metadata),
                preference.preference_id
            ))
        else:
            # Insert new preference
            cursor.execute('''
            INSERT INTO user_preferences
            (preference_id, user_id, preference_type, value, confidence, source, 
             created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                preference.preference_id,
                preference.user_id,
                preference.preference_type,
                preference.value,
                preference.confidence,
                preference.source,
                preference.created_at,
                preference.updated_at,
                json.dumps(preference.metadata)
            ))
        
        # Commit changes
        self.conn.commit()
        
        logger.info(f"Stored preference {preference.preference_id} for user {preference.user_id}")
    
    def get_preference(self, user_id: str, preference_type: str) -> Optional[UserPreference]:
        """Get a user preference by type"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
        SELECT * FROM user_preferences
        WHERE user_id = ? AND preference_type = ?
        ORDER BY confidence DESC, updated_at DESC
        LIMIT 1
        ''', (user_id, preference_type))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        # Convert row to UserPreference
        preference = UserPreference(
            preference_id=row[0],
            user_id=row[1],
            preference_type=row[2],
            value=row[3],
            confidence=row[4],
            source=row[5],
            created_at=row[6],
            updated_at=row[7],
            metadata=json.loads(row[8]) if row[8] else {}
        )
        
        return preference
    
    def get_all_user_preferences(self, user_id: str) -> List[UserPreference]:
        """Get all preferences for a user"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
        SELECT * FROM user_preferences
        WHERE user_id = ?
        ORDER BY confidence DESC, updated_at DESC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        
        preferences = []
        for row in rows:
            preference = UserPreference(
                preference_id=row[0],
                user_id=row[1],
                preference_type=row[2],
                value=row[3],
                confidence=row[4],
                source=row[5],
                created_at=row[6],
                updated_at=row[7],
                metadata=json.loads(row[8]) if row[8] else {}
            )
            preferences.append(preference)
            
        return preferences
    
    def search_preferences(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[UserPreference]:
        """Search for preferences using semantic search"""
        # Create embedding for query
        query_embedding = self._get_embedding(query)
        
        # Build filters
        filters = {}
        if user_id:
            filters["user_id"] = user_id
            
        # Search in vector database
        results = self.preference_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=filters if filters else None
        )
        
        # Convert results to UserPreference objects
        preferences = []
        for i, doc_id in enumerate(results["ids"][0]):
            # Get preference from SQLite
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM user_preferences WHERE preference_id = ?",
                (doc_id,)
            )
            
            row = cursor.fetchone()
            if row:
                preference = UserPreference(
                    preference_id=row[0],
                    user_id=row[1],
                    preference_type=row[2],
                    value=row[3],
                    confidence=row[4],
                    source=row[5],
                    created_at=row[6],
                    updated_at=row[7],
                    metadata=json.loads(row[8]) if row[8] else {}
                )
                preferences.append(preference)
                
        return preferences
    
    def infer_preference(
        self, 
        user_id: str, 
        text: str, 
        confidence: float = 0.6,
        source: str = "inferred"
    ) -> Optional[UserPreference]:
        """Infer a user preference from text"""
        # This is a placeholder for a more sophisticated inference system
        # In a real system, you would use an ML model to extract preferences
        
        # For now, we'll use a simple keyword-based approach
        preference_keywords = {
            "food": ["food", "cuisine", "dish", "restaurant", "eat", "meal", "dinner", "lunch"],
            "travel": ["travel", "vacation", "trip", "destination", "country", "city", "visit"],
            "hobby": ["hobby", "interest", "enjoy", "pastime", "sport", "game", "activity"],
            "music": ["music", "song", "artist", "band", "concert", "listen", "genre"],
            "movie": ["movie", "film", "watch", "cinema", "actor", "director", "genre"]
        }
        
        text_lower = text.lower()
        
        for preference_type, keywords in preference_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Extract a sentence containing the keyword
                    sentences = text.split(".")
                    relevant_sentence = ""
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            relevant_sentence = sentence.strip()
                            break
                            
                    if relevant_sentence:
                        # Create a preference
                        preference = UserPreference(
                            user_id=user_id,
                            preference_id=str(uuid.uuid4()),
                            preference_type=preference_type,
                            value=relevant_sentence,
                            confidence=confidence,
                            source=source
                        )
                        
                        # Store preference
                        self.store_preference(preference)
                        
                        return preference
                        
        return None
    
    def save_to_disk(self, path: str):
        """Save preferences to disk"""
        # SQLite database is already persistent
        # Vector DB is already persistent
        logger.info(f"LongTermMemory data is already persistent at {self.sqlite_path} and {self.vector_db_path}")
        
    def load_from_disk(self, path: str):
        """Load preferences from disk"""
        # SQLite database is already persistent
        # Vector DB is already persistent
        logger.info(f"LongTermMemory data is already loaded from {self.sqlite_path} and {self.vector_db_path}")

class EpisodicMemorySystem:
    """
    Episodic memory system for storing memories of past experiences.
    """
    def __init__(
        self,
        vector_db_path: str = "data/vector_db",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        sqlite_path: str = "data/episodic.db",
        vector_dimension: int = 768
    ):
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model
        self.sqlite_path = sqlite_path
        self.vector_dimension = vector_dimension
        
        # Initialize vector database
        self._init_vector_db()
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Initialize SQLite database
        self._init_sqlite_db()
        
        logger.info(f"Initialized EpisodicMemorySystem with vector DB at {vector_db_path}")
    
    def _init_vector_db(self):
        """Initialize vector database"""
        # Create directory if it doesn't exist
        Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.vector_db = chromadb.PersistentClient(path=self.vector_db_path)
        
        # Create collection if it doesn't exist
        try:
            self.memory_collection = self.vector_db.get_collection("episodic_memories")
            logger.info("Using existing episodic_memories collection in vector DB")
        except Exception:
            self.memory_collection = self.vector_db.create_collection(
                name="episodic_memories",
                metadata={"description": "Episodic memories"}
            )
            logger.info("Created new episodic_memories collection in vector DB")
    
    def _init_embedding_model(self):
        """Initialize embedding model"""
        # Load model
        self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.to("cuda")
            
        # Set to evaluation mode
        self.embedding_model.eval()
        
        logger.info(f"Loaded embedding model {self.embedding_model_name}")
    
    def _init_sqlite_db(self):
        """Initialize SQLite database"""
        # Create directory if it doesn't exist
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(self.sqlite_path)
        cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS episodic_memories (
            memory_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            importance REAL NOT NULL,
            emotion TEXT NOT NULL,
            metadata TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory_relations (
            memory_id TEXT NOT NULL,
            related_memory_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            weight REAL NOT NULL,
            PRIMARY KEY (memory_id, related_memory_id),
            FOREIGN KEY (memory_id) REFERENCES episodic_memories(memory_id),
            FOREIGN KEY (related_memory_id) REFERENCES episodic_memories(memory_id)
        )
        ''')
        
        # Create indexes
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_episodic_memories_user_id 
        ON episodic_memories(user_id)
        ''')
        
        # Commit changes
        self.conn.commit()
        
        logger.info(f"Initialized SQLite database at {self.sqlite_path}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        # Tokenize and encode
        tokens = self.embedding_model.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            tokens = {k: v.to("cuda") for k, v in tokens.items()}
            
        # Get embedding
        with torch.no_grad():
            model_output = self.embedding_model(**tokens)
            
        # Use mean pooling
        embedding = torch.mean(model_output.last_hidden_state, dim=1).cpu().numpy()[0]
        
        return embedding.tolist()
    
    def store_memory(self, memory: EpisodicMemory):
        """Store an episodic memory"""
        # Create embedding if not provided
        if not memory.embedding:
            text_to_embed = f"{memory.title}: {memory.content[:1000]}"  # Limit content length
            memory.embedding = self._get_embedding(text_to_embed)
        
        # Update vector database
        self.memory_collection.upsert(
            ids=[memory.memory_id],
            embeddings=[memory.embedding],
            metadatas=[{
                "user_id": memory.user_id,
                "title": memory.title,
                "importance": memory.importance,
                "emotion": memory.emotion
            }],
            documents=[memory.content[:1000]]  # Limit content length
        )
        
        # Update SQLite database
        cursor = self.conn.cursor()
        
        # Check if memory already exists
        cursor.execute(
            "SELECT memory_id FROM episodic_memories WHERE memory_id = ?",
            (memory.memory_id,)
        )
        
        if cursor.fetchone():
            # Update existing memory
            cursor.execute('''
            UPDATE episodic_memories
            SET title = ?, content = ?, importance = ?, emotion = ?, metadata = ?
            WHERE memory_id = ?
            ''', (
                memory.title,
                memory.content,
                memory.importance,
                memory.emotion,
                json.dumps(memory.metadata),
                memory.memory_id
            ))
        else:
            # Insert new memory
            cursor.execute('''
            INSERT INTO episodic_memories
            (memory_id, user_id, title, content, created_at, importance, emotion, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory.memory_id,
                memory.user_id,
                memory.title,
                memory.content,
                memory.created_at,
                memory.importance,
                memory.emotion,
                json.dumps(memory.metadata)
            ))
        
        # Store memory relations
        for related_memory_id in memory.related_memories:
            cursor.execute('''
            INSERT OR REPLACE INTO memory_relations
            (memory_id, related_memory_id, relation_type, weight)
            VALUES (?, ?, ?, ?)
            ''', (
                memory.memory_id,
                related_memory_id,
                "related",  # Default relation type
                0.5  # Default weight
            ))
        
        # Commit changes
        self.conn.commit()
        
        logger.info(f"Stored episodic memory {memory.memory_id} for user {memory.user_id}")
    
    """
    Memory Management System.
    Provides classes for managing different types of memory (short-term, long-term, episodic, procedural).
    """
    import os
    import json
    import time
    import sqlite3
    import logging
    from datetime import datetime
    from typing import Dict, List, Optional, Union, Any
    from collections import defaultdict
    import uuid
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    class Conversation:
        """Represents a conversation between a user and the AI."""
        
        def __init__(self, 
                     conversation_id: str, 
                     user_id: str, 
                     messages: Optional[List[Dict[str, str]]] = None,
                     created_at: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None):
            """
            Initialize a conversation.
            
            Args:
                conversation_id: Unique identifier for the conversation
                user_id: ID of the user involved in the conversation
                messages: List of message dictionaries with 'role', 'content', and 'timestamp'
                created_at: ISO format datetime string when conversation was created
                metadata: Additional metadata for the conversation
            """
            self.conversation_id = conversation_id
            self.user_id = user_id
            self.messages = messages or []
            self.created_at = created_at or datetime.now().isoformat()
            self.metadata = metadata or {}
        
        def add_message(self, role: str, content: str, timestamp: Optional[str] = None):
            """
            Add a message to the conversation.
            
            Args:
                role: Role of the message sender ('user', 'assistant', or 'system')
                content: Content of the message
                timestamp: ISO format datetime string when message was created
            """
            self.messages.append({
                "role": role,
                "content": content,
                "timestamp": timestamp or datetime.now().isoformat()
            })
        
        def get_last_user_message(self) -> Optional[Dict[str, str]]:
            """Get the last message from the user."""
            for message in reversed(self.messages):
                if message["role"] == "user":
                    return message
            return None
        
        def get_last_assistant_message(self) -> Optional[Dict[str, str]]:
            """Get the last message from the assistant."""
            for message in reversed(self.messages):
                if message["role"] == "assistant":
                    return message
            return None
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert the conversation to a dictionary."""
            return {
                "conversation_id": self.conversation_id,
                "user_id": self.user_id,
                "messages": self.messages,
                "created_at": self.created_at,
                "metadata": self.metadata
            }
    
    class UserPreference:
        """Represents a user preference in the long-term memory."""
        
        def __init__(self, 
                     preference_id: str,
                     user_id: str,
                     category: str,
                     preference_key: str,
                     preference_value: Any,
                     confidence: float = 0.0,
                     created_at: Optional[str] = None,
                     updated_at: Optional[str] = None,
                     source: str = "inference",
                     metadata: Optional[Dict[str, Any]] = None):
            """
            Initialize a user preference.
            
            Args:
                preference_id: Unique identifier for the preference
                user_id: ID of the user
                category: Category of the preference (e.g., 'dating', 'food')
                preference_key: Key for the preference (e.g., 'likes_dogs')
                preference_value: Value of the preference
                confidence: Confidence score for this preference (0.0-1.0)
                created_at: ISO format datetime string when preference was created
                updated_at: ISO format datetime string when preference was last updated
                source: Source of the preference ('explicit', 'inference', etc.)
                metadata: Additional metadata for the preference
            """
            self.preference_id = preference_id
            self.user_id = user_id
            self.category = category
            self.preference_key = preference_key
            self.preference_value = preference_value
            self.confidence = confidence
            self.created_at = created_at or datetime.now().isoformat()
            self.updated_at = updated_at or self.created_at
            self.source = source
            self.metadata = metadata or {}
        
        def update_value(self, new_value: Any, confidence: Optional[float] = None):
            """
            Update the value of the preference.
            
            Args:
                new_value: New value for the preference
                confidence: New confidence score (optional)
            """
            self.preference_value = new_value
            if confidence is not None:
                self.confidence = confidence
            self.updated_at = datetime.now().isoformat()
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert the preference to a dictionary."""
            return {
                "preference_id": self.preference_id,
                "user_id": self.user_id,
                "category": self.category,
                "preference_key": self.preference_key,
                "preference_value": self.preference_value,
                "confidence": self.confidence,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "source": self.source,
                "metadata": self.metadata
            }
    
    class EpisodicMemory:
        """Represents an episodic memory in the episodic memory system."""
        
        def __init__(self,
                     memory_id: str,
                     user_id: str,
                     title: str,
                     content: str,
                     created_at: Optional[str] = None,
                     importance: float = 0.5,
                     emotion: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     related_memories: Optional[List[str]] = None):
            """
            Initialize an episodic memory.
            
            Args:
                memory_id: Unique identifier for the memory
                user_id: ID of the user
                title: Title/summary of the memory
                content: Detailed content of the memory
                created_at: ISO format datetime string when memory was created
                importance: Importance score for this memory (0.0-1.0)
                emotion: Primary emotion associated with this memory
                metadata: Additional metadata for the memory
                related_memories: List of related memory IDs
            """
            self.memory_id = memory_id
            self.user_id = user_id
            self.title = title
            self.content = content
            self.created_at = created_at or datetime.now().isoformat()
            self.importance = importance
            self.emotion = emotion
            self.metadata = metadata or {}
            self.related_memories = related_memories or []
        
        def add_related_memory(self, memory_id: str):
            """Add a related memory ID."""
            if memory_id not in self.related_memories:
                self.related_memories.append(memory_id)
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert the memory to a dictionary."""
            return {
                "memory_id": self.memory_id,
                "user_id": self.user_id,
                "title": self.title,
                "content": self.content,
                "created_at": self.created_at,
                "importance": self.importance,
                "emotion": self.emotion,
                "metadata": self.metadata,
                "related_memories": self.related_memories
            }
    
    class ProceduralMemory:
        """Represents a procedural memory/protocol in the procedural memory system."""
        
        def __init__(self,
                     protocol_id: str,
                     name: str,
                     steps: List[Dict[str, Any]],
                     category: Optional[str] = None,
                     description: Optional[str] = None,
                     created_at: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None):
            """
            Initialize a procedural memory/protocol.
            
            Args:
                protocol_id: Unique identifier for the protocol
                name: Name of the protocol
                steps: List of steps to execute the protocol
                category: Category of the protocol
                description: Description of the protocol
                created_at: ISO format datetime string when protocol was created
                metadata: Additional metadata for the protocol
            """
            self.protocol_id = protocol_id
            self.name = name
            self.steps = steps
            self.category = category
            self.description = description
            self.created_at = created_at or datetime.now().isoformat()
            self.metadata = metadata or {}
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert the protocol to a dictionary."""
            return {
                "protocol_id": self.protocol_id,
                "name": self.name,
                "steps": self.steps,
                "category": self.category,
                "description": self.description,
                "created_at": self.created_at,
                "metadata": self.metadata
            }
    
    class ShortTermMemory:
        """Manages short-term memory (conversation history)."""
        
        def __init__(self, db_path: str, max_conversations: int = 100):
            """
            Initialize short-term memory.
            
            Args:
                db_path: Path to the SQLite database
                max_conversations: Maximum number of conversations to keep in memory
            """
            self.db_path = db_path
            self.max_conversations = max_conversations
            self.conn = self._init_db()
        
        def _init_db(self) -> sqlite3.Connection:
            """Initialize the database connection and tables."""
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            
            # Create tables if they don't exist
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                metadata TEXT
            )
            ''')
            
            # Messages table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
            )
            ''')
            
            conn.commit()
            return conn
        
        def create_conversation(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> Conversation:
            """
            Create a new conversation.
            
            Args:
                user_id: ID of the user
                metadata: Additional metadata for the conversation
                
            Returns:
                A new Conversation object
            """
            conversation_id = str(uuid.uuid4())
            created_at = datetime.now().isoformat()
            
            cursor = self.conn.cursor()
            
            # Insert conversation into database
            cursor.execute(
                "INSERT INTO conversations (conversation_id, user_id, created_at, last_updated, metadata) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, user_id, created_at, created_at, json.dumps(metadata or {}))
            )
            
            self.conn.commit()
            
            # Prune old conversations if needed
            self._prune_old_conversations()
            
            return Conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                created_at=created_at,
                metadata=metadata or {}
            )
        
        def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
            """
            Get a conversation by ID.
            
            Args:
                conversation_id: ID of the conversation
                
            Returns:
                Conversation object or None if not found
            """
            cursor = self.conn.cursor()
            
            # Get conversation from database
            cursor.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
                
            # Get messages for conversation
            cursor.execute(
                "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,)
            )
            
            messages = []
            for msg_row in cursor.fetchall():
                messages.append({
                    "role": msg_row[0],
                    "content": msg_row[1],
                    "timestamp": msg_row[2]
                })
            
            # Create conversation object
            conversation = Conversation(
                conversation_id=row[0],
                user_id=row[1],
                created_at=row[2],
                messages=messages,
                metadata=json.loads(row[4]) if row[4] else {}
            )
            
            return conversation
        
        def add_message(self, conversation_id: str, role: str, content: str) -> bool:
            """
            Add a message to a conversation.
            
            Args:
                conversation_id: ID of the conversation
                role: Role of the message sender ('user', 'assistant', or 'system')
                content: Content of the message
                
            Returns:
                True if successful, False otherwise
            """
            cursor = self.conn.cursor()
            
            # Check if conversation exists
            cursor.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (conversation_id,)
            )
            
            if not cursor.fetchone():
                return False
                
            # Add message to database
            message_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT INTO messages (message_id, conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content, timestamp)
            )
            
            # Update last_updated timestamp for conversation
            cursor.execute(
                "UPDATE conversations SET last_updated = ? WHERE conversation_id = ?",
                (timestamp, conversation_id)
            )
            
            self.conn.commit()
            
            return True
        
        def get_user_conversations(self, user_id: str, limit: int = 10) -> List[Conversation]:
            """
            Get conversations for a user.
            
            Args:
                user_id: ID of the user
                limit: Maximum number of conversations to return
                
            Returns:
                List of Conversation objects
            """
            cursor = self.conn.cursor()
            
            # Get conversations from database
            cursor.execute(
                "SELECT conversation_id FROM conversations WHERE user_id = ? ORDER BY last_updated DESC LIMIT ?",
                (user_id, limit)
            )
            
            conversations = []
            for row in cursor.fetchall():
                conversation = self.get_conversation(row[0])
                if conversation:
                    conversations.append(conversation)
            
            return conversations
        
        def _prune_old_conversations(self):
            """Prune old conversations to stay within the maximum limit."""
            cursor = self.conn.cursor()
            
            # Get count of conversations
            cursor.execute("SELECT COUNT(*) FROM conversations")
            count = cursor.fetchone()[0]
            
            if count <= self.max_conversations:
                return
                
            # Delete oldest conversations
            to_delete = count - self.max_conversations
            
            cursor.execute(
                """
                DELETE FROM messages 
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversations 
                    ORDER BY last_updated ASC 
                    LIMIT ?
                )
                """,
                (to_delete,)
            )
            
            cursor.execute(
                "DELETE FROM conversations ORDER BY last_updated ASC LIMIT ?",
                (to_delete,)
            )
            
            self.conn.commit()
    
    class LongTermMemory:
        """Manages long-term memory (user preferences and knowledge)."""
        
        def __init__(self, db_path: str):
            """
            Initialize long-term memory.
            
            Args:
                db_path: Path to the SQLite database
            """
            self.db_path = db_path
            self.conn = self._init_db()
        
        def _init_db(self) -> sqlite3.Connection:
            """Initialize the database connection and tables."""
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            
            # Create tables if they don't exist
            cursor = conn.cursor()
            
            # User preferences table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                preference_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(user_id, category, preference_key)
            )
            ''')
            
            conn.commit()
            return conn
        
        def set_preference(self, user_id: str, category: str, key: str, value: Any, 
                          confidence: float = 0.5, source: str = "inference", 
                          metadata: Optional[Dict[str, Any]] = None) -> UserPreference:
            """
            Set or update a user preference.
            
            Args:
                user_id: ID of the user
                category: Category of the preference
                key: Key for the preference
                value: Value of the preference
                confidence: Confidence score for this preference
                source: Source of the preference
                metadata: Additional metadata for the preference
                
            Returns:
                UserPreference object for the new/updated preference
            """
            cursor = self.conn.cursor()
            
            # Check if preference already exists
            cursor.execute(
                "SELECT preference_id, created_at FROM user_preferences WHERE user_id = ? AND category = ? AND preference_key = ?",
                (user_id, category, key)
            )
            
            row = cursor.fetchone()
            now = datetime.now().isoformat()
            
            if row:
                # Update existing preference
                preference_id = row[0]
                created_at = row[1]
                
                cursor.execute(
                    """
                    UPDATE user_preferences 
                    SET preference_value = ?, confidence = ?, updated_at = ?, source = ?, metadata = ?
                    WHERE preference_id = ?
                    """,
                    (
                        json.dumps(value), 
                        confidence, 
                        now, 
                        source, 
                        json.dumps(metadata or {}),
                        preference_id
                    )
                )
            else:
                # Create new preference
                preference_id = str(uuid.uuid4())
                created_at = now
                
                cursor.execute(
                    """
                    INSERT INTO user_preferences 
                    (preference_id, user_id, category, preference_key, preference_value, confidence, created_at, updated_at, source, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        preference_id,
                        user_id,
                        category,
                        key,
                        json.dumps(value),
                        confidence,
                        created_at,
                        now,
                        source,
                        json.dumps(metadata or {})
                    )
                )
            
            self.conn.commit()
            
            # Return the preference object
            return UserPreference(
                preference_id=preference_id,
                user_id=user_id,
                category=category,
                preference_key=key,
                preference_value=value,
                confidence=confidence,
                created_at=created_at,
                updated_at=now,
                source=source,
                metadata=metadata or {}
            )
        
        def get_preference(self, user_id: str, category: str, key: str) -> Optional[UserPreference]:
            """
            Get a user preference.
            
            Args:
                user_id: ID of the user
                category: Category of the preference
                key: Key for the preference
                
            Returns:
                UserPreference object or None if not found
            """
            cursor = self.conn.cursor()
            
            cursor.execute(
                "SELECT * FROM user_preferences WHERE user_id = ? AND category = ? AND preference_key = ?",
                (user_id, category, key)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
                
            # Parse preference value from JSON
            preference_value = json.loads(row[4])
            metadata = json.loads(row[9]) if row[9] else {}
            
            # Create preference object
            preference = UserPreference(
                preference_id=row[0],
                user_id=row[1],
                category=row[2],
                preference_key=row[3],
                preference_value=preference_value,
                confidence=row[5],
                created_at=row[6],
                updated_at=row[7],
                source=row[8],
                metadata=metadata
            )
            
            return preference
        
        def get_user_preferences(self, user_id: str, category: Optional[str] = None) -> List[UserPreference]:
            """
            Get all preferences for a user.
            
            Args:
                user_id: ID of the user
                category: Optional category filter
                
            Returns:
                List of UserPreference objects
            """
            cursor = self.conn.cursor()
            
            if category:
                cursor.execute(
                    "SELECT * FROM user_preferences WHERE user_id = ? AND category = ?",
                    (user_id, category)
                )
            else:
                cursor.execute(
                    "SELECT * FROM user_preferences WHERE user_id = ?",
                    (user_id,)
                )
            
            preferences = []
            for row in cursor.fetchall():
                # Parse preference value from JSON
                preference_value = json.loads(row[4])
                metadata = json.loads(row[9]) if row[9] else {}
                
                # Create preference object
                preference = UserPreference(
                    preference_id=row[0],
                    user_id=row[1],
                    category=row[2],
                    preference_key=row[3],
                    preference_value=preference_value,
                    confidence=row[5],
                    created_at=row[6],
                    updated_at=row[7],
                    source=row[8],
                    metadata=metadata
                )
                
                preferences.append(preference)
            
            return preferences
        
        def delete_preference(self, user_id: str, category: str, key: str) -> bool:
            """
            Delete a user preference.
            
            Args:
                user_id: ID of the user
                category: Category of the preference
                key: Key for the preference
                
            Returns:
                True if deleted, False if not found
            """
            cursor = self.conn.cursor()
            
            cursor.execute(
                "DELETE FROM user_preferences WHERE user_id = ? AND category = ? AND preference_key = ?",
                (user_id, category, key)
            )
            
            self.conn.commit()
            
            return cursor.rowcount > 0
    
    class EpisodicMemorySystem:
        """Manages episodic memory (significant events and experiences)."""
        
        def __init__(self, db_path: str):
            """
            Initialize episodic memory system.
            
            Args:
                db_path: Path to the SQLite database
            """
            self.db_path = db_path
            self.conn = self._init_db()
        
        def _init_db(self) -> sqlite3.Connection:
            """Initialize the database connection and tables."""
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            
            # Create tables if they don't exist
            cursor = conn.cursor()
            
            # Episodic memories table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodic_memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                importance REAL NOT NULL,
                emotion TEXT,
                metadata TEXT
            )
            ''')
            
            # Memory relations table (for connecting related memories)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_relations (
                relation_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                related_memory_id TEXT NOT NULL,
                relation_type TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES episodic_memories (memory_id),
                FOREIGN KEY (related_memory_id) REFERENCES episodic_memories (memory_id),
                UNIQUE(memory_id, related_memory_id)
            )
            ''')
            
            conn.commit()
            return conn
        
        def create_memory(self, user_id: str, title: str, content: str, importance: float = 0.5,
                         emotion: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> EpisodicMemory:
            """
            Create a new episodic memory.
            
            Args:
                user_id: ID of the user
                title: Title/summary of the memory
                content: Detailed content of the memory
                importance: Importance score for this memory (0.0-1.0)
                emotion: Primary emotion associated with this memory
                metadata: Additional metadata for the memory
                
            Returns:
                EpisodicMemory object for the new memory
            """
            memory_id = str(uuid.uuid4())
            created_at = datetime.now().isoformat()
            
            cursor = self.conn.cursor()
            
            # Insert memory into database
            cursor.execute(
                """
                INSERT INTO episodic_memories 
                (memory_id, user_id, title, content, created_at, importance, emotion, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    user_id,
                    title,
                    content,
                    created_at,
                    importance,
                    emotion,
                    json.dumps(metadata or {})
                )
            )
            
            self.conn.commit()
            
            # Return the memory object
            return EpisodicMemory(
                memory_id=memory_id,
                user_id=user_id,
                title=title,
                content=content,
                created_at=created_at,
                importance=importance,
                emotion=emotion,
                metadata=metadata or {}
            )
        
        def get_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        """Get an episodic memory by ID"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            "SELECT * FROM episodic_memories WHERE memory_id = ?",
            (memory_id,)
        )
        
        row = cursor.fetchone()
        if not row:
            return None
            
        # Get related memories
        cursor.execute(
            "SELECT related_memory_id FROM memory_relations WHERE memory_id = ?",
            (memory_id,)
        )
        
        related_memories = [rel[0] for rel in cursor.fetchall()]
        
        # Create memory object
        memory = EpisodicMemory(
            memory_id=row[0],
            user_id=row[1],
            title=row[2],
            content=row[3],
            created_at=row[4],
            importance=row[5],
            emotion=row[6],
            metadata=json.loads(row[7]) if row[7] else {},
            related_memories=related_memories
        )
        
        return memory
        
        def get_user_memories(self, user_id: str, limit: int = 10, sort_by: str = "created_at", 
                             min_importance: float = 0.0) -> List[EpisodicMemory]:
            """
            Get memories for a user.
            
            Args:
                user_id: ID of the user
                limit: Maximum number of memories to return
                sort_by: Field to sort by ('created_at', 'importance')
                min_importance: Minimum importance threshold
                
            Returns:
                List of EpisodicMemory objects
            """
            cursor = self.conn.cursor()
            
            # Validate sort field
            valid_sort_fields = {"created_at": "created_at DESC", "importance": "importance DESC"}
            sort_sql = valid_sort_fields.get(sort_by, "created_at DESC")
            
            # Get memories from database
            cursor.execute(
                f"""
                SELECT memory_id FROM episodic_memories 
                WHERE user_id = ? AND importance >= ?
                ORDER BY {sort_sql} LIMIT ?
                """,
                (user_id, min_importance, limit)
            )
            
            memories = []
            for row in cursor.fetchall():
                memory = self.get_memory(row[0])
                if memory:
                    memories.append(memory)
            
            return memories
        
        def relate_memories(self, memory_id: str, related_memory_id: str, relation_type: Optional[str] = None) -> bool:
            """
            Create a relation between two memories.
            
            Args:
                memory_id: ID of the first memory
                related_memory_id: ID of the related memory
                relation_type: Type of relation
                
            Returns:
                True if successful, False otherwise
            """
            cursor = self.conn.cursor()
            
            # Check if both memories exist
            cursor.execute(
                "SELECT 1 FROM episodic_memories WHERE memory_id IN (?, ?)",
                (memory_id, related_memory_id)
            )
            
            if len(cursor.fetchall()) != 2:
                return False
                
            # Create relation
            relation_id = str(uuid.uuid4())
            created_at = datetime.now().isoformat()
            
            try:
                cursor.execute(
                    """
                    INSERT INTO memory_relations 
                    (relation_id, memory_id, related_memory_id, relation_type, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        relation_id,
                        memory_id,
                        related_memory_id,
                        relation_type,
                        created_at
                    )
                )
                
                # Create bidirectional relation (if different relation type is needed, call this function again with swapped IDs)
                cursor.execute(
                    """
                    INSERT INTO memory_relations 
                    (relation_id, memory_id, related_memory_id, relation_type, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        related_memory_id,
                        memory_id,
                        relation_type,
                        created_at
                    )
                )
                
                self.conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Relation already exists
                return True
        
        class ProceduralMemorySystem:
        """Manages procedural memory (processes, protocols, and procedures)."""
        
        def __init__(self, db_path: str):
            """
            Initialize procedural memory system.
            
            Args:
                db_path: Path to the SQLite database
            """
            self.db_path = db_path
            self.conn = self._init_db()
        
        def _init_db(self) -> sqlite3.Connection:
            """Initialize the database connection and tables."""
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            
            # Create tables if they don't exist
            cursor = conn.cursor()
            
            # Procedural memories (protocols) table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS procedural_protocols (
                protocol_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                description TEXT,
                steps TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
            ''')
            
            conn.commit()
            return conn
        
        def create_protocol(self, name: str, steps: List[Dict[str, Any]], category: Optional[str] = None,
                           description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ProceduralMemory:
            """
            Create a new procedural protocol.
            
            Args:
                name: Name of the protocol
                steps: List of steps to execute the protocol
                category: Category of the protocol
                description: Description of the protocol
                metadata: Additional metadata for the protocol
                
            Returns:
                ProceduralMemory object for the new protocol
            """
            protocol_id = str(uuid.uuid4())
            created_at = datetime.now().isoformat()
            
            cursor = self.conn.cursor()
            
            # Insert protocol into database
            cursor.execute(
                """
                INSERT INTO procedural_protocols 
                (protocol_id, name, category, description, steps, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    protocol_id,
                    name,
                    category,
                    description,
                    json.dumps(steps),
                    created_at,
                    json.dumps(metadata or {})
                )
            )
            
            self.conn.commit()
            
            # Return the protocol object
            return ProceduralMemory(
                protocol_id=protocol_id,
                name=name,
                steps=steps,
                category=category,
                description=description,
                created_at=created_at,
                metadata=metadata or {}
            )
        
        def get_protocol(self, protocol_id: str) -> Optional[ProceduralMemory]:
            """
            Get a protocol by ID.
            
            Args:
                protocol_id: ID of the protocol
                
            Returns:
                ProceduralMemory object or None if not found
            """
            cursor = self.conn.cursor()
            
            cursor.execute(
                "SELECT * FROM procedural_protocols WHERE protocol_id = ?",
                (protocol_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
                
            # Parse steps from JSON
            steps = json.loads(row[4])
            metadata = json.loads(row[6]) if row[6] else {}
            
            # Create protocol object
            protocol = ProceduralMemory(
                protocol_id=row[0],
                name=row[1],
                category=row[2],
                description=row[3],
                steps=steps,
                created_at=row[5],
                metadata=metadata
            )
            
            return protocol
        
        def get_protocols_by_category(self, category: str) -> List[ProceduralMemory]:
            """
            Get protocols by category.
            
            Args:
                category: Category to filter by
                
            Returns:
                List of ProceduralMemory objects
            """
            cursor = self.conn.cursor()
            
            cursor.execute(
                "SELECT protocol_id FROM procedural_protocols WHERE category = ?",
                (category,)
            )
            
            protocols = []
            for row in cursor.fetchall():
                protocol = self.get_protocol(row[0])
                if protocol:
                    protocols.append(protocol)
            
            return protocols
        
        def get_all_protocols(self) -> List[ProceduralMemory]:
            """
            Get all protocols.
            
            Returns:
                List of ProceduralMemory objects
            """
            cursor = self.conn.cursor()
            
            cursor.execute("SELECT protocol_id FROM procedural_protocols")
            
            protocols = []
            for row in cursor.fetchall():
                protocol = self.get_protocol(row[0])
                if protocol:
                    protocols.append(protocol)
            
            return protocols
        
        def delete_protocol(self, protocol_id: str) -> bool:
            """
            Delete a protocol.
            
            Args:
                protocol_id: ID of the protocol
                
            Returns:
                True if deleted, False if not found
            """
            cursor = self.conn.cursor()
            
            cursor.execute(
                "DELETE FROM procedural_protocols WHERE protocol_id = ?",
                (protocol_id,)
            )
            
            self.conn.commit()
            
            return cursor.rowcount > 0
        
        class MemoryManager:
        """
        Unified memory management system that integrates short-term, long-term, 
        episodic, and procedural memory systems.
        """
        
        def __init__(self, base_path: str = "data/memory"):
            """
            Initialize memory manager.
            
            Args:
                base_path: Base directory for memory databases
            """
            self.base_path = base_path
            os.makedirs(base_path, exist_ok=True)
            
            # Initialize memory subsystems
            self.short_term = ShortTermMemory(os.path.join(base_path, "short_term.db"))
            self.long_term = LongTermMemory(os.path.join(base_path, "long_term.db"))
            self.episodic = EpisodicMemorySystem(os.path.join(base_path, "episodic.db"))
            self.procedural = ProceduralMemorySystem(os.path.join(base_path, "procedural.db"))
            
            logger.info(f"Memory Manager initialized at {base_path}")
        
        def create_conversation(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> Conversation:
            """Create a new conversation in short-term memory."""
            return self.short_term.create_conversation(user_id, metadata)
        
        def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
            """Get a conversation from short-term memory."""
            return self.short_term.get_conversation(conversation_id)
        
        def add_message(self, conversation_id: str, role: str, content: str) -> bool:
            """Add a message to a conversation in short-term memory."""
            return self.short_term.add_message(conversation_id, role, content)
        
        def set_user_preference(self, user_id: str, category: str, key: str, value: Any, 
                               confidence: float = 0.5, source: str = "inference", 
                               metadata: Optional[Dict[str, Any]] = None) -> UserPreference:
            """Set a user preference in long-term memory."""
            return self.long_term.set_preference(user_id, category, key, value, confidence, source, metadata)
        
        def get_user_preference(self, user_id: str, category: str, key: str) -> Optional[UserPreference]:
            """Get a user preference from long-term memory."""
            return self.long_term.get_preference(user_id, category, key)
        
        def create_episodic_memory(self, user_id: str, title: str, content: str, 
                                  importance: float = 0.5, emotion: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> EpisodicMemory:
            """Create a new episodic memory."""
            return self.episodic.create_memory(user_id, title, content, importance, emotion, metadata)
        
        def get_episodic_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
            """Get an episodic memory by ID."""
            return self.episodic.get_memory(memory_id)
        
        def create_procedural_protocol(self, name: str, steps: List[Dict[str, Any]], 
                                      category: Optional[str] = None, description: Optional[str] = None,
                                      metadata: Optional[Dict[str, Any]] = None) -> ProceduralMemory:
            """Create a new procedural protocol."""
            return self.procedural.create_protocol(name, steps, category, description, metadata)
        
        def get_procedural_protocol(self, protocol_id: str) -> Optional[ProceduralMemory]:
            """Get a procedural protocol by ID."""
            return self.procedural.get_protocol(protocol_id)
        
        def get_user_state(self, user_id: str) -> Dict[str, Any]:
            """
            Get comprehensive user state from all memory systems.
            
            Args:
                user_id: ID of the user
                
            Returns:
                Dictionary with user state information
            """
            # Get recent conversations
            recent_conversations = self.short_term.get_user_conversations(user_id, limit=3)
            
            # Get user preferences
            preferences = self.long_term.get_user_preferences(user_id)
            
            # Get significant memories
            memories = self.episodic.get_user_memories(user_id, limit=5, sort_by="importance", min_importance=0.7)
            
            # Format the state
            user_state = {
                "user_id": user_id,
                "recent_conversations": [
                    {
                        "conversation_id": conv.conversation_id,
                        "created_at": conv.created_at,
                        "messages_count": len(conv.messages),
                        "last_message": conv.messages[-1] if conv.messages else None
                    }
                    for conv in recent_conversations
                ],
                "preferences": [pref.to_dict() for pref in preferences],
                "significant_memories": [mem.to_dict() for mem in memories],
                "timestamp": datetime.now().isoformat()
            }
            
            return user_state
        
        def create_backup(self, backup_dir: Optional[str] = None) -> str:
            """
            Create a backup of all memory databases.
            
            Args:
                backup_dir: Directory to store the backup
                
            Returns:
                Path to the backup directory
            """
            if not backup_dir:
                backup_dir = os.path.join(self.base_path, "backups", datetime.now().strftime("%Y%m%d_%H%M%S"))
            
            os.makedirs(backup_dir, exist_ok=True)
            
            # Close existing connections
            for db in [self.short_term, self.long_term, self.episodic, self.procedural]:
                db.conn.close()
            
            # Copy database files
            import shutil
            for db_name, db_obj in [
                ("short_term.db", self.short_term),
                ("long_term.db", self.long_term),
                ("episodic.db", self.episodic),
                ("procedural.db", self.procedural)
            ]:
                src_path = db_obj.db_path
                dst_path = os.path.join(backup_dir, db_name)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
            
            # Reopen connections
            self.short_term.conn = self.short_term._init_db()
            self.long_term.conn = self.long_term._init_db()
            self.episodic.conn = self.episodic._init_db()
            self.procedural.conn = self.procedural._init_db()
            
            logger.info(f"Memory backup created at {backup_dir}")
            return backup_dir
        
        def restore_backup(self, backup_dir: str) -> bool:
            """
            Restore memory databases from a backup.
            
            Args:
                backup_dir: Directory containing the backup
                
            Returns:
                True if successful, False otherwise
            """
            if not os.path.isdir(backup_dir):
                logger.error(f"Backup directory not found: {backup_dir}")
                return False
            
            # Close existing connections
            for db in [self.short_term, self.long_term, self.episodic, self.procedural]:
                db.conn.close()
            
            # Restore database files
            import shutil
            for db_name, db_obj in [
                ("short_term.db", self.short_term),
                ("long_term.db", self.long_term),
                ("episodic.db", self.episodic),
                ("procedural.db", self.procedural)
            ]:
                src_path = os.path.join(backup_dir, db_name)
                dst_path = db_obj.db_path
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                else:
                    logger.warning(f"Backup file not found: {src_path}")
            
            # Reopen connections
            self.short_term.conn = self.short_term._init_db()
            self.long_term.conn = self.long_term._init_db()
            self.episodic.conn = self.episodic._init_db()
            self.procedural.conn = self.procedural._init_db()
            
            logger.info(f"Memory backup restored from {backup_dir}")
            return True
    
    def search_memories(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[EpisodicMemory]:
        """Search for memories using semantic search"""
        # Create embedding for query
        query_embedding = self._get_embedding(query)
        
        # Build filters
        filters = {}
        if user_id:
            filters["user_id"] = user_id
            
        # Search in vector database
        results = self.memory_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=filters if filters else None
        )
        
        # Convert results to EpisodicMemory objects
        memories = []
        for i, memory_id in enumerate(results["ids"][0]):
            memory = self.get_memory(memory_id)
            if memory:
                # Add embedding from search results
                memory.embedding = results["embeddings"][0][i]
                memories.append(memory)
                
        return memories
    
    def get_user_memories(
        self, 
        user_id: str, 
        limit: int = 100, 
        sort_by: str = "importance"
    ) -> List[EpisodicMemory]:
        """Get all memories for a user"""
        cursor = self.conn.cursor()
        
        # Build query based on sort criteria
        if sort_by == "importance":
            order_by = "importance DESC, created_at DESC"
        elif sort_by == "recent":
            order_by = "created_at DESC"
        else:
            order_by = "created_at DESC"
            
        cursor.execute(f'''
        SELECT * FROM episodic_memories
        WHERE user_id = ?
        ORDER BY {order_by}
        LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        
        memories = []
        for row in rows:
            memory_id = row[0]
            
            # Get related memories
            cursor.execute(
                "SELECT related_memory_id FROM memory_relations WHERE memory_id = ?",
                (memory_id,)
            )
            
            related_memories = [rel[0] for rel in cursor.fetchall()]
            
            # Create memory object
            memory = EpisodicMemory(
                memory_id=memory_id,
                user_id=row[1],
                title=row[2],
                content=row[3],
                created_at=row[4],
                importance=row[5],
                emotion=row[6],
                metadata=json.loads(row[7]) if row[7] else {},
                related_memories=related_memories
            )
            
            memories.append(memory)
            
        return memories
    
    def create_memory_from_conversation(
        self, 
        user_id: str, 
        conversation: Conversation,
        title: Optional[str] = None,
        importance: float = 0.5,
        emotion: str = "neutral"
    ) -> EpisodicMemory:
        """Create an episodic memory from a conversation"""
        # Generate a title if not provided
        if not title:
            # Use the first few words of the first user message
            for msg in conversation.messages:
                if msg["role"] == "user":
                    content = msg["content"]
                    title = " ".join(content.split()[:5]) + "..."
                    break
            
            # Fallback title if no user messages
            if not title:
                title = f"Conversation on {datetime.datetime.fromtimestamp(conversation.created_at).strftime('%Y-%m-%d')}"
        
        # Format conversation as content
        content = ""
        for msg in conversation.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content += f"{role}: {msg['content']}\n\n"
            
        # Create memory
        memory = EpisodicMemory(
            memory_id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            content=content,
            importance=importance,
            emotion=emotion,
            metadata={
                "source": "conversation",
                "conversation_id": conversation.conversation_id,
                "message_count": len(conversation.messages)
            }
        )
        
        # Store memory
        self.store_memory(memory)
        
        return memory
    
    def save_to_disk(self, path: str):
        """Save episodic memories to disk"""
        # SQLite database is already persistent
        # Vector DB is already persistent
        logger.info(f"EpisodicMemorySystem data is already persistent at {self.sqlite_path} and {self.vector_db_path}")
        
    def load_from_disk(self, path: str):
        """Load episodic memories from disk"""
        # SQLite database is already persistent
        # Vector DB is already persistent
        logger.info(f"EpisodicMemorySystem data is already loaded from {self.sqlite_path} and {self.vector_db_path}")

class ProceduralMemorySystem:
    """
    Procedural memory system for storing how-to knowledge and protocols.
    """
    def __init__(self, sqlite_path: str = "data/procedural.db"):
        self.sqlite_path = sqlite_path
        
        # Initialize SQLite database
        self._init_sqlite_db()
        
        logger.info(f"Initialized ProceduralMemorySystem at {sqlite_path}")
    
    def _init_sqlite_db(self):
        """Initialize SQLite database"""
        # Create directory if it doesn't exist
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(self.sqlite_path)
        cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS procedural_memories (
            protocol_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            tags TEXT NOT NULL,
            metadata TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS protocol_steps (
            step_id TEXT PRIMARY KEY,
            protocol_id TEXT NOT NULL,
            step_number INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            expected_result TEXT,
            FOREIGN KEY (protocol_id) REFERENCES procedural_memories(protocol_id)
        )
        ''')
        
        # Create indexes
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_procedural_memories_name
        ON procedural_memories(name)
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_protocol_steps_protocol_id
        ON protocol_steps(protocol_id)
        ''')
        
        # Commit changes
        self.conn.commit()
        
        logger.info(f"Initialized SQLite database at {self.sqlite_path}")
    
    def store_protocol(self, protocol: ProceduralMemory):
        """Store a procedural memory"""
        cursor = self.conn.cursor()
        
        # Check if protocol already exists
        cursor.execute(
            "SELECT protocol_id FROM procedural_memories WHERE protocol_id = ?",
            (protocol.protocol_id,)
        )
        
        if cursor.fetchone():
            # Update existing protocol
            cursor.execute('''
            UPDATE procedural_memories
            SET name = ?, description = ?, updated_at = ?, tags = ?, metadata = ?
            WHERE protocol_id = ?
            ''', (
                protocol.name,
                protocol.description,
                event_protocol = ProceduralMemory(
            protocol_id="event_planning",
            name="Event Planning",
            steps=[
            {
            "id": 1,
            "name": "Define event type",
            "description": "Ask user about event type (dinner, movie, activity)",
            "required_info":["event_type", "date", "time"]
            },
            ]