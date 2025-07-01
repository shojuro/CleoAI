"""
Pinecone service for vector storage and similarity search.
"""
import logging
from typing import Optional, Dict, Any, List, Tuple
import os
import pinecone
from pinecone import Index, ServerlessSpec
import numpy as np
import time

from src.utils.error_handling import retry_on_error, handle_errors, MemoryError

logger = logging.getLogger(__name__)


class PineconeMemoryService:
    """
    Pinecone-based memory service for vector storage and similarity search.
    
    This service provides high-performance vector storage with metadata
    filtering and namespace support for multi-tenancy.
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 environment: Optional[str] = None,
                 index_name: str = "cleoai-memories",
                 dimension: int = 1536,
                 metric: str = "cosine",
                 cloud: str = "aws",
                 region: str = "us-east-1"):
        """
        Initialize Pinecone memory service.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            dimension: Vector dimension (must match embedding model)
            metric: Distance metric (cosine, euclidean, dotproduct)
            cloud: Cloud provider for serverless
            region: Region for serverless
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        if not self.api_key:
            raise MemoryError("Pinecone API key is required")
        
        try:
            # Initialize Pinecone
            pinecone.init(api_key=self.api_key)
            
            # Create or connect to index
            self._initialize_index(cloud, region)
            
            # Connect to index
            self.index = pinecone.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index '{self.index_name}' with {stats.total_vector_count} vectors")
            
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise MemoryError(f"Pinecone connection failed: {e}")
    
    def _initialize_index(self, cloud: str, region: str):
        """Initialize Pinecone index if it doesn't exist."""
        try:
            # Check if index exists
            existing_indexes = pinecone.list_indexes()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                # Create serverless index
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    )
                )
                
                # Wait for index to be ready
                while not pinecone.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                
                logger.info(f"Created Pinecone index: {self.index_name}")
            else:
                # Verify dimension matches
                index_info = pinecone.describe_index(self.index_name)
                if index_info.dimension != self.dimension:
                    raise MemoryError(
                        f"Index dimension mismatch: expected {self.dimension}, "
                        f"got {index_info.dimension}"
                    )
                
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise
    
    @retry_on_error(max_attempts=3, delay=0.5)
    def upsert_memory(self,
                     memory_id: str,
                     embedding: List[float],
                     user_id: str,
                     metadata: Dict[str, Any],
                     namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Store or update a memory vector.
        
        Args:
            memory_id: Unique memory identifier
            embedding: Vector embedding
            user_id: User identifier
            metadata: Memory metadata
            namespace: Optional namespace (defaults to user_id)
            
        Returns:
            Upsert response
        """
        try:
            # Validate embedding dimension
            if len(embedding) != self.dimension:
                raise MemoryError(
                    f"Embedding dimension mismatch: expected {self.dimension}, "
                    f"got {len(embedding)}"
                )
            
            # Use user_id as namespace for isolation
            namespace = namespace or user_id
            
            # Add user_id to metadata
            metadata["user_id"] = user_id
            metadata["timestamp"] = int(time.time())
            
            # Ensure metadata values are compatible with Pinecone
            clean_metadata = self._clean_metadata(metadata)
            
            # Upsert vector
            response = self.index.upsert(
                vectors=[(memory_id, embedding, clean_metadata)],
                namespace=namespace
            )
            
            logger.info(f"Upserted memory {memory_id} for user {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to upsert memory: {e}")
            raise MemoryError(f"Failed to store in Pinecone: {e}")
    
    @retry_on_error(max_attempts=3, delay=0.5)
    def upsert_batch(self,
                    memories: List[Tuple[str, List[float], Dict[str, Any]]],
                    user_id: str,
                    namespace: Optional[str] = None,
                    batch_size: int = 100) -> Dict[str, Any]:
        """
        Batch upsert multiple memories.
        
        Args:
            memories: List of (id, embedding, metadata) tuples
            user_id: User identifier
            namespace: Optional namespace
            batch_size: Batch size for upserts
            
        Returns:
            Combined upsert response
        """
        try:
            namespace = namespace or user_id
            total_upserted = 0
            
            # Process in batches
            for i in range(0, len(memories), batch_size):
                batch = memories[i:i + batch_size]
                
                # Prepare vectors
                vectors = []
                for memory_id, embedding, metadata in batch:
                    # Validate dimension
                    if len(embedding) != self.dimension:
                        logger.warning(f"Skipping {memory_id}: dimension mismatch")
                        continue
                    
                    # Add user_id and timestamp
                    metadata["user_id"] = user_id
                    metadata["timestamp"] = int(time.time())
                    clean_metadata = self._clean_metadata(metadata)
                    
                    vectors.append((memory_id, embedding, clean_metadata))
                
                if vectors:
                    response = self.index.upsert(
                        vectors=vectors,
                        namespace=namespace
                    )
                    total_upserted += response["upserted_count"]
            
            logger.info(f"Batch upserted {total_upserted} memories for user {user_id}")
            return {"upserted_count": total_upserted}
            
        except Exception as e:
            logger.error(f"Failed to batch upsert memories: {e}")
            raise MemoryError(f"Failed to batch store in Pinecone: {e}")
    
    @handle_errors(default_return=[])
    def query_memories(self,
                      query_embedding: List[float],
                      user_id: str,
                      top_k: int = 10,
                      namespace: Optional[str] = None,
                      filter_dict: Optional[Dict[str, Any]] = None,
                      include_values: bool = False) -> List[Dict[str, Any]]:
        """
        Query similar memories.
        
        Args:
            query_embedding: Query vector
            user_id: User identifier
            top_k: Number of results
            namespace: Optional namespace
            filter_dict: Metadata filters
            include_values: Include vector values
            
        Returns:
            List of similar memories
        """
        try:
            # Validate dimension
            if len(query_embedding) != self.dimension:
                raise MemoryError(
                    f"Query dimension mismatch: expected {self.dimension}, "
                    f"got {len(query_embedding)}"
                )
            
            # Use user_id as namespace
            namespace = namespace or user_id
            
            # Build filter
            query_filter = {"user_id": {"$eq": user_id}}
            if filter_dict:
                query_filter.update(filter_dict)
            
            # Query index
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=query_filter,
                include_values=include_values,
                include_metadata=True
            )
            
            # Format results
            results = []
            for match in response.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                if include_values and hasattr(match, 'values'):
                    result["values"] = match.values
                    
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to query memories: {e}")
            return []
    
    @handle_errors(default_return=[])
    def query_by_metadata(self,
                         user_id: str,
                         metadata_filter: Dict[str, Any],
                         top_k: int = 100,
                         namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query memories by metadata without vector similarity.
        
        Args:
            user_id: User identifier
            metadata_filter: Metadata filters
            top_k: Maximum results
            namespace: Optional namespace
            
        Returns:
            List of matching memories
        """
        try:
            namespace = namespace or user_id
            
            # Create a random vector for metadata-only search
            dummy_vector = np.random.rand(self.dimension).tolist()
            
            # Build filter including user_id
            query_filter = {"user_id": {"$eq": user_id}}
            query_filter.update(metadata_filter)
            
            # Query with high top_k to get all matches
            response = self.index.query(
                vector=dummy_vector,
                top_k=min(top_k, 10000),  # Pinecone limit
                namespace=namespace,
                filter=query_filter,
                include_metadata=True
            )
            
            # Return results (ignoring scores since vector is random)
            results = []
            for match in response.matches:
                results.append({
                    "id": match.id,
                    "metadata": match.metadata
                })
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to query by metadata: {e}")
            return []
    
    def fetch_memories(self,
                      memory_ids: List[str],
                      user_id: str,
                      namespace: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Fetch specific memories by ID.
        
        Args:
            memory_ids: List of memory IDs
            user_id: User identifier
            namespace: Optional namespace
            
        Returns:
            Dictionary of memory data by ID
        """
        try:
            namespace = namespace or user_id
            
            # Fetch vectors
            response = self.index.fetch(
                ids=memory_ids,
                namespace=namespace
            )
            
            # Format results
            memories = {}
            for memory_id, data in response.vectors.items():
                # Verify user_id matches
                if data.metadata.get("user_id") == user_id:
                    memories[memory_id] = {
                        "values": data.values,
                        "metadata": data.metadata
                    }
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to fetch memories: {e}")
            return {}
    
    @retry_on_error(max_attempts=3)
    def update_memory_metadata(self,
                             memory_id: str,
                             user_id: str,
                             metadata_updates: Dict[str, Any],
                             namespace: Optional[str] = None) -> bool:
        """
        Update memory metadata.
        
        Args:
            memory_id: Memory identifier
            user_id: User identifier
            metadata_updates: Metadata to update
            namespace: Optional namespace
            
        Returns:
            True if successful
        """
        try:
            namespace = namespace or user_id
            
            # Fetch current memory
            current = self.index.fetch(ids=[memory_id], namespace=namespace)
            
            if memory_id not in current.vectors:
                logger.warning(f"Memory {memory_id} not found")
                return False
            
            # Verify ownership
            if current.vectors[memory_id].metadata.get("user_id") != user_id:
                logger.warning(f"Memory {memory_id} does not belong to user {user_id}")
                return False
            
            # Update metadata
            updated_metadata = current.vectors[memory_id].metadata.copy()
            updated_metadata.update(metadata_updates)
            updated_metadata["last_updated"] = int(time.time())
            
            # Clean metadata
            clean_metadata = self._clean_metadata(updated_metadata)
            
            # Re-upsert with updated metadata
            self.index.upsert(
                vectors=[(
                    memory_id,
                    current.vectors[memory_id].values,
                    clean_metadata
                )],
                namespace=namespace
            )
            
            logger.info(f"Updated metadata for memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory metadata: {e}")
            return False
    
    def delete_memory(self,
                     memory_id: str,
                     user_id: str,
                     namespace: Optional[str] = None) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory identifier
            user_id: User identifier
            namespace: Optional namespace
            
        Returns:
            True if successful
        """
        try:
            namespace = namespace or user_id
            
            # Delete from index
            self.index.delete(
                ids=[memory_id],
                namespace=namespace
            )
            
            logger.info(f"Deleted memory {memory_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False
    
    def delete_user_memories(self,
                           user_id: str,
                           namespace: Optional[str] = None) -> bool:
        """
        Delete all memories for a user.
        
        Args:
            user_id: User identifier
            namespace: Optional namespace
            
        Returns:
            True if successful
        """
        try:
            namespace = namespace or user_id
            
            # Delete entire namespace
            self.index.delete(
                delete_all=True,
                namespace=namespace
            )
            
            logger.info(f"Deleted all memories for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user memories: {e}")
            return False
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata for Pinecone compatibility.
        
        Pinecone metadata must be:
        - Strings, numbers, booleans, or lists of these
        - Keys must be strings
        - No nested dictionaries
        """
        clean = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list):
                # Only keep simple types in lists
                clean[key] = [
                    v for v in value 
                    if isinstance(v, (str, int, float, bool))
                ]
            elif isinstance(value, dict):
                # Flatten nested dicts with dot notation
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (str, int, float, bool)):
                        clean[f"{key}.{sub_key}"] = sub_value
            # Skip other types
            
        return clean
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone statistics."""
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "index_name": self.index_name,
                "dimension": self.dimension,
                "metric": self.metric,
                "total_vectors": stats.total_vector_count,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {},
                "index_fullness": stats.index_fullness
            }
            
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check Pinecone health."""
        try:
            # Try to get index stats
            self.index.describe_index_stats()
            return True
        except:
            return False
