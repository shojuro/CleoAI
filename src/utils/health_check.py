"""
Comprehensive health check for distributed memory backends.
"""
import os
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

import redis
import pymongo
from supabase import create_client, Client
import pinecone

logger = logging.getLogger(__name__)


@dataclass
class ServiceHealth:
    """Health status for a single service."""
    name: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: float
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    services: Dict[str, ServiceHealth]
    warnings: list
    version: str = "1.0.0"


class HealthChecker:
    """Performs comprehensive health checks on all backend services."""
    
    def __init__(self, memory_backend_config):
        self.config = memory_backend_config
        self.checkers = {
            "redis": self._check_redis,
            "mongodb": self._check_mongodb,
            "supabase": self._check_supabase,
            "pinecone": self._check_pinecone,
            "sqlite": self._check_sqlite,
            "chromadb": self._check_chromadb
        }
    
    async def check_all(self) -> SystemHealth:
        """Check health of all enabled services."""
        services = {}
        warnings = []
        
        # Check each enabled backend
        for backend, checker in self.checkers.items():
            env_var = f"USE_{backend.upper()}"
            if os.getenv(env_var, "false").lower() == "true":
                try:
                    service_health = await checker()
                    services[backend] = service_health
                except Exception as e:
                    logger.error(f"Failed to check {backend}: {e}")
                    services[backend] = ServiceHealth(
                        name=backend,
                        status="unhealthy",
                        latency_ms=0.0,
                        message=f"Health check failed: {str(e)}"
                    )
        
        # Determine overall status
        statuses = [s.status for s in services.values()]
        if all(s == "healthy" for s in statuses):
            overall_status = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall_status = "unhealthy"
            warnings.append("One or more services are unhealthy")
        else:
            overall_status = "degraded"
            warnings.append("Some services are experiencing issues")
        
        # Add warnings for missing services
        if not services:
            warnings.append("No backend services are enabled")
            overall_status = "unhealthy"
        
        # Check for vector storage
        has_vector = any(s in services for s in ["pinecone", "chromadb"])
        if not has_vector:
            warnings.append("No vector storage backend is enabled")
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            services=services,
            warnings=warnings
        )
    
    async def _check_redis(self) -> ServiceHealth:
        """Check Redis health."""
        start_time = time.time()
        
        try:
            r = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Ping test
            r.ping()
            
            # Write/read test
            test_key = "_health_check"
            test_value = str(time.time())
            r.setex(test_key, 10, test_value)
            read_value = r.get(test_key)
            
            if read_value.decode() != test_value:
                raise ValueError("Write/read test failed")
            
            # Get server info
            info = r.info()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                name="redis",
                status="healthy",
                latency_ms=latency_ms,
                details={
                    "version": info.get("redis_version", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "uptime_days": info.get("uptime_in_days", 0)
                }
            )
            
        except redis.ConnectionError:
            return ServiceHealth(
                name="redis",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message="Failed to connect to Redis"
            )
        except Exception as e:
            return ServiceHealth(
                name="redis",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message=str(e)
            )
    
    async def _check_mongodb(self) -> ServiceHealth:
        """Check MongoDB health."""
        start_time = time.time()
        
        try:
            client = pymongo.MongoClient(
                self.config.mongodb_connection_string,
                serverSelectionTimeoutMS=5000
            )
            
            # Ping test
            client.admin.command('ping')
            
            # Get server status
            db = client.get_database(self.config.mongodb_database)
            server_status = client.admin.command('serverStatus')
            
            # Test write/read
            test_collection = db['_health_check']
            test_doc = {"timestamp": time.time(), "test": True}
            result = test_collection.insert_one(test_doc)
            found = test_collection.find_one({"_id": result.inserted_id})
            
            if not found:
                raise ValueError("Write/read test failed")
            
            # Clean up
            test_collection.delete_one({"_id": result.inserted_id})
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                name="mongodb",
                status="healthy",
                latency_ms=latency_ms,
                details={
                    "version": server_status.get("version", "unknown"),
                    "uptime": server_status.get("uptime", 0),
                    "connections": server_status.get("connections", {}).get("current", 0)
                }
            )
            
        except pymongo.errors.ServerSelectionTimeoutError:
            return ServiceHealth(
                name="mongodb",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message="Failed to connect to MongoDB"
            )
        except Exception as e:
            return ServiceHealth(
                name="mongodb",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message=str(e)
            )
    
    async def _check_supabase(self) -> ServiceHealth:
        """Check Supabase health."""
        start_time = time.time()
        
        try:
            # Skip if using placeholder values
            if "your-project" in self.config.supabase_url:
                return ServiceHealth(
                    name="supabase",
                    status="unhealthy",
                    latency_ms=0.0,
                    message="Supabase not configured (placeholder values detected)"
                )
            
            supabase: Client = create_client(
                self.config.supabase_url,
                self.config.supabase_anon_key
            )
            
            # Test query
            response = supabase.table('_health_check').select("*").limit(1).execute()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                name="supabase",
                status="healthy",
                latency_ms=latency_ms,
                details={
                    "url": self.config.supabase_url,
                    "tables_accessible": True
                }
            )
            
        except Exception as e:
            # If it's just a missing table, service is still healthy
            if "_health_check" in str(e) and "not found" in str(e):
                return ServiceHealth(
                    name="supabase",
                    status="healthy",
                    latency_ms=(time.time() - start_time) * 1000,
                    message="Connected (health check table not found)",
                    details={"tables_accessible": False}
                )
            
            return ServiceHealth(
                name="supabase",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message=str(e)
            )
    
    async def _check_pinecone(self) -> ServiceHealth:
        """Check Pinecone health."""
        start_time = time.time()
        
        try:
            # Skip if using placeholder values
            if self.config.pinecone_api_key == "your-pinecone-api-key-here":
                return ServiceHealth(
                    name="pinecone",
                    status="unhealthy",
                    latency_ms=0.0,
                    message="Pinecone not configured (placeholder values detected)"
                )
            
            # Initialize Pinecone
            pinecone.init(
                api_key=self.config.pinecone_api_key,
                environment=self.config.pinecone_environment
            )
            
            # List indexes
            indexes = pinecone.list_indexes()
            
            # Check if our index exists
            index_exists = self.config.pinecone_index_name in indexes
            
            latency_ms = (time.time() - start_time) * 1000
            
            if index_exists:
                # Get index stats
                index = pinecone.Index(self.config.pinecone_index_name)
                stats = index.describe_index_stats()
                
                return ServiceHealth(
                    name="pinecone",
                    status="healthy",
                    latency_ms=latency_ms,
                    details={
                        "index_name": self.config.pinecone_index_name,
                        "total_vectors": stats.total_vector_count,
                        "dimension": stats.dimension,
                        "namespaces": list(stats.namespaces.keys()) if stats.namespaces else []
                    }
                )
            else:
                return ServiceHealth(
                    name="pinecone",
                    status="degraded",
                    latency_ms=latency_ms,
                    message=f"Index '{self.config.pinecone_index_name}' not found",
                    details={"available_indexes": indexes}
                )
            
        except Exception as e:
            return ServiceHealth(
                name="pinecone",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message=str(e)
            )
    
    async def _check_sqlite(self) -> ServiceHealth:
        """Check SQLite health."""
        start_time = time.time()
        
        try:
            import sqlite3
            
            # Connect to database
            db_path = os.path.join(self.config.sqlite_path, self.config.sqlite_db_name)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size();")
            db_size = cursor.fetchone()[0]
            
            conn.close()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                name="sqlite",
                status="healthy",
                latency_ms=latency_ms,
                details={
                    "database": db_path,
                    "tables": tables,
                    "size_bytes": db_size
                }
            )
            
        except Exception as e:
            return ServiceHealth(
                name="sqlite",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message=str(e)
            )
    
    async def _check_chromadb(self) -> ServiceHealth:
        """Check ChromaDB health."""
        start_time = time.time()
        
        try:
            import chromadb
            
            # Create client
            client = chromadb.PersistentClient(path=self.config.chromadb_path)
            
            # List collections
            collections = client.list_collections()
            collection_names = [c.name for c in collections]
            
            # Get heartbeat
            heartbeat = client.heartbeat()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                name="chromadb",
                status="healthy",
                latency_ms=latency_ms,
                details={
                    "collections": collection_names,
                    "heartbeat": heartbeat
                }
            )
            
        except Exception as e:
            return ServiceHealth(
                name="chromadb",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message=str(e)
            )


async def perform_health_check(memory_backend_config) -> Dict[str, Any]:
    """Perform a comprehensive health check and return results as dict."""
    checker = HealthChecker(memory_backend_config)
    health = await checker.check_all()
    
    # Convert to dict for JSON serialization
    result = asdict(health)
    result["services"] = {
        name: asdict(service) for name, service in health.services.items()
    }
    
    return result