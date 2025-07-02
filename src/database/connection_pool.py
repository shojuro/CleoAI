"""
Database connection pooling implementation for CleoAI.
Provides efficient connection management for PostgreSQL, MongoDB, and Redis.
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from queue import Queue, Empty, Full

import psycopg2
from psycopg2 import pool
import asyncpg
import pymongo
from motor.motor_asyncio import AsyncIOMotorClient
import redis
from redis.sentinel import Sentinel
import aioredis

from src.utils.config_validator import get_config
from src.utils.error_handling import DatabaseConnectionError
from src.monitoring.metrics import metrics_collector


logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Connection pool configuration."""
    min_size: int = 2
    max_size: int = 20
    max_overflow: int = 10
    timeout: float = 30.0
    idle_time: int = 3600
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 60


class ConnectionPoolManager:
    """Manages connection pools for all databases."""
    
    def __init__(self):
        self.config = get_config()
        self._pools: Dict[str, Any] = {}
        self._pool_configs: Dict[str, PoolConfig] = {}
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        self._lock = threading.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize all connection pools."""
        if self._initialized:
            return
        
        logger.info("Initializing database connection pools")
        
        try:
            # Initialize PostgreSQL pool
            await self._init_postgres_pool()
            
            # Initialize MongoDB pool
            await self._init_mongodb_pool()
            
            # Initialize Redis pool
            await self._init_redis_pool()
            
            # Start health check tasks
            await self._start_health_checks()
            
            self._initialized = True
            logger.info("All connection pools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pools: {e}")
            await self.shutdown()
            raise DatabaseConnectionError(f"Pool initialization failed: {e}")
    
    async def _init_postgres_pool(self):
        """Initialize PostgreSQL connection pool."""
        pg_config = self.config.get("database", {}).get("postgres", {})
        pool_config = PoolConfig(
            min_size=pg_config.get("pool_min_size", 2),
            max_size=pg_config.get("pool_max_size", 20),
            timeout=pg_config.get("pool_timeout", 30.0)
        )
        
        self._pool_configs["postgres"] = pool_config
        
        # Async pool for async operations
        self._pools["postgres_async"] = await asyncpg.create_pool(
            host=pg_config.get("host", "localhost"),
            port=pg_config.get("port", 5432),
            user=pg_config.get("user"),
            password=pg_config.get("password"),
            database=pg_config.get("database"),
            min_size=pool_config.min_size,
            max_size=pool_config.max_size,
            timeout=pool_config.timeout,
            command_timeout=60,
            server_settings={
                'application_name': 'cleoai',
                'jit': 'off'
            }
        )
        
        # Sync pool for sync operations
        self._pools["postgres_sync"] = psycopg2.pool.ThreadedConnectionPool(
            pool_config.min_size,
            pool_config.max_size,
            host=pg_config.get("host", "localhost"),
            port=pg_config.get("port", 5432),
            user=pg_config.get("user"),
            password=pg_config.get("password"),
            database=pg_config.get("database"),
            application_name='cleoai'
        )
        
        logger.info(f"PostgreSQL pool initialized (min: {pool_config.min_size}, max: {pool_config.max_size})")
    
    async def _init_mongodb_pool(self):
        """Initialize MongoDB connection pool."""
        mongo_config = self.config.get("database", {}).get("mongodb", {})
        pool_config = PoolConfig(
            min_size=mongo_config.get("pool_min_size", 2),
            max_size=mongo_config.get("pool_max_size", 100),
            timeout=mongo_config.get("pool_timeout", 30.0)
        )
        
        self._pool_configs["mongodb"] = pool_config
        
        # Build connection string
        host = mongo_config.get("host", "localhost")
        port = mongo_config.get("port", 27017)
        user = mongo_config.get("user")
        password = mongo_config.get("password")
        database = mongo_config.get("database")
        replica_set = mongo_config.get("replica_set")
        
        if user and password:
            conn_str = f"mongodb://{user}:{password}@{host}:{port}/{database}"
        else:
            conn_str = f"mongodb://{host}:{port}/{database}"
        
        if replica_set:
            conn_str += f"?replicaSet={replica_set}"
        
        # Async client
        self._pools["mongodb_async"] = AsyncIOMotorClient(
            conn_str,
            minPoolSize=pool_config.min_size,
            maxPoolSize=pool_config.max_size,
            waitQueueTimeoutMS=int(pool_config.timeout * 1000),
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=60000,
            appname='cleoai'
        )
        
        # Sync client
        self._pools["mongodb_sync"] = pymongo.MongoClient(
            conn_str,
            minPoolSize=pool_config.min_size,
            maxPoolSize=pool_config.max_size,
            waitQueueTimeoutMS=int(pool_config.timeout * 1000),
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=60000,
            appname='cleoai'
        )
        
        logger.info(f"MongoDB pool initialized (min: {pool_config.min_size}, max: {pool_config.max_size})")
    
    async def _init_redis_pool(self):
        """Initialize Redis connection pool."""
        redis_config = self.config.get("database", {}).get("redis", {})
        pool_config = PoolConfig(
            min_size=redis_config.get("pool_min_size", 2),
            max_size=redis_config.get("pool_max_size", 50),
            timeout=redis_config.get("pool_timeout", 30.0)
        )
        
        self._pool_configs["redis"] = pool_config
        
        host = redis_config.get("host", "localhost")
        port = redis_config.get("port", 6379)
        password = redis_config.get("password")
        db = redis_config.get("db", 0)
        sentinel_nodes = redis_config.get("sentinels", [])
        
        if sentinel_nodes:
            # Redis Sentinel configuration
            sentinel = Sentinel(
                [(node['host'], node['port']) for node in sentinel_nodes],
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            
            # Get master connection
            master_name = redis_config.get("master_name", "mymaster")
            
            # Async pool
            self._pools["redis_async"] = await aioredis.create_redis_pool(
                sentinel.discover_master(master_name),
                password=password,
                db=db,
                minsize=pool_config.min_size,
                maxsize=pool_config.max_size,
                timeout=pool_config.timeout
            )
            
            # Sync pool
            self._pools["redis_sync"] = sentinel.master_for(
                master_name,
                socket_timeout=5.0,
                password=password,
                db=db,
                connection_pool_kwargs={
                    'max_connections': pool_config.max_size,
                    'socket_connect_timeout': 5.0,
                    'socket_timeout': 5.0,
                    'retry_on_timeout': True
                }
            )
        else:
            # Direct Redis connection
            # Async pool
            self._pools["redis_async"] = await aioredis.create_redis_pool(
                f"redis://{host}:{port}/{db}",
                password=password,
                minsize=pool_config.min_size,
                maxsize=pool_config.max_size,
                timeout=pool_config.timeout
            )
            
            # Sync pool
            self._pools["redis_sync"] = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                connection_pool=redis.ConnectionPool(
                    host=host,
                    port=port,
                    password=password,
                    db=db,
                    max_connections=pool_config.max_size,
                    socket_connect_timeout=5.0,
                    socket_timeout=5.0,
                    retry_on_timeout=True
                )
            )
        
        logger.info(f"Redis pool initialized (min: {pool_config.min_size}, max: {pool_config.max_size})")
    
    async def _start_health_checks(self):
        """Start health check tasks for all pools."""
        for pool_name in ["postgres", "mongodb", "redis"]:
            task = asyncio.create_task(self._health_check_loop(pool_name))
            self._health_check_tasks[pool_name] = task
    
    async def _health_check_loop(self, pool_name: str):
        """Health check loop for a specific pool."""
        pool_config = self._pool_configs.get(pool_name, PoolConfig())
        
        while True:
            try:
                await asyncio.sleep(pool_config.health_check_interval)
                await self._check_pool_health(pool_name)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed for {pool_name}: {e}")
                metrics_collector.increment(
                    "db_pool_health_check_failed",
                    tags={"pool": pool_name}
                )
    
    async def _check_pool_health(self, pool_name: str):
        """Check health of a specific pool."""
        start_time = datetime.now()
        
        try:
            if pool_name == "postgres":
                async with self.get_postgres_connection() as conn:
                    await conn.fetchval("SELECT 1")
            elif pool_name == "mongodb":
                client = self.get_mongodb_client()
                await client.admin.command('ping')
            elif pool_name == "redis":
                async with self.get_redis_connection() as conn:
                    await conn.ping()
            
            latency = (datetime.now() - start_time).total_seconds()
            metrics_collector.histogram(
                "db_pool_health_check_latency",
                latency,
                tags={"pool": pool_name}
            )
            
            # Check pool statistics
            await self._collect_pool_metrics(pool_name)
            
        except Exception as e:
            logger.warning(f"Health check failed for {pool_name}: {e}")
            raise
    
    async def _collect_pool_metrics(self, pool_name: str):
        """Collect and report pool metrics."""
        if pool_name == "postgres":
            pool = self._pools.get("postgres_async")
            if pool:
                metrics_collector.gauge(
                    "db_pool_size",
                    pool.get_size(),
                    tags={"pool": "postgres", "type": "current"}
                )
                metrics_collector.gauge(
                    "db_pool_free",
                    pool.get_idle_size(),
                    tags={"pool": "postgres", "type": "free"}
                )
        
        # Similar metrics for other pools
    
    @asynccontextmanager
    async def get_postgres_connection(self):
        """Get PostgreSQL connection from pool."""
        pool = self._pools.get("postgres_async")
        if not pool:
            raise DatabaseConnectionError("PostgreSQL pool not initialized")
        
        conn = None
        start_time = datetime.now()
        
        try:
            # Acquire connection with timeout
            conn = await asyncio.wait_for(
                pool.acquire(),
                timeout=self._pool_configs["postgres"].timeout
            )
            
            acquisition_time = (datetime.now() - start_time).total_seconds()
            metrics_collector.histogram(
                "db_pool_acquisition_time",
                acquisition_time,
                tags={"pool": "postgres"}
            )
            
            yield conn
            
        except asyncio.TimeoutError:
            metrics_collector.increment(
                "db_pool_timeout",
                tags={"pool": "postgres"}
            )
            raise DatabaseConnectionError("Failed to acquire PostgreSQL connection: timeout")
        except Exception as e:
            metrics_collector.increment(
                "db_pool_error",
                tags={"pool": "postgres", "error": type(e).__name__}
            )
            raise DatabaseConnectionError(f"PostgreSQL connection error: {e}")
        finally:
            if conn:
                await pool.release(conn)
    
    @contextmanager
    def get_postgres_connection_sync(self):
        """Get synchronous PostgreSQL connection from pool."""
        pool = self._pools.get("postgres_sync")
        if not pool:
            raise DatabaseConnectionError("PostgreSQL sync pool not initialized")
        
        conn = None
        try:
            conn = pool.getconn()
            yield conn
        finally:
            if conn:
                pool.putconn(conn)
    
    def get_mongodb_client(self) -> AsyncIOMotorClient:
        """Get MongoDB client."""
        client = self._pools.get("mongodb_async")
        if not client:
            raise DatabaseConnectionError("MongoDB pool not initialized")
        return client
    
    def get_mongodb_client_sync(self) -> pymongo.MongoClient:
        """Get synchronous MongoDB client."""
        client = self._pools.get("mongodb_sync")
        if not client:
            raise DatabaseConnectionError("MongoDB sync pool not initialized")
        return client
    
    @asynccontextmanager
    async def get_redis_connection(self):
        """Get Redis connection from pool."""
        pool = self._pools.get("redis_async")
        if not pool:
            raise DatabaseConnectionError("Redis pool not initialized")
        
        try:
            # For aioredis, the pool itself acts as the connection
            yield pool
        except Exception as e:
            metrics_collector.increment(
                "db_pool_error",
                tags={"pool": "redis", "error": type(e).__name__}
            )
            raise DatabaseConnectionError(f"Redis connection error: {e}")
    
    def get_redis_connection_sync(self) -> redis.Redis:
        """Get synchronous Redis connection."""
        conn = self._pools.get("redis_sync")
        if not conn:
            raise DatabaseConnectionError("Redis sync pool not initialized")
        return conn
    
    async def execute_postgres(self, query: str, *args, **kwargs):
        """Execute PostgreSQL query with automatic connection management."""
        async with self.get_postgres_connection() as conn:
            return await conn.fetch(query, *args, **kwargs)
    
    async def execute_postgres_one(self, query: str, *args, **kwargs):
        """Execute PostgreSQL query returning single row."""
        async with self.get_postgres_connection() as conn:
            return await conn.fetchrow(query, *args, **kwargs)
    
    async def execute_postgres_val(self, query: str, *args, **kwargs):
        """Execute PostgreSQL query returning single value."""
        async with self.get_postgres_connection() as conn:
            return await conn.fetchval(query, *args, **kwargs)
    
    async def execute_postgres_many(self, query: str, *args, **kwargs):
        """Execute PostgreSQL query with many parameter sets."""
        async with self.get_postgres_connection() as conn:
            return await conn.executemany(query, args)
    
    async def transaction_postgres(self):
        """Create PostgreSQL transaction context."""
        async with self.get_postgres_connection() as conn:
            async with conn.transaction():
                yield conn
    
    def get_pool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all connection pools."""
        stats = {}
        
        # PostgreSQL stats
        pg_pool = self._pools.get("postgres_async")
        if pg_pool:
            stats["postgres"] = {
                "size": pg_pool.get_size(),
                "free": pg_pool.get_idle_size(),
                "used": pg_pool.get_size() - pg_pool.get_idle_size(),
                "max_size": self._pool_configs["postgres"].max_size
            }
        
        # MongoDB stats (Motor doesn't expose pool stats directly)
        stats["mongodb"] = {
            "configured": bool(self._pools.get("mongodb_async")),
            "max_size": self._pool_configs.get("mongodb", PoolConfig()).max_size
        }
        
        # Redis stats
        redis_pool = self._pools.get("redis_async")
        if redis_pool:
            stats["redis"] = {
                "size": redis_pool.size,
                "free": redis_pool.freesize,
                "used": redis_pool.size - redis_pool.freesize,
                "max_size": self._pool_configs["redis"].max_size
            }
        
        return stats
    
    async def warmup_pools(self):
        """Warm up connection pools by creating minimum connections."""
        logger.info("Warming up connection pools")
        
        # Warm up PostgreSQL
        tasks = []
        for _ in range(self._pool_configs["postgres"].min_size):
            tasks.append(self._check_pool_health("postgres"))
        
        # Warm up Redis
        for _ in range(self._pool_configs["redis"].min_size):
            tasks.append(self._check_pool_health("redis"))
        
        # Execute warmup tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Connection pools warmed up")
    
    async def shutdown(self):
        """Shutdown all connection pools."""
        logger.info("Shutting down connection pools")
        
        # Cancel health check tasks
        for task in self._health_check_tasks.values():
            task.cancel()
        
        await asyncio.gather(
            *self._health_check_tasks.values(),
            return_exceptions=True
        )
        
        # Close PostgreSQL pools
        if "postgres_async" in self._pools:
            await self._pools["postgres_async"].close()
        
        if "postgres_sync" in self._pools:
            self._pools["postgres_sync"].closeall()
        
        # Close MongoDB connections
        if "mongodb_async" in self._pools:
            self._pools["mongodb_async"].close()
        
        if "mongodb_sync" in self._pools:
            self._pools["mongodb_sync"].close()
        
        # Close Redis pools
        if "redis_async" in self._pools:
            self._pools["redis_async"].close()
            await self._pools["redis_async"].wait_closed()
        
        self._pools.clear()
        self._initialized = False
        
        logger.info("Connection pools shut down")


# Global connection pool manager instance
connection_pool_manager = ConnectionPoolManager()


# Convenience functions
async def get_postgres_connection():
    """Get PostgreSQL connection from global pool."""
    return connection_pool_manager.get_postgres_connection()


def get_postgres_connection_sync():
    """Get synchronous PostgreSQL connection from global pool."""
    return connection_pool_manager.get_postgres_connection_sync()


def get_mongodb_client() -> AsyncIOMotorClient:
    """Get MongoDB client from global pool."""
    return connection_pool_manager.get_mongodb_client()


def get_mongodb_client_sync() -> pymongo.MongoClient:
    """Get synchronous MongoDB client from global pool."""
    return connection_pool_manager.get_mongodb_client_sync()


async def get_redis_connection():
    """Get Redis connection from global pool."""
    return connection_pool_manager.get_redis_connection()


def get_redis_connection_sync() -> redis.Redis:
    """Get synchronous Redis connection from global pool."""
    return connection_pool_manager.get_redis_connection_sync()