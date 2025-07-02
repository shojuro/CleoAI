"""
Database connection management for CleoAI.

This module provides:
- Connection string building
- Connection pooling
- Session management
- Database initialization
"""
import os
import logging
from typing import Optional, Generator, Dict, Any
from contextlib import contextmanager
from urllib.parse import quote_plus

from sqlalchemy import create_engine, event, pool
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool

from ..monitoring import track_memory_operation, capture_exception

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration."""
    
    def __init__(self):
        """Initialize database configuration from environment."""
        # Connection details
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.user = os.getenv("POSTGRES_USER", "cleoai")
        self.password = os.getenv("POSTGRES_PASSWORD", "")
        self.database = os.getenv("POSTGRES_DB", "cleoai_memory")
        
        # Connection pool settings
        self.pool_size = int(os.getenv("POSTGRES_POOL_SIZE", "20"))
        self.max_overflow = int(os.getenv("POSTGRES_MAX_OVERFLOW", "40"))
        self.pool_timeout = int(os.getenv("POSTGRES_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("POSTGRES_POOL_RECYCLE", "3600"))
        
        # Performance settings
        self.echo = os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true"
        self.connect_timeout = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10"))
        self.command_timeout = int(os.getenv("POSTGRES_COMMAND_TIMEOUT", "30"))
        
        # SSL settings
        self.ssl_mode = os.getenv("POSTGRES_SSL_MODE", "prefer")
        self.ssl_cert = os.getenv("POSTGRES_SSL_CERT")
        self.ssl_key = os.getenv("POSTGRES_SSL_KEY")
        self.ssl_rootcert = os.getenv("POSTGRES_SSL_ROOTCERT")


def get_database_url(config: Optional[DatabaseConfig] = None) -> str:
    """
    Build database connection URL.
    
    Args:
        config: Database configuration
        
    Returns:
        PostgreSQL connection URL
    """
    if config is None:
        config = DatabaseConfig()
    
    # URL encode password to handle special characters
    password = quote_plus(config.password) if config.password else ""
    
    # Base URL
    url = f"postgresql://{config.user}:{password}@{config.host}:{config.port}/{config.database}"
    
    # Add connection parameters
    params = []
    
    # Timeouts
    params.append(f"connect_timeout={config.connect_timeout}")
    params.append(f"options=-c statement_timeout={config.command_timeout}000")  # Convert to ms
    
    # SSL
    if config.ssl_mode != "disable":
        params.append(f"sslmode={config.ssl_mode}")
        if config.ssl_cert:
            params.append(f"sslcert={config.ssl_cert}")
        if config.ssl_key:
            params.append(f"sslkey={config.ssl_key}")
        if config.ssl_rootcert:
            params.append(f"sslrootcert={config.ssl_rootcert}")
    
    # Application name
    params.append("application_name=cleoai")
    
    # Build final URL
    if params:
        url += "?" + "&".join(params)
    
    return url


def create_engine_with_pool(
    url: str,
    config: Optional[DatabaseConfig] = None
) -> Engine:
    """
    Create SQLAlchemy engine with connection pooling.
    
    Args:
        url: Database URL
        config: Database configuration
        
    Returns:
        SQLAlchemy Engine
    """
    if config is None:
        config = DatabaseConfig()
    
    # Engine arguments
    engine_args = {
        "echo": config.echo,
        "pool_pre_ping": True,  # Verify connections before use
        "pool_size": config.pool_size,
        "max_overflow": config.max_overflow,
        "pool_timeout": config.pool_timeout,
        "pool_recycle": config.pool_recycle,
    }
    
    # Use NullPool for testing/development
    if os.getenv("ENVIRONMENT") == "testing":
        engine_args["poolclass"] = NullPool
    else:
        engine_args["poolclass"] = QueuePool
    
    # Create engine
    engine = create_engine(url, **engine_args)
    
    # Add event listeners for monitoring
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_connection, connection_record):
        """Set session parameters on connect."""
        with dbapi_connection.cursor() as cursor:
            # Set search path
            cursor.execute("SET search_path TO public")
            # Set timezone
            cursor.execute("SET timezone = 'UTC'")
    
    @event.listens_for(engine, "checkout")
    def receive_checkout(dbapi_connection, connection_record, connection_proxy):
        """Track connection checkout."""
        connection_record.info['checkout_time'] = os.times()
    
    @event.listens_for(engine, "checkin")
    def receive_checkin(dbapi_connection, connection_record):
        """Track connection checkin."""
        if 'checkout_time' in connection_record.info:
            checkout_time = connection_record.info.pop('checkout_time')
            # Could log connection usage time here
    
    logger.info(f"Database engine created with pool_size={config.pool_size}")
    
    return engine


# Global engine and session factory
_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def init_database(config: Optional[DatabaseConfig] = None) -> Engine:
    """
    Initialize database connection.
    
    Args:
        config: Database configuration
        
    Returns:
        SQLAlchemy Engine
    """
    global _engine, _SessionFactory
    
    if _engine is not None:
        return _engine
    
    try:
        # Get database URL
        url = get_database_url(config)
        
        # Create engine
        _engine = create_engine_with_pool(url, config)
        
        # Create session factory
        _SessionFactory = sessionmaker(
            bind=_engine,
            expire_on_commit=False,
            autoflush=False
        )
        
        # Test connection
        with _engine.connect() as conn:
            result = conn.execute("SELECT 1")
            logger.info("Database connection established")
        
        return _engine
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        capture_exception(e, context={"component": "database_init"})
        raise


def get_session_factory() -> sessionmaker:
    """
    Get SQLAlchemy session factory.
    
    Returns:
        Session factory
    """
    if _SessionFactory is None:
        init_database()
    return _SessionFactory


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.
    
    Yields:
        Database session
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database session.
    
    Yields:
        Database session
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        with track_memory_operation("db_session", "postgresql"):
            yield session
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all_tables():
    """Create all database tables."""
    from .models import Base
    
    engine = init_database()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")


def drop_all_tables():
    """Drop all database tables (use with caution!)."""
    from .models import Base
    
    engine = init_database()
    Base.metadata.drop_all(bind=engine)
    logger.warning("All database tables dropped!")


def get_connection_stats() -> Dict[str, Any]:
    """
    Get database connection pool statistics.
    
    Returns:
        Connection statistics
    """
    if _engine is None:
        return {"status": "not_initialized"}
    
    pool = _engine.pool
    
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total": pool.size() + pool.overflow(),
        "status": "healthy" if pool.checkedin() > 0 else "exhausted"
    }