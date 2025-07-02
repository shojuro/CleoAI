"""
Database package for CleoAI.

This package provides:
- SQLAlchemy models for relational data
- Database connection management
- Migration utilities
- Connection pooling
"""

from .models import Base, User, Conversation, Message, UserPreference, AuditLog
from .connection import (
    get_database_url,
    create_engine,
    get_session_factory,
    get_db,
    init_database
)

__all__ = [
    # Models
    'Base',
    'User',
    'Conversation',
    'Message',
    'UserPreference',
    'AuditLog',
    
    # Connection utilities
    'get_database_url',
    'create_engine',
    'get_session_factory',
    'get_db',
    'init_database'
]