"""
SQLAlchemy database models for CleoAI.

This module defines the relational database schema for:
- User management
- Conversation tracking
- Message storage
- User preferences
- Audit logging
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, String, Text, DateTime, Boolean, Integer, Float,
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    JSON, Enum as SQLEnum, LargeBinary
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
import enum

Base = declarative_base()


class IsolationLevel(enum.Enum):
    """User data isolation levels."""
    NONE = "none"
    USER = "user"
    TENANT = "tenant"
    STRICT = "strict"


class MessageRole(enum.Enum):
    """Message roles in conversations."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class AuditAction(enum.Enum):
    """Audit log action types."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"


class User(Base):
    """User model with security and isolation features."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    username = Column(String(255), unique=True, nullable=True)
    
    # Security fields
    password_hash = Column(String(255), nullable=True)
    api_key_hash = Column(String(255), nullable=True, unique=True)
    roles = Column(JSONB, default=lambda: ["user"])
    isolation_level = Column(
        SQLEnum(IsolationLevel),
        default=IsolationLevel.USER,
        nullable=False
    )
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)  # Soft delete
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_created', 'created_at'),
        CheckConstraint('failed_login_attempts >= 0', name='check_failed_attempts_positive'),
    )
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format."""
        if email and '@' not in email:
            raise ValueError("Invalid email format")
        return email.lower() if email else None


class Conversation(Base):
    """Conversation model for tracking chat sessions."""
    __tablename__ = 'conversations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Conversation metadata
    title = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    tags = Column(JSONB, default=list)
    metadata = Column(JSONB, default=dict)
    
    # Model information
    model_name = Column(String(255), nullable=True)
    model_version = Column(String(50), nullable=True)
    
    # Privacy and encryption
    is_encrypted = Column(Boolean, default=False, nullable=False)
    encryption_key_id = Column(String(255), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_message_at = Column(DateTime(timezone=True), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)  # Soft delete
    
    # Statistics
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_conversation_user', 'user_id', 'created_at'),
        Index('idx_conversation_updated', 'updated_at'),
        Index('idx_conversation_last_message', 'last_message_at'),
        UniqueConstraint('conversation_id', name='uq_conversation_id'),
    )


class Message(Base):
    """Message model for storing conversation messages."""
    __tablename__ = 'messages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False)
    
    # Message content
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)  # Encrypted if conversation.is_encrypted
    
    # Token tracking
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict)
    function_call = Column(JSONB, nullable=True)  # For function calling
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    edited_at = Column(DateTime(timezone=True), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)  # Soft delete
    
    # Performance metrics
    inference_time_ms = Column(Float, nullable=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    # Indexes
    __table_args__ = (
        Index('idx_message_conversation', 'conversation_id', 'created_at'),
        Index('idx_message_role', 'role'),
    )


class UserPreference(Base):
    """User preferences with encryption support."""
    __tablename__ = 'user_preferences'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Preference identification
    preference_type = Column(String(50), nullable=False)  # settings, privacy, model, etc.
    preference_key = Column(String(255), nullable=False)
    
    # Value storage (encrypted)
    preference_value = Column(Text, nullable=False)  # JSON string, encrypted
    is_encrypted = Column(Boolean, default=True, nullable=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="preferences")
    
    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'preference_type', 'preference_key', name='uq_user_preference'),
        Index('idx_preference_user_type', 'user_id', 'preference_type'),
    )


class AuditLog(Base):
    """Audit log for security and compliance tracking."""
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Actor information
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    ip_address = Column(String(45), nullable=True)  # Supports IPv6
    user_agent = Column(String(500), nullable=True)
    
    # Action details
    action = Column(SQLEnum(AuditAction), nullable=False)
    resource_type = Column(String(50), nullable=False)  # user, conversation, message, etc.
    resource_id = Column(String(255), nullable=True)
    
    # Additional context
    details = Column(JSONB, default=dict)
    status = Column(String(20), nullable=False, default='success')  # success, failure
    error_message = Column(Text, nullable=True)
    
    # Tracing
    trace_id = Column(String(64), nullable=True, index=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_user_action', 'user_id', 'action', 'created_at'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_created', 'created_at'),
        Index('idx_audit_trace', 'trace_id'),
    )


class ModelDeployment(Base):
    """Track model deployments and versions."""
    __tablename__ = 'model_deployments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Model identification
    model_name = Column(String(255), nullable=False)
    model_version = Column(String(50), nullable=False)
    model_hash = Column(String(64), nullable=True)  # SHA256 of model files
    
    # Deployment details
    deployment_type = Column(String(50), nullable=False)  # production, staging, development
    is_active = Column(Boolean, default=False, nullable=False)
    endpoint = Column(String(500), nullable=True)
    
    # Performance metrics
    avg_inference_time_ms = Column(Float, nullable=True)
    total_requests = Column(Integer, default=0)
    error_rate = Column(Float, default=0.0)
    
    # Metadata
    config = Column(JSONB, default=dict)
    deployed_by = Column(String(255), nullable=True)
    deployed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    retired_at = Column(DateTime(timezone=True), nullable=True)
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('model_name', 'model_version', 'deployment_type', name='uq_model_deployment'),
        Index('idx_deployment_active', 'is_active', 'deployment_type'),
        Index('idx_deployment_deployed', 'deployed_at'),
    )