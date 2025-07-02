"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2024-01-02 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # Create enum types
    op.execute("CREATE TYPE isolationlevel AS ENUM ('none', 'user', 'tenant', 'strict')")
    op.execute("CREATE TYPE messagerole AS ENUM ('user', 'assistant', 'system', 'function')")
    op.execute("CREATE TYPE auditaction AS ENUM ('create', 'read', 'update', 'delete', 'login', 'logout', 'export')")
    
    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('username', sa.String(length=255), nullable=True),
        sa.Column('password_hash', sa.String(length=255), nullable=True),
        sa.Column('api_key_hash', sa.String(length=255), nullable=True),
        sa.Column('roles', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('isolation_level', postgresql.ENUM('none', 'user', 'tenant', 'strict', name='isolationlevel'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=False),
        sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint('failed_login_attempts >= 0', name='check_failed_attempts_positive'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_user_created', 'users', ['created_at'], unique=False)
    op.create_index('idx_user_email_active', 'users', ['email', 'is_active'], unique=False)
    op.create_index(op.f('ix_users_api_key_hash'), 'users', ['api_key_hash'], unique=True)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_user_id'), 'users', ['user_id'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    
    # Create conversations table
    op.create_table('conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('conversation_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('model_name', sa.String(length=255), nullable=True),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('is_encrypted', sa.Boolean(), nullable=False),
        sa.Column('encryption_key_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_message_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('message_count', sa.Integer(), nullable=False),
        sa.Column('total_tokens', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('conversation_id', name='uq_conversation_id')
    )
    op.create_index('idx_conversation_last_message', 'conversations', ['last_message_at'], unique=False)
    op.create_index('idx_conversation_updated', 'conversations', ['updated_at'], unique=False)
    op.create_index('idx_conversation_user', 'conversations', ['user_id', 'created_at'], unique=False)
    op.create_index(op.f('ix_conversations_conversation_id'), 'conversations', ['conversation_id'], unique=True)
    
    # Create messages table
    op.create_table('messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', postgresql.ENUM('user', 'assistant', 'system', 'function', name='messagerole'), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('function_call', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('edited_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('inference_time_ms', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_message_conversation', 'messages', ['conversation_id', 'created_at'], unique=False)
    op.create_index('idx_message_role', 'messages', ['role'], unique=False)
    
    # Create user_preferences table
    op.create_table('user_preferences',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('preference_type', sa.String(length=50), nullable=False),
        sa.Column('preference_key', sa.String(length=255), nullable=False),
        sa.Column('preference_value', sa.Text(), nullable=False),
        sa.Column('is_encrypted', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'preference_type', 'preference_key', name='uq_user_preference')
    )
    op.create_index('idx_preference_user_type', 'user_preferences', ['user_id', 'preference_type'], unique=False)
    
    # Create audit_logs table
    op.create_table('audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('action', postgresql.ENUM('create', 'read', 'update', 'delete', 'login', 'logout', 'export', name='auditaction'), nullable=False),
        sa.Column('resource_type', sa.String(length=50), nullable=False),
        sa.Column('resource_id', sa.String(length=255), nullable=True),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('trace_id', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_audit_created', 'audit_logs', ['created_at'], unique=False)
    op.create_index('idx_audit_resource', 'audit_logs', ['resource_type', 'resource_id'], unique=False)
    op.create_index('idx_audit_trace', 'audit_logs', ['trace_id'], unique=False)
    op.create_index('idx_audit_user_action', 'audit_logs', ['user_id', 'action', 'created_at'], unique=False)
    op.create_index(op.f('ix_audit_logs_trace_id'), 'audit_logs', ['trace_id'], unique=False)
    
    # Create model_deployments table
    op.create_table('model_deployments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_name', sa.String(length=255), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column('model_hash', sa.String(length=64), nullable=True),
        sa.Column('deployment_type', sa.String(length=50), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('endpoint', sa.String(length=500), nullable=True),
        sa.Column('avg_inference_time_ms', sa.Float(), nullable=True),
        sa.Column('total_requests', sa.Integer(), nullable=True),
        sa.Column('error_rate', sa.Float(), nullable=True),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('deployed_by', sa.String(length=255), nullable=True),
        sa.Column('deployed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('retired_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_name', 'model_version', 'deployment_type', name='uq_model_deployment')
    )
    op.create_index('idx_deployment_active', 'model_deployments', ['is_active', 'deployment_type'], unique=False)
    op.create_index('idx_deployment_deployed', 'model_deployments', ['deployed_at'], unique=False)


def downgrade() -> None:
    """Drop all tables and types."""
    op.drop_index('idx_deployment_deployed', table_name='model_deployments')
    op.drop_index('idx_deployment_active', table_name='model_deployments')
    op.drop_table('model_deployments')
    
    op.drop_index(op.f('ix_audit_logs_trace_id'), table_name='audit_logs')
    op.drop_index('idx_audit_user_action', table_name='audit_logs')
    op.drop_index('idx_audit_trace', table_name='audit_logs')
    op.drop_index('idx_audit_resource', table_name='audit_logs')
    op.drop_index('idx_audit_created', table_name='audit_logs')
    op.drop_table('audit_logs')
    
    op.drop_index('idx_preference_user_type', table_name='user_preferences')
    op.drop_table('user_preferences')
    
    op.drop_index('idx_message_role', table_name='messages')
    op.drop_index('idx_message_conversation', table_name='messages')
    op.drop_table('messages')
    
    op.drop_index(op.f('ix_conversations_conversation_id'), table_name='conversations')
    op.drop_index('idx_conversation_user', table_name='conversations')
    op.drop_index('idx_conversation_updated', table_name='conversations')
    op.drop_index('idx_conversation_last_message', table_name='conversations')
    op.drop_table('conversations')
    
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_user_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index(op.f('ix_users_api_key_hash'), table_name='users')
    op.drop_index('idx_user_email_active', table_name='users')
    op.drop_index('idx_user_created', table_name='users')
    op.drop_table('users')
    
    # Drop enum types
    op.execute('DROP TYPE auditaction')
    op.execute('DROP TYPE messagerole')
    op.execute('DROP TYPE isolationlevel')