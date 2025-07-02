"""
Secure memory manager with data isolation and encryption.

This module wraps the standard memory manager to add row-level security
and field-level encryption for sensitive data.
"""
import logging
from typing import Dict, List, Optional, Any, Union
from functools import wraps

from .memory_manager import MemoryManager, Conversation
from ..security.data_isolation import (
    SecurityContext, DataIsolationManager, AccessType,
    get_security_context, get_isolation_manager, require_access
)
from ..security.encryption import EncryptionProvider, get_encryption_provider
from ..utils.error_handling import AuthorizationError, handle_errors

logger = logging.getLogger(__name__)

# Fields to encrypt in different data types
ENCRYPTED_FIELDS = {
    'conversation': ['messages'],
    'preference': ['preference_value'],
    'memory': ['content', 'embedding'],
    'user_profile': ['personal_info', 'preferences']
}


class SecureMemoryManager(MemoryManager):
    """
    Secure wrapper for MemoryManager with data isolation and encryption.
    
    This class ensures that:
    1. Users can only access their own data
    2. Sensitive fields are encrypted at rest
    3. All access is audited
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isolation_manager = get_isolation_manager()
        self.encryption_provider = get_encryption_provider()
        self._setup_isolation_policies()
    
    def _setup_isolation_policies(self):
        """Configure isolation policies for memory resources."""
        # Add custom rules if needed
        policy = self.isolation_manager.policies.get('conversation')
        if policy:
            # Example: Allow service accounts to access all conversations for analytics
            policy.add_rule(
                lambda ctx, resource: ctx.has_role('service') and ctx.metadata.get('purpose') == 'analytics'
            )
    
    def _get_context(self) -> SecurityContext:
        """Get current security context or raise error."""
        context = get_security_context()
        if not context:
            raise AuthorizationError("No security context available")
        return context
    
    def _encrypt_resource(self, resource_type: str, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in a resource."""
        fields = ENCRYPTED_FIELDS.get(resource_type, [])
        if fields:
            return self.encryption_provider.encrypt_fields(resource, fields)
        return resource
    
    def _decrypt_resource(self, resource_type: str, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in a resource."""
        fields = ENCRYPTED_FIELDS.get(resource_type, [])
        if fields:
            return self.encryption_provider.decrypt_fields(resource, fields)
        return resource
    
    @handle_errors(default_return=None)
    def add_message(self,
                   conversation_id: str,
                   user_id: str,
                   role: str,
                   content: str,
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Add a message to a conversation with security checks."""
        context = self._get_context()
        
        # Check if user can access this conversation
        conversation = self.get_conversation(conversation_id, user_id)
        if conversation:
            self.isolation_manager.check_access(
                'conversation',
                conversation.to_dict(),
                AccessType.WRITE,
                context
            )
        else:
            # New conversation - check if user can create for this user_id
            if not context.can_access_user(user_id):
                raise AuthorizationError(f"Cannot create conversation for user {user_id}")
        
        # Add message with encryption
        message_data = {
            'role': role,
            'content': content,
            'metadata': metadata
        }
        
        # Encrypt message content
        encrypted_message = self.encryption_provider.encrypt_fields(
            message_data,
            ['content']
        )
        
        # Call parent method
        return super().add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role=encrypted_message['role'],
            content=encrypted_message['content'],
            metadata=encrypted_message.get('metadata')
        )
    
    @handle_errors(default_return=None)
    def get_conversation(self,
                        conversation_id: str,
                        user_id: str) -> Optional[Conversation]:
        """Get a conversation with security checks and decryption."""
        context = self._get_context()
        
        # Get conversation
        conversation = super().get_conversation(conversation_id, user_id)
        if not conversation:
            return None
        
        # Check access
        self.isolation_manager.check_access(
            'conversation',
            conversation.to_dict(),
            AccessType.READ,
            context
        )
        
        # Decrypt messages
        for message in conversation.messages:
            if isinstance(message.get('content'), dict) and 'ciphertext' in message['content']:
                decrypted_content = self.encryption_provider.decrypt(
                    message['content'],
                    return_type=str
                )
                message['content'] = decrypted_content
        
        return conversation
    
    @handle_errors(default_return=[])
    def get_recent_conversations(self,
                               user_id: str,
                               limit: int = 10) -> List[Conversation]:
        """Get recent conversations with security filtering."""
        context = self._get_context()
        
        # Check if user can access this user's data
        if not context.can_access_user(user_id):
            logger.warning(f"User {context.user_id} attempted to access conversations for {user_id}")
            return []
        
        # Get conversations
        conversations = super().get_recent_conversations(user_id, limit)
        
        # Filter based on access control
        filtered = []
        for conv in conversations:
            try:
                self.isolation_manager.check_access(
                    'conversation',
                    conv.to_dict(),
                    AccessType.READ,
                    context
                )
                
                # Decrypt messages
                for message in conv.messages:
                    if isinstance(message.get('content'), dict) and 'ciphertext' in message['content']:
                        decrypted_content = self.encryption_provider.decrypt(
                            message['content'],
                            return_type=str
                        )
                        message['content'] = decrypted_content
                
                filtered.append(conv)
            except AuthorizationError:
                continue
        
        return filtered
    
    @handle_errors(default_return=None)
    def store_user_preference(self,
                            user_id: str,
                            preference_type: str,
                            preference_key: str,
                            preference_value: Any,
                            confidence: float = 0.5,
                            source: str = "inferred",
                            metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Store user preference with encryption."""
        context = self._get_context()
        
        # Check access
        if not context.can_access_user(user_id):
            raise AuthorizationError(f"Cannot store preferences for user {user_id}")
        
        # Encrypt preference value
        encrypted_pref = self.encryption_provider.encrypt(preference_value)
        
        # Store encrypted
        return super().store_user_preference(
            user_id=user_id,
            preference_type=preference_type,
            preference_key=preference_key,
            preference_value=encrypted_pref.to_dict(),
            confidence=confidence,
            source=source,
            metadata=metadata
        )
    
    @handle_errors(default_return=None)
    def get_user_preference(self,
                          user_id: str,
                          preference_type: str,
                          preference_key: str) -> Optional[Any]:
        """Get user preference with decryption."""
        context = self._get_context()
        
        # Check access
        if not context.can_access_user(user_id):
            logger.warning(f"User {context.user_id} attempted to access preferences for {user_id}")
            return None
        
        # Get preference
        pref_value = super().get_user_preference(
            user_id=user_id,
            preference_type=preference_type,
            preference_key=preference_key
        )
        
        if pref_value and isinstance(pref_value, dict) and 'ciphertext' in pref_value:
            # Decrypt
            return self.encryption_provider.decrypt(pref_value)
        
        return pref_value
    
    @handle_errors(default_return=[])
    def search_memories(self,
                       user_id: str,
                       query: str,
                       memory_types: Optional[List[str]] = None,
                       limit: int = 10,
                       min_relevance: float = 0.7) -> List[Dict[str, Any]]:
        """Search memories with security filtering and decryption."""
        context = self._get_context()
        
        # Check access
        if not context.can_access_user(user_id):
            logger.warning(f"User {context.user_id} attempted to search memories for {user_id}")
            return []
        
        # Search memories
        memories = super().search_memories(
            user_id=user_id,
            query=query,
            memory_types=memory_types,
            limit=limit * 2,  # Get extra to account for filtering
            min_relevance=min_relevance
        )
        
        # Filter and decrypt
        filtered = []
        for memory in memories:
            try:
                # Check access
                self.isolation_manager.check_access(
                    'memory',
                    memory,
                    AccessType.READ,
                    context
                )
                
                # Decrypt content
                decrypted = self._decrypt_resource('memory', memory)
                filtered.append(decrypted)
                
                if len(filtered) >= limit:
                    break
                    
            except AuthorizationError:
                continue
        
        return filtered
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics with access control."""
        context = self._get_context()
        
        # Check access
        if not context.can_access_user(user_id) and not context.is_admin():
            raise AuthorizationError(f"Cannot access memory stats for user {user_id}")
        
        # Get stats
        stats = super().get_memory_stats(user_id)
        
        # Add security info if admin
        if context.is_admin():
            # Get access logs for this user
            access_logs = self.isolation_manager.get_access_logs(user_id=user_id)
            stats['security'] = {
                'total_access_attempts': len(access_logs),
                'successful_accesses': len([l for l in access_logs if l.success]),
                'failed_accesses': len([l for l in access_logs if not l.success]),
                'last_access': max([l.timestamp for l in access_logs]) if access_logs else None
            }
        
        return stats
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data (for GDPR compliance)."""
        context = self._get_context()
        
        # Only user themselves or admin can export
        if not context.can_access_user(user_id) and not context.is_admin():
            raise AuthorizationError(f"Cannot export data for user {user_id}")
        
        # Log this sensitive operation
        logger.info(f"Data export requested for user {user_id} by {context.user_id}")
        
        # Collect all user data
        data = {
            'user_id': user_id,
            'export_timestamp': time.time(),
            'conversations': [],
            'preferences': [],
            'memories': []
        }
        
        # Get conversations
        conversations = super().get_recent_conversations(user_id, limit=1000)
        for conv in conversations:
            # Decrypt messages
            conv_dict = conv.to_dict()
            for message in conv_dict['messages']:
                if isinstance(message.get('content'), dict) and 'ciphertext' in message['content']:
                    message['content'] = self.encryption_provider.decrypt(
                        message['content'],
                        return_type=str
                    )
            data['conversations'].append(conv_dict)
        
        # Get preferences
        preferences = super().get_all_user_preferences(user_id)
        for pref in preferences:
            if isinstance(pref.get('preference_value'), dict) and 'ciphertext' in pref['preference_value']:
                pref['preference_value'] = self.encryption_provider.decrypt(pref['preference_value'])
            data['preferences'].append(pref)
        
        # Get memories (if implemented)
        # data['memories'] = self._get_all_user_memories(user_id)
        
        return data
    
    def delete_user_data(self, user_id: str, confirm: bool = False) -> bool:
        """Delete all user data (for GDPR compliance)."""
        context = self._get_context()
        
        # Only user themselves or admin can delete
        if not context.can_access_user(user_id) and not context.is_admin():
            raise AuthorizationError(f"Cannot delete data for user {user_id}")
        
        if not confirm:
            raise ValueError("Must confirm deletion by setting confirm=True")
        
        # Log this sensitive operation
        logger.warning(f"Data deletion requested for user {user_id} by {context.user_id}")
        
        # TODO: Implement actual deletion across all backends
        # This would need to:
        # 1. Delete from all memory backends
        # 2. Delete from vector stores
        # 3. Clear caches
        # 4. Log the deletion
        
        return True


def create_secure_memory_manager(*args, **kwargs) -> SecureMemoryManager:
    """Factory function to create a secure memory manager."""
    return SecureMemoryManager(*args, **kwargs)