"""
Unit tests for secure memory manager with data isolation and encryption.
"""
import pytest
import time
import json
from unittest.mock import Mock, patch

from src.memory.secure_memory_manager import SecureMemoryManager
from src.security.data_isolation import (
    SecurityContext, IsolationLevel, set_security_context,
    init_isolation_manager, AccessType
)
from src.security.encryption import init_encryption_provider
from src.utils.error_handling import AuthorizationError


class TestSecureMemoryManager:
    """Test suite for secure memory manager."""
    
    @pytest.fixture
    def security_context_user1(self):
        """Create security context for user1."""
        return SecurityContext(
            user_id="user1",
            roles=["user"],
            isolation_level=IsolationLevel.USER
        )
    
    @pytest.fixture
    def security_context_user2(self):
        """Create security context for user2."""
        return SecurityContext(
            user_id="user2",
            roles=["user"],
            isolation_level=IsolationLevel.USER
        )
    
    @pytest.fixture
    def security_context_admin(self):
        """Create security context for admin."""
        return SecurityContext(
            user_id="admin",
            roles=["admin"],
            isolation_level=IsolationLevel.NONE
        )
    
    @pytest.fixture
    def secure_memory_manager(self):
        """Create secure memory manager instance."""
        # Initialize security components
        init_isolation_manager()
        init_encryption_provider()
        
        # Create manager with mock backends
        manager = SecureMemoryManager()
        return manager
    
    def test_user_isolation_conversations(self, secure_memory_manager, 
                                        security_context_user1, 
                                        security_context_user2):
        """Test that users can only access their own conversations."""
        # Set context for user1
        set_security_context(security_context_user1)
        
        # User1 creates a conversation
        conversation_id = "conv1"
        secure_memory_manager.add_message(
            conversation_id=conversation_id,
            user_id="user1",
            role="user",
            content="Hello, this is user1"
        )
        
        # User1 can access their conversation
        conv = secure_memory_manager.get_conversation(conversation_id, "user1")
        assert conv is not None
        assert len(conv.messages) == 1
        
        # Switch to user2
        set_security_context(security_context_user2)
        
        # User2 cannot access user1's conversation
        with pytest.raises(AuthorizationError):
            secure_memory_manager.get_conversation(conversation_id, "user1")
        
        # User2 cannot add to user1's conversation
        with pytest.raises(AuthorizationError):
            secure_memory_manager.add_message(
                conversation_id=conversation_id,
                user_id="user1",
                role="user",
                content="This should fail"
            )
    
    def test_admin_access(self, secure_memory_manager,
                         security_context_user1,
                         security_context_admin):
        """Test that admins can access any user's data."""
        # User1 creates data
        set_security_context(security_context_user1)
        
        conversation_id = "conv1"
        secure_memory_manager.add_message(
            conversation_id=conversation_id,
            user_id="user1",
            role="user",
            content="User1's private message"
        )
        
        # Switch to admin
        set_security_context(security_context_admin)
        
        # Admin can access user1's conversation
        conv = secure_memory_manager.get_conversation(conversation_id, "user1")
        assert conv is not None
        assert len(conv.messages) == 1
        
        # Admin can get stats for any user
        stats = secure_memory_manager.get_memory_stats("user1")
        assert stats is not None
        assert 'security' in stats  # Admin gets security info
    
    def test_encryption_decryption(self, secure_memory_manager,
                                  security_context_user1):
        """Test that sensitive data is encrypted and decrypted properly."""
        set_security_context(security_context_user1)
        
        # Add a message
        conversation_id = "conv1"
        content = "This is sensitive information"
        
        secure_memory_manager.add_message(
            conversation_id=conversation_id,
            user_id="user1",
            role="user",
            content=content
        )
        
        # Mock the internal storage to verify encryption
        # In real implementation, we'd check the actual storage backend
        # Here we verify that encryption/decryption works end-to-end
        
        # Retrieve and verify decryption
        conv = secure_memory_manager.get_conversation(conversation_id, "user1")
        assert conv is not None
        assert len(conv.messages) == 1
        assert conv.messages[0]['content'] == content
    
    def test_user_preferences_encryption(self, secure_memory_manager,
                                       security_context_user1):
        """Test that user preferences are encrypted."""
        set_security_context(security_context_user1)
        
        # Store preference
        pref_value = {"theme": "dark", "language": "en"}
        secure_memory_manager.store_user_preference(
            user_id="user1",
            preference_type="settings",
            preference_key="ui_preferences",
            preference_value=pref_value
        )
        
        # Retrieve preference
        retrieved = secure_memory_manager.get_user_preference(
            user_id="user1",
            preference_type="settings",
            preference_key="ui_preferences"
        )
        
        # Should be decrypted back to original
        assert retrieved == json.dumps(pref_value)
    
    def test_memory_search_with_isolation(self, secure_memory_manager,
                                        security_context_user1,
                                        security_context_user2):
        """Test that memory search respects user isolation."""
        # Create memories for user1
        set_security_context(security_context_user1)
        
        # Mock some memories (in real implementation these would be stored)
        # For now, test that search respects access control
        
        # User1 searches their memories
        results = secure_memory_manager.search_memories(
            user_id="user1",
            query="test"
        )
        # Results would be filtered by access control
        
        # Switch to user2
        set_security_context(security_context_user2)
        
        # User2 cannot search user1's memories
        results = secure_memory_manager.search_memories(
            user_id="user1",
            query="test"
        )
        assert results == []  # Empty due to access control
    
    def test_audit_logging(self, secure_memory_manager,
                         security_context_user1):
        """Test that access is properly audited."""
        set_security_context(security_context_user1)
        
        # Perform some operations
        conversation_id = "conv1"
        secure_memory_manager.add_message(
            conversation_id=conversation_id,
            user_id="user1",
            role="user",
            content="Test message"
        )
        
        # Get access logs
        isolation_manager = secure_memory_manager.isolation_manager
        logs = isolation_manager.get_access_logs(user_id="user1")
        
        # Should have access logs
        assert len(logs) > 0
        assert any(log.resource_type == "conversation" for log in logs)
        assert any(log.access_type == AccessType.WRITE for log in logs)
    
    def test_data_export(self, secure_memory_manager,
                        security_context_user1,
                        security_context_user2):
        """Test GDPR-compliant data export."""
        # Create some data as user1
        set_security_context(security_context_user1)
        
        secure_memory_manager.add_message(
            conversation_id="conv1",
            user_id="user1",
            role="user",
            content="Private message"
        )
        
        secure_memory_manager.store_user_preference(
            user_id="user1",
            preference_type="privacy",
            preference_key="data_sharing",
            preference_value="minimal"
        )
        
        # User1 can export their own data
        export = secure_memory_manager.export_user_data("user1")
        assert export['user_id'] == "user1"
        assert len(export['conversations']) > 0
        assert len(export['preferences']) > 0
        
        # Switch to user2
        set_security_context(security_context_user2)
        
        # User2 cannot export user1's data
        with pytest.raises(AuthorizationError):
            secure_memory_manager.export_user_data("user1")
    
    def test_concurrent_access(self, secure_memory_manager,
                             security_context_user1,
                             security_context_user2):
        """Test that concurrent access from different users is properly isolated."""
        import threading
        
        results = {'user1': None, 'user2': None, 'errors': []}
        
        def user1_operation():
            try:
                set_security_context(security_context_user1)
                secure_memory_manager.add_message(
                    conversation_id="conv1",
                    user_id="user1",
                    role="user",
                    content="User1 message"
                )
                conv = secure_memory_manager.get_conversation("conv1", "user1")
                results['user1'] = len(conv.messages) if conv else 0
            except Exception as e:
                results['errors'].append(('user1', str(e)))
        
        def user2_operation():
            try:
                set_security_context(security_context_user2)
                # This should fail
                conv = secure_memory_manager.get_conversation("conv1", "user1")
                results['user2'] = 'should_not_reach_here'
            except AuthorizationError:
                results['user2'] = 'access_denied'
            except Exception as e:
                results['errors'].append(('user2', str(e)))
        
        # Run operations concurrently
        t1 = threading.Thread(target=user1_operation)
        t2 = threading.Thread(target=user2_operation)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        # Verify results
        assert results['user1'] == 1  # User1 successfully added message
        assert results['user2'] == 'access_denied'  # User2 was denied
        assert len(results['errors']) == 0  # No unexpected errors


class TestEncryptionProvider:
    """Test encryption functionality."""
    
    def test_encrypt_decrypt_string(self):
        """Test basic string encryption/decryption."""
        from src.security.encryption import EncryptionProvider
        
        provider = EncryptionProvider()
        plaintext = "This is a secret message"
        
        # Encrypt
        encrypted = provider.encrypt(plaintext)
        assert encrypted.ciphertext != plaintext
        assert encrypted.key_version is not None
        
        # Decrypt
        decrypted = provider.decrypt(encrypted, return_type=str)
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_dict(self):
        """Test dictionary encryption/decryption."""
        from src.security.encryption import EncryptionProvider
        
        provider = EncryptionProvider()
        data = {"user": "test", "preferences": {"theme": "dark"}}
        
        # Encrypt
        encrypted = provider.encrypt(data)
        
        # Decrypt
        decrypted = provider.decrypt(encrypted, return_type=dict)
        assert decrypted == data
    
    def test_field_encryption(self):
        """Test selective field encryption."""
        from src.security.encryption import EncryptionProvider
        
        provider = EncryptionProvider()
        data = {
            "public_field": "visible",
            "secret_field": "hidden",
            "another_secret": "also hidden"
        }
        
        # Encrypt specific fields
        encrypted_data = provider.encrypt_fields(
            data,
            ["secret_field", "another_secret"]
        )
        
        # Public field should be unchanged
        assert encrypted_data["public_field"] == "visible"
        
        # Secret fields should be encrypted
        assert isinstance(encrypted_data["secret_field"], dict)
        assert "ciphertext" in encrypted_data["secret_field"]
        assert encrypted_data["secret_field_encrypted"] is True
        
        # Decrypt fields
        decrypted_data = provider.decrypt_fields(
            encrypted_data,
            ["secret_field", "another_secret"]
        )
        
        # Should be back to original
        assert decrypted_data == data
        assert "secret_field_encrypted" not in decrypted_data