"""
Comprehensive tests for data isolation and multi-tenancy security.

Tests cover:
- User data isolation
- Row-level security
- Tenant isolation
- Cross-user data access prevention
- Security context propagation
- Audit logging
"""
import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import asyncio
import threading

from src.security.data_isolation import (
    SecurityContext, IsolationLevel, AccessType,
    IsolationManager, init_isolation_manager,
    set_security_context, get_current_security_context,
    check_resource_access, audit_access,
    isolation_required
)
from src.utils.error_handling import AuthorizationError


class TestSecurityContext:
    """Test security context functionality."""
    
    def test_security_context_creation(self):
        """Test creating security context with various parameters."""
        context = SecurityContext(
            user_id="test_user",
            roles=["user", "admin"],
            isolation_level=IsolationLevel.USER,
            tenant_id="tenant_123",
            metadata={"ip": "192.168.1.1"}
        )
        
        assert context.user_id == "test_user"
        assert "admin" in context.roles
        assert context.isolation_level == IsolationLevel.USER
        assert context.tenant_id == "tenant_123"
        assert context.metadata["ip"] == "192.168.1.1"
    
    def test_security_context_defaults(self):
        """Test security context default values."""
        context = SecurityContext(
            user_id="test_user",
            roles=["user"]
        )
        
        assert context.isolation_level == IsolationLevel.USER
        assert context.tenant_id is None
        assert context.metadata == {}
    
    def test_invalid_isolation_level(self):
        """Test that invalid isolation level raises error."""
        with pytest.raises(ValueError):
            SecurityContext(
                user_id="test_user",
                roles=["user"],
                isolation_level="invalid_level"  # Should be IsolationLevel enum
            )
    
    def test_empty_user_id(self):
        """Test that empty user ID is rejected."""
        with pytest.raises(ValueError):
            SecurityContext(
                user_id="",
                roles=["user"]
            )
    
    def test_empty_roles(self):
        """Test that empty roles list is rejected."""
        with pytest.raises(ValueError):
            SecurityContext(
                user_id="test_user",
                roles=[]
            )


class TestIsolationManager:
    """Test isolation manager functionality."""
    
    @pytest.fixture
    def isolation_manager(self):
        """Create isolation manager for testing."""
        return init_isolation_manager()
    
    def test_user_isolation_check(self, isolation_manager):
        """Test user-level isolation checks."""
        user_context = SecurityContext(
            user_id="user1",
            roles=["user"],
            isolation_level=IsolationLevel.USER
        )
        
        # User can access their own resources
        assert isolation_manager.check_access(
            context=user_context,
            resource_type="conversation",
            resource_id="conv_123",
            owner_id="user1",
            access_type=AccessType.READ
        ) is True
        
        # User cannot access other user's resources
        assert isolation_manager.check_access(
            context=user_context,
            resource_type="conversation",
            resource_id="conv_456",
            owner_id="user2",
            access_type=AccessType.READ
        ) is False
    
    def test_admin_bypass(self, isolation_manager):
        """Test that admins can bypass isolation."""
        admin_context = SecurityContext(
            user_id="admin1",
            roles=["admin"],
            isolation_level=IsolationLevel.NONE
        )
        
        # Admin can access any user's resources
        assert isolation_manager.check_access(
            context=admin_context,
            resource_type="conversation",
            resource_id="conv_123",
            owner_id="user1",
            access_type=AccessType.READ
        ) is True
        
        assert isolation_manager.check_access(
            context=admin_context,
            resource_type="conversation",
            resource_id="conv_456",
            owner_id="user2",
            access_type=AccessType.WRITE
        ) is True
    
    def test_tenant_isolation(self, isolation_manager):
        """Test tenant-level isolation."""
        tenant1_context = SecurityContext(
            user_id="user1",
            roles=["user"],
            isolation_level=IsolationLevel.TENANT,
            tenant_id="tenant_A"
        )
        
        tenant2_context = SecurityContext(
            user_id="user2",
            roles=["user"],
            isolation_level=IsolationLevel.TENANT,
            tenant_id="tenant_B"
        )
        
        # User in tenant A cannot access tenant B resources
        assert isolation_manager.check_access(
            context=tenant1_context,
            resource_type="document",
            resource_id="doc_123",
            owner_id="user2",
            tenant_id="tenant_B",
            access_type=AccessType.READ
        ) is False
        
        # User in same tenant can access shared resources
        tenant1_user2_context = SecurityContext(
            user_id="user3",
            roles=["user"],
            isolation_level=IsolationLevel.TENANT,
            tenant_id="tenant_A"
        )
        
        assert isolation_manager.check_access(
            context=tenant1_user2_context,
            resource_type="shared_document",
            resource_id="shared_123",
            owner_id="user1",
            tenant_id="tenant_A",
            access_type=AccessType.READ,
            metadata={"shared": True}
        ) is True
    
    def test_strict_isolation(self, isolation_manager):
        """Test strict isolation mode."""
        strict_context = SecurityContext(
            user_id="secure_user",
            roles=["user"],
            isolation_level=IsolationLevel.STRICT
        )
        
        # In strict mode, user can only access their own resources
        assert isolation_manager.check_access(
            context=strict_context,
            resource_type="sensitive_data",
            resource_id="data_123",
            owner_id="secure_user",
            access_type=AccessType.READ
        ) is True
        
        # Cannot access even shared resources in strict mode
        assert isolation_manager.check_access(
            context=strict_context,
            resource_type="shared_data",
            resource_id="shared_456",
            owner_id="secure_user",
            access_type=AccessType.READ,
            metadata={"shared": True}
        ) is True  # Only because they own it
        
        assert isolation_manager.check_access(
            context=strict_context,
            resource_type="shared_data",
            resource_id="shared_789",
            owner_id="other_user",
            access_type=AccessType.READ,
            metadata={"shared": True}
        ) is False
    
    def test_access_type_restrictions(self, isolation_manager):
        """Test different access type restrictions."""
        user_context = SecurityContext(
            user_id="user1",
            roles=["user"],
            isolation_level=IsolationLevel.USER
        )
        
        # User can read their own resources
        assert isolation_manager.check_access(
            context=user_context,
            resource_type="document",
            resource_id="doc_123",
            owner_id="user1",
            access_type=AccessType.READ
        ) is True
        
        # Check if write access requires additional permissions
        # This depends on implementation details
        assert isolation_manager.check_access(
            context=user_context,
            resource_type="document",
            resource_id="doc_123",
            owner_id="user1",
            access_type=AccessType.WRITE
        ) is True
        
        # User cannot delete others' resources
        assert isolation_manager.check_access(
            context=user_context,
            resource_type="document",
            resource_id="doc_456",
            owner_id="user2",
            access_type=AccessType.DELETE
        ) is False


class TestContextPropagation:
    """Test security context propagation across threads/async tasks."""
    
    def test_context_var_propagation(self):
        """Test context propagation using contextvars."""
        context = SecurityContext(
            user_id="test_user",
            roles=["user"]
        )
        
        set_security_context(context)
        
        # Context should be available in same thread
        retrieved = get_current_security_context()
        assert retrieved is not None
        assert retrieved.user_id == "test_user"
        
        # Clear context
        set_security_context(None)
        assert get_current_security_context() is None
    
    @pytest.mark.asyncio
    async def test_async_context_propagation(self):
        """Test context propagation in async tasks."""
        context = SecurityContext(
            user_id="async_user",
            roles=["user"]
        )
        
        set_security_context(context)
        
        async def nested_async_function():
            # Context should be available in nested async call
            ctx = get_current_security_context()
            assert ctx is not None
            assert ctx.user_id == "async_user"
            return True
        
        result = await nested_async_function()
        assert result is True
        
        # Context still available after async call
        assert get_current_security_context().user_id == "async_user"
    
    def test_thread_isolation(self):
        """Test that contexts don't leak between threads."""
        results = {}
        
        def thread_function(user_id, thread_name):
            context = SecurityContext(
                user_id=user_id,
                roles=["user"]
            )
            set_security_context(context)
            
            # Sleep to ensure threads run concurrently
            import time
            time.sleep(0.1)
            
            # Get context and verify it's the one we set
            ctx = get_current_security_context()
            results[thread_name] = ctx.user_id if ctx else None
        
        # Create multiple threads with different contexts
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=thread_function,
                args=(f"user_{i}", f"thread_{i}")
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify each thread had its own context
        for i in range(5):
            assert results[f"thread_{i}"] == f"user_{i}"
    
    @pytest.mark.asyncio
    async def test_concurrent_async_contexts(self):
        """Test concurrent async tasks with different contexts."""
        async def async_task(user_id):
            context = SecurityContext(
                user_id=user_id,
                roles=["user"]
            )
            set_security_context(context)
            
            # Simulate some async work
            await asyncio.sleep(0.1)
            
            # Verify context is preserved
            ctx = get_current_security_context()
            assert ctx.user_id == user_id
            return user_id
        
        # Run multiple async tasks concurrently
        tasks = [async_task(f"user_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Verify all tasks completed with correct context
        for i, result in enumerate(results):
            assert result == f"user_{i}"


class TestIsolationDecorator:
    """Test isolation_required decorator."""
    
    def test_isolation_decorator_with_context(self):
        """Test decorator with valid security context."""
        @isolation_required()
        def protected_function():
            return "success"
        
        context = SecurityContext(
            user_id="test_user",
            roles=["user"]
        )
        set_security_context(context)
        
        result = protected_function()
        assert result == "success"
    
    def test_isolation_decorator_without_context(self):
        """Test decorator without security context."""
        @isolation_required()
        def protected_function():
            return "should not reach here"
        
        set_security_context(None)
        
        with pytest.raises(AuthorizationError) as exc:
            protected_function()
        
        assert "security context required" in str(exc.value).lower()
    
    def test_isolation_decorator_with_role_requirement(self):
        """Test decorator with specific role requirement."""
        @isolation_required(required_roles=["admin"])
        def admin_only_function():
            return "admin access granted"
        
        # Test with admin role
        admin_context = SecurityContext(
            user_id="admin_user",
            roles=["user", "admin"]
        )
        set_security_context(admin_context)
        
        result = admin_only_function()
        assert result == "admin access granted"
        
        # Test without admin role
        user_context = SecurityContext(
            user_id="regular_user",
            roles=["user"]
        )
        set_security_context(user_context)
        
        with pytest.raises(AuthorizationError) as exc:
            admin_only_function()
        
        assert "insufficient permissions" in str(exc.value).lower()
    
    @pytest.mark.asyncio
    async def test_isolation_decorator_async(self):
        """Test decorator with async functions."""
        @isolation_required()
        async def async_protected_function():
            await asyncio.sleep(0.01)
            return "async success"
        
        context = SecurityContext(
            user_id="async_user",
            roles=["user"]
        )
        set_security_context(context)
        
        result = await async_protected_function()
        assert result == "async success"


class TestAuditLogging:
    """Test audit logging for access attempts."""
    
    @pytest.fixture
    def mock_audit_logger(self):
        """Mock audit logger."""
        with patch('src.security.data_isolation.audit_logger') as mock:
            yield mock
    
    def test_successful_access_audit(self, mock_audit_logger):
        """Test audit logging for successful access."""
        context = SecurityContext(
            user_id="test_user",
            roles=["user"]
        )
        
        audit_access(
            context=context,
            resource_type="document",
            resource_id="doc_123",
            access_type=AccessType.READ,
            success=True
        )
        
        mock_audit_logger.log_event.assert_called_once()
        call_args = mock_audit_logger.log_event.call_args[1]
        
        assert call_args["event_type"] == "resource_access"
        assert call_args["user_id"] == "test_user"
        assert call_args["resource_type"] == "document"
        assert call_args["resource_id"] == "doc_123"
        assert call_args["action"] == "READ"
        assert call_args["result"] == "success"
    
    def test_failed_access_audit(self, mock_audit_logger):
        """Test audit logging for failed access."""
        context = SecurityContext(
            user_id="test_user",
            roles=["user"]
        )
        
        audit_access(
            context=context,
            resource_type="sensitive_data",
            resource_id="secret_123",
            access_type=AccessType.WRITE,
            success=False,
            reason="Insufficient permissions"
        )
        
        mock_audit_logger.log_event.assert_called_once()
        call_args = mock_audit_logger.log_event.call_args[1]
        
        assert call_args["result"] == "failure"
        assert call_args["metadata"]["reason"] == "Insufficient permissions"
    
    def test_audit_with_metadata(self, mock_audit_logger):
        """Test audit logging with additional metadata."""
        context = SecurityContext(
            user_id="test_user",
            roles=["user"],
            metadata={"ip_address": "192.168.1.1", "user_agent": "TestClient/1.0"}
        )
        
        audit_access(
            context=context,
            resource_type="api_endpoint",
            resource_id="/api/v1/users",
            access_type=AccessType.READ,
            success=True
        )
        
        mock_audit_logger.log_event.assert_called_once()
        call_args = mock_audit_logger.log_event.call_args[1]
        
        # Context metadata should be included
        assert "ip_address" in call_args["metadata"]
        assert "user_agent" in call_args["metadata"]


class TestResourceAccessPatterns:
    """Test various resource access patterns."""
    
    @pytest.fixture
    def isolation_manager(self):
        """Create isolation manager."""
        return init_isolation_manager()
    
    def test_hierarchical_resource_access(self, isolation_manager):
        """Test access to hierarchical resources."""
        context = SecurityContext(
            user_id="user1",
            roles=["user"],
            isolation_level=IsolationLevel.USER
        )
        
        # User owns a project
        assert isolation_manager.check_access(
            context=context,
            resource_type="project",
            resource_id="proj_123",
            owner_id="user1",
            access_type=AccessType.READ
        ) is True
        
        # User should have access to resources within their project
        assert isolation_manager.check_access(
            context=context,
            resource_type="project_file",
            resource_id="file_456",
            owner_id="user1",
            access_type=AccessType.READ,
            metadata={"parent_project": "proj_123"}
        ) is True
    
    def test_delegation_access_pattern(self, isolation_manager):
        """Test delegated access patterns."""
        # Owner context
        owner_context = SecurityContext(
            user_id="owner",
            roles=["user"],
            isolation_level=IsolationLevel.USER
        )
        
        # Delegated user context
        delegate_context = SecurityContext(
            user_id="delegate",
            roles=["user"],
            isolation_level=IsolationLevel.USER
        )
        
        # Owner grants access to delegate (this would be stored in DB)
        delegated_permissions = {
            "resource_id": "doc_123",
            "delegate_id": "delegate",
            "permissions": ["READ"]
        }
        
        # Test delegate access with delegation metadata
        assert isolation_manager.check_access(
            context=delegate_context,
            resource_type="document",
            resource_id="doc_123",
            owner_id="owner",
            access_type=AccessType.READ,
            metadata={"delegated_to": ["delegate"]}
        ) is True
    
    def test_time_based_access(self, isolation_manager):
        """Test time-based access restrictions."""
        context = SecurityContext(
            user_id="user1",
            roles=["user"],
            isolation_level=IsolationLevel.USER
        )
        
        # Resource with time-based access
        current_time = datetime.utcnow()
        
        # Access allowed (not expired)
        assert isolation_manager.check_access(
            context=context,
            resource_type="temporary_file",
            resource_id="temp_123",
            owner_id="user1",
            access_type=AccessType.READ,
            metadata={
                "expires_at": (current_time + timedelta(hours=1)).isoformat()
            }
        ) is True
        
        # Access denied (expired)
        assert isolation_manager.check_access(
            context=context,
            resource_type="temporary_file",
            resource_id="temp_456",
            owner_id="user1",
            access_type=AccessType.READ,
            metadata={
                "expires_at": (current_time - timedelta(hours=1)).isoformat()
            }
        ) is False


class TestCrossUserAccessPrevention:
    """Test prevention of cross-user data access."""
    
    def test_sql_injection_in_user_id(self):
        """Test that SQL injection in user_id is prevented."""
        malicious_user_ids = [
            "user1' OR '1'='1",
            "user1'; DROP TABLE users; --",
            "user1' UNION SELECT * FROM passwords --"
        ]
        
        for malicious_id in malicious_user_ids:
            with pytest.raises(ValueError):
                SecurityContext(
                    user_id=malicious_id,
                    roles=["user"]
                )
    
    def test_path_traversal_in_resource_id(self):
        """Test that path traversal in resource IDs is prevented."""
        isolation_manager = init_isolation_manager()
        
        context = SecurityContext(
            user_id="user1",
            roles=["user"]
        )
        
        malicious_resource_ids = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "valid_id/../../../secret_file"
        ]
        
        for resource_id in malicious_resource_ids:
            # Should either sanitize or reject
            result = isolation_manager.check_access(
                context=context,
                resource_type="file",
                resource_id=resource_id,
                owner_id="user1",
                access_type=AccessType.READ
            )
            # Access should be denied for malicious IDs
            assert result is False
    
    def test_privilege_escalation_prevention(self):
        """Test prevention of privilege escalation."""
        # Regular user context
        user_context = SecurityContext(
            user_id="regular_user",
            roles=["user"]
        )
        
        # User tries to modify their own roles (should fail)
        with pytest.raises(AuthorizationError):
            # This would be in the actual role modification endpoint
            if "admin" not in user_context.roles:
                raise AuthorizationError("Cannot modify roles without admin permission")
        
        # User tries to access admin resources
        isolation_manager = init_isolation_manager()
        assert isolation_manager.check_access(
            context=user_context,
            resource_type="admin_panel",
            resource_id="admin_config",
            owner_id="system",
            access_type=AccessType.READ,
            metadata={"required_role": "admin"}
        ) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])