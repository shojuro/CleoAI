"""
Data isolation module for row-level security in CleoAI.

This module implements user data isolation to ensure that users can only
access their own data, with proper audit logging and access control.
"""
import logging
import time
import uuid
from typing import Optional, Dict, Any, List, Set, Callable
from dataclasses import dataclass, field
from functools import wraps
from contextvars import ContextVar
from enum import Enum
import json

from ..utils.error_handling import AuthorizationError
from ..api.auth import UserRole

logger = logging.getLogger(__name__)

# Context variable for current security context
_security_context: ContextVar[Optional['SecurityContext']] = ContextVar('security_context', default=None)


class IsolationLevel(Enum):
    """Data isolation levels."""
    NONE = "none"  # No isolation (admin only)
    USER = "user"  # User-level isolation
    TENANT = "tenant"  # Tenant-level isolation
    STRICT = "strict"  # Strictest isolation with audit


class AccessType(Enum):
    """Types of data access."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class SecurityContext:
    """
    Security context for the current request/operation.
    
    This context is used to enforce data isolation and access control
    throughout the application.
    """
    user_id: str
    tenant_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    isolation_level: IsolationLevel = IsolationLevel.USER
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_role(self, role: str) -> bool:
        """Check if context has a specific role."""
        return role in self.roles
    
    def is_admin(self) -> bool:
        """Check if context has admin privileges."""
        return UserRole.ADMIN in self.roles
    
    def can_access_user(self, target_user_id: str) -> bool:
        """Check if context can access data for a specific user."""
        # Admins can access any user
        if self.is_admin():
            return True
        
        # Users can only access their own data
        return self.user_id == target_user_id
    
    def can_access_tenant(self, target_tenant_id: str) -> bool:
        """Check if context can access data for a specific tenant."""
        if self.is_admin():
            return True
        
        if self.isolation_level == IsolationLevel.TENANT:
            return self.tenant_id == target_tenant_id
        
        return False


@dataclass
class AccessLog:
    """Audit log entry for data access."""
    timestamp: float
    user_id: str
    resource_type: str
    resource_id: str
    access_type: AccessType
    success: bool
    context: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class IsolationPolicy:
    """
    Policy for data isolation rules.
    
    Defines what data can be accessed by whom and under what conditions.
    """
    
    def __init__(self, 
                 resource_type: str,
                 isolation_level: IsolationLevel = IsolationLevel.USER,
                 audit_enabled: bool = True):
        self.resource_type = resource_type
        self.isolation_level = isolation_level
        self.audit_enabled = audit_enabled
        self.custom_rules: List[Callable] = []
    
    def add_rule(self, rule: Callable[[SecurityContext, Dict[str, Any]], bool]):
        """Add a custom access rule."""
        self.custom_rules.append(rule)
    
    def check_access(self, 
                     context: SecurityContext, 
                     resource: Dict[str, Any],
                     access_type: AccessType) -> bool:
        """
        Check if access is allowed based on the policy.
        
        Args:
            context: Current security context
            resource: Resource being accessed
            access_type: Type of access requested
            
        Returns:
            True if access is allowed
        """
        # Admin bypass
        if context.is_admin() and access_type != AccessType.DELETE:
            return True
        
        # Check isolation level
        if self.isolation_level == IsolationLevel.USER:
            resource_user_id = resource.get('user_id')
            if not resource_user_id or not context.can_access_user(resource_user_id):
                return False
        
        elif self.isolation_level == IsolationLevel.TENANT:
            resource_tenant_id = resource.get('tenant_id')
            if not resource_tenant_id or not context.can_access_tenant(resource_tenant_id):
                return False
        
        elif self.isolation_level == IsolationLevel.STRICT:
            # Strict mode requires explicit permission
            if access_type == AccessType.DELETE and not context.is_admin():
                return False
            
            resource_user_id = resource.get('user_id')
            if not resource_user_id or resource_user_id != context.user_id:
                return False
        
        # Check custom rules
        for rule in self.custom_rules:
            if not rule(context, resource):
                return False
        
        return True


class DataIsolationManager:
    """
    Central manager for data isolation and access control.
    
    This class manages security policies and enforces data isolation
    across the application.
    """
    
    def __init__(self, audit_backend: Optional[Any] = None):
        self.policies: Dict[str, IsolationPolicy] = {}
        self.audit_backend = audit_backend
        self.access_logs: List[AccessLog] = []
        self._setup_default_policies()
    
    def _setup_default_policies(self):
        """Set up default isolation policies."""
        # User data policies
        self.register_policy("conversation", IsolationPolicy(
            "conversation",
            IsolationLevel.USER,
            audit_enabled=True
        ))
        
        self.register_policy("memory", IsolationPolicy(
            "memory",
            IsolationLevel.USER,
            audit_enabled=True
        ))
        
        self.register_policy("preference", IsolationPolicy(
            "preference",
            IsolationLevel.USER,
            audit_enabled=True
        ))
        
        # System resources (admin only)
        self.register_policy("model", IsolationPolicy(
            "model",
            IsolationLevel.NONE,
            audit_enabled=True
        ))
    
    def register_policy(self, resource_type: str, policy: IsolationPolicy):
        """Register an isolation policy for a resource type."""
        self.policies[resource_type] = policy
        logger.info(f"Registered isolation policy for {resource_type}")
    
    def check_access(self,
                     resource_type: str,
                     resource: Dict[str, Any],
                     access_type: AccessType = AccessType.READ,
                     context: Optional[SecurityContext] = None) -> bool:
        """
        Check if access to a resource is allowed.
        
        Args:
            resource_type: Type of resource
            resource: Resource data
            access_type: Type of access
            context: Security context (uses current if not provided)
            
        Returns:
            True if access is allowed
            
        Raises:
            AuthorizationError: If access is denied
        """
        # Get current context if not provided
        if context is None:
            context = get_security_context()
            if context is None:
                raise AuthorizationError("No security context available")
        
        # Get policy
        policy = self.policies.get(resource_type)
        if not policy:
            logger.warning(f"No isolation policy for resource type: {resource_type}")
            # Default deny for unknown resources
            return False
        
        # Check access
        allowed = policy.check_access(context, resource, access_type)
        
        # Audit log
        if policy.audit_enabled:
            self._log_access(
                context=context,
                resource_type=resource_type,
                resource_id=resource.get('id', 'unknown'),
                access_type=access_type,
                success=allowed
            )
        
        if not allowed:
            raise AuthorizationError(
                f"Access denied to {resource_type} resource for user {context.user_id}"
            )
        
        return True
    
    def filter_resources(self,
                        resource_type: str,
                        resources: List[Dict[str, Any]],
                        context: Optional[SecurityContext] = None) -> List[Dict[str, Any]]:
        """
        Filter a list of resources based on access control.
        
        Args:
            resource_type: Type of resources
            resources: List of resources
            context: Security context
            
        Returns:
            Filtered list of accessible resources
        """
        if context is None:
            context = get_security_context()
            if context is None:
                return []
        
        policy = self.policies.get(resource_type)
        if not policy:
            return []
        
        filtered = []
        for resource in resources:
            try:
                if policy.check_access(context, resource, AccessType.READ):
                    filtered.append(resource)
            except Exception as e:
                logger.debug(f"Access check failed for resource: {e}")
        
        return filtered
    
    def _log_access(self,
                    context: SecurityContext,
                    resource_type: str,
                    resource_id: str,
                    access_type: AccessType,
                    success: bool,
                    error_message: Optional[str] = None):
        """Log an access attempt."""
        log_entry = AccessLog(
            timestamp=time.time(),
            user_id=context.user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            access_type=access_type,
            success=success,
            context={
                'request_id': context.request_id,
                'roles': context.roles,
                'tenant_id': context.tenant_id
            },
            error_message=error_message
        )
        
        self.access_logs.append(log_entry)
        
        # Send to audit backend if configured
        if self.audit_backend:
            try:
                self.audit_backend.log(log_entry)
            except Exception as e:
                logger.error(f"Failed to send audit log: {e}")
        
        # Log to standard logger
        if success:
            logger.info(f"Access granted: {context.user_id} -> {resource_type}:{resource_id} ({access_type.value})")
        else:
            logger.warning(f"Access denied: {context.user_id} -> {resource_type}:{resource_id} ({access_type.value})")
    
    def get_access_logs(self,
                       user_id: Optional[str] = None,
                       resource_type: Optional[str] = None,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None) -> List[AccessLog]:
        """Get access logs with optional filtering."""
        logs = self.access_logs
        
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        
        if resource_type:
            logs = [l for l in logs if l.resource_type == resource_type]
        
        if start_time:
            logs = [l for l in logs if l.timestamp >= start_time]
        
        if end_time:
            logs = [l for l in logs if l.timestamp <= end_time]
        
        return logs


# Global isolation manager instance
_isolation_manager: Optional[DataIsolationManager] = None


def init_isolation_manager(audit_backend: Optional[Any] = None) -> DataIsolationManager:
    """Initialize the global isolation manager."""
    global _isolation_manager
    _isolation_manager = DataIsolationManager(audit_backend)
    return _isolation_manager


def get_isolation_manager() -> DataIsolationManager:
    """Get the global isolation manager."""
    if _isolation_manager is None:
        return init_isolation_manager()
    return _isolation_manager


def set_security_context(context: SecurityContext):
    """Set the current security context."""
    _security_context.set(context)


def get_security_context() -> Optional[SecurityContext]:
    """Get the current security context."""
    return _security_context.get()


def with_security_context(context: SecurityContext):
    """Decorator to run a function with a specific security context."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            token = _security_context.set(context)
            try:
                return func(*args, **kwargs)
            finally:
                _security_context.reset(token)
        return wrapper
    return decorator


def require_access(resource_type: str, access_type: AccessType = AccessType.READ):
    """
    Decorator to enforce access control on a function.
    
    The decorated function must accept a 'resource' parameter or
    return a resource dict with appropriate user_id/tenant_id fields.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the resource from arguments or execute function
            resource = kwargs.get('resource')
            if resource is None:
                # Execute function to get resource
                result = func(*args, **kwargs)
                if isinstance(result, dict):
                    resource = result
                else:
                    raise ValueError("Cannot determine resource for access control")
            
            # Check access
            manager = get_isolation_manager()
            manager.check_access(resource_type, resource, access_type)
            
            # Execute function if not already done
            if resource is kwargs.get('resource'):
                return func(*args, **kwargs)
            else:
                return result
        
        return wrapper
    return decorator