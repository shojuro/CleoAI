"""
Error tracking integration for CleoAI using Sentry.

This module provides comprehensive error tracking with:
- Automatic error capture and reporting
- Performance monitoring
- Custom context and user tracking
- Integration with logging and tracing
- Sensitive data scrubbing
"""
import os
import logging
from typing import Optional, Dict, Any, Callable, List
from functools import wraps
from contextlib import contextmanager

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.pymongo import PyMongoIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.pure_eval import PureEvalIntegration
from sentry_sdk.integrations.stdlib import StdlibIntegration
from sentry_sdk.integrations.excepthook import ExcepthookIntegration

from ..security.data_isolation import get_current_security_context
from .tracing import get_current_trace_id

logger = logging.getLogger(__name__)


class ErrorTrackingConfig:
    """Configuration for error tracking."""
    
    def __init__(self):
        """Initialize error tracking configuration from environment."""
        self.enabled = os.getenv("ERROR_TRACKING_ENABLED", "true").lower() == "true"
        self.dsn = os.getenv("SENTRY_DSN", "")
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.release = os.getenv("SERVICE_VERSION", "2.0.0")
        self.traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0"))
        self.profiles_sample_rate = float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "1.0"))
        self.attach_stacktrace = os.getenv("SENTRY_ATTACH_STACKTRACE", "true").lower() == "true"
        self.send_default_pii = os.getenv("SENTRY_SEND_PII", "false").lower() == "true"
        self.debug = os.getenv("SENTRY_DEBUG", "false").lower() == "true"
        
        # Performance monitoring
        self.enable_tracing = os.getenv("SENTRY_ENABLE_TRACING", "true").lower() == "true"
        self.slow_request_threshold = int(os.getenv("SENTRY_SLOW_REQUEST_MS", "1000"))
        
        # Data scrubbing
        self.scrub_defaults = True  # Always scrub sensitive data
        self.scrub_ip_addresses = True
        
        # Custom tags
        self.default_tags = {
            "service": "cleoai",
            "component": os.getenv("COMPONENT_NAME", "api"),
            "deployment": os.getenv("DEPLOYMENT_TYPE", "docker"),
        }


def before_send(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process event before sending to Sentry.
    
    This function:
    - Adds custom context
    - Scrubs sensitive data
    - Filters unwanted events
    - Enhances error information
    """
    # Add security context if available
    security_context = get_current_security_context()
    if security_context:
        event.setdefault("user", {}).update({
            "id": security_context.user_id,
            "username": security_context.user_id,
            "ip_address": security_context.metadata.get("ip_address"),
        })
        event.setdefault("tags", {})["user_roles"] = ",".join(security_context.roles)
        event.setdefault("extra", {})["isolation_level"] = security_context.isolation_level.value
    
    # Add trace context
    trace_id = get_current_trace_id()
    if trace_id:
        event.setdefault("contexts", {}).setdefault("trace", {})["trace_id"] = trace_id
        event.setdefault("tags", {})["trace_id"] = trace_id
    
    # Scrub sensitive data from request data
    if "request" in event and "data" in event["request"]:
        event["request"]["data"] = scrub_sensitive_data(event["request"]["data"])
    
    # Scrub sensitive data from extra context
    if "extra" in event:
        event["extra"] = scrub_sensitive_data(event["extra"])
    
    # Filter out certain errors
    if "exception" in event:
        for exception in event["exception"].get("values", []):
            # Don't report client disconnections
            if exception.get("type") == "ConnectionError":
                return None
            
            # Don't report cancelled operations
            if "CancelledError" in exception.get("type", ""):
                return None
    
    # Add additional context for model errors
    if hint and "exc_info" in hint:
        exc_type, exc_value, exc_tb = hint["exc_info"]
        if "model" in str(exc_type).lower() or "inference" in str(exc_type).lower():
            event.setdefault("tags", {})["error_category"] = "model_error"
            event.setdefault("extra", {})["model_context"] = {
                "model_loaded": hasattr(exc_value, "model_name"),
                "inference_active": hasattr(exc_value, "inference_id"),
            }
    
    return event


def before_send_transaction(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process transaction before sending to Sentry.
    
    This function enhances performance monitoring data.
    """
    # Add custom measurements
    if "measurements" not in event:
        event["measurements"] = {}
    
    # Add security context to transactions
    security_context = get_current_security_context()
    if security_context:
        event.setdefault("tags", {})["user_id"] = security_context.user_id
        event.setdefault("tags", {})["user_roles"] = ",".join(security_context.roles)
    
    # Filter out health check transactions
    if event.get("transaction", "").endswith("/health"):
        return None
    
    return event


def scrub_sensitive_data(data: Any) -> Any:
    """
    Recursively scrub sensitive data from dictionaries and lists.
    
    Args:
        data: Data to scrub
        
    Returns:
        Scrubbed data
    """
    if isinstance(data, dict):
        scrubbed = {}
        sensitive_keys = {
            'password', 'passwd', 'pwd', 'secret', 'token', 'api_key',
            'apikey', 'access_token', 'refresh_token', 'private_key',
            'authorization', 'cookie', 'session', 'csrf', 'xsrf',
            'credit_card', 'cc_number', 'cvv', 'ssn', 'tax_id'
        }
        
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                scrubbed[key] = "[REDACTED]"
            else:
                scrubbed[key] = scrub_sensitive_data(value)
        return scrubbed
    
    elif isinstance(data, list):
        return [scrub_sensitive_data(item) for item in data]
    
    elif isinstance(data, str):
        # Scrub potential secrets in strings
        if len(data) > 20 and data.isalnum():
            # Might be a token or key
            return "[POSSIBLE_SECRET_REDACTED]"
        return data
    
    return data


def initialize_error_tracking(config: Optional[ErrorTrackingConfig] = None) -> bool:
    """
    Initialize Sentry error tracking.
    
    Args:
        config: Error tracking configuration
        
    Returns:
        True if initialized successfully
    """
    if config is None:
        config = ErrorTrackingConfig()
    
    if not config.enabled:
        logger.info("Error tracking disabled")
        return False
    
    if not config.dsn:
        logger.warning("Error tracking enabled but SENTRY_DSN not provided")
        return False
    
    try:
        # Configure logging integration
        logging_integration = LoggingIntegration(
            level=logging.INFO,  # Capture info and above
            event_level=logging.ERROR  # Send errors as events
        )
        
        # Initialize Sentry
        sentry_sdk.init(
            dsn=config.dsn,
            environment=config.environment,
            release=config.release,
            traces_sample_rate=config.traces_sample_rate if config.enable_tracing else 0.0,
            profiles_sample_rate=config.profiles_sample_rate,
            attach_stacktrace=config.attach_stacktrace,
            send_default_pii=config.send_default_pii,
            debug=config.debug,
            
            # Integrations
            integrations=[
                FastApiIntegration(
                    transaction_style="endpoint",
                    failed_request_status_codes={400, 401, 403, 404, 429, 500, 502, 503, 504}
                ),
                logging_integration,
                RedisIntegration(),
                PyMongoIntegration(),
                SqlalchemyIntegration(),
                PureEvalIntegration(),
                StdlibIntegration(),
                ExcepthookIntegration(always_run=True),
            ],
            
            # Callbacks
            before_send=before_send,
            before_send_transaction=before_send_transaction,
            
            # Performance
            traces_sampler=traces_sampler,
            
            # Options
            max_breadcrumbs=100,
            request_bodies="medium",
            with_locals=True,
            
            # Default tags
            _experiments={
                "profiles_sample_rate": config.profiles_sample_rate,
            }
        )
        
        # Set default tags
        for key, value in config.default_tags.items():
            sentry_sdk.set_tag(key, value)
        
        logger.info(f"Error tracking initialized (environment: {config.environment})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize error tracking: {e}")
        return False


def traces_sampler(sampling_context: Dict[str, Any]) -> float:
    """
    Dynamic sampling for performance monitoring.
    
    Args:
        sampling_context: Context for sampling decision
        
    Returns:
        Sample rate (0.0 to 1.0)
    """
    # Always trace errors
    if sampling_context.get("parent_sampled") is True:
        return 1.0
    
    # Get transaction name
    transaction_name = sampling_context.get("transaction_context", {}).get("name", "")
    
    # Don't sample health checks
    if "/health" in transaction_name:
        return 0.0
    
    # Higher sampling for critical endpoints
    critical_endpoints = ["/graphql", "/api/inference", "/api/train"]
    if any(endpoint in transaction_name for endpoint in critical_endpoints):
        return 1.0
    
    # Lower sampling for high-volume endpoints
    high_volume_endpoints = ["/metrics", "/api/memory/search"]
    if any(endpoint in transaction_name for endpoint in high_volume_endpoints):
        return 0.1
    
    # Default sampling
    return float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0"))


def capture_exception(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = "error",
    fingerprint: Optional[List[str]] = None
) -> Optional[str]:
    """
    Capture an exception with additional context.
    
    Args:
        error: Exception to capture
        context: Additional context
        level: Error level
        fingerprint: Custom fingerprint for grouping
        
    Returns:
        Event ID if captured
    """
    with sentry_sdk.push_scope() as scope:
        # Set level
        scope.level = level
        
        # Add context
        if context:
            for key, value in context.items():
                scope.set_context(key, value)
        
        # Set fingerprint for custom grouping
        if fingerprint:
            scope.fingerprint = fingerprint
        
        # Add security context
        security_context = get_current_security_context()
        if security_context:
            scope.user = {
                "id": security_context.user_id,
                "username": security_context.user_id,
            }
            scope.set_tag("user_roles", ",".join(security_context.roles))
        
        # Capture
        return sentry_sdk.capture_exception(error)


def capture_message(
    message: str,
    level: str = "info",
    context: Optional[Dict[str, Any]] = None,
    fingerprint: Optional[List[str]] = None
) -> Optional[str]:
    """
    Capture a message with additional context.
    
    Args:
        message: Message to capture
        level: Message level
        context: Additional context
        fingerprint: Custom fingerprint for grouping
        
    Returns:
        Event ID if captured
    """
    with sentry_sdk.push_scope() as scope:
        # Set level
        scope.level = level
        
        # Add context
        if context:
            for key, value in context.items():
                scope.set_context(key, value)
        
        # Set fingerprint
        if fingerprint:
            scope.fingerprint = fingerprint
        
        # Capture
        return sentry_sdk.capture_message(message, level=level)


def set_user_context(user_id: str, email: Optional[str] = None, username: Optional[str] = None):
    """
    Set user context for error tracking.
    
    Args:
        user_id: User ID
        email: User email
        username: Username
    """
    sentry_sdk.set_user({
        "id": user_id,
        "email": email,
        "username": username or user_id,
    })


def add_breadcrumb(
    message: str,
    category: str,
    level: str = "info",
    data: Optional[Dict[str, Any]] = None
):
    """
    Add a breadcrumb for tracking user actions.
    
    Args:
        message: Breadcrumb message
        category: Category (e.g., 'auth', 'navigation')
        level: Breadcrumb level
        data: Additional data
    """
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {}
    )


@contextmanager
def error_tracking_context(
    operation: str,
    data: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None
):
    """
    Context manager for error tracking with automatic breadcrumbs.
    
    Args:
        operation: Operation name
        data: Operation data
        user_id: User performing operation
    """
    # Add breadcrumb for operation start
    add_breadcrumb(
        message=f"Starting {operation}",
        category="operation",
        data=data
    )
    
    # Set user if provided
    if user_id:
        set_user_context(user_id)
    
    try:
        yield
        # Add success breadcrumb
        add_breadcrumb(
            message=f"Completed {operation}",
            category="operation",
            level="info"
        )
    except Exception as e:
        # Add failure breadcrumb
        add_breadcrumb(
            message=f"Failed {operation}: {str(e)}",
            category="operation",
            level="error",
            data={"error": str(e)}
        )
        raise


def track_error(func: Callable) -> Callable:
    """
    Decorator to automatically track errors in functions.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        operation = f"{func.__module__}.{func.__name__}"
        with error_tracking_context(operation):
            return func(*args, **kwargs)
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        operation = f"{func.__module__}.{func.__name__}"
        with error_tracking_context(operation):
            return await func(*args, **kwargs)
    
    import asyncio
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class ErrorMetrics:
    """Helper class for tracking error metrics."""
    
    @staticmethod
    def track_model_error(
        model_name: str,
        error_type: str,
        inference_id: Optional[str] = None,
        input_tokens: Optional[int] = None
    ):
        """Track model-specific errors."""
        capture_message(
            f"Model error: {error_type}",
            level="error",
            context={
                "model": {
                    "name": model_name,
                    "error_type": error_type,
                    "inference_id": inference_id,
                    "input_tokens": input_tokens,
                }
            },
            fingerprint=["model_error", model_name, error_type]
        )
    
    @staticmethod
    def track_memory_error(
        backend: str,
        operation: str,
        error_type: str,
        user_id: Optional[str] = None
    ):
        """Track memory operation errors."""
        capture_message(
            f"Memory error: {error_type}",
            level="error",
            context={
                "memory": {
                    "backend": backend,
                    "operation": operation,
                    "error_type": error_type,
                    "user_id": user_id,
                }
            },
            fingerprint=["memory_error", backend, operation, error_type]
        )
    
    @staticmethod
    def track_api_error(
        endpoint: str,
        method: str,
        status_code: int,
        error_message: str,
        user_id: Optional[str] = None
    ):
        """Track API errors."""
        capture_message(
            f"API error: {status_code} on {endpoint}",
            level="error" if status_code >= 500 else "warning",
            context={
                "api": {
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": status_code,
                    "error_message": error_message,
                    "user_id": user_id,
                }
            },
            fingerprint=["api_error", endpoint, method, str(status_code)]
        )


# Global error metrics instance
error_metrics = ErrorMetrics()