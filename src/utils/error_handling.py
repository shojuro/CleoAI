"""
Comprehensive error handling utilities for CleoAI.

This module provides centralized error handling, retry logic, and
graceful degradation patterns for the entire system.
"""
import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from enum import Enum
import torch

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CleoAIError(Exception):
    """Base exception for all CleoAI errors."""
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None
    ) -> None:
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = time.time()
        
        # Log the error
        logger.error(
            f"{self.__class__.__name__}: {message}",
            extra={
                "severity": severity.value,
                "context": context,
                "recovery_suggestions": recovery_suggestions
            }
        )


class ModelError(CleoAIError):
    """Errors related to model operations."""
    pass


class MemoryError(CleoAIError):
    """Errors related to memory operations."""
    pass


class InferenceError(CleoAIError):
    """Errors related to inference operations."""
    pass


class TrainingError(CleoAIError):
    """Errors related to training operations."""
    pass


class ConfigurationError(CleoAIError):
    """Errors related to configuration."""
    pass


def retry_on_error(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """
    Decorator for retrying functions on error.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to retry on
        on_retry: Optional callback function called on each retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.2f} seconds..."
                        )
                        
                        if on_retry:
                            on_retry(e, attempt + 1)
                        
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            # If we get here, all attempts failed
            raise last_exception
        
        return wrapper
    return decorator


def handle_errors(
    default_return: Any = None,
    log_errors: bool = True,
    raise_on_critical: bool = True,
    error_map: Optional[Dict[Type[Exception], Callable[[Exception], Any]]] = None
) -> Callable:
    """
    Decorator for comprehensive error handling.
    
    Args:
        default_return: Default value to return on error
        log_errors: Whether to log errors
        raise_on_critical: Whether to re-raise critical errors
        error_map: Mapping of exception types to handler functions
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__}: {e}",
                        exc_info=True,
                        extra={"traceback": traceback.format_exc()}
                    )
                
                # Check if it's a critical error
                if isinstance(e, CleoAIError) and e.severity == ErrorSeverity.CRITICAL:
                    if raise_on_critical:
                        raise
                
                # Check error map for specific handlers
                if error_map:
                    for error_type, handler in error_map.items():
                        if isinstance(e, error_type):
                            return handler(e)
                
                return default_return
        
        return wrapper
    return decorator


class ErrorContext:
    """Context manager for error handling with cleanup."""
    
    def __init__(
        self,
        operation_name: str,
        cleanup_func: Optional[Callable[[], None]] = None,
        suppress_errors: bool = False
    ) -> None:
        self.operation_name = operation_name
        self.cleanup_func = cleanup_func
        self.suppress_errors = suppress_errors
        self.start_time = None
        
    def __enter__(self) -> 'ErrorContext':
        self.start_time = time.time()
        logger.debug(f"Starting operation: {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.time() - self.start_time
        
        if exc_type is None:
            logger.debug(
                f"Operation '{self.operation_name}' completed successfully in {duration:.2f}s"
            )
        else:
            logger.error(
                f"Operation '{self.operation_name}' failed after {duration:.2f}s: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
            
            # Run cleanup if provided
            if self.cleanup_func:
                try:
                    logger.info(f"Running cleanup for '{self.operation_name}'")
                    self.cleanup_func()
                except Exception as cleanup_error:
                    logger.error(f"Cleanup failed: {cleanup_error}")
        
        return self.suppress_errors


def validate_input(
    validation_rules: Dict[str, Callable[[Any], bool]],
    error_messages: Optional[Dict[str, str]] = None
) -> Callable:
    """
    Decorator for input validation.
    
    Args:
        validation_rules: Dict mapping parameter names to validation functions
        error_messages: Optional custom error messages
        
    Returns:
        Decorated function with input validation
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validation_rules.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        error_msg = (
                            error_messages.get(param_name) if error_messages 
                            else f"Invalid value for parameter '{param_name}': {value}"
                        )
                        raise ValueError(error_msg)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def graceful_degradation(
    fallback_func: Callable[..., T],
    condition: Optional[Callable[[Exception], bool]] = None
) -> Callable:
    """
    Decorator for graceful degradation to fallback behavior.
    
    Args:
        fallback_func: Function to call when primary function fails
        condition: Optional condition to check if fallback should be used
        
    Returns:
        Decorated function with fallback behavior
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if condition is None or condition(e):
                    logger.warning(
                        f"Primary function {func.__name__} failed, using fallback: {e}"
                    )
                    return fallback_func(*args, **kwargs)
                else:
                    raise
        
        return wrapper
    return decorator


class ResourceManager:
    """Manages resources with automatic cleanup."""
    
    def __init__(self) -> None:
        self.resources: Dict[str, Any] = {}
        self.cleanup_funcs: Dict[str, Callable[[], None]] = {}
    
    def register_resource(
        self,
        name: str,
        resource: Any,
        cleanup_func: Optional[Callable[[], None]] = None
    ) -> None:
        """Register a resource with optional cleanup function."""
        self.resources[name] = resource
        if cleanup_func:
            self.cleanup_funcs[name] = cleanup_func
        logger.debug(f"Registered resource: {name}")
    
    def get_resource(self, name: str) -> Any:
        """Get a registered resource."""
        if name not in self.resources:
            raise KeyError(f"Resource '{name}' not found")
        return self.resources[name]
    
    def cleanup_resource(self, name: str) -> None:
        """Clean up a specific resource."""
        if name in self.cleanup_funcs:
            try:
                logger.info(f"Cleaning up resource: {name}")
                self.cleanup_funcs[name]()
            except Exception as e:
                logger.error(f"Error cleaning up resource '{name}': {e}")
        
        if name in self.resources:
            del self.resources[name]
        if name in self.cleanup_funcs:
            del self.cleanup_funcs[name]
    
    def cleanup_all(self) -> None:
        """Clean up all registered resources."""
        for name in list(self.resources.keys()):
            self.cleanup_resource(name)
    
    def __enter__(self) -> 'ResourceManager':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup_all()


def handle_gpu_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle GPU-specific errors.
    
    Automatically falls back to CPU on GPU errors.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "CUDA" in str(e) or "out of memory" in str(e):
                logger.warning(f"GPU error in {func.__name__}: {e}. Falling back to CPU.")
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Modify kwargs to use CPU
                if 'device' in kwargs:
                    kwargs['device'] = 'cpu'
                
                # Retry on CPU
                return func(*args, **kwargs)
            else:
                raise
    
    return wrapper


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                    logger.info(f"Circuit breaker for {func.__name__} entering half-open state")
                else:
                    raise RuntimeError(
                        f"Circuit breaker is open for {func.__name__}. "
                        f"Retry after {self.recovery_timeout}s"
                    )
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failure count
                if self.state == "half-open":
                    self.state = "closed"
                    logger.info(f"Circuit breaker for {func.__name__} is now closed")
                self.failure_count = 0
                
                return result
                
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error(
                        f"Circuit breaker for {func.__name__} is now open after "
                        f"{self.failure_count} failures"
                    )
                
                raise
        
        return wrapper