"""
Error handling middleware for CleoAI API.
Catches exceptions and returns sanitized error responses.
"""

import time
import logging
from typing import Callable, Optional, Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
import traceback

from src.utils.error_sanitizer import error_sanitizer, ErrorSeverity
from src.monitoring.metrics import metrics_collector
from src.utils.error_handling import (
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    ResourceNotFoundError,
    RateLimitError,
    DatabaseConnectionError
)


logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware to handle exceptions and return sanitized error responses."""
    
    # HTTP status code mappings for different error types
    ERROR_STATUS_CODES = {
        # Client errors (4xx)
        ValidationError: 400,
        AuthenticationError: 401,
        AuthorizationError: 403,
        ResourceNotFoundError: 404,
        RateLimitError: 429,
        
        # Server errors (5xx)
        DatabaseConnectionError: 503,
        MemoryError: 507,
        TimeoutError: 504,
        
        # Default
        Exception: 500
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        """Process request and handle any exceptions."""
        start_time = time.time()
        
        # Extract request context for error tracking
        context = self._extract_request_context(request)
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Track successful requests
            self._track_request(request, response.status_code, time.time() - start_time)
            
            return response
            
        except HTTPException as exc:
            # Handle Starlette HTTP exceptions
            return await self._handle_http_exception(exc, context)
            
        except Exception as exc:
            # Handle all other exceptions
            return await self._handle_exception(exc, context, request)
    
    def _extract_request_context(self, request: Request) -> Dict[str, Any]:
        """Extract context information from request."""
        context = {
            'method': request.method,
            'path': request.url.path,
            'client_host': request.client.host if request.client else None,
            'user_agent': request.headers.get('user-agent', 'Unknown')
        }
        
        # Add user context if authenticated
        if hasattr(request.state, 'user'):
            context['user_id'] = getattr(request.state.user, 'id', None)
            context['user_roles'] = getattr(request.state.user, 'roles', [])
        
        return context
    
    async def _handle_http_exception(
        self,
        exc: HTTPException,
        context: Dict[str, Any]
    ) -> JSONResponse:
        """Handle Starlette HTTP exceptions."""
        # Create sanitized error response
        error_response = error_sanitizer.create_error_response(
            exc,
            status_code=exc.status_code,
            context=context
        )
        
        # Track the error
        self._track_error(exc, exc.status_code, context)
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response,
            headers=exc.headers if hasattr(exc, 'headers') else None
        )
    
    async def _handle_exception(
        self,
        exc: Exception,
        context: Dict[str, Any],
        request: Request
    ) -> JSONResponse:
        """Handle general exceptions."""
        # Determine status code
        status_code = self._get_status_code(exc)
        
        # Create sanitized error response
        error_response = error_sanitizer.create_error_response(
            exc,
            status_code=status_code,
            context=context
        )
        
        # Log the full error for debugging
        if status_code >= 500:
            logger.error(
                f"Internal server error on {request.method} {request.url.path}",
                exc_info=exc,
                extra={
                    'error_id': error_response['error']['error_id'],
                    'context': context
                }
            )
        else:
            logger.warning(
                f"Client error on {request.method} {request.url.path}: {type(exc).__name__}",
                extra={
                    'error_id': error_response['error']['error_id'],
                    'context': context
                }
            )
        
        # Track the error
        self._track_error(exc, status_code, context)
        
        # Add security headers
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'Cache-Control': 'no-store'
        }
        
        return JSONResponse(
            status_code=status_code,
            content=error_response,
            headers=headers
        )
    
    def _get_status_code(self, exc: Exception) -> int:
        """Get appropriate HTTP status code for exception."""
        # Check exact type match first
        exc_type = type(exc)
        if exc_type in self.ERROR_STATUS_CODES:
            return self.ERROR_STATUS_CODES[exc_type]
        
        # Check inheritance
        for error_type, code in self.ERROR_STATUS_CODES.items():
            if isinstance(exc, error_type):
                return code
        
        # Default to 500
        return 500
    
    def _track_request(self, request: Request, status_code: int, duration: float):
        """Track request metrics."""
        metrics_collector.histogram(
            'http_request_duration_seconds',
            duration,
            tags={
                'method': request.method,
                'path': request.url.path,
                'status': str(status_code)
            }
        )
        
        metrics_collector.increment(
            'http_requests_total',
            tags={
                'method': request.method,
                'path': request.url.path,
                'status': str(status_code)
            }
        )
    
    def _track_error(self, exc: Exception, status_code: int, context: Dict[str, Any]):
        """Track error metrics."""
        metrics_collector.increment(
            'errors_total',
            tags={
                'error_type': type(exc).__name__,
                'status_code': str(status_code),
                'path': context.get('path', 'unknown')
            }
        )
        
        # Track specific error types
        if isinstance(exc, ValidationError):
            metrics_collector.increment('validation_errors_total')
        elif isinstance(exc, (AuthenticationError, AuthorizationError)):
            metrics_collector.increment('auth_errors_total')
        elif isinstance(exc, RateLimitError):
            metrics_collector.increment('rate_limit_errors_total')
        elif isinstance(exc, DatabaseConnectionError):
            metrics_collector.increment('database_errors_total')


class GraphQLErrorFormatter:
    """Format GraphQL errors with sanitization."""
    
    @staticmethod
    def format_error(error: Exception, debug: bool = False) -> Dict[str, Any]:
        """Format GraphQL error for response."""
        # Sanitize the error
        sanitized = error_sanitizer.sanitize_error(error)
        
        # Build GraphQL error format
        formatted = {
            'message': sanitized.user_message,
            'extensions': {
                'code': sanitized.error_code,
                'error_id': sanitized.error_id,
                'timestamp': sanitized.timestamp.isoformat()
            }
        }
        
        # Add path if available
        if hasattr(error, 'path'):
            formatted['path'] = error.path
        
        # Add locations if available
        if hasattr(error, 'locations'):
            formatted['locations'] = error.locations
        
        # Add debug info if enabled
        if debug and sanitized.debug_info:
            formatted['extensions']['debug'] = sanitized.debug_info
        
        return formatted


def setup_error_handlers(app):
    """Setup error handlers for the application."""
    
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        """Handle validation errors."""
        error_response = error_sanitizer.create_error_response(
            exc,
            status_code=400,
            context={'path': request.url.path}
        )
        
        return JSONResponse(
            status_code=400,
            content=error_response
        )
    
    @app.exception_handler(AuthenticationError)
    async def auth_error_handler(request: Request, exc: AuthenticationError):
        """Handle authentication errors."""
        error_response = error_sanitizer.create_error_response(
            exc,
            status_code=401,
            context={'path': request.url.path}
        )
        
        return JSONResponse(
            status_code=401,
            content=error_response,
            headers={'WWW-Authenticate': 'Bearer'}
        )
    
    @app.exception_handler(AuthorizationError)
    async def authz_error_handler(request: Request, exc: AuthorizationError):
        """Handle authorization errors."""
        error_response = error_sanitizer.create_error_response(
            exc,
            status_code=403,
            context={'path': request.url.path}
        )
        
        return JSONResponse(
            status_code=403,
            content=error_response
        )
    
    @app.exception_handler(ResourceNotFoundError)
    async def not_found_handler(request: Request, exc: ResourceNotFoundError):
        """Handle resource not found errors."""
        error_response = error_sanitizer.create_error_response(
            exc,
            status_code=404,
            context={'path': request.url.path}
        )
        
        return JSONResponse(
            status_code=404,
            content=error_response
        )
    
    @app.exception_handler(RateLimitError)
    async def rate_limit_handler(request: Request, exc: RateLimitError):
        """Handle rate limit errors."""
        error_response = error_sanitizer.create_error_response(
            exc,
            status_code=429,
            context={'path': request.url.path}
        )
        
        # Add rate limit headers
        headers = {
            'X-RateLimit-Limit': str(exc.limit) if hasattr(exc, 'limit') else '100',
            'X-RateLimit-Remaining': '0',
            'X-RateLimit-Reset': str(exc.reset_time) if hasattr(exc, 'reset_time') else ''
        }
        
        return JSONResponse(
            status_code=429,
            content=error_response,
            headers=headers
        )
    
    @app.exception_handler(500)
    async def internal_server_error_handler(request: Request, exc: Exception):
        """Handle internal server errors."""
        error_response = error_sanitizer.create_error_response(
            exc,
            status_code=500,
            context={'path': request.url.path}
        )
        
        # Log the full error
        logger.error(
            f"Internal server error on {request.url.path}",
            exc_info=exc,
            extra={'error_id': error_response['error']['error_id']}
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )