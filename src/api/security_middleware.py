"""
Security middleware for FastAPI to inject security context.

This middleware creates a security context for each request based on
the authenticated user information.
"""
import logging
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ..security.data_isolation import SecurityContext, set_security_context, IsolationLevel
from .auth import get_current_user, TokenData

logger = logging.getLogger(__name__)


class SecurityContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to create and inject security context for each request.
    
    This ensures that all operations have access to the current user's
    security context for data isolation and access control.
    """
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with security context."""
        # Try to get authenticated user
        user_data: Optional[TokenData] = None
        
        # Check if we have auth headers
        auth_header = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key")
        
        if auth_header or api_key:
            try:
                # This would normally use the FastAPI dependency injection
                # For middleware, we need to manually extract
                from .auth import decode_token, validate_api_key
                
                if auth_header and auth_header.startswith("Bearer "):
                    token = auth_header[7:]
                    user_data = decode_token(token)
                elif api_key:
                    service = validate_api_key(api_key)
                    if service:
                        user_data = TokenData(
                            user_id=f"service:{service}",
                            roles=["service"]
                        )
            except Exception as e:
                logger.debug(f"Failed to extract user from request: {e}")
        
        # Create security context
        if user_data:
            context = SecurityContext(
                user_id=user_data.user_id,
                roles=user_data.roles,
                isolation_level=IsolationLevel.USER,
                metadata={
                    'ip_address': request.client.host if request.client else None,
                    'user_agent': request.headers.get('User-Agent'),
                    'request_path': str(request.url.path)
                }
            )
            
            # Set context for this request
            set_security_context(context)
            
            # Add to request state for easy access
            request.state.security_context = context
            request.state.user = user_data
            
            logger.debug(f"Security context created for user {user_data.user_id}")
        else:
            # Anonymous request
            context = SecurityContext(
                user_id="anonymous",
                roles=["anonymous"],
                isolation_level=IsolationLevel.STRICT,
                metadata={
                    'ip_address': request.client.host if request.client else None,
                    'user_agent': request.headers.get('User-Agent'),
                    'request_path': str(request.url.path)
                }
            )
            set_security_context(context)
            request.state.security_context = context
            
            logger.debug("Anonymous security context created")
        
        # Process request
        response = await call_next(request)
        
        # Clear context after request
        set_security_context(None)
        
        return response


def get_request_security_context(request: Request) -> Optional[SecurityContext]:
    """
    Get security context from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Security context if available
    """
    return getattr(request.state, 'security_context', None)