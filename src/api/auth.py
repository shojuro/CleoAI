"""
Authentication and authorization module for CleoAI API.

This module provides JWT-based authentication, API key validation,
role-based access control (RBAC), and rate limiting.
"""
import os
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone
from functools import wraps
import secrets
import hashlib

import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from strawberry.permission import BasePermission
from strawberry.types import Info
import redis

from src.utils.secrets_manager import get_secrets_manager
from src.utils.error_handling import AuthenticationError, AuthorizationError

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Get secrets
secrets_manager = get_secrets_manager()
JWT_SECRET_KEY = secrets_manager.get_secret("JWT_SECRET_KEY") or "fallback-dev-secret"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Redis for rate limiting
redis_client = None
try:
    redis_host = secrets_manager.get_secret("REDIS_HOST") or "localhost"
    redis_port = int(secrets_manager.get_secret("REDIS_PORT") or 6379)
    redis_password = secrets_manager.get_secret("REDIS_PASSWORD")
    
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Connected to Redis for rate limiting")
except Exception as e:
    logger.warning(f"Redis not available for rate limiting: {e}")


class UserRole:
    """User roles for RBAC."""
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"
    READONLY = "readonly"


class TokenData:
    """Token payload data."""
    def __init__(self, 
                 user_id: str,
                 roles: List[str],
                 exp: Optional[datetime] = None,
                 iat: Optional[datetime] = None,
                 jti: Optional[str] = None):
        self.user_id = user_id
        self.roles = roles
        self.exp = exp or (datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS))
        self.iat = iat or datetime.now(timezone.utc)
        self.jti = jti or secrets.token_urlsafe(16)


def create_access_token(user_id: str, roles: List[str] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        user_id: User identifier
        roles: List of user roles
        
    Returns:
        JWT token string
    """
    if roles is None:
        roles = [UserRole.USER]
    
    token_data = TokenData(user_id=user_id, roles=roles)
    
    payload = {
        "sub": user_id,
        "roles": roles,
        "exp": token_data.exp,
        "iat": token_data.iat,
        "jti": token_data.jti
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    # Store token in Redis for revocation support
    if redis_client:
        try:
            redis_client.setex(
                f"token:{token_data.jti}",
                int(JWT_EXPIRATION_HOURS * 3600),
                user_id
            )
        except Exception as e:
            logger.error(f"Failed to store token in Redis: {e}")
    
    return token


def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        TokenData object
        
    Raises:
        AuthenticationError: If token is invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Check if token is revoked
        jti = payload.get("jti")
        if jti and redis_client:
            try:
                if not redis_client.exists(f"token:{jti}"):
                    raise AuthenticationError("Token has been revoked")
            except redis.RedisError:
                pass  # Continue if Redis is down
        
        return TokenData(
            user_id=payload["sub"],
            roles=payload.get("roles", [UserRole.USER]),
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            jti=jti
        )
        
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {e}")


def revoke_token(token: str) -> bool:
    """
    Revoke a JWT token.
    
    Args:
        token: JWT token to revoke
        
    Returns:
        True if revoked successfully
    """
    try:
        token_data = decode_token(token)
        if token_data.jti and redis_client:
            redis_client.delete(f"token:{token_data.jti}")
            return True
    except Exception as e:
        logger.error(f"Failed to revoke token: {e}")
    return False


def hash_password(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def validate_api_key(api_key: str) -> Optional[str]:
    """
    Validate an API key.
    
    Args:
        api_key: API key to validate
        
    Returns:
        Service name if valid, None otherwise
    """
    # In production, store API keys in database with hashed values
    # For now, check against environment variable
    valid_api_key = secrets_manager.get_secret("API_KEY")
    
    if valid_api_key and secrets.compare_digest(api_key, valid_api_key):
        return "default-service"
    
    return None


class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, 
                 requests_per_minute: int = 60,
                 requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
    
    def check_rate_limit(self, identifier: str) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: Unique identifier (user_id, IP, etc.)
            
        Returns:
            Tuple of (allowed, limits_info)
        """
        if not redis_client:
            return True, {}
        
        try:
            now = int(time.time())
            minute_key = f"rate:{identifier}:minute:{now // 60}"
            hour_key = f"rate:{identifier}:hour:{now // 3600}"
            
            # Check minute limit
            minute_count = redis_client.incr(minute_key)
            if minute_count == 1:
                redis_client.expire(minute_key, 60)
            
            # Check hour limit
            hour_count = redis_client.incr(hour_key)
            if hour_count == 1:
                redis_client.expire(hour_key, 3600)
            
            if minute_count > self.requests_per_minute:
                return False, {
                    "limit": self.requests_per_minute,
                    "window": "minute",
                    "retry_after": 60 - (now % 60)
                }
            
            if hour_count > self.requests_per_hour:
                return False, {
                    "limit": self.requests_per_hour,
                    "window": "hour",
                    "retry_after": 3600 - (now % 3600)
                }
            
            return True, {
                "minute_remaining": self.requests_per_minute - minute_count,
                "hour_remaining": self.requests_per_hour - hour_count
            }
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return True, {}  # Allow request if rate limiting fails


# FastAPI dependencies
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header)
) -> Optional[TokenData]:
    """
    Get current authenticated user from JWT or API key.
    
    Returns:
        TokenData if authenticated, None otherwise
    """
    # Try JWT first
    if credentials and credentials.credentials:
        try:
            return decode_token(credentials.credentials)
        except AuthenticationError:
            pass
    
    # Try API key
    if api_key:
        service = validate_api_key(api_key)
        if service:
            return TokenData(
                user_id=f"service:{service}",
                roles=[UserRole.SERVICE]
            )
    
    return None


async def require_user(
    current_user: Optional[TokenData] = Depends(get_current_user)
) -> TokenData:
    """Require authenticated user."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


async def require_roles(
    allowed_roles: List[str],
    current_user: TokenData = Depends(require_user)
) -> TokenData:
    """Require specific roles."""
    if not any(role in current_user.roles for role in allowed_roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user


# Strawberry GraphQL permissions
class IsAuthenticated(BasePermission):
    """Permission class requiring authentication."""
    
    message = "Authentication required"
    
    def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        request = info.context["request"]
        
        # Check for user in request context
        if hasattr(request, "user") and request.user:
            return True
        
        # Try to authenticate from headers
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                token = auth_header[7:]
                token_data = decode_token(token)
                request.user = token_data
                return True
            except AuthenticationError:
                pass
        
        # Try API key
        api_key = request.headers.get("X-API-Key")
        if api_key and validate_api_key(api_key):
            request.user = TokenData(
                user_id=f"service:{api_key[:8]}",
                roles=[UserRole.SERVICE]
            )
            return True
        
        return False


class HasRole(BasePermission):
    """Permission class requiring specific roles."""
    
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles
        self.message = f"Requires one of roles: {', '.join(allowed_roles)}"
    
    def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        # First check authentication
        if not IsAuthenticated().has_permission(source, info, **kwargs):
            return False
        
        request = info.context["request"]
        user = getattr(request, "user", None)
        
        if not user:
            return False
        
        return any(role in user.roles for role in self.allowed_roles)


class RateLimited(BasePermission):
    """Permission class for rate limiting."""
    
    def __init__(self, 
                 requests_per_minute: int = 60,
                 requests_per_hour: int = 1000):
        self.rate_limiter = RateLimiter(requests_per_minute, requests_per_hour)
        self.message = "Rate limit exceeded"
    
    def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        request = info.context["request"]
        
        # Get identifier (user ID or IP)
        identifier = None
        if hasattr(request, "user") and request.user:
            identifier = request.user.user_id
        else:
            identifier = request.client.host
        
        allowed, limits = self.rate_limiter.check_rate_limit(identifier)
        
        if not allowed:
            self.message = f"Rate limit exceeded: {limits['limit']} per {limits['window']}"
            # Add rate limit headers to response
            if hasattr(request, "state"):
                request.state.rate_limit_info = limits
        
        return allowed


# Middleware for adding security headers
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Rate limit headers
    if hasattr(request.state, "rate_limit_info"):
        info = request.state.rate_limit_info
        if "retry_after" in info:
            response.headers["Retry-After"] = str(info["retry_after"])
        if "minute_remaining" in info:
            response.headers["X-RateLimit-Remaining-Minute"] = str(info["minute_remaining"])
            response.headers["X-RateLimit-Remaining-Hour"] = str(info["hour_remaining"])
    
    return response