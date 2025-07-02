"""
API router for CleoAI FastAPI application with authentication.
"""
import os
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from .graphql import create_graphql_router
from .auth import (
    security_headers_middleware,
    get_current_user,
    require_user,
    TokenData
)
from .security_middleware import SecurityContextMiddleware
from ..utils.secrets_manager import get_secrets_manager

logger = logging.getLogger(__name__)

# Get configuration
secrets_manager = get_secrets_manager()
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting CleoAI API")
    yield
    # Shutdown
    logger.info("Shutting down CleoAI API")


def create_api_app(
    inference_engine=None,
    memory_adapter=None,
    cors_origins: Optional[List[str]] = None,
    debug: bool = False
) -> FastAPI:
    """
    Create FastAPI application with GraphQL endpoint and authentication.
    
    Args:
        inference_engine: Inference engine instance
        memory_adapter: Memory adapter instance
        cors_origins: List of allowed CORS origins
        debug: Enable debug mode
        
    Returns:
        FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title="CleoAI API",
        description="Secure GraphQL API for CleoAI Autonomous Agent",
        version="2.0.0",
        debug=debug,
        lifespan=lifespan,
        docs_url="/docs" if ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if ENVIRONMENT != "production" else None
    )
    
    # Configure CORS with proper origins
    if cors_origins is None:
        cors_origins = ALLOWED_ORIGINS
    
    # Security: Restrict CORS in production
    if ENVIRONMENT == "production" and "*" in cors_origins:
        logger.warning("CORS configured with wildcard in production - this is insecure!")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        max_age=86400  # 24 hours
    )
    
    # Add security headers middleware
    app.middleware("http")(security_headers_middleware)
    
    # Add security context middleware
    app.add_middleware(SecurityContextMiddleware)
    
    # Add trusted host middleware in production
    if ENVIRONMENT == "production":
        allowed_hosts = os.getenv("ALLOWED_HOSTS", "").split(",")
        if allowed_hosts:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=allowed_hosts
            )
    
    # Create and mount GraphQL router
    graphql_router = create_graphql_router()
    app.include_router(graphql_router, prefix="/graphql")
    
    # Health check endpoint (public)
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        from ..utils.health_check import perform_health_check
        from config import memory_backend_config
        
        try:
            # Run health check
            health_result = await perform_health_check(memory_backend_config)
            
            # Add API info
            health_result["service"] = "CleoAI API"
            health_result["version"] = "2.0.0"
            health_result["environment"] = ENVIRONMENT
            
            # Set HTTP status based on health
            if health_result["status"] == "unhealthy":
                raise HTTPException(status_code=503, detail=health_result)
            
            return health_result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "error": "Health check failed",
                    "service": "CleoAI API"
                }
            )
    
    # Detailed health check endpoint (requires authentication)
    @app.get("/health/detailed")
    async def health_check_detailed(
        current_user: TokenData = Depends(require_user)
    ):
        """Detailed health check with full service information."""
        from ..utils.health_check import perform_health_check
        from config import memory_backend_config
        
        try:
            # Run comprehensive health check
            health_result = await perform_health_check(memory_backend_config)
            
            # Add system information
            import platform
            import psutil
            
            health_result["system"] = {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": {
                    "percent": psutil.disk_usage('/').percent,
                    "free_gb": psutil.disk_usage('/').free / (1024**3)
                }
            }
            
            # Add configuration info (sanitized)
            health_result["configuration"] = {
                "backends_enabled": {
                    "redis": memory_backend_config.use_redis,
                    "mongodb": memory_backend_config.use_mongodb,
                    "supabase": memory_backend_config.use_supabase,
                    "pinecone": memory_backend_config.use_pinecone,
                    "sqlite": memory_backend_config.use_sqlite,
                    "chromadb": memory_backend_config.use_chromadb
                },
                "auth_enabled": True,
                "rate_limiting_enabled": bool(os.getenv("RATE_LIMIT_ENABLED", "true"))
            }
            
            # Add user context
            health_result["request_user"] = {
                "user_id": current_user.user_id,
                "roles": current_user.roles
            }
            
            return health_result
            
        except Exception as e:
            logger.error(f"Detailed health check failed: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "error": "Health check failed",
                    "service": "CleoAI API"
                }
            )
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "CleoAI API",
            "version": "2.0.0",
            "environment": ENVIRONMENT,
            "graphql_endpoint": "/graphql",
            "graphql_playground": "/graphql" if ENVIRONMENT != "production" else None,
            "documentation": {
                "type": "GraphQL Schema",
                "introspection": ENVIRONMENT != "production",
                "openapi": "/docs" if ENVIRONMENT != "production" else None
            },
            "authentication": {
                "type": "JWT Bearer Token",
                "header": "Authorization: Bearer <token>",
                "alternative": "X-API-Key header"
            }
        }
    
    # Authentication test endpoint
    @app.get("/auth/test")
    async def auth_test(
        current_user: Optional[TokenData] = Depends(get_current_user)
    ):
        """Test authentication status."""
        if current_user:
            return {
                "authenticated": True,
                "user_id": current_user.user_id,
                "roles": current_user.roles
            }
        else:
            return {
                "authenticated": False,
                "message": "No valid authentication provided"
            }
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        # Don't expose internal errors in production
        if ENVIRONMENT == "production":
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred"
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "error": type(exc).__name__,
                    "message": str(exc)
                }
            )
    
    logger.info(f"Created secure API app with debug={debug}, environment={ENVIRONMENT}")
    
    return app