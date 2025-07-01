"""
API router for CleoAI FastAPI application.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ariadne.asgi import GraphQL
from typing import Optional
import logging

from .graphql_schema import schema, initialize_api

logger = logging.getLogger(__name__)


def create_api_app(
    inference_engine=None,
    memory_adapter=None,
    cors_origins: Optional[list] = None,
    debug: bool = False
) -> FastAPI:
    """
    Create FastAPI application with GraphQL endpoint.
    
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
        description="GraphQL API for CleoAI Autonomous Agent",
        version="1.0.0",
        debug=debug
    )
    
    # Configure CORS
    if cors_origins is None:
        cors_origins = ["*"]  # Allow all origins in development
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize GraphQL with services
    if inference_engine and memory_adapter:
        initialize_api(inference_engine, memory_adapter)
    
    # Create GraphQL app
    graphql_app = GraphQL(schema, debug=debug)
    
    # Mount GraphQL endpoint
    app.mount("/graphql", graphql_app)
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Comprehensive health check endpoint."""
        from ..utils.health_check import perform_health_check
        from config import memory_backend_config
        
        try:
            # Run comprehensive health check
            health_result = await perform_health_check(memory_backend_config)
            
            # Add API-specific info
            health_result["service"] = "CleoAI API"
            health_result["endpoints"] = {
                "graphql": "/graphql",
                "health": "/health",
                "health_detailed": "/health/detailed"
            }
            
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
                    "error": str(e),
                    "service": "CleoAI API"
                }
            )
    
    # Detailed health check endpoint
    @app.get("/health/detailed")
    async def health_check_detailed():
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
                }
            }
            
            return health_result
            
        except Exception as e:
            logger.error(f"Detailed health check failed: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "error": str(e),
                    "service": "CleoAI API"
                }
            )
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "CleoAI API",
            "version": "1.0.0",
            "graphql_endpoint": "/graphql",
            "graphql_playground": "/graphql" if debug else None,
            "documentation": {
                "type": "GraphQL Schema",
                "introspection": debug
            }
        }
    
    logger.info(f"Created API app with debug={debug}")
    
    return app
