"""
Minimal API server for testing without ML dependencies.
"""
import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CleoAI API (Minimal)",
    description="Minimal API server for testing without ML dependencies",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CleoAI API (Minimal Mode)",
        "version": "1.0.0",
        "status": "running",
        "mode": "minimal",
        "endpoints": {
            "root": "/",
            "health": "/health",
            "graphql": "/graphql (disabled in minimal mode)"
        },
        "message": "API is running in minimal mode without ML dependencies"
    }

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    import platform
    
    try:
        import psutil
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
    except:
        system_info = {"status": "psutil not available"}
    
    return {
        "status": "healthy",
        "service": "CleoAI API (Minimal)",
        "mode": "minimal",
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "system": system_info,
        "backends": {
            "ml_models": "disabled",
            "memory_backends": "limited functionality"
        }
    }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working."""
    return {
        "message": "API is working!",
        "timestamp": str(Path(__file__).stat().st_mtime),
        "features": {
            "ml_inference": False,
            "memory_storage": False,
            "graphql": False
        }
    }

@app.post("/api/echo")
async def echo_endpoint(data: dict):
    """Echo endpoint for testing POST requests."""
    return {
        "echo": data,
        "message": "Data received successfully"
    }

if __name__ == "__main__":
    # Parse simple arguments
    import sys
    
    host = "0.0.0.0"
    port = 8000
    debug = "--debug" in sys.argv
    
    logger.info(f"Starting minimal API server on {host}:{port}")
    logger.info("This is a minimal mode without ML dependencies")
    logger.info("To use full features, install all requirements with: pip install -r requirements.txt")
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
        reload=debug
    )