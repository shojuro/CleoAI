"""
Prometheus metrics exporter for CleoAI.

This module provides HTTP endpoints for Prometheus to scrape metrics
and includes middleware for automatic metric collection.
"""
import os
import logging
from typing import Optional
from fastapi import FastAPI, Response, Request
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from .metrics import (
    get_metrics, track_api_metrics, api_request_size_bytes,
    api_response_size_bytes, api_requests_total, api_request_duration_seconds,
    metrics_collector, CONTENT_TYPE_LATEST
)

logger = logging.getLogger(__name__)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically collect API metrics for all requests.
    """
    
    def __init__(self, app: FastAPI, excluded_paths: Optional[set] = None):
        """
        Initialize Prometheus middleware.
        
        Args:
            app: FastAPI application
            excluded_paths: Set of paths to exclude from metrics
        """
        super().__init__(app)
        self.excluded_paths = excluded_paths or {'/metrics', '/health', '/'}
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and collect metrics."""
        # Skip metrics for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Start timing
        import time
        start_time = time.time()
        
        # Get request details
        method = request.method
        path = request.url.path
        
        # Track request size
        content_length = request.headers.get('content-length')
        if content_length:
            api_request_size_bytes.labels(
                method=method,
                endpoint=path
            ).observe(int(content_length))
        
        # Process request
        status_code = 500  # Default to error
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Track response size
            if hasattr(response, 'headers') and 'content-length' in response.headers:
                api_response_size_bytes.labels(
                    method=method,
                    endpoint=path
                ).observe(int(response.headers['content-length']))
            
            return response
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
            
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            api_requests_total.labels(
                method=method,
                endpoint=path,
                status_code=str(status_code)
            ).inc()
            
            api_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)


def setup_prometheus_endpoint(app: FastAPI, metrics_path: str = "/metrics"):
    """
    Add Prometheus metrics endpoint to FastAPI app.
    
    Args:
        app: FastAPI application
        metrics_path: Path for metrics endpoint
    """
    @app.get(metrics_path, include_in_schema=False)
    async def metrics():
        """Prometheus metrics endpoint."""
        metrics_data = get_metrics()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST,
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
    
    logger.info(f"Prometheus metrics endpoint configured at {metrics_path}")


def create_metrics_app() -> FastAPI:
    """
    Create a standalone FastAPI app for metrics.
    
    This can be used to run metrics on a separate port from the main API.
    """
    app = FastAPI(
        title="CleoAI Metrics",
        description="Prometheus metrics endpoint for CleoAI",
        docs_url=None,
        redoc_url=None,
        openapi_url=None
    )
    
    # Add metrics endpoint
    setup_prometheus_endpoint(app)
    
    # Health check for the metrics service
    @app.get("/health")
    async def health():
        """Health check for metrics service."""
        return {"status": "healthy", "service": "metrics"}
    
    return app


class MetricsConfig:
    """Configuration for metrics collection."""
    
    def __init__(self):
        """Initialize metrics configuration from environment."""
        self.enabled = os.getenv("METRICS_ENABLED", "true").lower() == "true"
        self.port = int(os.getenv("METRICS_PORT", "9090"))
        self.path = os.getenv("METRICS_PATH", "/metrics")
        self.collect_system_metrics = os.getenv(
            "COLLECT_SYSTEM_METRICS", "true"
        ).lower() == "true"
        self.system_metrics_interval = int(
            os.getenv("SYSTEM_METRICS_INTERVAL", "60")
        )
        
    def apply_to_app(self, app: FastAPI):
        """
        Apply metrics configuration to FastAPI app.
        
        Args:
            app: FastAPI application
        """
        if not self.enabled:
            logger.info("Metrics collection disabled")
            return
        
        # Add Prometheus middleware
        app.add_middleware(PrometheusMiddleware)
        
        # Add metrics endpoint
        setup_prometheus_endpoint(app, self.path)
        
        # Start system metrics collection if enabled
        if self.collect_system_metrics:
            self._start_system_metrics_collection()
    
    def _start_system_metrics_collection(self):
        """Start background task for system metrics collection."""
        import asyncio
        import threading
        
        def collect_loop():
            """Background loop for collecting system metrics."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def collect():
                while True:
                    try:
                        metrics_collector.collect_system_metrics()
                    except Exception as e:
                        logger.error(f"Failed to collect system metrics: {e}")
                    
                    await asyncio.sleep(self.system_metrics_interval)
            
            loop.run_until_complete(collect())
        
        # Start collection in background thread
        thread = threading.Thread(target=collect_loop, daemon=True)
        thread.start()
        
        logger.info(
            f"System metrics collection started "
            f"(interval: {self.system_metrics_interval}s)"
        )