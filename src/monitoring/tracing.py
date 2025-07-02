"""
OpenTelemetry distributed tracing implementation for CleoAI.

This module provides tracing instrumentation for all components including:
- FastAPI/HTTP requests
- GraphQL operations
- Model inference
- Memory operations
- Database queries
"""
import os
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

# Global tracer instance
tracer: Optional[trace.Tracer] = None


class TracingConfig:
    """Configuration for OpenTelemetry tracing."""
    
    def __init__(self):
        """Initialize tracing configuration from environment."""
        self.enabled = os.getenv("TRACING_ENABLED", "true").lower() == "true"
        self.service_name = os.getenv("SERVICE_NAME", "cleoai")
        self.service_version = os.getenv("SERVICE_VERSION", "2.0.0")
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Exporter configuration
        self.exporter_type = os.getenv("TRACE_EXPORTER", "otlp")  # otlp, jaeger, zipkin, console
        self.exporter_endpoint = os.getenv("TRACE_EXPORTER_ENDPOINT", "localhost:4317")
        
        # Sampling configuration
        self.sampling_rate = float(os.getenv("TRACE_SAMPLING_RATE", "1.0"))
        
        # Additional settings
        self.log_correlation = os.getenv("TRACE_LOG_CORRELATION", "true").lower() == "true"
        self.propagate_traces = os.getenv("TRACE_PROPAGATION", "true").lower() == "true"


def initialize_tracing(config: Optional[TracingConfig] = None) -> Optional[trace.Tracer]:
    """
    Initialize OpenTelemetry tracing with configured exporters.
    
    Args:
        config: Tracing configuration
        
    Returns:
        Configured tracer instance
    """
    global tracer
    
    if config is None:
        config = TracingConfig()
    
    if not config.enabled:
        logger.info("Tracing disabled")
        return None
    
    # Create resource with service information
    resource = Resource.create({
        SERVICE_NAME: config.service_name,
        SERVICE_VERSION: config.service_version,
        "service.environment": config.environment,
        "service.instance.id": os.getenv("HOSTNAME", "local"),
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Configure exporter based on type
    exporter = None
    if config.exporter_type == "otlp":
        exporter = OTLPSpanExporter(
            endpoint=config.exporter_endpoint,
            insecure=True  # Use secure=False for development
        )
    elif config.exporter_type == "jaeger":
        # Parse Jaeger endpoint
        if ":" in config.exporter_endpoint:
            host, port = config.exporter_endpoint.split(":")
            exporter = JaegerExporter(
                agent_host_name=host,
                agent_port=int(port),
                collector_endpoint=None  # Use agent, not collector
            )
        else:
            exporter = JaegerExporter(
                agent_host_name=config.exporter_endpoint,
                agent_port=6831  # Default Jaeger agent port
            )
    elif config.exporter_type == "zipkin":
        exporter = ZipkinExporter(
            endpoint=f"http://{config.exporter_endpoint}/api/v2/spans"
        )
    elif config.exporter_type == "console":
        exporter = ConsoleSpanExporter()
    else:
        logger.warning(f"Unknown exporter type: {config.exporter_type}, using console")
        exporter = ConsoleSpanExporter()
    
    # Add span processor
    if config.exporter_type == "console":
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    
    # Set the tracer provider
    trace.set_tracer_provider(provider)
    
    # Get tracer
    tracer = trace.get_tracer(
        config.service_name,
        config.service_version
    )
    
    # Configure propagation
    if config.propagate_traces:
        from opentelemetry.propagate import set_global_textmap
        set_global_textmap(TraceContextTextMapPropagator())
    
    # Enable log correlation
    if config.log_correlation:
        LoggingInstrumentor().instrument()
    
    logger.info(
        f"Tracing initialized with {config.exporter_type} exporter "
        f"(endpoint: {config.exporter_endpoint})"
    )
    
    return tracer


def instrument_app(app: Any, config: Optional[TracingConfig] = None):
    """
    Instrument FastAPI application with OpenTelemetry.
    
    Args:
        app: FastAPI application
        config: Tracing configuration
    """
    if config is None:
        config = TracingConfig()
    
    if not config.enabled:
        return
    
    # Initialize tracing if not already done
    global tracer
    if tracer is None:
        initialize_tracing(config)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    # Instrument HTTP client
    RequestsInstrumentor().instrument()
    
    # Instrument databases
    try:
        RedisInstrumentor().instrument()
    except Exception:
        logger.debug("Redis instrumentation not available")
    
    try:
        PymongoInstrumentor().instrument()
    except Exception:
        logger.debug("MongoDB instrumentation not available")
    
    try:
        Psycopg2Instrumentor().instrument()
    except Exception:
        logger.debug("PostgreSQL instrumentation not available")
    
    logger.info("Application instrumented for tracing")


def trace_operation(
    name: str,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Decorator to trace function execution.
    
    Args:
        name: Span name
        kind: Span kind (INTERNAL, SERVER, CLIENT, etc.)
        attributes: Additional span attributes
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if tracer is None:
                return await func(*args, **kwargs)
            
            with tracer.start_as_current_span(
                name,
                kind=kind,
                attributes=attributes or {}
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(
                        Status(StatusCode.ERROR, str(e))
                    )
                    span.record_exception(e)
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if tracer is None:
                return func(*args, **kwargs)
            
            with tracer.start_as_current_span(
                name,
                kind=kind,
                attributes=attributes or {}
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(
                        Status(StatusCode.ERROR, str(e))
                    )
                    span.record_exception(e)
                    raise
        
        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


@contextmanager
def trace_context(
    name: str,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Context manager for tracing a block of code.
    
    Args:
        name: Span name
        kind: Span kind
        attributes: Additional span attributes
    """
    if tracer is None:
        yield None
        return
    
    with tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=attributes or {}
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(
                Status(StatusCode.ERROR, str(e))
            )
            span.record_exception(e)
            raise


def add_span_attributes(attributes: Dict[str, Any]):
    """
    Add attributes to the current span.
    
    Args:
        attributes: Dictionary of attributes to add
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Add an event to the current span.
    
    Args:
        name: Event name
        attributes: Event attributes
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        span.add_event(name, attributes=attributes or {})


class ModelInferenceTracer:
    """Specialized tracer for model inference operations."""
    
    @staticmethod
    @contextmanager
    def trace_inference(
        model_name: str,
        model_version: str,
        input_tokens: Optional[int] = None
    ):
        """
        Trace model inference operation.
        
        Args:
            model_name: Name of the model
            model_version: Model version
            input_tokens: Number of input tokens
        """
        attributes = {
            "model.name": model_name,
            "model.version": model_version,
            "inference.type": "text_generation",
        }
        
        if input_tokens:
            attributes["model.input_tokens"] = input_tokens
        
        with trace_context(
            f"model_inference/{model_name}",
            kind=trace.SpanKind.INTERNAL,
            attributes=attributes
        ) as span:
            yield span
            
            # Add output metrics if available
            if span and hasattr(span, '_output_tokens'):
                span.set_attribute("model.output_tokens", span._output_tokens)


class MemoryOperationTracer:
    """Specialized tracer for memory operations."""
    
    @staticmethod
    @contextmanager
    def trace_memory_op(
        operation: str,
        backend: str,
        user_id: Optional[str] = None
    ):
        """
        Trace memory operation.
        
        Args:
            operation: Operation type (read, write, search, etc.)
            backend: Memory backend name
            user_id: User ID if applicable
        """
        attributes = {
            "memory.operation": operation,
            "memory.backend": backend,
        }
        
        if user_id:
            attributes["user.id"] = user_id
        
        with trace_context(
            f"memory/{operation}",
            kind=trace.SpanKind.CLIENT,
            attributes=attributes
        ) as span:
            yield span


def create_trace_id() -> str:
    """
    Create a new trace ID for correlation.
    
    Returns:
        Trace ID as string
    """
    span = trace.get_current_span()
    if span:
        context = span.get_span_context()
        if context and context.is_valid:
            return format(context.trace_id, '032x')
    
    # Generate new trace ID if no active span
    import random
    return format(random.getrandbits(128), '032x')


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID if available.
    
    Returns:
        Current trace ID or None
    """
    span = trace.get_current_span()
    if span:
        context = span.get_span_context()
        if context and context.is_valid:
            return format(context.trace_id, '032x')
    return None