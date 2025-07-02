"""
CleoAI Monitoring Package.

This package provides comprehensive monitoring capabilities including:
- Prometheus metrics collection and export
- OpenTelemetry distributed tracing
- Centralized structured logging
- Performance monitoring
- Audit logging
"""

from .metrics import (
    # Metric instances
    api_requests_total,
    api_request_duration_seconds,
    model_inference_requests_total,
    model_inference_duration_seconds,
    memory_operations_total,
    memory_operation_duration_seconds,
    errors_total,
    
    # Decorators and helpers
    track_api_metrics,
    track_model_inference,
    track_memory_operation,
    track_message,
    track_error,
    update_business_metrics,
    get_metrics,
    
    # Collector
    metrics_collector
)

from .prometheus_exporter import (
    PrometheusMiddleware,
    setup_prometheus_endpoint,
    create_metrics_app,
    MetricsConfig
)

from .tracing import (
    TracingConfig,
    initialize_tracing,
    instrument_app,
    trace_operation,
    trace_context,
    add_span_attributes,
    add_span_event,
    create_trace_id,
    get_current_trace_id,
    ModelInferenceTracer,
    MemoryOperationTracer
)

from .logging import (
    LogConfig,
    setup_logging,
    get_logger,
    audit_logger,
    performance_logger,
    SensitiveDataFilter,
    TraceContextFilter
)

from .error_tracking import (
    ErrorTrackingConfig,
    initialize_error_tracking,
    capture_exception,
    capture_message,
    set_user_context,
    add_breadcrumb,
    error_tracking_context,
    track_error,
    error_metrics
)

__all__ = [
    # Metrics
    'api_requests_total',
    'api_request_duration_seconds',
    'model_inference_requests_total',
    'model_inference_duration_seconds',
    'memory_operations_total',
    'memory_operation_duration_seconds',
    'errors_total',
    'track_api_metrics',
    'track_model_inference',
    'track_memory_operation',
    'track_message',
    'track_error',
    'update_business_metrics',
    'get_metrics',
    'metrics_collector',
    
    # Prometheus
    'PrometheusMiddleware',
    'setup_prometheus_endpoint',
    'create_metrics_app',
    'MetricsConfig',
    
    # Tracing
    'TracingConfig',
    'initialize_tracing',
    'instrument_app',
    'trace_operation',
    'trace_context',
    'add_span_attributes',
    'add_span_event',
    'create_trace_id',
    'get_current_trace_id',
    'ModelInferenceTracer',
    'MemoryOperationTracer',
    
    # Logging
    'LogConfig',
    'setup_logging',
    'get_logger',
    'audit_logger',
    'performance_logger',
    'SensitiveDataFilter',
    'TraceContextFilter',
    
    # Error Tracking
    'ErrorTrackingConfig',
    'initialize_error_tracking',
    'capture_exception',
    'capture_message',
    'set_user_context',
    'add_breadcrumb',
    'error_tracking_context',
    'track_error',
    'error_metrics'
]