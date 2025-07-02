"""
Prometheus metrics definitions and collectors for CleoAI.

This module defines all application metrics including:
- API request metrics (latency, errors, throughput)
- Model inference metrics (time, token count, queue size)
- Memory operation metrics (queries, updates, size)
- System resource metrics (CPU, memory, disk, GPU)
"""
import time
import psutil
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager

from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST
)

logger = logging.getLogger(__name__)

# Create a custom registry to avoid conflicts
REGISTRY = CollectorRegistry()

# API Metrics
api_requests_total = Counter(
    'cleoai_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code'],
    registry=REGISTRY
)

api_request_duration_seconds = Histogram(
    'cleoai_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY
)

api_request_size_bytes = Summary(
    'cleoai_api_request_size_bytes',
    'Size of API requests in bytes',
    ['method', 'endpoint'],
    registry=REGISTRY
)

api_response_size_bytes = Summary(
    'cleoai_api_response_size_bytes',
    'Size of API responses in bytes',
    ['method', 'endpoint'],
    registry=REGISTRY
)

# Model Metrics
model_inference_requests_total = Counter(
    'cleoai_model_inference_requests_total',
    'Total number of model inference requests',
    ['model_name', 'model_version', 'status'],
    registry=REGISTRY
)

model_inference_duration_seconds = Histogram(
    'cleoai_model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_name', 'model_version'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY
)

model_tokens_processed_total = Counter(
    'cleoai_model_tokens_processed_total',
    'Total number of tokens processed',
    ['model_name', 'model_version', 'token_type'],
    registry=REGISTRY
)

model_queue_size = Gauge(
    'cleoai_model_queue_size',
    'Current size of the model inference queue',
    ['model_name'],
    registry=REGISTRY
)

# Memory Metrics
memory_operations_total = Counter(
    'cleoai_memory_operations_total',
    'Total number of memory operations',
    ['operation_type', 'backend', 'status'],
    registry=REGISTRY
)

memory_operation_duration_seconds = Histogram(
    'cleoai_memory_operation_duration_seconds',
    'Memory operation duration in seconds',
    ['operation_type', 'backend'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY
)

memory_store_size_bytes = Gauge(
    'cleoai_memory_store_size_bytes',
    'Current size of memory store in bytes',
    ['backend', 'data_type'],
    registry=REGISTRY
)

memory_cache_hits_total = Counter(
    'cleoai_memory_cache_hits_total',
    'Total number of memory cache hits',
    ['cache_type'],
    registry=REGISTRY
)

memory_cache_misses_total = Counter(
    'cleoai_memory_cache_misses_total',
    'Total number of memory cache misses',
    ['cache_type'],
    registry=REGISTRY
)

# System Metrics
system_cpu_usage_percent = Gauge(
    'cleoai_system_cpu_usage_percent',
    'CPU usage percentage',
    registry=REGISTRY
)

system_memory_usage_bytes = Gauge(
    'cleoai_system_memory_usage_bytes',
    'Memory usage in bytes',
    ['memory_type'],
    registry=REGISTRY
)

system_disk_usage_bytes = Gauge(
    'cleoai_system_disk_usage_bytes',
    'Disk usage in bytes',
    ['mount_point', 'usage_type'],
    registry=REGISTRY
)

system_network_io_bytes = Counter(
    'cleoai_system_network_io_bytes',
    'Network I/O in bytes',
    ['direction', 'interface'],
    registry=REGISTRY
)

# GPU Metrics (if available)
gpu_usage_percent = Gauge(
    'cleoai_gpu_usage_percent',
    'GPU usage percentage',
    ['gpu_id', 'metric_type'],
    registry=REGISTRY
)

gpu_memory_usage_bytes = Gauge(
    'cleoai_gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id', 'memory_type'],
    registry=REGISTRY
)

# Business Metrics
conversations_active = Gauge(
    'cleoai_conversations_active',
    'Number of active conversations',
    registry=REGISTRY
)

users_active = Gauge(
    'cleoai_users_active',
    'Number of active users',
    ['time_window'],
    registry=REGISTRY
)

messages_processed_total = Counter(
    'cleoai_messages_processed_total',
    'Total number of messages processed',
    ['message_type', 'status'],
    registry=REGISTRY
)

# Error Metrics
errors_total = Counter(
    'cleoai_errors_total',
    'Total number of errors',
    ['error_type', 'component', 'severity'],
    registry=REGISTRY
)


class MetricsCollector:
    """Collector for system and application metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.enabled = True
        self._gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU metrics are available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except Exception:
            logger.info("GPU metrics not available")
            return False
    
    def collect_system_metrics(self):
        """Collect system resource metrics."""
        if not self.enabled:
            return
            
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage_percent.set(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            system_memory_usage_bytes.labels(memory_type='used').set(memory.used)
            system_memory_usage_bytes.labels(memory_type='available').set(memory.available)
            system_memory_usage_bytes.labels(memory_type='total').set(memory.total)
            
            # Disk metrics
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    system_disk_usage_bytes.labels(
                        mount_point=partition.mountpoint,
                        usage_type='used'
                    ).set(usage.used)
                    system_disk_usage_bytes.labels(
                        mount_point=partition.mountpoint,
                        usage_type='free'
                    ).set(usage.free)
                    system_disk_usage_bytes.labels(
                        mount_point=partition.mountpoint,
                        usage_type='total'
                    ).set(usage.total)
                except PermissionError:
                    continue
            
            # Network metrics
            net_io = psutil.net_io_counters(pernic=True)
            for interface, counters in net_io.items():
                system_network_io_bytes.labels(
                    direction='sent',
                    interface=interface
                ).inc(counters.bytes_sent)
                system_network_io_bytes.labels(
                    direction='received',
                    interface=interface
                ).inc(counters.bytes_recv)
                
            # GPU metrics if available
            if self._gpu_available:
                self._collect_gpu_metrics()
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            errors_total.labels(
                error_type='metric_collection',
                component='system',
                severity='warning'
            ).inc()
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics using pynvml."""
        try:
            import pynvml
            
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usage_percent.labels(
                    gpu_id=str(i),
                    metric_type='compute'
                ).set(utilization.gpu)
                gpu_usage_percent.labels(
                    gpu_id=str(i),
                    metric_type='memory'
                ).set(utilization.memory)
                
                # GPU memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_usage_bytes.labels(
                    gpu_id=str(i),
                    memory_type='used'
                ).set(mem_info.used)
                gpu_memory_usage_bytes.labels(
                    gpu_id=str(i),
                    memory_type='free'
                ).set(mem_info.free)
                gpu_memory_usage_bytes.labels(
                    gpu_id=str(i),
                    memory_type='total'
                ).set(mem_info.total)
                
        except Exception as e:
            logger.error(f"Failed to collect GPU metrics: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()


# Decorators for metric collection
def track_api_metrics(endpoint: str):
    """Decorator to track API request metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = getattr(e, 'status_code', 500)
                raise
            finally:
                duration = time.time() - start_time
                method = kwargs.get('request', {}).get('method', 'UNKNOWN')
                
                api_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=str(status_code)
                ).inc()
                
                api_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = getattr(e, 'status_code', 500)
                raise
            finally:
                duration = time.time() - start_time
                method = kwargs.get('request', {}).get('method', 'UNKNOWN')
                
                api_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=str(status_code)
                ).inc()
                
                api_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def track_model_inference(model_name: str, model_version: str):
    """Decorator to track model inference metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = await func(*args, **kwargs)
                
                # Track tokens if available
                if hasattr(result, 'input_tokens'):
                    model_tokens_processed_total.labels(
                        model_name=model_name,
                        model_version=model_version,
                        token_type='input'
                    ).inc(result.input_tokens)
                
                if hasattr(result, 'output_tokens'):
                    model_tokens_processed_total.labels(
                        model_name=model_name,
                        model_version=model_version,
                        token_type='output'
                    ).inc(result.output_tokens)
                
                return result
            except Exception as e:
                status = 'error'
                errors_total.labels(
                    error_type='model_inference',
                    component=model_name,
                    severity='error'
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                
                model_inference_requests_total.labels(
                    model_name=model_name,
                    model_version=model_version,
                    status=status
                ).inc()
                
                model_inference_duration_seconds.labels(
                    model_name=model_name,
                    model_version=model_version
                ).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                
                # Track tokens if available
                if hasattr(result, 'input_tokens'):
                    model_tokens_processed_total.labels(
                        model_name=model_name,
                        model_version=model_version,
                        token_type='input'
                    ).inc(result.input_tokens)
                
                if hasattr(result, 'output_tokens'):
                    model_tokens_processed_total.labels(
                        model_name=model_name,
                        model_version=model_version,
                        token_type='output'
                    ).inc(result.output_tokens)
                
                return result
            except Exception as e:
                status = 'error'
                errors_total.labels(
                    error_type='model_inference',
                    component=model_name,
                    severity='error'
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                
                model_inference_requests_total.labels(
                    model_name=model_name,
                    model_version=model_version,
                    status=status
                ).inc()
                
                model_inference_duration_seconds.labels(
                    model_name=model_name,
                    model_version=model_version
                ).observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@contextmanager
def track_memory_operation(operation_type: str, backend: str):
    """Context manager to track memory operation metrics."""
    start_time = time.time()
    status = 'success'
    
    try:
        yield
    except Exception as e:
        status = 'error'
        errors_total.labels(
            error_type='memory_operation',
            component=backend,
            severity='error'
        ).inc()
        raise
    finally:
        duration = time.time() - start_time
        
        memory_operations_total.labels(
            operation_type=operation_type,
            backend=backend,
            status=status
        ).inc()
        
        memory_operation_duration_seconds.labels(
            operation_type=operation_type,
            backend=backend
        ).observe(duration)


def update_business_metrics(active_conversations: int, active_users: Dict[str, int]):
    """Update business-related metrics."""
    conversations_active.set(active_conversations)
    
    for time_window, count in active_users.items():
        users_active.labels(time_window=time_window).set(count)


def track_message(message_type: str, status: str):
    """Track message processing."""
    messages_processed_total.labels(
        message_type=message_type,
        status=status
    ).inc()


def track_error(error_type: str, component: str, severity: str = 'error'):
    """Track application errors."""
    errors_total.labels(
        error_type=error_type,
        component=component,
        severity=severity
    ).inc()


def get_metrics() -> bytes:
    """Generate current metrics in Prometheus format."""
    # Collect latest system metrics
    metrics_collector.collect_system_metrics()
    
    # Generate metrics
    return generate_latest(REGISTRY)


# Fix missing import
import asyncio