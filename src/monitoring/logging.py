"""
Centralized logging configuration for CleoAI with ELK stack integration.

This module provides structured logging with:
- JSON formatting for ELK ingestion
- Log correlation with trace IDs
- Sensitive data masking
- Performance logging
- Audit logging
"""
import os
import sys
import json
import logging
import re
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pythonjsonlogger import jsonlogger
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from .tracing import get_current_trace_id


class SensitiveDataFilter(logging.Filter):
    """Filter to mask sensitive data in logs."""
    
    # Patterns for sensitive data
    PATTERNS = [
        # API keys and tokens
        (r'(api[_-]?key|token|bearer)\s*[:=]\s*["\']?([^"\'\s]+)', r'\1=***REDACTED***'),
        (r'(Authorization|X-API-Key):\s*([^\s]+)', r'\1: ***REDACTED***'),
        
        # Passwords
        (r'(password|passwd|pwd)\s*[:=]\s*["\']?([^"\'\s]+)', r'\1=***REDACTED***'),
        
        # Credit cards
        (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', r'****-****-****-****'),
        
        # SSN
        (r'\b\d{3}-\d{2}-\d{4}\b', r'***-**-****'),
        
        # Email addresses (partial masking)
        (r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', 
         lambda m: f"{m.group(1)[:3]}***@{m.group(2)}"),
        
        # JWT tokens
        (r'eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+', r'***JWT_REDACTED***'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Apply sensitive data masking to log record."""
        # Mask message
        if hasattr(record, 'msg'):
            record.msg = self._mask_sensitive_data(str(record.msg))
        
        # Mask exception info
        if record.exc_info and record.exc_text:
            record.exc_text = self._mask_sensitive_data(record.exc_text)
        
        # Mask extra fields
        for key in record.__dict__:
            if key not in ('msg', 'exc_info', 'exc_text'):
                value = getattr(record, key)
                if isinstance(value, str):
                    setattr(record, key, self._mask_sensitive_data(value))
                elif isinstance(value, dict):
                    setattr(record, key, self._mask_dict(value))
        
        return True
    
    def _mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data in text."""
        for pattern, replacement in self.PATTERNS:
            if callable(replacement):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            else:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def _mask_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask sensitive data in dictionary."""
        masked = {}
        sensitive_keys = {
            'password', 'passwd', 'pwd', 'token', 'api_key', 'apikey',
            'secret', 'private_key', 'access_token', 'refresh_token'
        }
        
        for key, value in data.items():
            if any(sk in key.lower() for sk in sensitive_keys):
                masked[key] = '***REDACTED***'
            elif isinstance(value, str):
                masked[key] = self._mask_sensitive_data(value)
            elif isinstance(value, dict):
                masked[key] = self._mask_dict(value)
            elif isinstance(value, list):
                masked[key] = [
                    self._mask_sensitive_data(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                masked[key] = value
        
        return masked


class TraceContextFilter(logging.Filter):
    """Filter to add trace context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add trace ID to log record."""
        trace_id = get_current_trace_id()
        if trace_id:
            record.trace_id = trace_id
        return True


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # Add service information
        log_record['service'] = {
            'name': os.getenv('SERVICE_NAME', 'cleoai'),
            'version': os.getenv('SERVICE_VERSION', '2.0.0'),
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'instance_id': os.getenv('HOSTNAME', 'local')
        }
        
        # Add level as both name and number
        log_record['level'] = record.levelname
        log_record['level_value'] = record.levelno
        
        # Add source information
        log_record['source'] = {
            'file': record.pathname,
            'line': record.lineno,
            'function': record.funcName,
            'module': record.module
        }
        
        # Add thread/process info
        log_record['thread'] = {
            'id': record.thread,
            'name': record.threadName
        }
        log_record['process'] = {
            'id': record.process,
            'name': record.processName
        }
        
        # Move message to proper field
        if 'message' in log_record:
            log_record['msg'] = log_record.pop('message')


class LogConfig:
    """Logging configuration."""
    
    def __init__(self):
        """Initialize logging configuration from environment."""
        self.level = os.getenv('LOG_LEVEL', 'INFO').upper()
        self.format = os.getenv('LOG_FORMAT', 'json')  # json or text
        self.output = os.getenv('LOG_OUTPUT', 'stdout')  # stdout, file, both
        self.file_path = os.getenv('LOG_FILE_PATH', 'logs/cleoai.log')
        self.max_file_size = int(os.getenv('LOG_MAX_FILE_SIZE', '100'))  # MB
        self.backup_count = int(os.getenv('LOG_BACKUP_COUNT', '10'))
        self.mask_sensitive = os.getenv('LOG_MASK_SENSITIVE', 'true').lower() == 'true'
        self.include_trace = os.getenv('LOG_INCLUDE_TRACE', 'true').lower() == 'true'


def setup_logging(config: Optional[LogConfig] = None) -> None:
    """
    Configure application logging.
    
    Args:
        config: Logging configuration
    """
    if config is None:
        config = LogConfig()
    
    # Create logs directory if needed
    if config.output in ('file', 'both'):
        log_dir = os.path.dirname(config.file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Create formatter
    if config.format == 'json':
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            rename_fields={'levelname': 'level', 'msg': 'message'}
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(trace_id)s - %(message)s' if config.include_trace
            else '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Add filters
    filters = []
    if config.mask_sensitive:
        filters.append(SensitiveDataFilter())
    if config.include_trace:
        filters.append(TraceContextFilter())
    
    # Configure handlers
    handlers = []
    
    # Console handler
    if config.output in ('stdout', 'both'):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        for filter in filters:
            console_handler.addFilter(filter)
        handlers.append(console_handler)
    
    # File handler
    if config.output in ('file', 'both'):
        file_handler = RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size * 1024 * 1024,  # Convert MB to bytes
            backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        for filter in filters:
            file_handler.addFilter(filter)
        handlers.append(file_handler)
    
    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Configure specific loggers
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            'config': {
                'level': config.level,
                'format': config.format,
                'output': config.output,
                'mask_sensitive': config.mask_sensitive,
                'include_trace': config.include_trace
            }
        }
    )


class AuditLogger:
    """Specialized logger for audit events."""
    
    def __init__(self, name: str = 'audit'):
        """Initialize audit logger."""
        self.logger = logging.getLogger(f'cleoai.{name}')
        
        # Ensure audit logs are always recorded
        self.logger.setLevel(logging.INFO)
        
        # Add dedicated audit file handler
        audit_handler = TimedRotatingFileHandler(
            'logs/audit.log',
            when='midnight',
            interval=1,
            backupCount=90  # Keep 90 days of audit logs
        )
        
        # Use JSON format for audit logs
        audit_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        
        # Add sensitive data filter
        audit_handler.addFilter(SensitiveDataFilter())
        
        self.logger.addHandler(audit_handler)
    
    def log_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an audit event.
        
        Args:
            event_type: Type of event (login, access, modify, etc.)
            user_id: ID of user performing action
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            action: Action performed
            result: Result of action (success, failure, etc.)
            metadata: Additional event metadata
        """
        event = {
            'event_type': event_type,
            'user_id': user_id,
            'resource': {
                'type': resource_type,
                'id': resource_id
            },
            'action': action,
            'result': result,
            'metadata': metadata or {}
        }
        
        self.logger.info(
            f"Audit event: {event_type}",
            extra={'audit_event': event}
        )


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self, name: str = 'performance'):
        """Initialize performance logger."""
        self.logger = logging.getLogger(f'cleoai.{name}')
    
    def log_operation(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a performance metric.
        
        Args:
            operation: Name of operation
            duration_ms: Duration in milliseconds
            success: Whether operation succeeded
            metadata: Additional metadata
        """
        self.logger.info(
            f"Performance: {operation}",
            extra={
                'performance': {
                    'operation': operation,
                    'duration_ms': duration_ms,
                    'success': success,
                    'metadata': metadata or {}
                }
            }
        )


# Global instances
audit_logger = AuditLogger()
performance_logger = PerformanceLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with proper configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    return logging.getLogger(f'cleoai.{name}')