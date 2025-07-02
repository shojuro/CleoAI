"""
Error message sanitization for CleoAI.
Prevents information disclosure through error messages while maintaining useful debugging info.
"""

import re
import logging
import traceback
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass
import hashlib
import json
from datetime import datetime

from src.utils.config_validator import get_config


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SanitizedError:
    """Sanitized error information."""
    error_code: str
    user_message: str
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    details: Optional[Dict[str, Any]] = None
    debug_info: Optional[Dict[str, Any]] = None


class ErrorSanitizer:
    """Sanitizes error messages to prevent information disclosure."""
    
    # Patterns that might leak sensitive information
    SENSITIVE_PATTERNS = [
        # File paths
        (r'(/[a-zA-Z0-9_\-./]+)+\.(py|js|sql|conf|yaml|yml|json)', '[FILE_PATH]'),
        (r'[A-Z]:\\[^\\]+\\[^\\]+', '[FILE_PATH]'),
        
        # Database connection strings
        (r'(postgresql|postgres|mysql|mongodb|redis)://[^@]+@[^/]+/\w+', '[DATABASE_URL]'),
        (r'Server=[\w\-\.]+;Database=\w+;', '[CONNECTION_STRING]'),
        
        # IP addresses
        (r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '[IP_ADDRESS]'),
        (r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b', '[IPV6_ADDRESS]'),
        
        # Ports
        (r':(?:6379|5432|3306|27017|11211)\b', ':[PORT]'),
        
        # Credentials
        (r'password[\'"\s]*[:=][\'"\s]*[^\s\'"]+', 'password=[REDACTED]'),
        (r'token[\'"\s]*[:=][\'"\s]*[^\s\'"]+', 'token=[REDACTED]'),
        (r'api[_-]?key[\'"\s]*[:=][\'"\s]*[^\s\'"]+', 'api_key=[REDACTED]'),
        (r'secret[\'"\s]*[:=][\'"\s]*[^\s\'"]+', 'secret=[REDACTED]'),
        
        # AWS specific
        (r'AKIA[0-9A-Z]{16}', '[AWS_ACCESS_KEY]'),
        (r'arn:aws:[^:]+:[^:]+:[^:]+:[^:]+', '[AWS_ARN]'),
        
        # Stack traces with line numbers
        (r'File "[^"]+", line \d+', 'File "[REDACTED]", line [N]'),
        (r'at line \d+ column \d+', 'at line [N] column [N]'),
        
        # SQL queries
        (r'SELECT .+ FROM .+ WHERE', 'SELECT [QUERY]'),
        (r'INSERT INTO .+ VALUES', 'INSERT [QUERY]'),
        (r'UPDATE .+ SET', 'UPDATE [QUERY]'),
        (r'DELETE FROM .+ WHERE', 'DELETE [QUERY]'),
        
        # Table/Column names
        (r'\btable\s+[\'"`]?\w+[\'"`]?', 'table [TABLE]'),
        (r'\bcolumn\s+[\'"`]?\w+[\'"`]?', 'column [COLUMN]'),
        
        # Environment variables
        (r'\$\{?[A-Z_]+\}?', '[ENV_VAR]'),
        
        # URLs with parameters
        (r'https?://[^\s]+\?[^\s]+', '[URL_WITH_PARAMS]'),
        
        # Email addresses
        (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]'),
        
        # UUIDs
        (r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '[UUID]'),
        
        # Memory addresses
        (r'0x[0-9a-fA-F]+', '[MEMORY_ADDRESS]'),
        
        # Version numbers that might reveal vulnerabilities
        (r'(version|v)\s*[0-9]+\.[0-9]+\.[0-9]+', 'version [VERSION]'),
    ]
    
    # Error mappings for common exceptions
    ERROR_MAPPINGS = {
        # Database errors
        'IntegrityError': ('DB001', 'A data conflict occurred. Please try again.', ErrorSeverity.MEDIUM),
        'OperationalError': ('DB002', 'Database connection issue. Please try again later.', ErrorSeverity.HIGH),
        'DataError': ('DB003', 'Invalid data format provided.', ErrorSeverity.MEDIUM),
        'ProgrammingError': ('DB004', 'An internal error occurred.', ErrorSeverity.HIGH),
        
        # Authentication errors
        'AuthenticationError': ('AUTH001', 'Authentication failed. Please check your credentials.', ErrorSeverity.MEDIUM),
        'AuthorizationError': ('AUTH002', 'You do not have permission to perform this action.', ErrorSeverity.MEDIUM),
        'TokenExpiredError': ('AUTH003', 'Your session has expired. Please log in again.', ErrorSeverity.LOW),
        'InvalidTokenError': ('AUTH004', 'Invalid authentication token.', ErrorSeverity.MEDIUM),
        
        # Validation errors
        'ValidationError': ('VAL001', 'The provided data is invalid.', ErrorSeverity.LOW),
        'InputValidationError': ('VAL002', 'Invalid input provided.', ErrorSeverity.LOW),
        'SchemaValidationError': ('VAL003', 'Data does not match expected format.', ErrorSeverity.LOW),
        
        # File/Resource errors
        'FileNotFoundError': ('RES001', 'The requested resource was not found.', ErrorSeverity.LOW),
        'PermissionError': ('RES002', 'Access to the resource is denied.', ErrorSeverity.MEDIUM),
        'ResourceNotFoundError': ('RES003', 'The requested resource does not exist.', ErrorSeverity.LOW),
        
        # Network errors
        'ConnectionError': ('NET001', 'Network connection failed. Please check your connection.', ErrorSeverity.MEDIUM),
        'TimeoutError': ('NET002', 'The request timed out. Please try again.', ErrorSeverity.MEDIUM),
        'DNSError': ('NET003', 'Network resolution failed.', ErrorSeverity.HIGH),
        
        # System errors
        'MemoryError': ('SYS001', 'System resources exhausted. Please try again later.', ErrorSeverity.CRITICAL),
        'SystemError': ('SYS002', 'An internal system error occurred.', ErrorSeverity.CRITICAL),
        'OSError': ('SYS003', 'System operation failed.', ErrorSeverity.HIGH),
        
        # Application errors
        'KeyError': ('APP001', 'Required data is missing.', ErrorSeverity.MEDIUM),
        'ValueError': ('APP002', 'Invalid value provided.', ErrorSeverity.LOW),
        'TypeError': ('APP003', 'Invalid data type provided.', ErrorSeverity.MEDIUM),
        'AttributeError': ('APP004', 'Invalid operation requested.', ErrorSeverity.MEDIUM),
        'IndexError': ('APP005', 'Invalid data access attempted.', ErrorSeverity.MEDIUM),
        'RuntimeError': ('APP006', 'An unexpected error occurred.', ErrorSeverity.HIGH),
        
        # Default
        'Exception': ('ERR001', 'An unexpected error occurred. Please try again.', ErrorSeverity.MEDIUM),
    }
    
    def __init__(self):
        """Initialize error sanitizer."""
        self.config = get_config()
        self.debug_mode = self.config.get("debug", False)
        self.log_sanitized_errors = self.config.get("logging", {}).get("log_sanitized_errors", True)
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Compile regex patterns for better performance."""
        compiled = []
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            try:
                compiled.append((re.compile(pattern, re.IGNORECASE), replacement))
            except re.error as e:
                logger.error(f"Failed to compile pattern {pattern}: {e}")
        return compiled
    
    def sanitize_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        include_type: bool = False
    ) -> SanitizedError:
        """
        Sanitize an error for safe display to users.
        
        Args:
            error: The exception to sanitize
            context: Additional context about the error
            include_type: Whether to include error type in debug mode
            
        Returns:
            SanitizedError object with safe information
        """
        error_type = type(error).__name__
        error_str = str(error)
        
        # Get error mapping
        error_code, user_message, severity = self.ERROR_MAPPINGS.get(
            error_type,
            self.ERROR_MAPPINGS['Exception']
        )
        
        # Generate unique error ID
        error_id = self._generate_error_id(error, context)
        
        # Sanitize error string
        sanitized_str = self._sanitize_string(error_str)
        
        # Prepare sanitized error
        sanitized_error = SanitizedError(
            error_code=error_code,
            user_message=user_message,
            error_id=error_id,
            timestamp=datetime.utcnow(),
            severity=severity,
            details={}
        )
        
        # Add limited details based on error type
        if isinstance(error, (ValueError, TypeError, KeyError)):
            # These are usually safe to include some details
            sanitized_error.details = {
                'type': error_type if include_type else 'validation_error',
                'sanitized_message': sanitized_str[:100]  # Limit length
            }
        
        # In debug mode, include more information
        if self.debug_mode:
            sanitized_error.debug_info = {
                'error_type': error_type,
                'sanitized_message': sanitized_str,
                'context': self._sanitize_context(context) if context else None,
                'stack_trace': self._sanitize_stack_trace()
            }
        
        # Log the sanitized error
        if self.log_sanitized_errors:
            self._log_sanitized_error(error, sanitized_error)
        
        return sanitized_error
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize a string by removing sensitive information."""
        if not text:
            return ""
        
        sanitized = text
        
        # Apply all compiled patterns
        for pattern, replacement in self._compiled_patterns:
            sanitized = pattern.sub(replacement, sanitized)
        
        # Additional sanitization for common patterns
        # Remove specific error details that might leak information
        sanitized = re.sub(r'near ".+"', 'near [REDACTED]', sanitized)
        sanitized = re.sub(r'constraint ".+"', 'constraint [REDACTED]', sanitized)
        sanitized = re.sub(r'relation ".+"', 'relation [REDACTED]', sanitized)
        
        return sanitized
    
    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize context dictionary."""
        if not context:
            return {}
        
        sanitized_context = {}
        
        for key, value in context.items():
            # Skip sensitive keys
            if any(sensitive in key.lower() for sensitive in ['password', 'token', 'secret', 'key', 'auth']):
                sanitized_context[key] = '[REDACTED]'
                continue
            
            # Sanitize string values
            if isinstance(value, str):
                sanitized_context[key] = self._sanitize_string(value)
            elif isinstance(value, dict):
                sanitized_context[key] = self._sanitize_context(value)
            elif isinstance(value, list):
                sanitized_context[key] = [
                    self._sanitize_string(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized_context[key] = value
        
        return sanitized_context
    
    def _sanitize_stack_trace(self) -> List[str]:
        """Sanitize stack trace information."""
        tb = traceback.format_exc()
        lines = tb.split('\n')
        
        sanitized_lines = []
        for line in lines:
            # Sanitize each line
            sanitized_line = self._sanitize_string(line)
            
            # Further sanitize file paths in traceback
            sanitized_line = re.sub(
                r'File "([^"]+)"',
                lambda m: f'File "{self._sanitize_file_path(m.group(1))}"',
                sanitized_line
            )
            
            sanitized_lines.append(sanitized_line)
        
        return sanitized_lines
    
    def _sanitize_file_path(self, path: str) -> str:
        """Sanitize file path to show only relevant parts."""
        # Keep only the last 2-3 components of the path
        parts = path.replace('\\', '/').split('/')
        if len(parts) > 3:
            return '.../' + '/'.join(parts[-2:])
        return '/'.join(parts)
    
    def _generate_error_id(self, error: Exception, context: Optional[Dict[str, Any]]) -> str:
        """Generate unique error ID for tracking."""
        # Create a hash of error details
        error_data = {
            'type': type(error).__name__,
            'message': str(error)[:100],  # Limit message length
            'timestamp': datetime.utcnow().isoformat(),
            'context_keys': sorted(context.keys()) if context else []
        }
        
        error_json = json.dumps(error_data, sort_keys=True)
        error_hash = hashlib.sha256(error_json.encode()).hexdigest()[:12]
        
        return f"ERR-{datetime.utcnow().strftime('%Y%m%d')}-{error_hash}"
    
    def _log_sanitized_error(self, original_error: Exception, sanitized_error: SanitizedError):
        """Log sanitized error for debugging."""
        logger.error(
            f"Sanitized error: {sanitized_error.error_code} - {sanitized_error.user_message}",
            extra={
                'error_id': sanitized_error.error_id,
                'error_type': type(original_error).__name__,
                'severity': sanitized_error.severity.value,
                'sanitized_details': sanitized_error.details
            },
            exc_info=True
        )
    
    def create_error_response(
        self,
        error: Exception,
        status_code: int = 500,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            error: The exception to handle
            status_code: HTTP status code
            context: Additional context
            
        Returns:
            Dictionary suitable for API response
        """
        sanitized = self.sanitize_error(error, context)
        
        response = {
            'error': {
                'code': sanitized.error_code,
                'message': sanitized.user_message,
                'error_id': sanitized.error_id,
                'timestamp': sanitized.timestamp.isoformat()
            }
        }
        
        # Add details if available
        if sanitized.details:
            response['error']['details'] = sanitized.details
        
        # Add debug info in debug mode
        if self.debug_mode and sanitized.debug_info:
            response['debug'] = sanitized.debug_info
        
        return response
    
    def sanitize_log_message(self, message: str) -> str:
        """Sanitize a log message before writing."""
        return self._sanitize_string(message)


# Global error sanitizer instance
error_sanitizer = ErrorSanitizer()


def sanitize_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> SanitizedError:
    """Convenience function to sanitize errors."""
    return error_sanitizer.sanitize_error(error, context)


def create_error_response(
    error: Exception,
    status_code: int = 500,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function to create error responses."""
    return error_sanitizer.create_error_response(error, status_code, context)