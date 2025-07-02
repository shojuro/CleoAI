"""
Input validation and sanitization module for CleoAI.

This module provides comprehensive input validation to prevent
injection attacks, XSS, and other security vulnerabilities.
"""
import re
import html
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import bleach
from pydantic import BaseModel, validator, Field, constr, conint, confloat

logger = logging.getLogger(__name__)

# Regular expressions for validation
USERNAME_REGEX = re.compile(r'^[a-zA-Z0-9_-]{3,32}$')
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
UUID_REGEX = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$')
SAFE_STRING_REGEX = re.compile(r'^[a-zA-Z0-9\s\-_.,!?\'\"]+$')

# Dangerous patterns to detect
SQL_INJECTION_PATTERNS = [
    r"(\bunion\b.*\bselect\b|\bselect\b.*\bfrom\b|\binsert\b.*\binto\b|\bupdate\b.*\bset\b|\bdelete\b.*\bfrom\b|\bdrop\b.*\btable\b)",
    r"(--|#|\/\*|\*\/|@@|@|\bchar\b|\bnchar\b|\bvarchar\b|\bnvarchar\b|\balter\b|\bbegin\b|\bcast\b|\bcreate\b|\bcursor\b|\bdeclare\b|\bdelete\b|\bdrop\b|\bexec\b|\bexecute\b|\bfetch\b|\binsert\b|\bkill\b|\bselect\b|\bsys\b|\bsysobjects\b|\bsyscol

umns\b|\btable\b|\bupdate\b)",
    r"(\bor\b\s*\d+\s*=\s*\d+|\band\b\s*\d+\s*=\s*\d+)",
    r"(;|\||&&|\|\||>|<|--)"
]

XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe[^>]*>",
    r"<object[^>]*>",
    r"<embed[^>]*>",
    r"<link[^>]*>",
    r"<meta[^>]*>"
]

# Allowed HTML tags for rich text (if needed)
ALLOWED_TAGS = ['p', 'br', 'strong', 'em', 'u', 'a', 'ul', 'ol', 'li', 'blockquote', 'code', 'pre']
ALLOWED_ATTRIBUTES = {'a': ['href', 'title']}


class ValidationError(Exception):
    """Custom validation error."""
    pass


# Pydantic models for input validation
class UserInput(BaseModel):
    """User identification input."""
    user_id: constr(min_length=1, max_length=128)
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("User ID cannot be empty")
        # Allow various formats but sanitize
        v = v.strip()
        if len(v) > 128:
            raise ValueError("User ID too long")
        return v


class ConversationInput(BaseModel):
    """Conversation input validation."""
    conversation_id: Optional[constr(regex=UUID_REGEX)] = None
    message: constr(min_length=1, max_length=10000)
    
    @validator('message')
    def validate_message(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Message cannot be empty")
        # Check for dangerous patterns
        if detect_sql_injection(v):
            raise ValueError("Invalid input detected")
        if detect_xss(v):
            raise ValueError("Invalid input detected")
        return v.strip()


class PreferenceInput(BaseModel):
    """User preference input validation."""
    category: constr(regex=r'^[a-zA-Z0-9_-]{1,64}$')
    key: constr(regex=r'^[a-zA-Z0-9_-]{1,64}$')
    value: Union[str, int, float, bool, Dict[str, Any], List[Any]]
    confidence: confloat(ge=0.0, le=1.0) = 0.5
    
    @validator('value')
    def validate_value(cls, v):
        if isinstance(v, str):
            if len(v) > 1000:
                raise ValueError("String value too long")
            if detect_sql_injection(v) or detect_xss(v):
                raise ValueError("Invalid value detected")
        elif isinstance(v, (dict, list)):
            # Validate JSON-serializable
            try:
                json.dumps(v)
            except:
                raise ValueError("Value must be JSON-serializable")
            # Check size
            if len(json.dumps(v)) > 10000:
                raise ValueError("Value too large")
        return v


class MemoryInput(BaseModel):
    """Memory input validation."""
    title: constr(min_length=1, max_length=200)
    content: constr(min_length=1, max_length=10000)
    importance: confloat(ge=0.0, le=1.0) = 0.5
    emotion: Optional[constr(regex=r'^[a-zA-Z]{1,32}$')] = None
    tags: Optional[List[constr(regex=r'^[a-zA-Z0-9_-]{1,32}$')]] = None
    
    @validator('title', 'content')
    def validate_text(cls, v, field):
        if detect_sql_injection(v) or detect_xss(v):
            raise ValueError(f"Invalid {field.name} detected")
        return v.strip()
    
    @validator('tags')
    def validate_tags(cls, v):
        if v and len(v) > 20:
            raise ValueError("Too many tags (max 20)")
        return v


class SearchInput(BaseModel):
    """Search query input validation."""
    query: constr(min_length=1, max_length=500)
    limit: conint(ge=1, le=100) = 10
    offset: conint(ge=0) = 0
    
    @validator('query')
    def validate_query(cls, v):
        # Remove special characters that could be used for injection
        v = re.sub(r'[^\w\s\-\'\".,!?]', '', v)
        if detect_sql_injection(v):
            raise ValueError("Invalid search query")
        return v.strip()


def sanitize_string(text: str, max_length: int = 1000) -> str:
    """
    Basic string sanitization.
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not text:
        return ""
    
    # Truncate
    text = text[:max_length]
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def sanitize_html(html_content: str) -> str:
    """
    Sanitize HTML content to prevent XSS.
    
    Args:
        html_content: HTML content to sanitize
        
    Returns:
        Safe HTML
    """
    if not html_content:
        return ""
    
    # Use bleach to clean HTML
    cleaned = bleach.clean(
        html_content,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True
    )
    
    return cleaned


def escape_html(text: str) -> str:
    """
    Escape HTML special characters.
    
    Args:
        text: Text to escape
        
    Returns:
        HTML-escaped text
    """
    return html.escape(text, quote=True)


def detect_sql_injection(text: str) -> bool:
    """
    Detect potential SQL injection attempts.
    
    Args:
        text: Text to check
        
    Returns:
        True if suspicious pattern detected
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.warning(f"Potential SQL injection detected: {text[:100]}...")
            return True
    
    return False


def detect_xss(text: str) -> bool:
    """
    Detect potential XSS attempts.
    
    Args:
        text: Text to check
        
    Returns:
        True if suspicious pattern detected
    """
    if not text:
        return False
    
    for pattern in XSS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning(f"Potential XSS detected: {text[:100]}...")
            return True
    
    return False


def validate_json(json_str: str) -> Union[Dict, List]:
    """
    Validate and parse JSON string.
    
    Args:
        json_str: JSON string to validate
        
    Returns:
        Parsed JSON object
        
    Raises:
        ValidationError: If JSON is invalid
    """
    try:
        data = json.loads(json_str)
        
        # Additional size check
        if len(json_str) > 100000:  # 100KB limit
            raise ValidationError("JSON too large")
        
        return data
        
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {e}")


def validate_file_path(path: str) -> bool:
    """
    Validate file path to prevent directory traversal.
    
    Args:
        path: File path to validate
        
    Returns:
        True if path is safe
    """
    # Remove any directory traversal attempts
    if '..' in path or path.startswith('/'):
        return False
    
    # Check for null bytes
    if '\x00' in path:
        return False
    
    # Only allow alphanumeric, dash, underscore, dot, and forward slash
    if not re.match(r'^[a-zA-Z0-9_\-./]+$', path):
        return False
    
    return True


def validate_environment() -> Dict[str, bool]:
    """
    Validate environment variables for security.
    
    Returns:
        Dict of validation results
    """
    import os
    
    results = {}
    
    # Check for debug mode
    results['debug_disabled'] = os.getenv('DEBUG', 'false').lower() != 'true'
    
    # Check for secure cookies
    results['secure_cookies'] = os.getenv('SESSION_SECURE_COOKIE', 'true').lower() == 'true'
    
    # Check for proper environment
    results['production_env'] = os.getenv('ENVIRONMENT', '').lower() == 'production'
    
    # Check for rate limiting
    results['rate_limiting_enabled'] = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    
    # Check for introspection disabled
    results['introspection_disabled'] = os.getenv('ENABLE_INTROSPECTION', 'false').lower() != 'true'
    
    return results


# Validation decorators for functions
def validate_inputs(model_class: BaseModel):
    """
    Decorator to validate function inputs using Pydantic model.
    
    Args:
        model_class: Pydantic model class for validation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Validate kwargs against model
                validated = model_class(**kwargs)
                # Convert back to dict and call function
                return func(*args, **validated.dict())
            except Exception as e:
                logger.error(f"Input validation failed: {e}")
                raise ValidationError(f"Invalid input: {e}")
        return wrapper
    return decorator


# Pre-compiled validators for performance
class InputValidator:
    """High-performance input validator."""
    
    def __init__(self):
        self.sql_patterns = [re.compile(p, re.IGNORECASE) for p in SQL_INJECTION_PATTERNS]
        self.xss_patterns = [re.compile(p, re.IGNORECASE) for p in XSS_PATTERNS]
    
    def is_safe_string(self, text: str, max_length: int = 1000) -> bool:
        """Quick safety check for strings."""
        if not text or len(text) > max_length:
            return False
        
        # Check for null bytes
        if '\x00' in text:
            return False
        
        # Check SQL injection
        text_lower = text.lower()
        for pattern in self.sql_patterns:
            if pattern.search(text_lower):
                return False
        
        # Check XSS
        for pattern in self.xss_patterns:
            if pattern.search(text):
                return False
        
        return True
    
    def sanitize_for_log(self, text: str, max_length: int = 200) -> str:
        """Sanitize text for safe logging."""
        if not text:
            return ""
        
        # Truncate
        text = text[:max_length]
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Escape newlines for single-line logs
        text = text.replace('\n', '\\n').replace('\r', '\\r')
        
        return text


# Global validator instance
validator = InputValidator()