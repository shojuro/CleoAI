"""
Configuration validation for distributed memory backends.
"""
import os
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigValidator:
    """Validates configuration for distributed memory backends."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> ValidationResult:
        """Validate all backend configurations."""
        self.errors = []
        self.warnings = []
        
        # Check each backend if enabled
        if self._is_enabled("USE_REDIS"):
            self._validate_redis()
        
        if self._is_enabled("USE_MONGODB"):
            self._validate_mongodb()
        
        if self._is_enabled("USE_SUPABASE"):
            self._validate_supabase()
        
        if self._is_enabled("USE_PINECONE"):
            self._validate_pinecone()
        
        # Check for conflicting configurations
        self._check_conflicts()
        
        return ValidationResult(
            is_valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings
        )
    
    def _is_enabled(self, env_var: str) -> bool:
        """Check if a backend is enabled."""
        return os.getenv(env_var, "false").lower() == "true"
    
    def _check_env(self, var_name: str, required: bool = True) -> bool:
        """Check if environment variable is set."""
        value = os.getenv(var_name)
        if not value:
            if required:
                self.errors.append(f"Missing required environment variable: {var_name}")
            else:
                self.warnings.append(f"Missing optional environment variable: {var_name}")
            return False
        return True
    
    def _validate_redis(self):
        """Validate Redis configuration."""
        self._check_env("REDIS_HOST")
        self._check_env("REDIS_PORT")
        self._check_env("REDIS_PASSWORD", required=False)
        
        # Check port is valid
        port = os.getenv("REDIS_PORT", "")
        if port and not port.isdigit():
            self.errors.append(f"Invalid REDIS_PORT: {port} (must be numeric)")
    
    def _validate_mongodb(self):
        """Validate MongoDB configuration."""
        conn_string = os.getenv("MONGODB_CONNECTION_STRING", "")
        
        if not conn_string:
            self.errors.append("Missing MONGODB_CONNECTION_STRING")
        elif not (conn_string.startswith("mongodb://") or 
                  conn_string.startswith("mongodb+srv://")):
            self.errors.append("Invalid MONGODB_CONNECTION_STRING format")
    
    def _validate_supabase(self):
        """Validate Supabase configuration."""
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_ANON_KEY", "")
        
        if not url:
            self.errors.append("Missing SUPABASE_URL")
        elif not url.startswith("https://"):
            self.errors.append("SUPABASE_URL must use HTTPS")
        elif "your-project" in url:
            self.errors.append("SUPABASE_URL contains placeholder value")
        
        if not key:
            self.errors.append("Missing SUPABASE_ANON_KEY")
        elif key == "your-anon-key-here":
            self.errors.append("SUPABASE_ANON_KEY contains placeholder value")
    
    def _validate_pinecone(self):
        """Validate Pinecone configuration."""
        api_key = os.getenv("PINECONE_API_KEY", "")
        env = os.getenv("PINECONE_ENVIRONMENT", "")
        
        if not api_key:
            self.errors.append("Missing PINECONE_API_KEY")
        elif api_key == "your-pinecone-api-key-here":
            self.errors.append("PINECONE_API_KEY contains placeholder value")
        
        if not env:
            self.errors.append("Missing PINECONE_ENVIRONMENT")
        elif env not in ["us-east-1", "us-west-1", "eu-west-1", "asia-southeast-1"]:
            self.warnings.append(f"Unusual PINECONE_ENVIRONMENT: {env}")
    
    def _check_conflicts(self):
        """Check for conflicting configurations."""
        # Warn if all backends are disabled
        backends = ["USE_REDIS", "USE_MONGODB", "USE_SUPABASE", 
                   "USE_PINECONE", "USE_SQLITE", "USE_CHROMADB"]
        
        enabled_count = sum(1 for b in backends if self._is_enabled(b))
        
        if enabled_count == 0:
            self.errors.append("No memory backends are enabled!")
        
        # Warn about using only legacy backends
        legacy_only = (self._is_enabled("USE_SQLITE") or self._is_enabled("USE_CHROMADB")) and \
                     not any(self._is_enabled(b) for b in ["USE_REDIS", "USE_MONGODB", 
                                                           "USE_SUPABASE", "USE_PINECONE"])
        
        if legacy_only:
            self.warnings.append("Only legacy backends are enabled. Consider enabling distributed backends for better performance.")
        
        # Check for vector storage
        has_vector = self._is_enabled("USE_PINECONE") or self._is_enabled("USE_CHROMADB")
        if not has_vector:
            self.warnings.append("No vector storage backend enabled. Semantic search will not be available.")


def validate_configuration() -> ValidationResult:
    """Validate the current configuration."""
    validator = ConfigValidator()
    result = validator.validate_all()
    
    # Log results
    if result.errors:
        logger.error("Configuration validation failed:")
        for error in result.errors:
            logger.error(f"  - {error}")
    
    if result.warnings:
        logger.warning("Configuration warnings:")
        for warning in result.warnings:
            logger.warning(f"  - {warning}")
    
    if result.is_valid:
        logger.info("Configuration validation passed")
    
    return result


def ensure_valid_configuration():
    """Ensure configuration is valid, exit if not."""
    result = validate_configuration()
    
    if not result.is_valid:
        print("\n❌ Configuration validation failed!\n")
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
        
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        print("\nPlease fix the configuration errors before starting.")
        raise SystemExit(1)
    
    if result.warnings:
        print("\n⚠️  Configuration warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
        print()