"""
Secrets management module for secure credential handling.

This module provides a unified interface for accessing secrets from various sources:
- Environment variables (development)
- AWS Secrets Manager (production)
- HashiCorp Vault (enterprise)
- Azure Key Vault (Azure deployments)
"""
import os
import json
import logging
from typing import Dict, Any, Optional, Union
from functools import lru_cache
from datetime import datetime, timedelta
import base64

logger = logging.getLogger(__name__)


class SecretsManager:
    """
    Unified secrets management interface.
    
    Provides a consistent API for retrieving secrets from various backends
    with caching, rotation support, and fallback mechanisms.
    """
    
    def __init__(self, provider: str = "env", **kwargs):
        """
        Initialize secrets manager with specified provider.
        
        Args:
            provider: Secrets provider ('env', 'aws', 'vault', 'azure')
            **kwargs: Provider-specific configuration
        """
        self.provider = provider.lower()
        self._secrets_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(minutes=5)  # Default cache TTL
        
        # Initialize provider
        if self.provider == "aws":
            self._init_aws_secrets_manager(**kwargs)
        elif self.provider == "vault":
            self._init_hashicorp_vault(**kwargs)
        elif self.provider == "azure":
            self._init_azure_key_vault(**kwargs)
        elif self.provider == "env":
            self._init_env_provider()
        else:
            raise ValueError(f"Unknown secrets provider: {provider}")
            
        logger.info(f"Initialized secrets manager with provider: {self.provider}")
    
    def _init_aws_secrets_manager(self, region: str = None, **kwargs):
        """Initialize AWS Secrets Manager client."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            self.region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            self.client = boto3.client("secretsmanager", region_name=self.region)
            self.ClientError = ClientError
            logger.info(f"Initialized AWS Secrets Manager in region: {self.region}")
        except ImportError:
            raise ImportError("boto3 is required for AWS Secrets Manager. Install with: pip install boto3")
    
    def _init_hashicorp_vault(self, url: str = None, token: str = None, **kwargs):
        """Initialize HashiCorp Vault client."""
        try:
            import hvac
            
            self.vault_url = url or os.getenv("VAULT_ADDR", "http://localhost:8200")
            self.vault_token = token or os.getenv("VAULT_TOKEN")
            
            if not self.vault_token:
                raise ValueError("Vault token is required")
            
            self.client = hvac.Client(url=self.vault_url, token=self.vault_token)
            
            if not self.client.is_authenticated():
                raise ValueError("Failed to authenticate with Vault")
                
            logger.info(f"Initialized HashiCorp Vault client: {self.vault_url}")
        except ImportError:
            raise ImportError("hvac is required for HashiCorp Vault. Install with: pip install hvac")
    
    def _init_azure_key_vault(self, vault_url: str = None, **kwargs):
        """Initialize Azure Key Vault client."""
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential
            
            self.vault_url = vault_url or os.getenv("AZURE_KEY_VAULT_URL")
            if not self.vault_url:
                raise ValueError("Azure Key Vault URL is required")
            
            credential = DefaultAzureCredential()
            self.client = SecretClient(vault_url=self.vault_url, credential=credential)
            logger.info(f"Initialized Azure Key Vault client: {self.vault_url}")
        except ImportError:
            raise ImportError("azure-keyvault-secrets is required. Install with: pip install azure-keyvault-secrets azure-identity")
    
    def _init_env_provider(self):
        """Initialize environment variable provider."""
        logger.info("Using environment variables for secrets")
    
    @lru_cache(maxsize=128)
    def get_secret(self, secret_name: str, version: Optional[str] = None) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Retrieve a secret value.
        
        Args:
            secret_name: Name/ID of the secret
            version: Optional version/stage of the secret
            
        Returns:
            Secret value (string or dict)
        """
        # Check cache first
        cache_key = f"{secret_name}:{version or 'latest'}"
        if cache_key in self._secrets_cache:
            cached = self._secrets_cache[cache_key]
            if datetime.now() < cached["expires"]:
                return cached["value"]
        
        try:
            # Retrieve from provider
            if self.provider == "aws":
                value = self._get_aws_secret(secret_name, version)
            elif self.provider == "vault":
                value = self._get_vault_secret(secret_name, version)
            elif self.provider == "azure":
                value = self._get_azure_secret(secret_name, version)
            else:  # env
                value = self._get_env_secret(secret_name)
            
            # Cache the result
            if value is not None:
                self._secrets_cache[cache_key] = {
                    "value": value,
                    "expires": datetime.now() + self._cache_ttl
                }
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            # Try fallback to environment variable
            return self._get_env_secret(secret_name)
    
    def _get_aws_secret(self, secret_name: str, version: Optional[str] = None) -> Optional[Union[str, Dict[str, Any]]]:
        """Retrieve secret from AWS Secrets Manager."""
        try:
            kwargs = {"SecretId": secret_name}
            if version:
                kwargs["VersionId"] = version
            
            response = self.client.get_secret_value(**kwargs)
            
            # AWS can store either strings or binary
            if "SecretString" in response:
                secret = response["SecretString"]
                # Try to parse as JSON
                try:
                    return json.loads(secret)
                except json.JSONDecodeError:
                    return secret
            else:
                # Binary secret
                return base64.b64decode(response["SecretBinary"]).decode("utf-8")
                
        except self.ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.warning(f"Secret not found: {secret_name}")
                return None
            raise
    
    def _get_vault_secret(self, secret_name: str, version: Optional[str] = None) -> Optional[Union[str, Dict[str, Any]]]:
        """Retrieve secret from HashiCorp Vault."""
        try:
            # Vault paths typically use forward slashes
            path = secret_name.replace("_", "/")
            if version:
                response = self.client.secrets.kv.v2.read_secret_version(
                    path=path,
                    version=version
                )
            else:
                response = self.client.secrets.kv.v2.read_secret_version(path=path)
            
            return response["data"]["data"]
            
        except Exception as e:
            logger.warning(f"Failed to get Vault secret {secret_name}: {e}")
            return None
    
    def _get_azure_secret(self, secret_name: str, version: Optional[str] = None) -> Optional[str]:
        """Retrieve secret from Azure Key Vault."""
        try:
            # Azure uses hyphens instead of underscores
            name = secret_name.replace("_", "-")
            secret = self.client.get_secret(name, version=version)
            return secret.value
        except Exception as e:
            logger.warning(f"Failed to get Azure secret {secret_name}: {e}")
            return None
    
    def _get_env_secret(self, secret_name: str) -> Optional[str]:
        """Retrieve secret from environment variables."""
        # Try exact match first
        value = os.getenv(secret_name)
        if value:
            return value
        
        # Try uppercase
        value = os.getenv(secret_name.upper())
        if value:
            return value
        
        # Try with prefixes
        for prefix in ["CLEOAI_", "APP_"]:
            value = os.getenv(f"{prefix}{secret_name.upper()}")
            if value:
                return value
        
        return None
    
    def get_database_credentials(self) -> Dict[str, Any]:
        """Get all database credentials."""
        return {
            "mongodb": {
                "connection_string": self.get_secret("MONGODB_CONNECTION_STRING"),
                "username": self.get_secret("MONGO_ROOT_USERNAME"),
                "password": self.get_secret("MONGO_ROOT_PASSWORD"),
            },
            "redis": {
                "host": self.get_secret("REDIS_HOST") or "localhost",
                "port": int(self.get_secret("REDIS_PORT") or 6379),
                "password": self.get_secret("REDIS_PASSWORD"),
            },
            "supabase": {
                "url": self.get_secret("SUPABASE_URL"),
                "anon_key": self.get_secret("SUPABASE_ANON_KEY"),
                "service_key": self.get_secret("SUPABASE_SERVICE_KEY"),
            },
            "pinecone": {
                "api_key": self.get_secret("PINECONE_API_KEY"),
                "environment": self.get_secret("PINECONE_ENVIRONMENT"),
            }
        }
    
    def get_api_credentials(self) -> Dict[str, Any]:
        """Get API-related credentials."""
        return {
            "jwt_secret": self.get_secret("JWT_SECRET_KEY"),
            "api_key": self.get_secret("API_KEY"),
            "allowed_origins": self.get_secret("ALLOWED_ORIGINS"),
            "sentry_dsn": self.get_secret("SENTRY_DSN"),
            "opentelemetry_endpoint": self.get_secret("OTEL_EXPORTER_OTLP_ENDPOINT"),
        }
    
    def rotate_secret(self, secret_name: str) -> bool:
        """
        Trigger secret rotation (provider-specific).
        
        Args:
            secret_name: Name of the secret to rotate
            
        Returns:
            True if rotation was triggered successfully
        """
        try:
            if self.provider == "aws":
                self.client.rotate_secret(SecretId=secret_name)
                logger.info(f"Triggered rotation for AWS secret: {secret_name}")
                return True
            elif self.provider == "vault":
                # Vault rotation is typically handled by external processes
                logger.warning("Vault secret rotation should be handled externally")
                return False
            else:
                logger.warning(f"Secret rotation not supported for provider: {self.provider}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to rotate secret {secret_name}: {e}")
            return False
    
    def clear_cache(self):
        """Clear the secrets cache."""
        self._secrets_cache.clear()
        self.get_secret.cache_clear()
        logger.info("Cleared secrets cache")


# Global instance
_secrets_manager: Optional[SecretsManager] = None


def init_secrets_manager(provider: str = None, **kwargs) -> SecretsManager:
    """
    Initialize the global secrets manager.
    
    Args:
        provider: Secrets provider (defaults to SECRETS_PROVIDER env var)
        **kwargs: Provider-specific configuration
        
    Returns:
        SecretsManager instance
    """
    global _secrets_manager
    
    if provider is None:
        provider = os.getenv("SECRETS_PROVIDER", "env")
    
    _secrets_manager = SecretsManager(provider=provider, **kwargs)
    return _secrets_manager


def get_secrets_manager() -> SecretsManager:
    """
    Get the global secrets manager instance.
    
    Returns:
        SecretsManager instance
        
    Raises:
        RuntimeError: If secrets manager not initialized
    """
    if _secrets_manager is None:
        # Auto-initialize with defaults
        return init_secrets_manager()
    return _secrets_manager


def get_secret(secret_name: str, version: Optional[str] = None) -> Optional[Union[str, Dict[str, Any]]]:
    """
    Convenience function to get a secret.
    
    Args:
        secret_name: Name of the secret
        version: Optional version
        
    Returns:
        Secret value
    """
    return get_secrets_manager().get_secret(secret_name, version)