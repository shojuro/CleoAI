"""
Encryption module for data at rest in CleoAI.

This module provides field-level encryption for sensitive data using
AES-256-GCM with key management and rotation support.
"""
import os
import base64
import json
import logging
import time
from typing import Any, Dict, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import secrets

from ..utils.secrets_manager import get_secrets_manager
from ..utils.error_handling import handle_errors

logger = logging.getLogger(__name__)

# Encryption configuration
ENCRYPTION_ALGORITHM = "AES-256-GCM"
KEY_SIZE = 32  # 256 bits
NONCE_SIZE = 12  # 96 bits for GCM
TAG_SIZE = 16  # 128 bits
SALT_SIZE = 32  # 256 bits
ITERATIONS = 100000  # PBKDF2 iterations


@dataclass
class EncryptedData:
    """Container for encrypted data with metadata."""
    ciphertext: str  # Base64 encoded
    nonce: str  # Base64 encoded
    tag: str  # Base64 encoded
    key_version: str
    algorithm: str = ENCRYPTION_ALGORITHM
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'ciphertext': self.ciphertext,
            'nonce': self.nonce,
            'tag': self.tag,
            'key_version': self.key_version,
            'algorithm': self.algorithm,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedData':
        """Create from dictionary."""
        return cls(
            ciphertext=data['ciphertext'],
            nonce=data['nonce'],
            tag=data['tag'],
            key_version=data['key_version'],
            algorithm=data.get('algorithm', ENCRYPTION_ALGORITHM),
            timestamp=data.get('timestamp', time.time())
        )


@dataclass
class EncryptionKey:
    """Encryption key with metadata."""
    key: bytes
    key_id: str
    version: str
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    active: bool = True
    
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class KeyManager:
    """
    Manages encryption keys with rotation support.
    
    Integrates with external key management services (AWS KMS, Vault, etc.)
    or uses local key derivation for development.
    """
    
    def __init__(self, provider: str = "local"):
        self.provider = provider
        self.keys: Dict[str, EncryptionKey] = {}
        self.current_key_version: Optional[str] = None
        self.secrets_manager = get_secrets_manager()
        self._initialize_keys()
    
    def _initialize_keys(self):
        """Initialize encryption keys."""
        if self.provider == "local":
            # Local development mode - derive from master key
            master_key = self._get_master_key()
            self._derive_data_key(master_key)
        elif self.provider == "aws_kms":
            # AWS KMS integration
            self._initialize_aws_kms()
        elif self.provider == "vault":
            # HashiCorp Vault integration
            self._initialize_vault()
        else:
            raise ValueError(f"Unknown key provider: {self.provider}")
    
    def _get_master_key(self) -> bytes:
        """Get or generate master key."""
        # Try to get from secrets manager
        master_key_b64 = self.secrets_manager.get_secret("ENCRYPTION_MASTER_KEY")
        
        if master_key_b64:
            return base64.b64decode(master_key_b64)
        
        # Generate new master key for development
        logger.warning("Generating new master key - this should only happen in development!")
        master_key = secrets.token_bytes(KEY_SIZE)
        
        # Save for future use (in development only)
        if os.getenv("ENVIRONMENT") == "development":
            logger.info("Saving master key to secrets manager")
            self.secrets_manager.get_secret.cache_clear()
            os.environ["ENCRYPTION_MASTER_KEY"] = base64.b64encode(master_key).decode()
        
        return master_key
    
    def _derive_data_key(self, master_key: bytes, version: str = "v1"):
        """Derive a data encryption key from master key."""
        # Use PBKDF2 to derive key
        salt = self.secrets_manager.get_secret(f"ENCRYPTION_SALT_{version}")
        if not salt:
            salt = secrets.token_bytes(SALT_SIZE)
            if os.getenv("ENVIRONMENT") == "development":
                os.environ[f"ENCRYPTION_SALT_{version}"] = base64.b64encode(salt).decode()
        else:
            salt = base64.b64decode(salt)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            iterations=ITERATIONS,
            backend=default_backend()
        )
        
        derived_key = kdf.derive(master_key)
        
        # Create key object
        key_obj = EncryptionKey(
            key=derived_key,
            key_id=f"local-{version}",
            version=version,
            expires_at=time.time() + (90 * 24 * 3600)  # 90 days
        )
        
        self.keys[version] = key_obj
        self.current_key_version = version
        
        logger.info(f"Initialized encryption key version {version}")
    
    def _initialize_aws_kms(self):
        """Initialize AWS KMS key management."""
        # TODO: Implement AWS KMS integration
        raise NotImplementedError("AWS KMS integration not yet implemented")
    
    def _initialize_vault(self):
        """Initialize HashiCorp Vault key management."""
        # TODO: Implement Vault integration
        raise NotImplementedError("Vault integration not yet implemented")
    
    def get_current_key(self) -> EncryptionKey:
        """Get the current active encryption key."""
        if not self.current_key_version:
            raise ValueError("No active encryption key")
        
        key = self.keys.get(self.current_key_version)
        if not key or not key.active or key.is_expired():
            raise ValueError("Current key is not valid")
        
        return key
    
    def get_key_by_version(self, version: str) -> Optional[EncryptionKey]:
        """Get a specific key version for decryption."""
        return self.keys.get(version)
    
    def rotate_keys(self) -> str:
        """
        Rotate encryption keys.
        
        Returns:
            New key version
        """
        # Generate new version number
        current_version = int(self.current_key_version[1:]) if self.current_key_version else 0
        new_version = f"v{current_version + 1}"
        
        # Generate new key
        if self.provider == "local":
            master_key = self._get_master_key()
            self._derive_data_key(master_key, new_version)
        else:
            raise NotImplementedError(f"Key rotation not implemented for {self.provider}")
        
        # Mark old key as inactive (but keep for decryption)
        if self.current_key_version:
            old_key = self.keys.get(self.current_key_version)
            if old_key:
                old_key.active = False
        
        self.current_key_version = new_version
        logger.info(f"Rotated encryption key to version {new_version}")
        
        return new_version


class EncryptionProvider:
    """
    Main encryption provider for field-level encryption.
    
    Handles encryption/decryption of data with automatic key management.
    """
    
    def __init__(self, key_manager: Optional[KeyManager] = None):
        self.key_manager = key_manager or KeyManager()
        self.backend = default_backend()
    
    def encrypt(self, plaintext: Union[str, bytes, Dict, List]) -> EncryptedData:
        """
        Encrypt data using the current key.
        
        Args:
            plaintext: Data to encrypt (will be JSON-serialized if not bytes/str)
            
        Returns:
            EncryptedData object
        """
        # Convert to bytes
        if isinstance(plaintext, (dict, list)):
            plaintext = json.dumps(plaintext).encode('utf-8')
        elif isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        elif not isinstance(plaintext, bytes):
            plaintext = str(plaintext).encode('utf-8')
        
        # Get current key
        key = self.key_manager.get_current_key()
        
        # Generate nonce
        nonce = os.urandom(NONCE_SIZE)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key.key),
            modes.GCM(nonce),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        # Encrypt
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Get tag
        tag = encryptor.tag
        
        # Create encrypted data object
        return EncryptedData(
            ciphertext=base64.b64encode(ciphertext).decode('utf-8'),
            nonce=base64.b64encode(nonce).decode('utf-8'),
            tag=base64.b64encode(tag).decode('utf-8'),
            key_version=key.version
        )
    
    def decrypt(self, encrypted_data: Union[EncryptedData, Dict[str, Any]], 
                return_type: type = str) -> Any:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data object or dict
            return_type: Type to return (str, bytes, dict, list)
            
        Returns:
            Decrypted data in requested type
        """
        # Convert dict to EncryptedData if needed
        if isinstance(encrypted_data, dict):
            encrypted_data = EncryptedData.from_dict(encrypted_data)
        
        # Get key for this version
        key = self.key_manager.get_key_by_version(encrypted_data.key_version)
        if not key:
            raise ValueError(f"No key found for version {encrypted_data.key_version}")
        
        # Decode from base64
        ciphertext = base64.b64decode(encrypted_data.ciphertext)
        nonce = base64.b64decode(encrypted_data.nonce)
        tag = base64.b64decode(encrypted_data.tag)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key.key),
            modes.GCM(nonce, tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        # Decrypt
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Convert to requested type
        if return_type == bytes:
            return plaintext
        elif return_type == str:
            return plaintext.decode('utf-8')
        elif return_type in (dict, list):
            return json.loads(plaintext.decode('utf-8'))
        else:
            # Try to convert
            try:
                return return_type(plaintext.decode('utf-8'))
            except:
                return plaintext.decode('utf-8')
    
    def encrypt_fields(self, data: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        """
        Encrypt specific fields in a dictionary.
        
        Args:
            data: Dictionary containing data
            fields: List of field names to encrypt
            
        Returns:
            Dictionary with encrypted fields
        """
        result = data.copy()
        
        for field_name in fields:
            if field_name in result and result[field_name] is not None:
                encrypted = self.encrypt(result[field_name])
                result[field_name] = encrypted.to_dict()
                result[f"{field_name}_encrypted"] = True
        
        return result
    
    def decrypt_fields(self, data: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        """
        Decrypt specific fields in a dictionary.
        
        Args:
            data: Dictionary containing encrypted data
            fields: List of field names to decrypt
            
        Returns:
            Dictionary with decrypted fields
        """
        result = data.copy()
        
        for field_name in fields:
            if f"{field_name}_encrypted" in result and result.get(f"{field_name}_encrypted"):
                if field_name in result and isinstance(result[field_name], dict):
                    # Determine return type from original data structure
                    encrypted_data = result[field_name]
                    # Try to infer type from metadata or default to str
                    decrypted = self.decrypt(encrypted_data, return_type=str)
                    result[field_name] = decrypted
                    del result[f"{field_name}_encrypted"]
        
        return result


class FieldEncryption:
    """
    Decorator and context manager for automatic field encryption.
    """
    
    def __init__(self, fields: List[str], provider: Optional[EncryptionProvider] = None):
        self.fields = fields
        self.provider = provider or EncryptionProvider()
    
    def __call__(self, func):
        """Decorator to automatically encrypt/decrypt fields."""
        def wrapper(*args, **kwargs):
            # Encrypt input fields
            if len(args) > 0 and isinstance(args[0], dict):
                args = (self.provider.encrypt_fields(args[0], self.fields),) + args[1:]
            
            # Call function
            result = func(*args, **kwargs)
            
            # Decrypt output fields if result is a dict
            if isinstance(result, dict):
                result = self.provider.decrypt_fields(result, self.fields)
            
            return result
        
        return wrapper


@dataclass
class EncryptedField:
    """
    Descriptor for encrypted fields in dataclasses.
    """
    field_name: str
    provider: Optional[EncryptionProvider] = None
    
    def __post_init__(self):
        if self.provider is None:
            self.provider = EncryptionProvider()
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        # Get encrypted value
        encrypted_value = getattr(obj, f"_{self.field_name}_encrypted", None)
        if encrypted_value is None:
            return getattr(obj, f"_{self.field_name}", None)
        
        # Decrypt
        return self.provider.decrypt(encrypted_value)
    
    def __set__(self, obj, value):
        if value is None:
            setattr(obj, f"_{self.field_name}", None)
            setattr(obj, f"_{self.field_name}_encrypted", None)
        else:
            # Encrypt and store
            encrypted = self.provider.encrypt(value)
            setattr(obj, f"_{self.field_name}", None)
            setattr(obj, f"_{self.field_name}_encrypted", encrypted)


# Global encryption provider
_encryption_provider: Optional[EncryptionProvider] = None


def init_encryption_provider(key_manager: Optional[KeyManager] = None) -> EncryptionProvider:
    """Initialize the global encryption provider."""
    global _encryption_provider
    _encryption_provider = EncryptionProvider(key_manager)
    return _encryption_provider


def get_encryption_provider() -> EncryptionProvider:
    """Get the global encryption provider."""
    if _encryption_provider is None:
        return init_encryption_provider()
    return _encryption_provider