"""
Comprehensive tests for encryption and key management.

Tests cover:
- Field-level encryption
- Key rotation
- Encryption provider backends
- Data integrity
- Performance impact
- Key security
"""
import pytest
import json
import os
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import secrets
import base64

from src.security.encryption import (
    EncryptionProvider, EncryptedData, KeyManager,
    init_encryption_provider, get_encryption_key,
    EncryptionError
)
from src.utils.secrets_manager import get_secrets_manager


class TestEncryptionProvider:
    """Test encryption provider functionality."""
    
    @pytest.fixture
    def encryption_provider(self):
        """Create encryption provider for testing."""
        # Use a fixed key for testing
        with patch('src.security.encryption.get_encryption_key') as mock_key:
            mock_key.return_value = base64.urlsafe_b64encode(secrets.token_bytes(32))
            provider = EncryptionProvider()
            return provider
    
    def test_encrypt_decrypt_string(self, encryption_provider):
        """Test basic string encryption and decryption."""
        plaintext = "This is sensitive data that needs encryption"
        
        # Encrypt
        encrypted = encryption_provider.encrypt(plaintext)
        
        assert isinstance(encrypted, EncryptedData)
        assert encrypted.ciphertext != plaintext
        assert encrypted.ciphertext != ""
        assert encrypted.nonce is not None
        assert encrypted.tag is not None
        assert encrypted.key_version is not None
        
        # Decrypt
        decrypted = encryption_provider.decrypt(encrypted, return_type=str)
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_dict(self, encryption_provider):
        """Test dictionary encryption and decryption."""
        data = {
            "user_id": "12345",
            "ssn": "123-45-6789",
            "credit_card": "4111-1111-1111-1111",
            "preferences": {
                "theme": "dark",
                "language": "en"
            }
        }
        
        # Encrypt
        encrypted = encryption_provider.encrypt(data)
        
        # Verify it's encrypted
        assert isinstance(encrypted, EncryptedData)
        assert encrypted.ciphertext != json.dumps(data)
        
        # Decrypt
        decrypted = encryption_provider.decrypt(encrypted, return_type=dict)
        assert decrypted == data
        assert decrypted["preferences"]["theme"] == "dark"
    
    def test_encrypt_empty_data(self, encryption_provider):
        """Test encryption of empty data."""
        # Empty string
        encrypted = encryption_provider.encrypt("")
        decrypted = encryption_provider.decrypt(encrypted, return_type=str)
        assert decrypted == ""
        
        # Empty dict
        encrypted = encryption_provider.encrypt({})
        decrypted = encryption_provider.decrypt(encrypted, return_type=dict)
        assert decrypted == {}
    
    def test_encrypt_large_data(self, encryption_provider):
        """Test encryption of large data."""
        # Create large data (1MB)
        large_data = "A" * (1024 * 1024)
        
        # Measure encryption time
        start_time = time.time()
        encrypted = encryption_provider.encrypt(large_data)
        encryption_time = time.time() - start_time
        
        # Measure decryption time
        start_time = time.time()
        decrypted = encryption_provider.decrypt(encrypted, return_type=str)
        decryption_time = time.time() - start_time
        
        assert decrypted == large_data
        
        # Performance check - should be reasonably fast
        assert encryption_time < 1.0  # Less than 1 second for 1MB
        assert decryption_time < 1.0
    
    def test_tampered_ciphertext(self, encryption_provider):
        """Test that tampered ciphertext is detected."""
        plaintext = "Original data"
        encrypted = encryption_provider.encrypt(plaintext)
        
        # Tamper with ciphertext
        tampered = EncryptedData(
            ciphertext=encrypted.ciphertext[:-4] + "XXXX",  # Change last 4 chars
            nonce=encrypted.nonce,
            tag=encrypted.tag,
            key_version=encrypted.key_version
        )
        
        # Decryption should fail
        with pytest.raises(EncryptionError) as exc:
            encryption_provider.decrypt(tampered)
        
        assert "authentication" in str(exc.value).lower() or "verify" in str(exc.value).lower()
    
    def test_tampered_tag(self, encryption_provider):
        """Test that tampered authentication tag is detected."""
        plaintext = "Original data"
        encrypted = encryption_provider.encrypt(plaintext)
        
        # Tamper with tag
        tampered = EncryptedData(
            ciphertext=encrypted.ciphertext,
            nonce=encrypted.nonce,
            tag=base64.b64encode(b"fake_tag").decode(),
            key_version=encrypted.key_version
        )
        
        # Decryption should fail
        with pytest.raises(EncryptionError):
            encryption_provider.decrypt(tampered)
    
    def test_wrong_nonce(self, encryption_provider):
        """Test that wrong nonce prevents decryption."""
        plaintext = "Original data"
        encrypted = encryption_provider.encrypt(plaintext)
        
        # Use different nonce
        wrong_nonce = EncryptedData(
            ciphertext=encrypted.ciphertext,
            nonce=base64.b64encode(secrets.token_bytes(12)).decode(),
            tag=encrypted.tag,
            key_version=encrypted.key_version
        )
        
        # Decryption should fail
        with pytest.raises(EncryptionError):
            encryption_provider.decrypt(wrong_nonce)
    
    def test_deterministic_encryption_disabled(self, encryption_provider):
        """Test that encryption is non-deterministic (different each time)."""
        plaintext = "Same data encrypted multiple times"
        
        # Encrypt same data multiple times
        encrypted1 = encryption_provider.encrypt(plaintext)
        encrypted2 = encryption_provider.encrypt(plaintext)
        encrypted3 = encryption_provider.encrypt(plaintext)
        
        # All ciphertexts should be different
        assert encrypted1.ciphertext != encrypted2.ciphertext
        assert encrypted2.ciphertext != encrypted3.ciphertext
        assert encrypted1.ciphertext != encrypted3.ciphertext
        
        # But all should decrypt to same plaintext
        assert encryption_provider.decrypt(encrypted1) == plaintext
        assert encryption_provider.decrypt(encrypted2) == plaintext
        assert encryption_provider.decrypt(encrypted3) == plaintext


class TestFieldLevelEncryption:
    """Test field-level encryption functionality."""
    
    @pytest.fixture
    def encryption_provider(self):
        """Create encryption provider."""
        with patch('src.security.encryption.get_encryption_key') as mock_key:
            mock_key.return_value = base64.urlsafe_b64encode(secrets.token_bytes(32))
            return EncryptionProvider()
    
    def test_encrypt_specific_fields(self, encryption_provider):
        """Test encrypting specific fields in a document."""
        document = {
            "id": "12345",
            "name": "John Doe",
            "email": "john@example.com",
            "ssn": "123-45-6789",
            "salary": 75000,
            "department": "Engineering"
        }
        
        # Encrypt sensitive fields
        encrypted_doc = encryption_provider.encrypt_fields(
            document,
            fields_to_encrypt=["ssn", "salary"]
        )
        
        # Non-sensitive fields should be unchanged
        assert encrypted_doc["id"] == "12345"
        assert encrypted_doc["name"] == "John Doe"
        assert encrypted_doc["email"] == "john@example.com"
        assert encrypted_doc["department"] == "Engineering"
        
        # Sensitive fields should be encrypted
        assert isinstance(encrypted_doc["ssn"], dict)
        assert "ciphertext" in encrypted_doc["ssn"]
        assert encrypted_doc["ssn_encrypted"] is True
        
        assert isinstance(encrypted_doc["salary"], dict)
        assert "ciphertext" in encrypted_doc["salary"]
        assert encrypted_doc["salary_encrypted"] is True
    
    def test_decrypt_specific_fields(self, encryption_provider):
        """Test decrypting specific fields."""
        document = {
            "id": "12345",
            "name": "John Doe",
            "ssn": "123-45-6789",
            "salary": 75000
        }
        
        # Encrypt fields
        encrypted_doc = encryption_provider.encrypt_fields(
            document,
            fields_to_encrypt=["ssn", "salary"]
        )
        
        # Decrypt fields
        decrypted_doc = encryption_provider.decrypt_fields(
            encrypted_doc,
            fields_to_decrypt=["ssn", "salary"]
        )
        
        # Should match original
        assert decrypted_doc == document
        assert "ssn_encrypted" not in decrypted_doc
        assert "salary_encrypted" not in decrypted_doc
    
    def test_nested_field_encryption(self, encryption_provider):
        """Test encryption of nested fields."""
        document = {
            "id": "user123",
            "profile": {
                "name": "John Doe",
                "personal": {
                    "ssn": "123-45-6789",
                    "dob": "1990-01-01"
                }
            },
            "preferences": {
                "theme": "dark",
                "payment": {
                    "card_number": "4111-1111-1111-1111",
                    "cvv": "123"
                }
            }
        }
        
        # Encrypt nested fields
        encrypted_doc = encryption_provider.encrypt_fields(
            document,
            fields_to_encrypt=[
                "profile.personal.ssn",
                "preferences.payment.card_number",
                "preferences.payment.cvv"
            ]
        )
        
        # Check encryption
        assert isinstance(encrypted_doc["profile"]["personal"]["ssn"], dict)
        assert isinstance(encrypted_doc["preferences"]["payment"]["card_number"], dict)
        assert isinstance(encrypted_doc["preferences"]["payment"]["cvv"], dict)
        
        # Other fields unchanged
        assert encrypted_doc["profile"]["name"] == "John Doe"
        assert encrypted_doc["preferences"]["theme"] == "dark"
    
    def test_partial_field_decryption(self, encryption_provider):
        """Test decrypting only some fields while leaving others encrypted."""
        document = {
            "id": "12345",
            "public_data": "visible",
            "private_data": "hidden",
            "secret_data": "top secret"
        }
        
        # Encrypt multiple fields
        encrypted_doc = encryption_provider.encrypt_fields(
            document,
            fields_to_encrypt=["private_data", "secret_data"]
        )
        
        # Decrypt only one field
        partially_decrypted = encryption_provider.decrypt_fields(
            encrypted_doc.copy(),
            fields_to_decrypt=["private_data"]
        )
        
        # private_data should be decrypted
        assert partially_decrypted["private_data"] == "hidden"
        assert "private_data_encrypted" not in partially_decrypted
        
        # secret_data should still be encrypted
        assert isinstance(partially_decrypted["secret_data"], dict)
        assert "ciphertext" in partially_decrypted["secret_data"]
        assert partially_decrypted["secret_data_encrypted"] is True


class TestKeyRotation:
    """Test encryption key rotation."""
    
    @pytest.fixture
    def key_manager(self):
        """Create key manager for testing."""
        with patch('src.security.encryption.get_secrets_manager') as mock_secrets:
            mock_secrets.return_value.get_secret.return_value = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            return KeyManager()
    
    def test_key_versioning(self, key_manager):
        """Test that keys have versions."""
        key1 = key_manager.get_current_key()
        version1 = key_manager.current_key_version
        
        # Simulate key rotation
        key_manager._rotate_key()
        
        key2 = key_manager.get_current_key()
        version2 = key_manager.current_key_version
        
        assert key1 != key2
        assert version2 > version1
    
    def test_decrypt_with_old_key(self, key_manager):
        """Test decryption with old key versions."""
        provider = EncryptionProvider(key_manager=key_manager)
        
        # Encrypt with current key
        plaintext = "Data encrypted with key v1"
        encrypted_v1 = provider.encrypt(plaintext)
        
        # Rotate key
        key_manager._rotate_key()
        
        # Should still decrypt data encrypted with old key
        decrypted = provider.decrypt(encrypted_v1)
        assert decrypted == plaintext
    
    def test_multiple_key_rotations(self, key_manager):
        """Test multiple key rotations."""
        provider = EncryptionProvider(key_manager=key_manager)
        
        encrypted_data = []
        
        # Encrypt data with different key versions
        for i in range(5):
            data = f"Data encrypted with key v{i+1}"
            encrypted = provider.encrypt(data)
            encrypted_data.append((data, encrypted))
            
            # Rotate key
            key_manager._rotate_key()
        
        # All data should still be decryptable
        for original, encrypted in encrypted_data:
            decrypted = provider.decrypt(encrypted)
            assert decrypted == original
    
    def test_key_rotation_metadata(self, key_manager):
        """Test key rotation metadata tracking."""
        initial_version = key_manager.current_key_version
        
        # Perform rotation
        rotation_time = datetime.utcnow()
        key_manager._rotate_key()
        
        # Check metadata
        assert key_manager.current_key_version == initial_version + 1
        assert key_manager.last_rotation_time is not None
        assert (datetime.utcnow() - key_manager.last_rotation_time).total_seconds() < 1
    
    def test_key_derivation(self, key_manager):
        """Test that keys are properly derived."""
        # Keys should be derived from master key with salt
        key1 = key_manager.get_key_for_purpose("encryption")
        key2 = key_manager.get_key_for_purpose("signing")
        
        # Different purposes should yield different keys
        assert key1 != key2
        
        # Same purpose should yield same key
        key1_again = key_manager.get_key_for_purpose("encryption")
        assert key1 == key1_again


class TestEncryptionProviderBackends:
    """Test different encryption provider backends."""
    
    def test_local_key_provider(self):
        """Test local key storage provider."""
        with patch.dict(os.environ, {"ENCRYPTION_KEY_PROVIDER": "local"}):
            provider = init_encryption_provider()
            
            # Should work with local keys
            data = "Test data"
            encrypted = provider.encrypt(data)
            decrypted = provider.decrypt(encrypted)
            assert decrypted == data
    
    def test_aws_kms_provider(self):
        """Test AWS KMS provider."""
        with patch.dict(os.environ, {"ENCRYPTION_KEY_PROVIDER": "aws_kms"}):
            with patch('boto3.client') as mock_boto:
                # Mock KMS client
                mock_kms = MagicMock()
                mock_kms.generate_data_key.return_value = {
                    'Plaintext': secrets.token_bytes(32),
                    'CiphertextBlob': b'encrypted_key'
                }
                mock_kms.decrypt.return_value = {
                    'Plaintext': secrets.token_bytes(32)
                }
                mock_boto.return_value = mock_kms
                
                provider = init_encryption_provider()
                
                # Should use KMS for key management
                data = "Test data with KMS"
                encrypted = provider.encrypt(data)
                assert encrypted.metadata.get("key_provider") == "aws_kms"
    
    def test_vault_provider(self):
        """Test HashiCorp Vault provider."""
        with patch.dict(os.environ, {"ENCRYPTION_KEY_PROVIDER": "vault"}):
            with patch('hvac.Client') as mock_hvac:
                # Mock Vault client
                mock_vault = MagicMock()
                mock_vault.is_authenticated.return_value = True
                mock_vault.secrets.transit.generate_data_key.return_value = {
                    'data': {
                        'plaintext': base64.b64encode(secrets.token_bytes(32)).decode()
                    }
                }
                mock_hvac.return_value = mock_vault
                
                provider = init_encryption_provider()
                
                # Should use Vault for encryption
                data = "Test data with Vault"
                encrypted = provider.encrypt(data)
                assert encrypted.metadata.get("key_provider") == "vault"


class TestDataIntegrity:
    """Test data integrity features."""
    
    @pytest.fixture
    def encryption_provider(self):
        """Create encryption provider."""
        return init_encryption_provider()
    
    def test_checksum_verification(self, encryption_provider):
        """Test that data integrity is verified."""
        data = {"important": "data", "value": 12345}
        
        # Encrypt with checksum
        encrypted = encryption_provider.encrypt(data)
        
        # Verify checksum is included
        assert encrypted.checksum is not None
        
        # Tamper with data after encryption
        # In a real scenario, this would detect corruption
        tampered = EncryptedData(
            ciphertext=encrypted.ciphertext[:-1] + "X",  # Change last char
            nonce=encrypted.nonce,
            tag=encrypted.tag,
            key_version=encrypted.key_version,
            checksum=encrypted.checksum
        )
        
        # Should detect tampering
        with pytest.raises(EncryptionError):
            encryption_provider.decrypt(tampered)
    
    def test_metadata_integrity(self, encryption_provider):
        """Test that metadata is protected."""
        data = "Sensitive data"
        metadata = {"owner": "user123", "classification": "confidential"}
        
        # Encrypt with metadata
        encrypted = encryption_provider.encrypt(data, metadata=metadata)
        
        # Metadata should be included but protected
        assert encrypted.metadata == metadata
        
        # Tamper with metadata
        encrypted.metadata["classification"] = "public"
        
        # Should detect metadata tampering
        with pytest.raises(EncryptionError) as exc:
            encryption_provider.decrypt(encrypted)
        
        assert "metadata" in str(exc.value).lower() or "integrity" in str(exc.value).lower()


class TestEncryptionPerformance:
    """Test encryption performance characteristics."""
    
    @pytest.fixture
    def encryption_provider(self):
        """Create encryption provider."""
        return init_encryption_provider()
    
    def test_bulk_encryption_performance(self, encryption_provider):
        """Test performance with bulk encryption."""
        num_records = 1000
        record_size = 1024  # 1KB per record
        
        records = [
            {"id": i, "data": "X" * record_size}
            for i in range(num_records)
        ]
        
        # Measure encryption time
        start_time = time.time()
        encrypted_records = [
            encryption_provider.encrypt(record)
            for record in records
        ]
        encryption_time = time.time() - start_time
        
        # Measure decryption time
        start_time = time.time()
        decrypted_records = [
            encryption_provider.decrypt(encrypted, return_type=dict)
            for encrypted in encrypted_records
        ]
        decryption_time = time.time() - start_time
        
        # Performance assertions
        avg_encryption_time = encryption_time / num_records
        avg_decryption_time = decryption_time / num_records
        
        # Should be fast enough for production use
        assert avg_encryption_time < 0.01  # Less than 10ms per record
        assert avg_decryption_time < 0.01
        
        # Verify correctness
        for i, decrypted in enumerate(decrypted_records):
            assert decrypted == records[i]
    
    def test_parallel_encryption(self, encryption_provider):
        """Test thread-safe parallel encryption."""
        import concurrent.futures
        
        def encrypt_decrypt(data):
            encrypted = encryption_provider.encrypt(data)
            decrypted = encryption_provider.decrypt(encrypted)
            return data == decrypted
        
        # Run encryption/decryption in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(encrypt_decrypt, f"Data item {i}")
                for i in range(100)
            ]
            
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All operations should succeed
        assert all(results)
        assert len(results) == 100


class TestKeySecurityMeasures:
    """Test key security measures."""
    
    def test_key_not_in_logs(self):
        """Test that encryption keys are not logged."""
        import logging
        
        # Set up test logger
        test_logger = logging.getLogger("test_encryption")
        handler = logging.StreamHandler()
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        
        # Create provider with logging
        with patch('src.security.encryption.logger') as mock_logger:
            provider = init_encryption_provider()
            
            # Perform operations
            encrypted = provider.encrypt("test data")
            
            # Check logs don't contain key material
            for call in mock_logger.debug.call_args_list:
                log_message = str(call)
                assert "key" not in log_message.lower() or "redacted" in log_message.lower()
    
    def test_key_memory_cleanup(self):
        """Test that keys are cleaned from memory."""
        # This is a placeholder for actual memory cleanup testing
        # In production, would use secure memory allocation
        pass
    
    def test_key_permissions(self):
        """Test that key files have proper permissions."""
        # This would test file permissions on key storage
        # Placeholder for actual implementation
        pass


class TestEncryptionCompliance:
    """Test encryption compliance requirements."""
    
    def test_fips_compliance(self):
        """Test FIPS 140-2 compliance."""
        # Verify using approved algorithms
        provider = init_encryption_provider()
        
        # Should use AES-256-GCM (FIPS approved)
        assert provider.algorithm == "AES-256-GCM"
        assert provider.key_size == 256
    
    def test_key_length_requirements(self):
        """Test minimum key length requirements."""
        # Keys should be at least 256 bits
        key_manager = KeyManager()
        key = key_manager.get_current_key()
        
        assert len(key) >= 32  # 256 bits
    
    def test_encryption_mandatory_fields(self):
        """Test that certain fields are always encrypted."""
        provider = init_encryption_provider()
        
        # List of fields that must always be encrypted
        mandatory_encrypted_fields = [
            "ssn",
            "credit_card",
            "bank_account",
            "password",
            "api_key"
        ]
        
        document = {
            "name": "John Doe",
            "ssn": "123-45-6789",
            "credit_card": "4111-1111-1111-1111"
        }
        
        # These fields should be automatically encrypted
        encrypted_doc = provider.auto_encrypt_document(
            document,
            mandatory_fields=mandatory_encrypted_fields
        )
        
        assert isinstance(encrypted_doc["ssn"], dict)
        assert isinstance(encrypted_doc["credit_card"], dict)
        assert encrypted_doc["name"] == "John Doe"  # Not in mandatory list


if __name__ == "__main__":
    pytest.main([__file__, "-v"])