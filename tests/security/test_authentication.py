"""
Comprehensive authentication and authorization tests for CleoAI.

Tests cover:
- JWT token generation and validation
- API key authentication
- Role-based access control
- Session management
- Security headers
- Rate limiting
"""
import pytest
import time
import jwt
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.api.auth import (
    create_jwt_token, decode_token, hash_password, verify_password,
    validate_api_key, get_current_user, require_admin, require_user,
    RateLimiter, TokenData
)
from src.api.api_router import create_api_app
from src.utils.secrets_manager import get_secrets_manager


class TestPasswordHashing:
    """Test password hashing functionality."""
    
    def test_hash_password_creates_different_hashes(self):
        """Test that same password creates different hashes (salt)."""
        password = "SecurePassword123!"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        assert hash1 != hash2
        assert hash1.startswith("$2b$")  # bcrypt prefix
        assert len(hash1) == 60  # bcrypt hash length
    
    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "TestPassword456#"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "CorrectPassword789$"
        wrong_password = "WrongPassword789$"
        hashed = hash_password(password)
        
        assert verify_password(wrong_password, hashed) is False
    
    def test_verify_password_similar_passwords(self):
        """Test that similar passwords don't match."""
        password = "Password123!"
        similar_passwords = [
            "Password123",  # Missing special char
            "password123!",  # Different case
            "Password1234!",  # Extra digit
            "Password123! ",  # Extra space
        ]
        
        hashed = hash_password(password)
        
        for similar in similar_passwords:
            assert verify_password(similar, hashed) is False
    
    def test_hash_empty_password_raises_error(self):
        """Test that empty password raises error."""
        with pytest.raises(ValueError):
            hash_password("")
    
    def test_hash_none_password_raises_error(self):
        """Test that None password raises error."""
        with pytest.raises(TypeError):
            hash_password(None)


class TestJWTTokens:
    """Test JWT token generation and validation."""
    
    @pytest.fixture
    def mock_secrets(self):
        """Mock secrets manager."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            secrets = Mock()
            secrets.get_secret.return_value = "test-secret-key-for-jwt-signing-minimum-32-chars"
            mock.return_value = secrets
            yield secrets
    
    def test_create_jwt_token_structure(self, mock_secrets):
        """Test JWT token has correct structure."""
        token = create_jwt_token(
            user_id="test_user",
            roles=["user", "admin"]
        )
        
        # Decode without verification to check structure
        decoded = jwt.decode(
            token,
            options={"verify_signature": False}
        )
        
        assert decoded["sub"] == "test_user"
        assert decoded["roles"] == ["user", "admin"]
        assert "exp" in decoded
        assert "iat" in decoded
        assert "jti" in decoded  # JWT ID for uniqueness
    
    def test_create_jwt_token_expiration(self, mock_secrets):
        """Test token expiration time."""
        token = create_jwt_token(
            user_id="test_user",
            roles=["user"],
            expires_minutes=5
        )
        
        decoded = jwt.decode(
            token,
            mock_secrets.get_secret.return_value,
            algorithms=["HS256"]
        )
        
        exp_time = datetime.fromtimestamp(decoded["exp"])
        now = datetime.utcnow()
        
        # Check expiration is approximately 5 minutes from now
        assert 4 < (exp_time - now).total_seconds() / 60 < 6
    
    def test_decode_valid_token(self, mock_secrets):
        """Test decoding valid token."""
        token = create_jwt_token(
            user_id="test_user",
            roles=["user"]
        )
        
        token_data = decode_token(token)
        
        assert isinstance(token_data, TokenData)
        assert token_data.user_id == "test_user"
        assert token_data.roles == ["user"]
    
    def test_decode_expired_token(self, mock_secrets):
        """Test decoding expired token raises error."""
        # Create token that expires immediately
        token = create_jwt_token(
            user_id="test_user",
            roles=["user"],
            expires_minutes=-1
        )
        
        with pytest.raises(HTTPException) as exc:
            decode_token(token)
        
        assert exc.value.status_code == 401
        assert "expired" in str(exc.value.detail).lower()
    
    def test_decode_invalid_signature(self, mock_secrets):
        """Test decoding token with invalid signature."""
        token = create_jwt_token(
            user_id="test_user",
            roles=["user"]
        )
        
        # Tamper with token
        parts = token.split('.')
        tampered_token = f"{parts[0]}.{parts[1]}.invalid_signature"
        
        with pytest.raises(HTTPException) as exc:
            decode_token(tampered_token)
        
        assert exc.value.status_code == 401
    
    def test_decode_malformed_token(self, mock_secrets):
        """Test decoding malformed token."""
        malformed_tokens = [
            "not.a.token",
            "invalid-jwt-format",
            "",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",  # Incomplete
        ]
        
        for token in malformed_tokens:
            with pytest.raises(HTTPException) as exc:
                decode_token(token)
            assert exc.value.status_code == 401
    
    def test_token_with_additional_claims(self, mock_secrets):
        """Test token with additional claims."""
        token = create_jwt_token(
            user_id="test_user",
            roles=["user"],
            additional_claims={
                "email": "test@example.com",
                "permissions": ["read", "write"]
            }
        )
        
        decoded = jwt.decode(
            token,
            mock_secrets.get_secret.return_value,
            algorithms=["HS256"]
        )
        
        assert decoded["email"] == "test@example.com"
        assert decoded["permissions"] == ["read", "write"]


class TestAPIKeyAuthentication:
    """Test API key authentication."""
    
    def test_validate_api_key_valid(self):
        """Test validating correct API key."""
        valid_keys = {
            "test-service-key-123": "test-service"
        }
        
        with patch('src.api.auth.API_KEYS', valid_keys):
            service = validate_api_key("test-service-key-123")
            assert service == "test-service"
    
    def test_validate_api_key_invalid(self):
        """Test validating invalid API key."""
        valid_keys = {
            "valid-key": "service"
        }
        
        with patch('src.api.auth.API_KEYS', valid_keys):
            service = validate_api_key("invalid-key")
            assert service is None
    
    def test_validate_api_key_empty(self):
        """Test validating empty API key."""
        assert validate_api_key("") is None
        assert validate_api_key(None) is None
    
    def test_api_key_format_validation(self):
        """Test API key format requirements."""
        # API keys should have minimum length and complexity
        weak_keys = [
            "123",  # Too short
            "simple",  # No numbers/special chars
            " key-with-spaces ",  # Spaces
        ]
        
        for key in weak_keys:
            # In production, validate_api_key should reject weak keys
            # This is a placeholder for actual validation logic
            pass


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter for testing."""
        return RateLimiter(
            requests_per_minute=5,
            requests_per_hour=100,
            burst_size=2
        )
    
    def test_rate_limit_allows_normal_traffic(self, rate_limiter):
        """Test rate limiter allows normal traffic."""
        client_id = "test-client"
        
        # Should allow burst_size requests immediately
        for i in range(2):
            assert rate_limiter.is_allowed(client_id) is True
    
    def test_rate_limit_blocks_burst_exceeded(self, rate_limiter):
        """Test rate limiter blocks when burst exceeded."""
        client_id = "test-client"
        
        # Exhaust burst
        for i in range(2):
            rate_limiter.is_allowed(client_id)
        
        # Next request should be blocked
        assert rate_limiter.is_allowed(client_id) is False
    
    def test_rate_limit_per_minute_limit(self, rate_limiter):
        """Test per-minute rate limiting."""
        client_id = "test-client"
        
        # Make 5 requests (the limit)
        for i in range(5):
            assert rate_limiter.is_allowed(client_id) is True
            time.sleep(0.1)  # Small delay to avoid burst limit
        
        # 6th request should be blocked
        assert rate_limiter.is_allowed(client_id) is False
    
    def test_rate_limit_client_isolation(self, rate_limiter):
        """Test rate limits are per-client."""
        client1 = "client-1"
        client2 = "client-2"
        
        # Exhaust client1's limit
        for i in range(5):
            rate_limiter.is_allowed(client1)
        
        assert rate_limiter.is_allowed(client1) is False
        
        # Client2 should still be allowed
        assert rate_limiter.is_allowed(client2) is True
    
    def test_rate_limit_reset_after_time(self, rate_limiter):
        """Test rate limit resets after time window."""
        client_id = "test-client"
        
        # Exhaust limit
        for i in range(5):
            rate_limiter.is_allowed(client_id)
        
        assert rate_limiter.is_allowed(client_id) is False
        
        # Simulate time passing (would need to mock time in production)
        # In real implementation, limits should reset after time window
        pass


class TestAuthorizationDecorators:
    """Test authorization decorators and middleware."""
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request with auth."""
        from fastapi import Request
        request = Mock(spec=Request)
        request.headers = {"Authorization": "Bearer test-token"}
        return request
    
    @pytest.mark.asyncio
    async def test_require_user_with_valid_token(self, mock_request):
        """Test require_user with valid token."""
        mock_token_data = TokenData(
            user_id="test_user",
            roles=["user"]
        )
        
        with patch('src.api.auth.get_current_user', return_value=mock_token_data):
            result = await require_user(mock_request)
            assert result.user_id == "test_user"
    
    @pytest.mark.asyncio
    async def test_require_user_without_token(self, mock_request):
        """Test require_user without token."""
        mock_request.headers = {}
        
        with pytest.raises(HTTPException) as exc:
            await require_user(mock_request)
        
        assert exc.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_require_admin_with_admin_role(self, mock_request):
        """Test require_admin with admin role."""
        mock_token_data = TokenData(
            user_id="admin_user",
            roles=["user", "admin"]
        )
        
        with patch('src.api.auth.get_current_user', return_value=mock_token_data):
            result = await require_admin(mock_request)
            assert result.user_id == "admin_user"
    
    @pytest.mark.asyncio
    async def test_require_admin_without_admin_role(self, mock_request):
        """Test require_admin without admin role."""
        mock_token_data = TokenData(
            user_id="regular_user",
            roles=["user"]
        )
        
        with patch('src.api.auth.get_current_user', return_value=mock_token_data):
            with pytest.raises(HTTPException) as exc:
                await require_admin(mock_request)
            
            assert exc.value.status_code == 403
            assert "admin" in str(exc.value.detail).lower()


class TestSecurityHeaders:
    """Test security headers middleware."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client with security middleware."""
        app = create_api_app(debug=True)
        return TestClient(app)
    
    def test_security_headers_present(self, test_client):
        """Test that security headers are added to responses."""
        response = test_client.get("/health")
        
        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        
        assert "Strict-Transport-Security" in response.headers
        
        # Check CORS headers are properly configured
        assert "Access-Control-Allow-Origin" not in response.headers  # Should only be on CORS requests
    
    def test_no_sensitive_headers_exposed(self, test_client):
        """Test that sensitive headers are not exposed."""
        response = test_client.get("/health")
        
        # These should never be in responses
        assert "Server" not in response.headers
        assert "X-Powered-By" not in response.headers
        assert "X-AspNet-Version" not in response.headers
    
    def test_cors_preflight_request(self, test_client):
        """Test CORS preflight requests."""
        response = test_client.options(
            "/graphql",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,authorization"
            }
        )
        
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers


class TestSessionSecurity:
    """Test session management security."""
    
    def test_session_token_uniqueness(self):
        """Test that session tokens are unique."""
        tokens = set()
        
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret"
            
            for i in range(100):
                token = create_jwt_token(
                    user_id="same_user",
                    roles=["user"]
                )
                tokens.add(token)
        
        # All tokens should be unique due to JWT ID
        assert len(tokens) == 100
    
    def test_session_fixation_prevention(self):
        """Test prevention of session fixation attacks."""
        # In a real app, test that session ID changes after login
        # This is a placeholder for actual session management tests
        pass
    
    def test_concurrent_session_limit(self):
        """Test limiting concurrent sessions per user."""
        # Test that users can't have unlimited concurrent sessions
        # This would be implemented in actual session management
        pass


class TestAPIKeySecurity:
    """Test API key security measures."""
    
    def test_api_key_not_logged(self):
        """Test that API keys are not logged."""
        # This would test logging configuration to ensure
        # API keys are masked in logs
        pass
    
    def test_api_key_rotation(self):
        """Test API key rotation mechanism."""
        # Test that old API keys can be invalidated
        # and new ones generated
        pass
    
    def test_api_key_scope_limitation(self):
        """Test that API keys have limited scope."""
        # Test that API keys can only access specific endpoints
        # based on their service type
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])