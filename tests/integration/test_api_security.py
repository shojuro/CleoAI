"""
Integration tests for API security features.

Tests cover:
- End-to-end authentication flow
- Authorization across endpoints
- Security headers and CORS
- Rate limiting in practice
- Session management
- API key usage
"""
import pytest
import time
import json
import jwt
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from src.api.api_router import create_api_app
from src.api.auth import create_jwt_token, hash_password
from src.database.models import User
from src.security.data_isolation import SecurityContext, IsolationLevel


class TestAPIAuthentication:
    """Test API authentication flows."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_api_app(debug=True)
        return TestClient(app)
    
    @pytest.fixture
    def valid_token(self):
        """Create valid JWT token."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret-key"
            return create_jwt_token(
                user_id="test_user",
                roles=["user"]
            )
    
    @pytest.fixture
    def admin_token(self):
        """Create admin JWT token."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret-key"
            return create_jwt_token(
                user_id="admin_user",
                roles=["user", "admin"]
            )
    
    def test_unauthenticated_access_blocked(self, client):
        """Test that unauthenticated access is blocked."""
        # Try to access protected endpoint without auth
        response = client.post(
            "/graphql",
            json={"query": "{ currentUser { id } }"}
        )
        
        assert response.status_code == 401
        assert "unauthorized" in response.json()["detail"].lower()
    
    def test_authenticated_access_allowed(self, client, valid_token):
        """Test that authenticated access is allowed."""
        response = client.post(
            "/graphql",
            json={"query": "{ __typename }"},
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        
        assert response.status_code == 200
    
    def test_expired_token_rejected(self, client):
        """Test that expired tokens are rejected."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret-key"
            expired_token = create_jwt_token(
                user_id="test_user",
                roles=["user"],
                expires_minutes=-1  # Already expired
            )
        
        response = client.post(
            "/graphql",
            json={"query": "{ __typename }"},
            headers={"Authorization": f"Bearer {expired_token}"}
        )
        
        assert response.status_code == 401
        assert "expired" in response.json()["detail"].lower()
    
    def test_malformed_token_rejected(self, client):
        """Test that malformed tokens are rejected."""
        malformed_tokens = [
            "not.a.jwt",
            "Bearer malformed",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",  # Incomplete JWT
            "null",
            ""
        ]
        
        for token in malformed_tokens:
            response = client.post(
                "/graphql",
                json={"query": "{ __typename }"},
                headers={"Authorization": f"Bearer {token}"}
            )
            
            assert response.status_code == 401
    
    def test_token_with_invalid_signature(self, client, valid_token):
        """Test that tokens with invalid signatures are rejected."""
        # Decode token, modify payload, re-encode with wrong secret
        parts = valid_token.split('.')
        
        # Decode payload
        payload = json.loads(
            base64.urlsafe_b64decode(parts[1] + '=' * (4 - len(parts[1]) % 4))
        )
        
        # Modify payload
        payload["roles"] = ["admin"]  # Try to escalate privileges
        
        # Re-encode with wrong secret
        fake_token = jwt.encode(payload, "wrong-secret", algorithm="HS256")
        
        response = client.post(
            "/graphql",
            json={"query": "{ __typename }"},
            headers={"Authorization": f"Bearer {fake_token}"}
        )
        
        assert response.status_code == 401
    
    def test_api_key_authentication(self, client):
        """Test API key authentication."""
        with patch('src.api.auth.validate_api_key') as mock_validate:
            mock_validate.return_value = "test-service"
            
            response = client.get(
                "/health",
                headers={"X-API-Key": "valid-api-key"}
            )
            
            assert response.status_code == 200
            mock_validate.assert_called_with("valid-api-key")
    
    def test_invalid_api_key_rejected(self, client):
        """Test that invalid API keys are rejected."""
        with patch('src.api.auth.validate_api_key') as mock_validate:
            mock_validate.return_value = None
            
            response = client.post(
                "/graphql",
                json={"query": "{ __typename }"},
                headers={"X-API-Key": "invalid-key"}
            )
            
            assert response.status_code == 401


class TestAPIAuthorization:
    """Test API authorization and access control."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_api_app(debug=True)
        return TestClient(app)
    
    @pytest.fixture
    def user_token(self):
        """Create regular user token."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret-key"
            return create_jwt_token(
                user_id="regular_user",
                roles=["user"]
            )
    
    @pytest.fixture
    def admin_token(self):
        """Create admin token."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret-key"
            return create_jwt_token(
                user_id="admin_user",
                roles=["user", "admin"]
            )
    
    def test_role_based_access_control(self, client, user_token, admin_token):
        """Test role-based access control."""
        # Admin-only endpoint
        admin_query = """
        mutation {
            deleteUser(userId: "some_user") {
                success
            }
        }
        """
        
        # Regular user should be denied
        response = client.post(
            "/graphql",
            json={"query": admin_query},
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "errors" in data
        assert "permission" in str(data["errors"]).lower()
        
        # Admin should be allowed
        with patch('src.api.graphql.delete_user_resolver') as mock_resolver:
            mock_resolver.return_value = {"success": True}
            
            response = client.post(
                "/graphql",
                json={"query": admin_query},
                headers={"Authorization": f"Bearer {admin_token}"}
            )
            
            # Would succeed if resolver exists
    
    def test_resource_ownership_check(self, client, user_token):
        """Test that users can only access their own resources."""
        # Query for user's own data
        own_data_query = """
        query {
            conversation(id: "conv_123") {
                id
                messages {
                    content
                }
            }
        }
        """
        
        with patch('src.api.graphql.get_conversation_resolver') as mock_resolver:
            # Mock returns conversation owned by user
            mock_resolver.return_value = {
                "id": "conv_123",
                "user_id": "regular_user",
                "messages": []
            }
            
            response = client.post(
                "/graphql",
                json={"query": own_data_query},
                headers={"Authorization": f"Bearer {user_token}"}
            )
            
            # Should succeed for own data
            assert response.status_code == 200
    
    def test_cross_user_access_denied(self, client, user_token):
        """Test that cross-user access is denied."""
        # Try to access another user's data
        other_user_query = """
        query {
            userConversations(userId: "other_user") {
                id
            }
        }
        """
        
        response = client.post(
            "/graphql",
            json={"query": other_user_query},
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        # Should either return empty or error
        if "errors" in data:
            assert "permission" in str(data["errors"]).lower() or "unauthorized" in str(data["errors"]).lower()
        else:
            assert data["data"]["userConversations"] == []
    
    def test_field_level_authorization(self, client, user_token, admin_token):
        """Test field-level authorization in GraphQL."""
        sensitive_query = """
        query {
            user(id: "test_user") {
                id
                email
                roles
                apiKeys {  # Admin only field
                    key
                    createdAt
                }
            }
        }
        """
        
        # Regular user should not see apiKeys
        response = client.post(
            "/graphql",
            json={"query": sensitive_query},
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        data = response.json()
        if "errors" not in data:
            user_data = data["data"]["user"]
            assert "apiKeys" not in user_data or user_data["apiKeys"] is None


class TestSecurityHeaders:
    """Test security headers and CORS."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_api_app(debug=False)  # Production mode
        return TestClient(app)
    
    def test_security_headers_present(self, client):
        """Test that all security headers are present."""
        response = client.get("/health")
        
        # Required security headers
        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": lambda v: v is not None,
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
        for header, expected in required_headers.items():
            assert header in response.headers
            if callable(expected):
                assert expected(response.headers[header])
            else:
                assert response.headers[header] == expected
    
    def test_cors_configuration(self, client):
        """Test CORS configuration."""
        # Allowed origin
        response = client.options(
            "/graphql",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,authorization"
            }
        )
        
        assert response.status_code == 200
        assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        assert "POST" in response.headers["Access-Control-Allow-Methods"]
        assert "authorization" in response.headers["Access-Control-Allow-Headers"].lower()
        
        # Disallowed origin
        response = client.options(
            "/graphql",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        assert "Access-Control-Allow-Origin" not in response.headers or \
               response.headers["Access-Control-Allow-Origin"] != "http://evil.com"
    
    def test_no_sensitive_headers_leaked(self, client):
        """Test that sensitive headers are not leaked."""
        response = client.get("/health")
        
        # These should never be present
        forbidden_headers = [
            "Server",
            "X-Powered-By",
            "X-AspNet-Version",
            "X-AspNetMvc-Version",
            "X-Drupal-Cache",
            "X-Generator"
        ]
        
        for header in forbidden_headers:
            assert header not in response.headers


class TestRateLimiting:
    """Test rate limiting implementation."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_api_app(debug=True)
        return TestClient(app)
    
    @pytest.fixture
    def token(self):
        """Create test token."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret-key"
            return create_jwt_token(user_id="test_user", roles=["user"])
    
    def test_rate_limit_enforcement(self, client, token):
        """Test that rate limits are enforced."""
        # Configure tight rate limit for testing
        with patch('src.api.auth.RATE_LIMIT_PER_MINUTE', 5):
            headers = {"Authorization": f"Bearer {token}"}
            
            # Make requests up to limit
            for i in range(5):
                response = client.get("/health", headers=headers)
                assert response.status_code == 200
            
            # Next request should be rate limited
            response = client.get("/health", headers=headers)
            assert response.status_code == 429
            assert "rate limit" in response.json()["detail"].lower()
    
    def test_rate_limit_headers(self, client, token):
        """Test that rate limit headers are included."""
        response = client.get(
            "/health",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Should include rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
    
    def test_rate_limit_per_user(self, client):
        """Test that rate limits are per-user."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret-key"
            
            token1 = create_jwt_token(user_id="user1", roles=["user"])
            token2 = create_jwt_token(user_id="user2", roles=["user"])
        
        with patch('src.api.auth.RATE_LIMIT_PER_MINUTE', 2):
            # User 1 makes 2 requests (hits limit)
            for _ in range(2):
                response = client.get(
                    "/health",
                    headers={"Authorization": f"Bearer {token1}"}
                )
                assert response.status_code == 200
            
            # User 1's next request is blocked
            response = client.get(
                "/health",
                headers={"Authorization": f"Bearer {token1}"}
            )
            assert response.status_code == 429
            
            # User 2 can still make requests
            response = client.get(
                "/health",
                headers={"Authorization": f"Bearer {token2}"}
            )
            assert response.status_code == 200


class TestSessionSecurity:
    """Test session management security."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_api_app(debug=True)
        return TestClient(app)
    
    def test_session_token_uniqueness(self):
        """Test that session tokens are unique."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret-key"
            
            tokens = set()
            for _ in range(100):
                token = create_jwt_token(
                    user_id="same_user",
                    roles=["user"]
                )
                tokens.add(token)
            
            # All tokens should be unique
            assert len(tokens) == 100
    
    def test_concurrent_session_handling(self, client):
        """Test handling of concurrent sessions."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret-key"
            
            # Create multiple tokens for same user
            tokens = [
                create_jwt_token(user_id="test_user", roles=["user"])
                for _ in range(3)
            ]
        
        # All tokens should work independently
        for token in tokens:
            response = client.get(
                "/health",
                headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 200
    
    def test_token_refresh_flow(self, client):
        """Test token refresh flow."""
        # Would test actual refresh endpoint
        # This is a placeholder for implementation
        pass


class TestAPIInputValidation:
    """Test API input validation and sanitization."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_api_app(debug=True)
        return TestClient(app)
    
    @pytest.fixture
    def token(self):
        """Create test token."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret-key"
            return create_jwt_token(user_id="test_user", roles=["user"])
    
    def test_graphql_query_validation(self, client, token):
        """Test GraphQL query validation."""
        malicious_queries = [
            # Deeply nested query (complexity attack)
            """
            query {
                user {
                    conversations {
                        messages {
                            user {
                                conversations {
                                    messages {
                                        user {
                                            id
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            """,
            # Invalid query
            "{ invalid query syntax }",
            # Introspection (might be disabled in production)
            "{ __schema { types { name } } }"
        ]
        
        for query in malicious_queries:
            response = client.post(
                "/graphql",
                json={"query": query},
                headers={"Authorization": f"Bearer {token}"}
            )
            
            # Should either reject or handle safely
            assert response.status_code in [200, 400]
            if response.status_code == 200:
                assert "errors" in response.json()
    
    def test_request_size_limits(self, client, token):
        """Test request size limits."""
        # Create large request
        large_query = "{ " + "a" * (1024 * 1024) + " }"  # 1MB query
        
        response = client.post(
            "/graphql",
            json={"query": large_query},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Should reject oversized requests
        assert response.status_code in [413, 400]
    
    def test_injection_prevention(self, client, token):
        """Test injection attack prevention."""
        injection_attempts = [
            {
                "query": 'mutation { createUser(name: "\'; DROP TABLE users; --") { id } }'
            },
            {
                "query": 'query { user(id: "1 OR 1=1") { email } }'
            },
            {
                "query": 'mutation { updateProfile(bio: "<script>alert(1)</script>") { success } }'
            }
        ]
        
        for attempt in injection_attempts:
            response = client.post(
                "/graphql",
                json=attempt,
                headers={"Authorization": f"Bearer {token}"}
            )
            
            # Should handle safely (sanitize or reject)
            if response.status_code == 200:
                # If accepted, check that it's sanitized
                data = response.json()
                if "data" in data and data["data"]:
                    # Verify no actual SQL/XSS in response
                    response_str = json.dumps(data)
                    assert "<script>" not in response_str
                    assert "DROP TABLE" not in response_str


class TestAPIKeySecurity:
    """Test API key security measures."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_api_app(debug=True)
        return TestClient(app)
    
    def test_api_key_scoping(self, client):
        """Test that API keys have limited scope."""
        with patch('src.api.auth.validate_api_key') as mock_validate:
            # Service API key
            mock_validate.return_value = "metrics-service"
            
            # Should be able to access metrics endpoint
            response = client.get(
                "/metrics",
                headers={"X-API-Key": "metrics-key"}
            )
            # Would succeed if metrics endpoint exists
            
            # Should not be able to access user data
            response = client.post(
                "/graphql",
                json={"query": "{ users { id email } }"},
                headers={"X-API-Key": "metrics-key"}
            )
            
            # Should be denied or limited
            assert response.status_code in [401, 403] or \
                   ("errors" in response.json() and "permission" in str(response.json()["errors"]).lower())
    
    def test_api_key_rotation_handling(self, client):
        """Test API key rotation."""
        # Would test actual key rotation mechanism
        # Placeholder for implementation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])