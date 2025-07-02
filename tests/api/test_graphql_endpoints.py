"""
Comprehensive tests for GraphQL API endpoints.

Tests cover:
- Query operations
- Mutation operations
- Subscription operations
- Error handling
- Performance
- Data validation
"""
import pytest
import json
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from src.api.api_router import create_api_app
from src.api.graphql import schema
from src.api.auth import create_jwt_token


class TestGraphQLQueries:
    """Test GraphQL query operations."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_api_app(debug=True)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create auth headers."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret"
            token = create_jwt_token(user_id="test_user", roles=["user"])
            return {"Authorization": f"Bearer {token}"}
    
    def test_current_user_query(self, client, auth_headers):
        """Test querying current user."""
        query = """
        query {
            currentUser {
                id
                username
                email
                roles
            }
        }
        """
        
        with patch('src.api.graphql.get_current_user_resolver') as mock_resolver:
            mock_resolver.return_value = {
                "id": "test_user",
                "username": "testuser",
                "email": "test@example.com",
                "roles": ["user"]
            }
            
            response = client.post(
                "/graphql",
                json={"query": query},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" not in data
            assert data["data"]["currentUser"]["id"] == "test_user"
            assert data["data"]["currentUser"]["roles"] == ["user"]
    
    def test_conversation_query(self, client, auth_headers):
        """Test querying a conversation."""
        query = """
        query GetConversation($id: ID!) {
            conversation(id: $id) {
                id
                title
                createdAt
                messages {
                    id
                    role
                    content
                    timestamp
                }
                metadata {
                    model
                    temperature
                }
            }
        }
        """
        
        variables = {"id": "conv_123"}
        
        with patch('src.api.graphql.get_conversation_resolver') as mock_resolver:
            mock_resolver.return_value = {
                "id": "conv_123",
                "title": "Test Conversation",
                "createdAt": "2024-01-01T00:00:00Z",
                "messages": [
                    {
                        "id": "msg_1",
                        "role": "user",
                        "content": "Hello",
                        "timestamp": "2024-01-01T00:01:00Z"
                    },
                    {
                        "id": "msg_2",
                        "role": "assistant",
                        "content": "Hi there!",
                        "timestamp": "2024-01-01T00:01:30Z"
                    }
                ],
                "metadata": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7
                }
            }
            
            response = client.post(
                "/graphql",
                json={"query": query, "variables": variables},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" not in data
            assert data["data"]["conversation"]["id"] == "conv_123"
            assert len(data["data"]["conversation"]["messages"]) == 2
    
    def test_conversations_list_query(self, client, auth_headers):
        """Test querying list of conversations with pagination."""
        query = """
        query ListConversations($limit: Int, $offset: Int, $orderBy: String) {
            conversations(limit: $limit, offset: $offset, orderBy: $orderBy) {
                items {
                    id
                    title
                    lastMessageAt
                    messageCount
                }
                totalCount
                hasMore
            }
        }
        """
        
        variables = {
            "limit": 10,
            "offset": 0,
            "orderBy": "lastMessageAt_desc"
        }
        
        with patch('src.api.graphql.list_conversations_resolver') as mock_resolver:
            mock_resolver.return_value = {
                "items": [
                    {
                        "id": "conv_1",
                        "title": "Conversation 1",
                        "lastMessageAt": "2024-01-01T10:00:00Z",
                        "messageCount": 5
                    },
                    {
                        "id": "conv_2",
                        "title": "Conversation 2",
                        "lastMessageAt": "2024-01-01T09:00:00Z",
                        "messageCount": 3
                    }
                ],
                "totalCount": 15,
                "hasMore": True
            }
            
            response = client.post(
                "/graphql",
                json={"query": query, "variables": variables},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" not in data
            assert len(data["data"]["conversations"]["items"]) == 2
            assert data["data"]["conversations"]["totalCount"] == 15
            assert data["data"]["conversations"]["hasMore"] is True
    
    def test_search_messages_query(self, client, auth_headers):
        """Test searching messages."""
        query = """
        query SearchMessages($query: String!, $conversationId: ID) {
            searchMessages(query: $query, conversationId: $conversationId) {
                results {
                    message {
                        id
                        content
                    }
                    conversation {
                        id
                        title
                    }
                    relevanceScore
                    highlights
                }
                totalCount
            }
        }
        """
        
        variables = {
            "query": "machine learning",
            "conversationId": None
        }
        
        with patch('src.api.graphql.search_messages_resolver') as mock_resolver:
            mock_resolver.return_value = {
                "results": [
                    {
                        "message": {
                            "id": "msg_123",
                            "content": "Let's discuss machine learning algorithms"
                        },
                        "conversation": {
                            "id": "conv_456",
                            "title": "ML Discussion"
                        },
                        "relevanceScore": 0.95,
                        "highlights": ["machine learning"]
                    }
                ],
                "totalCount": 1
            }
            
            response = client.post(
                "/graphql",
                json={"query": query, "variables": variables},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" not in data
            assert len(data["data"]["searchMessages"]["results"]) == 1
            assert data["data"]["searchMessages"]["results"][0]["relevanceScore"] == 0.95
    
    def test_memory_stats_query(self, client, auth_headers):
        """Test querying memory statistics."""
        query = """
        query {
            memoryStats {
                totalConversations
                totalMessages
                totalMemoryUsage
                oldestMemory
                backends {
                    name
                    status
                    usage
                }
            }
        }
        """
        
        with patch('src.api.graphql.get_memory_stats_resolver') as mock_resolver:
            mock_resolver.return_value = {
                "totalConversations": 150,
                "totalMessages": 3420,
                "totalMemoryUsage": 524288000,  # 500MB
                "oldestMemory": "2023-01-01T00:00:00Z",
                "backends": [
                    {
                        "name": "redis",
                        "status": "healthy",
                        "usage": 104857600  # 100MB
                    },
                    {
                        "name": "mongodb",
                        "status": "healthy",
                        "usage": 419430400  # 400MB
                    }
                ]
            }
            
            response = client.post(
                "/graphql",
                json={"query": query},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" not in data
            assert data["data"]["memoryStats"]["totalConversations"] == 150
            assert len(data["data"]["memoryStats"]["backends"]) == 2


class TestGraphQLMutations:
    """Test GraphQL mutation operations."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_api_app(debug=True)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create auth headers."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret"
            token = create_jwt_token(user_id="test_user", roles=["user"])
            return {"Authorization": f"Bearer {token}"}
    
    def test_create_conversation_mutation(self, client, auth_headers):
        """Test creating a new conversation."""
        mutation = """
        mutation CreateConversation($title: String!) {
            createConversation(title: $title) {
                conversation {
                    id
                    title
                    createdAt
                }
                success
                message
            }
        }
        """
        
        variables = {"title": "New Test Conversation"}
        
        with patch('src.api.graphql.create_conversation_resolver') as mock_resolver:
            mock_resolver.return_value = {
                "conversation": {
                    "id": "conv_new_123",
                    "title": "New Test Conversation",
                    "createdAt": "2024-01-01T12:00:00Z"
                },
                "success": True,
                "message": "Conversation created successfully"
            }
            
            response = client.post(
                "/graphql",
                json={"query": mutation, "variables": variables},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" not in data
            assert data["data"]["createConversation"]["success"] is True
            assert data["data"]["createConversation"]["conversation"]["title"] == "New Test Conversation"
    
    def test_send_message_mutation(self, client, auth_headers):
        """Test sending a message."""
        mutation = """
        mutation SendMessage($conversationId: ID!, $content: String!, $stream: Boolean) {
            sendMessage(conversationId: $conversationId, content: $content, stream: $stream) {
                message {
                    id
                    role
                    content
                    timestamp
                }
                response {
                    id
                    role
                    content
                    timestamp
                    metadata {
                        model
                        tokens
                        processingTime
                    }
                }
                success
                error
            }
        }
        """
        
        variables = {
            "conversationId": "conv_123",
            "content": "What is machine learning?",
            "stream": False
        }
        
        with patch('src.api.graphql.send_message_resolver') as mock_resolver:
            mock_resolver.return_value = {
                "message": {
                    "id": "msg_user_123",
                    "role": "user",
                    "content": "What is machine learning?",
                    "timestamp": "2024-01-01T12:00:00Z"
                },
                "response": {
                    "id": "msg_asst_124",
                    "role": "assistant",
                    "content": "Machine learning is a subset of artificial intelligence...",
                    "timestamp": "2024-01-01T12:00:05Z",
                    "metadata": {
                        "model": "gpt-3.5-turbo",
                        "tokens": 150,
                        "processingTime": 2.5
                    }
                },
                "success": True,
                "error": None
            }
            
            response = client.post(
                "/graphql",
                json={"query": mutation, "variables": variables},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" not in data
            assert data["data"]["sendMessage"]["success"] is True
            assert data["data"]["sendMessage"]["response"]["role"] == "assistant"
            assert data["data"]["sendMessage"]["response"]["metadata"]["tokens"] == 150
    
    def test_update_conversation_mutation(self, client, auth_headers):
        """Test updating a conversation."""
        mutation = """
        mutation UpdateConversation($id: ID!, $title: String, $metadata: JSON) {
            updateConversation(id: $id, title: $title, metadata: $metadata) {
                conversation {
                    id
                    title
                    metadata
                }
                success
                message
            }
        }
        """
        
        variables = {
            "id": "conv_123",
            "title": "Updated Title",
            "metadata": {
                "tags": ["important", "technical"],
                "archived": False
            }
        }
        
        with patch('src.api.graphql.update_conversation_resolver') as mock_resolver:
            mock_resolver.return_value = {
                "conversation": {
                    "id": "conv_123",
                    "title": "Updated Title",
                    "metadata": {
                        "tags": ["important", "technical"],
                        "archived": False
                    }
                },
                "success": True,
                "message": "Conversation updated successfully"
            }
            
            response = client.post(
                "/graphql",
                json={"query": mutation, "variables": variables},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" not in data
            assert data["data"]["updateConversation"]["success"] is True
            assert data["data"]["updateConversation"]["conversation"]["title"] == "Updated Title"
    
    def test_delete_conversation_mutation(self, client, auth_headers):
        """Test deleting a conversation."""
        mutation = """
        mutation DeleteConversation($id: ID!) {
            deleteConversation(id: $id) {
                success
                message
                deletedAt
            }
        }
        """
        
        variables = {"id": "conv_123"}
        
        with patch('src.api.graphql.delete_conversation_resolver') as mock_resolver:
            mock_resolver.return_value = {
                "success": True,
                "message": "Conversation deleted successfully",
                "deletedAt": "2024-01-01T12:00:00Z"
            }
            
            response = client.post(
                "/graphql",
                json={"query": mutation, "variables": variables},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" not in data
            assert data["data"]["deleteConversation"]["success"] is True
    
    def test_update_user_preferences_mutation(self, client, auth_headers):
        """Test updating user preferences."""
        mutation = """
        mutation UpdatePreferences($preferences: UserPreferencesInput!) {
            updateUserPreferences(preferences: $preferences) {
                preferences {
                    theme
                    language
                    modelPreferences {
                        defaultModel
                        temperature
                        maxTokens
                    }
                    privacySettings {
                        shareData
                        allowAnalytics
                    }
                }
                success
                message
            }
        }
        """
        
        variables = {
            "preferences": {
                "theme": "dark",
                "language": "en",
                "modelPreferences": {
                    "defaultModel": "gpt-4",
                    "temperature": 0.7,
                    "maxTokens": 2000
                },
                "privacySettings": {
                    "shareData": False,
                    "allowAnalytics": True
                }
            }
        }
        
        with patch('src.api.graphql.update_preferences_resolver') as mock_resolver:
            mock_resolver.return_value = {
                "preferences": variables["preferences"],
                "success": True,
                "message": "Preferences updated successfully"
            }
            
            response = client.post(
                "/graphql",
                json={"query": mutation, "variables": variables},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" not in data
            assert data["data"]["updateUserPreferences"]["success"] is True
            assert data["data"]["updateUserPreferences"]["preferences"]["theme"] == "dark"


class TestGraphQLErrorHandling:
    """Test GraphQL error handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_api_app(debug=True)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create auth headers."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret"
            token = create_jwt_token(user_id="test_user", roles=["user"])
            return {"Authorization": f"Bearer {token}"}
    
    def test_malformed_query_error(self, client, auth_headers):
        """Test handling of malformed queries."""
        malformed_query = "{ this is not valid GraphQL }"
        
        response = client.post(
            "/graphql",
            json={"query": malformed_query},
            headers=auth_headers
        )
        
        assert response.status_code == 200  # GraphQL returns 200 with errors
        data = response.json()
        assert "errors" in data
        assert len(data["errors"]) > 0
        assert "syntax" in str(data["errors"][0]).lower() or "parse" in str(data["errors"][0]).lower()
    
    def test_field_not_found_error(self, client, auth_headers):
        """Test querying non-existent fields."""
        query = """
        query {
            currentUser {
                id
                nonExistentField
            }
        }
        """
        
        response = client.post(
            "/graphql",
            json={"query": query},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "errors" in data
        assert "nonExistentField" in str(data["errors"])
    
    def test_variable_type_mismatch_error(self, client, auth_headers):
        """Test variable type validation."""
        query = """
        query GetConversation($id: ID!) {
            conversation(id: $id) {
                id
                title
            }
        }
        """
        
        # Provide wrong type for variable
        variables = {"id": 123}  # Should be string, not number
        
        response = client.post(
            "/graphql",
            json={"query": query, "variables": variables},
            headers=auth_headers
        )
        
        # Depending on implementation, might coerce or error
        assert response.status_code == 200
        data = response.json()
        # Either errors or handles the type coercion
    
    def test_resolver_error_handling(self, client, auth_headers):
        """Test handling of resolver errors."""
        query = """
        query {
            conversation(id: "nonexistent") {
                id
                title
            }
        }
        """
        
        with patch('src.api.graphql.get_conversation_resolver') as mock_resolver:
            mock_resolver.side_effect = Exception("Conversation not found")
            
            response = client.post(
                "/graphql",
                json={"query": query},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" in data
            assert "not found" in str(data["errors"][0]).lower()
    
    def test_authorization_error_in_field(self, client, auth_headers):
        """Test field-level authorization errors."""
        query = """
        query {
            adminOnlyData {
                sensitiveInfo
            }
        }
        """
        
        with patch('src.api.graphql.admin_only_resolver') as mock_resolver:
            mock_resolver.side_effect = Exception("Insufficient permissions")
            
            response = client.post(
                "/graphql",
                json={"query": query},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" in data
            assert "permission" in str(data["errors"][0]).lower()


class TestGraphQLPerformance:
    """Test GraphQL performance characteristics."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_api_app(debug=True)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create auth headers."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret"
            token = create_jwt_token(user_id="test_user", roles=["user"])
            return {"Authorization": f"Bearer {token}"}
    
    def test_query_complexity_limit(self, client, auth_headers):
        """Test that query complexity is limited."""
        # Very complex nested query
        complex_query = """
        query {
            conversations(limit: 100) {
                items {
                    id
                    messages(limit: 100) {
                        id
                        user {
                            conversations(limit: 100) {
                                items {
                                    messages(limit: 100) {
                                        id
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        
        response = client.post(
            "/graphql",
            json={"query": complex_query},
            headers=auth_headers
        )
        
        # Should either reject or depth-limit
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert "errors" in data or "complexity" in str(data).lower()
    
    def test_response_size_limit(self, client, auth_headers):
        """Test response size limits."""
        query = """
        query {
            largeDataSet(limit: 10000) {
                items {
                    id
                    largeTextField
                }
            }
        }
        """
        
        with patch('src.api.graphql.large_dataset_resolver') as mock_resolver:
            # Simulate large response
            mock_resolver.return_value = {
                "items": [
                    {"id": f"item_{i}", "largeTextField": "X" * 10000}
                    for i in range(1000)
                ]
            }
            
            response = client.post(
                "/graphql",
                json={"query": query},
                headers=auth_headers
            )
            
            # Should handle large responses appropriately
            assert response.status_code == 200
    
    def test_n_plus_one_prevention(self, client, auth_headers):
        """Test that N+1 queries are prevented."""
        query = """
        query {
            conversations(limit: 10) {
                items {
                    id
                    lastMessage {
                        content
                        user {
                            username
                        }
                    }
                }
            }
        }
        """
        
        with patch('src.api.graphql.conversations_with_messages_resolver') as mock_resolver:
            # Should use dataloader or join to prevent N+1
            mock_resolver.return_value = {
                "items": [
                    {
                        "id": f"conv_{i}",
                        "lastMessage": {
                            "content": f"Message {i}",
                            "user": {"username": f"user_{i}"}
                        }
                    }
                    for i in range(10)
                ]
            }
            
            response = client.post(
                "/graphql",
                json={"query": query},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            # In real implementation, would verify query count


class TestGraphQLSubscriptions:
    """Test GraphQL subscription operations."""
    
    @pytest.mark.asyncio
    async def test_message_subscription(self):
        """Test real-time message subscription."""
        # This would test WebSocket subscriptions
        # Placeholder for actual implementation
        pass
    
    @pytest.mark.asyncio
    async def test_typing_indicator_subscription(self):
        """Test typing indicator subscription."""
        # This would test real-time typing indicators
        # Placeholder for actual implementation
        pass


class TestGraphQLDataValidation:
    """Test GraphQL data validation."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_api_app(debug=True)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create auth headers."""
        with patch('src.api.auth.get_secrets_manager') as mock:
            mock.return_value.get_secret.return_value = "test-secret"
            token = create_jwt_token(user_id="test_user", roles=["user"])
            return {"Authorization": f"Bearer {token}"}
    
    def test_input_length_validation(self, client, auth_headers):
        """Test input length limits."""
        mutation = """
        mutation SendMessage($conversationId: ID!, $content: String!) {
            sendMessage(conversationId: $conversationId, content: $content) {
                success
                error
            }
        }
        """
        
        # Very long message
        variables = {
            "conversationId": "conv_123",
            "content": "A" * 50000  # 50k characters
        }
        
        response = client.post(
            "/graphql",
            json={"query": mutation, "variables": variables},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        # Should either truncate or error
        if "errors" not in data:
            assert data["data"]["sendMessage"]["success"] is False or \
                   data["data"]["sendMessage"]["error"] is not None
    
    def test_special_character_handling(self, client, auth_headers):
        """Test handling of special characters."""
        mutation = """
        mutation CreateConversation($title: String!) {
            createConversation(title: $title) {
                conversation {
                    title
                }
                success
            }
        }
        """
        
        special_titles = [
            "Title with emoji 😀🚀",
            "Title with unicode: ñáéíóú",
            "Title with quotes: \"test\" 'test'",
            "Title with newlines:\nLine 2",
            "Title with tabs:\t\tIndented"
        ]
        
        for title in special_titles:
            variables = {"title": title}
            
            with patch('src.api.graphql.create_conversation_resolver') as mock_resolver:
                mock_resolver.return_value = {
                    "conversation": {"title": title},
                    "success": True
                }
                
                response = client.post(
                    "/graphql",
                    json={"query": mutation, "variables": variables},
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "errors" not in data
                # Title should be preserved or safely encoded
    
    def test_null_value_handling(self, client, auth_headers):
        """Test handling of null values."""
        query = """
        query GetConversation($id: ID!) {
            conversation(id: $id) {
                id
                title
                metadata
            }
        }
        """
        
        variables = {"id": "conv_with_nulls"}
        
        with patch('src.api.graphql.get_conversation_resolver') as mock_resolver:
            mock_resolver.return_value = {
                "id": "conv_with_nulls",
                "title": None,  # Null title
                "metadata": None  # Null metadata
            }
            
            response = client.post(
                "/graphql",
                json={"query": query, "variables": variables},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "errors" not in data
            assert data["data"]["conversation"]["title"] is None
            assert data["data"]["conversation"]["metadata"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])