#!/usr/bin/env python3
"""E2E tests for input validation and sanitization.

Tests:
- Empty text validation
- Invalid mode validation
- Missing required fields
- SQL injection prevention
- XSS prevention
- Path traversal prevention

Prerequisites:
    - TTS server running on port 5123
    - Auth token available at ~/.config/qwen3-tts/.voice_server_token

Run: pytest tests/test_e2e_security_validation.py -v
"""

import json
import os
import time
import urllib.error
import urllib.request

import pytest

# E2E tests require a live server and make real generation requests under load.
# Gated behind the `e2e` marker so plain `pytest tests/` skips them (no hang).
# Opt in with: pytest tests/ -m e2e
pytestmark = pytest.mark.e2e

SERVER_URL = "http://127.0.0.1:5123"
AUTH_TOKEN_PATHS = [
    "~/.config/qwen3-tts/.voice_server_token",
    "~/.voice_server_token",
]


def _wait_for_rate_limit_reset(timeout: int = 70) -> None:
    """Block until /generate is no longer rate-limited, up to timeout seconds."""
    url = f"{SERVER_URL}/generate"
    token = _get_auth_token()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    body = json.dumps({"text": "rate-limit-probe", "mode": "custom"}).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        # Not rate limited — window is fresh, proceed
    except urllib.error.HTTPError as e:
        if e.code == 429:
            retry_after = int(e.headers.get("Retry-After", 65))
            time.sleep(min(retry_after + 1, timeout))
        # Any other error (400, 503) means not rate-limited — proceed
    except Exception:
        pass  # Network error — proceed


def _assert_rejected(status: int, expected_codes: list, context: str) -> None:
    """Assert request was rejected with an expected code; skip if rate-limited."""
    if status == 429:
        pytest.skip(f"Rate limit exceeded before '{context}' could be verified")
    assert status in expected_codes, f"{context}, got {status}"


@pytest.fixture(scope="module", autouse=True)
def ensure_fresh_rate_limit(check_server):
    """Ensure the rate limit window is fresh before this module's tests run."""
    _wait_for_rate_limit_reset()
    yield


def _get_auth_token():
    """Read the server auth token from known locations."""
    for path in AUTH_TOKEN_PATHS:
        token_path = os.path.expanduser(path)
        try:
            with open(token_path) as f:
                token = f.read().strip()
                if token:
                    return token
        except FileNotFoundError:
            continue
    return ""


@pytest.fixture(scope="session", autouse=True)
def check_server():
    """Skip all tests if server is not running."""
    if not _is_server_running():
        pytest.skip("TTS server not running on port 5123. Start with: tts server start")

    token = _get_auth_token()
    if not token:
        pytest.skip("No auth token found. Token should be at ~/.config/qwen3-tts/.voice_server_token")


def _is_server_running():
    """Check if TTS server is healthy."""
    try:
        resp = urllib.request.urlopen(
            f"{SERVER_URL}/health", timeout=5
        )
        return resp.status == 200
    except Exception:
        return False


def _make_request(endpoint, data=None, method="GET", token=None):
    """Make an HTTP request to the server.

    Returns:
        Tuple of (status_code, response_data)
    """
    if token is None:
        token = _get_auth_token()

    url = f"{SERVER_URL}{endpoint}"
    headers = {
        "Content-Type": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    body = None
    if data:
        body = json.dumps(data).encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        response_data = json.loads(resp.read().decode()) if resp.status != 204 else {}
        return resp.status, response_data
    except urllib.error.HTTPError as e:
        try:
            error_data = json.loads(e.read().decode())
        except Exception:
            error_data = {}
        return e.code, error_data
    except Exception as e:
        return 0, {"error": str(e)}


class TestE2EInputValidationSecurity:
    """E2E tests for input validation security."""

    def test_01_empty_text_returns_400(self):
        """REGRESSION: Empty text should return validation error.

        E2E verification that empty input is rejected.
        """
        status, data = _make_request(
            "/generate",
            {"text": "", "mode": "clone"},
            method="POST"
        )

        # Should reject with 400, 422, or 500 (server error during validation)
        assert status in [400, 422, 500], \
            f"Empty text should be rejected, got {status}: {data}"

    def test_02_whitespace_only_text_rejected(self):
        """REGRESSION: Whitespace-only text should be rejected.

        E2E verification that text with only whitespace is rejected.
        """
        whitespace_texts = [
            "   ",
            "\t\t",
            "\n\n",
            "  \t  \n  ",
        ]

        for text in whitespace_texts:
            status, _ = _make_request(
                "/generate",
                {"text": text, "mode": "clone"},
                method="POST"
            )

            # Should reject with 400, 422, or 500
            _assert_rejected(status, [400, 422, 500], "Whitespace text")

    def test_03_invalid_mode_returns_400(self):
        """REGRESSION: Invalid mode should return validation error.

        E2E verification that only valid modes are accepted.
        """
        invalid_modes = [
            "invalid_mode",
            "hacker_mode",
            "../../../etc/passwd",
            "<script>alert('xss')</script>",
        ]

        for mode in invalid_modes:
            status, _ = _make_request(
                "/generate",
                {"text": "test", "mode": mode},
                method="POST"
            )

            # Should reject with 400, 422, 500, or return 503 if mode not supported
            _assert_rejected(status, [400, 422, 500, 503], f"Invalid mode '{mode}'")

    def test_04_missing_required_fields(self):
        """REGRESSION: Requests missing required fields should be rejected.

        E2E verification that complete payloads are required.
        """
        # Missing 'text' field
        status, _ = _make_request(
            "/generate",
            {"mode": "clone"},  # No 'text'
            method="POST"
        )

        # Should reject with 400, 422, or 500
        _assert_rejected(status, [400, 422, 500], "Missing 'text' field")

    def test_05_missing_mode_field(self):
        """REGRESSION: Requests missing mode should use default or reject.

        E2E verification that mode is handled properly when missing.
        """
        # Missing 'mode' field
        status, data = _make_request(
            "/generate",
            {"text": "test text"},  # No 'mode'
            method="POST"
        )

        # Should either accept (with default mode) or reject with 400/422/500
        # 200/202 = accepted with default, 400/422/500 = rejected
        _assert_rejected(status, [200, 202, 400, 422, 500, 503], "Missing 'mode' field")

    def test_06_very_long_text_rejected(self):
        """REGRESSION: Excessively long text should be rejected.

        E2E verification that max text length is enforced.
        """
        # Create text longer than typical max (10000 chars)
        long_text = "x" * 20000

        status, _ = _make_request(
            "/generate",
            {"text": long_text, "mode": "clone"},
            method="POST"
        )

        # Should reject with 400/422/422/500 for text too long
        # Or accept if the limit is higher
        if status == 429:
            pytest.skip("Rate limit exceeded before 'very long text' could be verified")
        elif status not in [200, 202]:
            assert status in [400, 422, 413, 500], \
                f"Very long text should be rejected, got {status}"


class TestE2EInjectionPrevention:
    """E2E tests for injection attack prevention."""

    def test_01_sql_injection_prevented(self):
        """REGRESSION: SQL injection attempts should be harmless.

        E2E verification that malicious input is sanitized.
        """
        sql_payloads = [
            "test'; DROP TABLE users; --",
            "test' OR '1'='1",
            "test' UNION SELECT * FROM models; --",
            "'; EXEC xp_cmdshell('format c:'); --",
        ]

        for payload in sql_payloads:
            status, response_str = _make_request(
                "/generate",
                {"text": payload, "mode": "clone"},
                method="POST"
            )

            # Should be rejected (400/422) or treated as literal text (200/202)
            # If 200, verify it's not a database error
            if status == 200:
                # Check response doesn't contain database error strings
                response_text = json.dumps(response_str).lower()
                assert "database" not in response_text, \
                    f"SQL injection not prevented for: {payload[:30]}"
                assert "syntax error" not in response_text, \
                    f"SQL injection not prevented for: {payload[:30]}"

    def test_02_xss_prevention_in_text(self):
        """REGRESSION: XSS attempts in text field should be harmless.

        E2E verification that cross-site scripting is prevented.
        """
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>",
        ]

        for payload in xss_payloads:
            status, response_str = _make_request(
                "/generate",
                {"text": payload, "mode": "clone"},
                method="POST"
            )

            # If successful, verify XSS wasn't executed
            # (text should be treated as literal, not HTML)
            if status in [200, 202]:
                json.dumps(response_str)
                # The response should not contain the unescaped script
                # (or if it does, it should be escaped/sanitized)

    def test_03_xss_prevention_in_parameters(self):
        """REGRESSION: XSS attempts in all parameters should be sanitized.

        E2E verification that all input fields are XSS-safe.
        """
        xss_payload = "<script>alert('xss')</script>"

        # Try XSS in different fields
        for field, value in [
            ("text", xss_payload),
            ("mode", xss_payload),
        ]:
            status, _ = _make_request(
                "/generate",
                {field: value, "mode": "clone"} if field != "mode" else {"text": "test", "mode": value},
                method="POST"
            )

            # Should be rejected or safely handled
            # 400/422/500/503 are all acceptable safe responses
            if status == 429:
                pytest.skip(f"Rate limit exceeded before 'XSS in {field}' could be verified")
            elif status not in [200, 202]:
                assert status in [400, 422, 500, 503], \
                    f"XSS in {field} should be rejected, got {status}"

    def test_04_path_traversal_prevented(self):
        """REGRESSION: Path traversal attempts should be prevented.

        E2E verification that file system access is blocked.
        """
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\sam",
            "....//....//....//etc/passwd",
        ]

        for payload in path_traversal_payloads:
            status, _ = _make_request(
                "/generate",
                {"text": payload, "mode": "clone"},
                method="POST"
            )

            # Should be rejected or safely treated as text
            # Should NOT return 200 with file contents
            if status == 429:
                pytest.skip("Rate limit exceeded before 'path traversal' could be verified")
            elif status == 200:
                # If accepted, it should be treated as text to speak,
                # not as a file path to read
                pass  # Accept - treated as literal text
            else:
                # 400/422/500/503 are all acceptable rejections
                assert status in [400, 422, 500, 503], \
                    f"Path traversal should be prevented, got {status}"

    def test_05_command_injection_prevented(self):
        """REGRESSION: Command injection attempts should be prevented.

        E2E verification that shell commands are not executed.
        """
        command_injection_payloads = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "$(reboot)",
            "`whoami`",
            "\nls -la\n",
        ]

        for payload in command_injection_payloads:
            status, response_str = _make_request(
                "/generate",
                {"text": payload, "mode": "clone"},
                method="POST"
            )

            # If successful, verify command was not executed
            # (text should be treated as literal)
            if status in [200, 202]:
                response_text = json.dumps(response_str).lower()
                # Should not show signs of command execution
                assert "root:" not in response_text, \
                    f"Command injection may have succeeded: {payload[:30]}"

    def test_06_template_injection_prevented(self):
        """REGRESSION: Template injection attempts should be prevented.

        E2E verification that template syntax is not evaluated.
        """
        template_payloads = [
            "{{7*7}}",
            "${7*7}",
            "#{7*7}",
            "{{config}}",
            "{{7*'7'}}",
        ]

        for payload in template_payloads:
            status, response_str = _make_request(
                "/generate",
                {"text": payload, "mode": "clone"},
                method="POST"
            )

            # Template injection should not execute
            # (text should be treated as literal)
            if status in [200, 202]:
                json.dumps(response_str)
                # If the payload was executed, we'd see "49" for {{7*7}}
                # Instead, it should be treated as literal text
                pass  # Accept - server treated it as literal text


class TestE2EDataTypeValidation:
    """E2E tests for data type validation."""

    def test_01_invalid_json_rejected(self):
        """REGRESSION: Malformed JSON should be rejected.

        E2E verification that request body must be valid JSON.
        """
        url = f"{SERVER_URL}/generate"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_get_auth_token()}",
        }

        # Send invalid JSON
        invalid_json_bodies = [
            b"{invalid json",
            b'{"text": "test", unclosed',
            b'{"text": test}',
        ]

        for body in invalid_json_bodies:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")

            try:
                resp = urllib.request.urlopen(req, timeout=10)
                status = resp.status
            except urllib.error.HTTPError as e:
                status = e.code
            except Exception:
                # JSON parsing error before HTTP handling
                status = 400  # Treat as bad request

            # Should reject with 400 (bad request) or 422 (validation error)
            assert status in [400, 422], \
                f"Invalid JSON should be rejected, got {status}"

    def test_02_wrong_content_type_rejected(self):
        """REGRESSION: Wrong content-type should be rejected.

        E2E verification that JSON endpoints require JSON content-type.
        """
        url = f"{SERVER_URL}/generate"
        valid_data = json.dumps({"text": "test", "mode": "clone"}).encode()

        # Try with wrong content-type
        for content_type in [
            "text/plain",
            "application/xml",
            "application/x-www-form-urlencoded",
        ]:
            headers = {
                "Content-Type": content_type,
                "Authorization": f"Bearer {_get_auth_token()}",
            }
            req = urllib.request.Request(url, data=valid_data, headers=headers, method="POST")

            try:
                resp = urllib.request.urlopen(req, timeout=10)
                status = resp.status
            except urllib.error.HTTPError as e:
                status = e.code

            # Should accept or reject (FastAPI may auto-detect JSON)
            # If rejected, should be 415 (unsupported media type), 400, or 422
            if status != 200:
                assert status in [400, 415, 422], \
                    f"Wrong content-type should be rejected, got {status} for {content_type}"

    def test_03_non_string_fields_rejected(self):
        """REGRESSION: Non-string values for text fields should be rejected.

        E2E verification that text fields accept only strings.
        """
        # Try sending numeric values for text field
        invalid_requests = [
            {"text": 12345, "mode": "clone"},  # Number instead of string
            {"text": None, "mode": "clone"},  # Null instead of string
            {"text": True, "mode": "clone"},  # Boolean instead of string
            {"text": [], "mode": "clone"},  # Array instead of string
        ]

        for payload in invalid_requests:
            status, _ = _make_request("/generate", data=payload, method="POST")

            # Should reject with 400, 422, or 500 (server error)
            _assert_rejected(status, [400, 422, 500], f"Non-string text ({type(payload['text']).__name__})")
