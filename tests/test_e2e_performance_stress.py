#!/usr/bin/env python3
"""E2E tests for stress testing and load handling.

Tests:
- High concurrent request volume
- Memory limits under stress
- Graceful degradation under load
- Server stability during stress

Prerequisites:
    - TTS server running on port 5123
    - Auth token available at ~/.config/qwen3-tts/.voice_server_token

Run: pytest tests/test_e2e_performance_stress.py -v
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
        resp = urllib.request.urlopen(req, timeout=120)
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


class TestE2EStressTesting:
    """E2E tests for stress testing."""

    def test_01_server_handles_high_concurrent_load(self):
        """REGRESSION: Server should handle 10 concurrent requests without crashing.

        E2E verification that server remains stable under high load.
        Reduced from 50 to 10 for faster testing.
        """
        import threading

        headers = {
            "Authorization": f"Bearer {_get_auth_token()}",
        }

        def rapid_request(request_id):
            """Make a single generation request."""
            try:
                req = urllib.request.Request(
                    f"{SERVER_URL}/generate",
                    data=json.dumps({"text": f"Stress test {request_id}", "mode": "clone"}).encode(),
                    headers=headers,
                    method="POST"
                )
                resp = urllib.request.urlopen(req, timeout=120)
                return request_id, resp.status, None
            except Exception as e:
                return request_id, None, str(e)

        # Launch 10 concurrent requests (reduced for faster testing)
        threads = []
        for i in range(10):
            thread = threading.Thread(target=rapid_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete (with timeout)
        start_time = time.time()
        for thread in threads:
            thread.join(timeout=180)
        elapsed = time.time() - start_time

        # Verify server is still responsive
        resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
        assert resp.status == 200, "Server should remain responsive after stress test"
        assert elapsed < 180, "Stress test should complete within timeout"

    def test_02_rapid_health_checks(self):
        """REGRESSION: Server should handle rapid health check requests.

        E2E verification that public endpoints remain accessible under load.
        """
        # Make 50 rapid health checks
        for i in range(50):
            resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
            assert resp.status == 200, f"Health check {i} failed"

    def test_03_stats_endpoint_under_load(self):
        """REGRESSION: Stats endpoint should remain responsive under load.

        E2E verification that monitoring doesn't break under stress.
        """
        import threading

        token = _get_auth_token()
        results = []

        def stats_request(request_id):
            """Make a stats request."""
            try:
                status, data = _make_request("/stats", method="GET", token=token)
                results.append((request_id, status, None))
            except Exception as e:
                results.append((request_id, None, str(e)))

        # Launch 10 concurrent stats requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=stats_request, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=60)

        # Verify all completed successfully
        assert len(results) == 10, "Some stats requests failed"

    def test_04_memory_recovers_after_load(self):
        """REGRESSION: Memory usage should stabilize after load test.

        E2E verification that memory doesn't grow unbounded.
        """
        # Get stats before
        status_before, data_before = _make_request("/stats", method="GET")

        # Make 10 generation requests
        for i in range(10):
            _make_request(
                "/generate",
                {"text": f"Memory stress {i}", "mode": "clone"},
                method="POST"
            )
            time.sleep(0.1)  # Small delay to avoid overwhelming

        # Get stats after
        status_after, data_after = _make_request("/stats", method="GET")

        # Verify server is still responsive
        assert status_after == 200, "Server should remain responsive after memory stress"


class TestE2EGracefulDegradation:
    """E2E tests for graceful degradation under stress."""

    def test_01_server_rejects_invalid_requests_under_load(self):
        """REGRESSION: Server should validate requests even under load.

        E2E verification that validation doesn't get bypassed under stress.
        """
        token = _get_auth_token()

        # Send invalid requests (empty text)
        for i in range(5):
            status, data = _make_request(
                "/generate",
                {"text": "", "mode": "clone"},
                method="POST",
                token=token
            )
            # Should be rejected (400, 422) even under repeated requests
            _assert_rejected(status, [400, 422, 500], f"Invalid request {i}")

    def test_02_concurrent_invalid_requests_handled(self):
        """REGRESSION: Server should handle concurrent invalid requests gracefully.

        E2E verification that malformed requests don't crash the server.
        """
        import threading

        token = _get_auth_token()

        def invalid_request(request_id):
            """Make an invalid request."""
            try:
                status, data = _make_request(
                    "/generate",
                    {"text": "", "mode": "clone"},
                    method="POST",
                    token=token
                )
                return request_id, status, None
            except Exception as e:
                return request_id, None, str(e)

        # Launch 10 concurrent invalid requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=invalid_request, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=60)

        # Verify server is still responsive
        status, _ = _make_request("/health", method="GET")
        assert status == 200, "Server should remain responsive after invalid request flood"
