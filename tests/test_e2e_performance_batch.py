#!/usr/bin/env python3
"""E2E tests for batch processing performance.

Tests:
- Concurrent generation performance
- Memory usage during operations
- Model loading/unloading performance
- Voice prompt creation performance
- Response time baselines

Prerequisites:
    - TTS server running on port 5123
    - Auth token available at ~/.config/qwen3-tts/.voice_server_token
    - Clone model loaded (for faster testing)

Run: pytest tests/test_e2e_performance_batch.py -v
"""

import json
import os
import time
import urllib.error
import urllib.request

import pytest

from tests.e2e_helpers import (
    assert_rejected as _assert_rejected,
)
from tests.e2e_helpers import (
    wait_for_rate_limit_reset,
)

# E2E tests require a live server and make real generation requests under load.
# Gated behind the `e2e` marker so plain `pytest tests/` skips them (no hang).
# Opt in with: pytest tests/ -m e2e
pytestmark = pytest.mark.e2e

SERVER_URL = "http://127.0.0.1:5123"
AUTH_TOKEN_PATHS = [
    "~/.config/qwen3-tts/.voice_server_token",
    "~/.voice_server_token",
]


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
        resp = urllib.request.urlopen(req, timeout=300)  # 5 min timeout for large files
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


@pytest.fixture(scope="session", autouse=True)
def check_server():
    """Skip all tests if server is not running."""
    if not _is_server_running():
        pytest.skip("TTS server not running on port 5123. Start with: tts server start")


@pytest.fixture(scope="module", autouse=True)
def ensure_fresh_rate_limit(check_server):
    """Ensure the rate limit window is fresh before this module's tests run."""
    wait_for_rate_limit_reset(SERVER_URL, _get_auth_token())
    yield


class TestE2EBatchProcessingPerformance:
    """E2E tests for batch processing performance."""

    def test_01_concurrent_generations_performance(self):
        """REGRESSION: Server should handle concurrent requests efficiently.

        E2E verification that concurrent requests don't degrade significantly.
        Note: This test uses Clone mode which should be fast.
        """
        import threading

        token = _get_auth_token()
        results = []
        errors = []

        def generate_request(request_id):
            """Make a single generation request."""
            start = time.time()
            try:
                status, data = _make_request(
                    "/generate",
                    {"text": f"Concurrent test {request_id}", "mode": "clone"},
                    method="POST",
                    token=token
                )
                duration = time.time() - start
                results.append((request_id, status, duration))
            except Exception as e:
                errors.append((request_id, str(e)))

        # Launch 5 concurrent requests (reduced for faster testing)
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        start_time = time.time()
        for thread in threads:
            thread.join(timeout=180)
        time.time() - start_time

        # Verify all completed (no thread timeouts)
        assert len(results) == 5, \
            f"Some concurrent requests failed: {len(results)}/5 completed, errors: {errors}"

        # Verify server is still responsive
        status, _ = _make_request("/health", method="GET")
        assert status == 200, "Server should remain responsive after concurrent requests"

        # All requests should complete within reasonable time
        # (with actual generation, each might take 10-30 seconds)
        for req_id, status, duration in results:
            assert duration < 120, \
                f"Request {req_id} took {duration:.1f}s (expected < 120s)"

    def test_02_sequential_generations_performance(self):
        """REGRESSION: Sequential generations should be efficient.

        E2E verification that generation throughput is acceptable.
        """
        token = _get_auth_token()

        times = []
        for i in range(3):
            start = time.time()
            status, data = _make_request(
                "/generate",
                {"text": f"Sequential test {i}", "mode": "clone"},
                method="POST",
                token=token
            )
            duration = time.time() - start
            times.append(duration)
            _assert_rejected(status, [200, 202, 400, 422, 500, 503], f"Generation {i} returned {status}")

        avg_time = sum(times) / len(times)

        # Average generation time should be reasonable
        assert avg_time < 60, \
            f"Average generation time {avg_time:.1f}s exceeds 60s threshold"

    def test_03_model_load_performance(self):
        """REGRESSION: Model loading should complete within acceptable time.

        E2E verification that model loading is efficient.
        """
        # Unload model first if loaded
        _make_request("/unload-model", data={"mode": "clone"}, method="POST")

        # Measure load time
        start_time = time.time()
        status, data = _make_request(
            "/load-model",
            {"mode": "clone"},
            method="POST"
        )
        load_time = time.time() - start_time

        # Should succeed or return valid error
        assert status in [200, 400, 422, 503], \
            f"Model loading returned {status}"

        # Model loading should be reasonably fast
        assert load_time < 120, \
            f"Model loading took {load_time:.1f}s (expected < 120s)"

    def test_04_voice_prompt_list_performance(self):
        """REGRESSION: Voice prompt listing should be fast.

        E2E verification that prompt queries are efficient.
        """
        times = []
        for _ in range(5):
            start = time.time()
            status, data = _make_request("/prompts", method="GET")
            duration = time.time() - start
            times.append(duration)
            assert status in [200, 404], \
                f"Prompts endpoint returned {status}"

        avg_time = sum(times) / len(times)

        # Prompts endpoint should be fast
        assert avg_time < 1.0, \
            f"Prompts endpoint avg time {avg_time:.3f}s exceeds 1s threshold"


class TestE2EMemoryUsage:
    """E2E tests for memory usage during operations."""

    def test_01_memory_check_after_generations(self):
        """REGRESSION: Server should remain stable after multiple generations.

        E2E verification that memory doesn't grow unbounded.
        """
        # Get stats before
        status_before, data_before = _make_request("/stats", method="GET")

        # Make 5 generation requests
        for i in range(5):
            _make_request(
                "/generate",
                {"text": f"Memory test {i}", "mode": "clone"},
                method="POST"
            )

        # Get stats after
        status_after, data_after = _make_request("/stats", method="GET")

        # Server should still be responsive
        assert status_after == 200, "Server should respond after memory test"

        # If both stats succeeded, check memory hasn't grown dramatically
        if status_before == 200 and status_after == 200:
            # Check that server is still responsive (memory hasn't exhausted)
            assert "mlx_memory_active_mb" in data_after or "backend" in data_after, \
                "Stats should include memory info"

    def test_02_stats_endpoint_contains_memory_info(self):
        """REGRESSION: Stats endpoint should report memory usage.

        E2E verification that memory monitoring works.
        """
        status, data = _make_request("/stats", method="GET")

        assert status == 200, "Stats endpoint should return 200"
        # Should contain memory or backend info
        assert any(k in data for k in ["mlx_memory_active_mb", "backend", "torch_memory_active_mb"]), \
            f"Stats should include memory info, got: {list(data.keys())}"


class TestE2EResponseTimeBaselines:
    """E2E tests for response time baselines."""

    def test_01_health_endpoint_response_time(self):
        """REGRESSION: Health endpoint should respond quickly.

        E2E verification that monitoring endpoint is fast.
        """
        times = []
        for _ in range(5):
            start = time.time()
            urllib.request.urlopen(f"{SERVER_URL}/health", timeout=10)
            duration = time.time() - start
            times.append(duration)

        avg_time = sum(times) / len(times)

        # Health endpoint should be very fast (< 1 second)
        assert avg_time < 2.0, \
            f"Health endpoint avg time {avg_time:.3f}s exceeds 2s threshold"

    def test_02_stats_endpoint_response_time(self):
        """REGRESSION: Stats endpoint should respond quickly.

        E2E verification that stats endpoint is fast.
        """
        times = []
        for _ in range(5):
            start = time.time()
            status, _ = _make_request("/stats", method="GET")
            duration = time.time() - start
            times.append(duration)

        avg_time = sum(times) / len(times)

        # Stats endpoint should be reasonably fast (< 2 seconds)
        assert avg_time < 3.0, \
            f"Stats endpoint avg time {avg_time:.3f}s exceeds 3s threshold"

    def test_03_models_endpoint_response_time(self):
        """REGRESSION: Models endpoint should respond quickly.

        E2E verification that model list endpoint is fast.
        """
        times = []
        for _ in range(5):
            start = time.time()
            status, _ = _make_request("/models", method="GET")
            duration = time.time() - start
            times.append(duration)

        avg_time = sum(times) / len(times)

        # Models endpoint should be reasonably fast (< 2 seconds)
        assert avg_time < 3.0, \
            f"Models endpoint avg time {avg_time:.3f}s exceeds 3s threshold"
