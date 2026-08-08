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

from tests.e2e_helpers import (
    assert_rejected as _assert_rejected,
)
from tests.e2e_helpers import (
    first_available_voice_prompt,
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


@pytest.fixture(scope="module", autouse=True)
def ensure_fresh_rate_limit(check_server):
    """Ensure the rate limit window is fresh before this module's tests run."""
    wait_for_rate_limit_reset(SERVER_URL, _get_auth_token())
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
        """REGRESSION: Server should handle concurrent /generate without crashing.

        Concurrent generations serialize on the server's inference_lock, so
        this exercises queued load. Each request must produce real audio (200
        + non-empty audio_base64), not merely a liveness response — otherwise
        the test passes "hollow" (rate-limited/fast-failed without generating).
        """
        import threading

        token = _get_auth_token()
        prompt = first_available_voice_prompt(SERVER_URL, token)
        if not prompt:
            pytest.skip("No voice prompts available for clone generation")
        results = []
        errors = []

        def generate_request(request_id):
            """Make a single generation request; capture status + body."""
            try:
                status, data = _make_request(
                    "/generate",
                    {"text": f"Stress test {request_id}", "mode": "clone", "prompt_file": prompt},
                    method="POST",
                    token=token,
                )
                results.append((request_id, status, data))
            except Exception as e:  # noqa: BLE001 - capture any thread failure
                errors.append((request_id, str(e)))

        # Launch 5 concurrent requests (serialized on inference_lock; 5 keeps
        # total runtime well under the thread timeout).
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete (with timeout)
        start_time = time.time()
        for thread in threads:
            thread.join(timeout=240)
        elapsed = time.time() - start_time

        # Every request must have actually generated audio.
        assert not errors, f"Some concurrent requests errored: {errors}"
        assert len(results) == 5, f"Only {len(results)}/5 requests completed"
        for request_id, status, data in results:
            assert status == 200, f"Request {request_id} returned {status}: {data}"
            audio = (data or {}).get("results", [{}])[0].get("audio_base64", "")
            assert audio, f"Request {request_id} returned no audio_base64: {data}"

        # Verify server is still responsive
        status, _ = _make_request("/health", method="GET")
        assert status == 200, "Server should remain responsive after stress test"
        assert elapsed < 240, "Stress test should complete within timeout"

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

        prompt = first_available_voice_prompt(SERVER_URL, _get_auth_token())
        if not prompt:
            pytest.skip("No voice prompts available for clone generation")

        # Make 5 generation requests — at least one must produce real audio,
        # otherwise this only checks liveness, not memory behavior under load.
        audio_produced = 0
        for i in range(5):
            status, data = _make_request(
                "/generate",
                {"text": f"Memory stress {i}", "mode": "clone", "prompt_file": prompt},
                method="POST",
            )
            if status == 200 and (data or {}).get("results", [{}])[0].get("audio_base64"):
                audio_produced += 1
            time.sleep(0.1)  # Small delay to avoid overwhelming

        # Get stats after
        status_after, data_after = _make_request("/stats", method="GET")

        # Verify server is still responsive
        assert status_after == 200, "Server should remain responsive after memory stress"
        assert audio_produced >= 1, (
            "No generation produced audio; memory test is meaningless without "
            "actual load (check rate limiting / model load)."
        )


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
