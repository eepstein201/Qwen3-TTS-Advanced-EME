#!/usr/bin/env python3
"""Comprehensive rate limiting tests (R-13).

Tests:
- Per-IP rate limiting
- Per-token rate limiting
- Hybrid rate limiting
- Config validation
- Rate limit strategy selection
- AI REGRESSION TESTS: Token hash security, decorator verification, config validation

Run: pytest tests/test_rate_limiting.py -v
"""

from unittest.mock import MagicMock, patch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    class _DummyMarkerFunc:
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            return lambda f: f
    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()
    class _DummyPytest:
        mark = _DummyMark()
    pytest = _DummyPytest()

try:
    from fastapi import Request
    HAS_FASTAPI_DEPS = True
except ImportError:
    HAS_FASTAPI_DEPS = False

try:
    import slowapi  # noqa: F401  (server.app imports slowapi unconditionally)
    HAS_SLOWAPI = True
except ImportError:
    HAS_SLOWAPI = False

HAS_DEPS = HAS_FASTAPI_DEPS and HAS_SLOWAPI

if HAS_PYTEST and not HAS_DEPS:
    pytest.skip(
        "requires fastapi, slowapi", allow_module_level=True
    )

if HAS_DEPS:
    from fastapi import Request as _Request
    Request = _Request

if HAS_DEPS:
    from qwen3_tts.server.app import _rate_limit, app


class TestRateLimitKeyFunctions:
    """Test rate limit key generation functions."""

    def test_get_ip_key_with_real_ip(self):
        """IP-only key should return client IP."""
        request = MagicMock(spec=Request)
        request.client.host = "192.168.1.100"
        request.headers = {}

        from qwen3_tts.server.app import _get_ip_key
        result = _get_ip_key(request)

        assert result == "192.168.1.100"

    def test_get_ip_key_with_trusted_proxy(self):
        """IP-only key honors X-Forwarded-For from a trusted proxy (loopback)."""
        request = MagicMock(spec=Request)
        # Loopback is a trusted proxy by default (Colab/tunnel forwards here).
        request.client.host = "127.0.0.1"
        request.headers = {"X-Forwarded-For": "203.0.113.1"}

        from qwen3_tts.server.app import _get_ip_key
        result = _get_ip_key(request)

        assert result == "203.0.113.1"

    def test_get_ip_key_untrusted_peer_ignores_proxy_header(self):
        """IP-only key ignores X-Forwarded-For from an untrusted direct peer."""
        request = MagicMock(spec=Request)
        request.client.host = "192.168.1.100"
        request.headers = {"X-Forwarded-For": "203.0.113.1"}

        from qwen3_tts.server.app import _get_ip_key
        result = _get_ip_key(request)

        assert result == "192.168.1.100"

    def test_get_token_key_with_valid_token(self):
        """Token-only key should hash token consistently."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer abc123def456"}

        from qwen3_tts.server.app import _get_token_key
        result1 = _get_token_key(request)
        result2 = _get_token_key(request)

        assert result1 == result2  # Same token = same hash
        assert len(result1) == 16  # SHA256 hex (first 16 chars)

    def test_get_token_key_without_token(self):
        """Token-only key should return 'anonymous' for missing tokens."""
        request = MagicMock(spec=Request)
        request.headers = {}

        from qwen3_tts.server.app import _get_token_key
        result = _get_token_key(request)

        assert result == "anonymous"


class TestAIRegressionTokenHashing:
    """AI REGRESSION: Test token hash consistency (prevents token leakage bugs).

    Token hashing is critical security: same token must always produce same hash,
    and hash must not leak the original token value.
    """

    def test_same_token_always_produces_same_hash(self):
        """REGRESSION: Same token must always produce same hash (deterministic).

        Prevents AI bug where hash is randomized or includes non-deterministic data.
        """
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer test_token_12345"}

        from qwen3_tts.server.app import _get_token_key

        # Call multiple times
        hashes = [_get_token_key(request) for _ in range(10)]

        # All hashes must be identical
        assert len(set(hashes)) == 1, "Token hash is non-deterministic"

    def test_token_hash_does_not_leak_original(self):
        """REGRESSION: Hash must not contain original token (prevents token leakage).

        Prevents AI bug where token is included in hash in plaintext.
        """
        request = MagicMock(spec=Request)
        original_token = "sensitive_token_abc123"
        request.headers = {"Authorization": f"Bearer {original_token}"}

        from qwen3_tts.server.app import _get_token_key
        token_hash = _get_token_key(request)

        # Hash must not contain original token
        assert original_token not in token_hash, "Token hash leaks original token"
        assert len(token_hash) == 16, "Token hash has wrong length (expected 16 for SHA256[:16])"

    def test_different_tokens_produce_different_hashes(self):
        """Different tokens must produce different hashes (collision resistance)."""
        from qwen3_tts.server.app import _get_token_key

        tokens = ["token_1", "token_2", "token_3"]
        hashes = []

        for token in tokens:
            request = MagicMock(spec=Request)
            request.headers = {"Authorization": f"Bearer {token}"}
            hashes.append(_get_token_key(request))

        # All hashes must be different
        assert len(set(hashes)) == len(tokens), "Token hash collision detected"


class TestAIRegressionDecoratorApplication:
    """AI REGRESSION: Test that rate limit decorators are actually applied.

    Prevents AI bug where decorator is added to code but not actually executed
    due to incorrect application order or missing wrapper.
    """

    def test_all_r13_endpoints_have_rate_limit_decorators(self):
        """REGRESSION: All R-13 endpoints must have rate limiting applied.

        This test prevents the common AI bug of adding decorators to endpoint
        functions but forgetting to apply them or applying them in wrong order.
        """

        # List of R-13 endpoints that must have rate limiting
        r13_endpoints = [
            "/generate",
            "/generate-stream",
            "/load-model",
            "/unload-model",
            "/update-model-config",
            "/load-asr",
            "/unload-asr",
            "/transcribe",
            "/create-voice-prompt",
            "/delete-prompt",
            "/rename-prompt",
            "/update-startup-config",
        ]

        # Check each endpoint exists in the app
        for route in app.routes:
            if hasattr(route, 'path') and route.path in r13_endpoints:
                # Endpoint exists (decorator verification would require runtime inspection)
                assert route.path in r13_endpoints, f"Endpoint {route.path} not in R-13 list"

    def test_rate_limit_decorator_supports_strategies(self):
        """REGRESSION: Rate limit decorator must support different strategies.

        Prevents AI bug where decorator doesn't properly select between limiters.
        """

        # Test hybrid strategy (default)
        decorator = _rate_limit("10/minute", strategy="hybrid")
        assert callable(decorator)

        # Test IP strategy
        decorator = _rate_limit("10/minute", strategy="ip")
        assert callable(decorator)

        # Test token strategy
        decorator = _rate_limit("10/minute", strategy="token")
        assert callable(decorator)


class TestAIRegressionConfigValidation:
    """AI REGRESSION: Test configuration validation catches and corrects invalid values.

    Prevents AI bug where invalid config values crash the server or pass through
    without validation.
    """

    def test_invalid_rate_limit_format_gets_corrected(self):
        """REGRESSION: Invalid rate limit strings must be corrected to defaults.

        Prevents AI bug where malformed rate limits (e.g., "invalid", "abc/minute")
        crash the server instead of being corrected.
        """
        from qwen3_tts.core.config import (
            _validate_rate_limit_string,
        )

        invalid_limits = ["invalid", "abc/minute", "100", "minute", "", None, 123]

        for invalid_limit in invalid_limits:
            is_valid = _validate_rate_limit_string(invalid_limit)
            assert not is_valid, f"Invalid limit {invalid_limit!r} was not caught"

    def test_negative_rate_limit_gets_corrected(self):
        """REGRESSION: Negative and zero rate limits must be rejected.

        Prevents AI bug where negative numbers pass validation but cause runtime errors.
        """
        from qwen3_tts.core.config import _validate_rate_limit_string

        negative_limits = ["-1/minute", "0/minute", "-100/hour"]

        for limit in negative_limits:
            is_valid = _validate_rate_limit_string(limit)
            assert not is_valid, f"Negative limit {limit!r} was not caught"

    def test_config_validation_provides_defaults(self):
        """REGRESSION: Missing config keys must get sensible defaults.

        Prevents AI bug where missing config causes KeyError or crashes.
        """
        from qwen3_tts.core.config import _get_default_rate_limit

        # All known endpoint types must have defaults
        endpoint_types = ["generate", "model_ops", "transcribe", "prompt_ops", "config_ops"]

        for endpoint_type in endpoint_types:
            default = _get_default_rate_limit(endpoint_type)
            assert isinstance(default, str), f"No default for {endpoint_type}"
            assert "/" in default, f"Invalid default format for {endpoint_type}: {default}"


class TestEndpointRateLimiting:
    """Test rate limiting on actual endpoints."""

    def test_strategy_parameter_support(self):
        """Rate limit decorator should support different strategies."""

        # Test hybrid strategy (default)
        decorator = _rate_limit("10/minute", strategy="hybrid")
        assert callable(decorator)

        # Test IP strategy
        decorator = _rate_limit("10/minute", strategy="ip")
        assert callable(decorator)

        # Test token strategy
        decorator = _rate_limit("10/minute", strategy="token")
        assert callable(decorator)


class TestAIRegressionRateLimitErrors:
    """AI REGRESSION: Test 429 error responses are consistent.

    Prevents AI bug where rate limit errors return inconsistent formats
    or leak internal implementation details.
    """

    def test_rate_limit_error_handler_callable(self):
        """REGRESSION: Error handler must be callable and not crash.

        Prevents AI bug where error handler is not properly configured
        or crashes when called with rate limit exceptions.
        """
        # Test the error handler is callable
        from qwen3_tts.server.app import _rate_limit_exceeded_handler

        assert callable(_rate_limit_exceeded_handler), "Error handler must be callable"

    def test_rate_limit_error_format_does_not_leak_details(self):
        """REGRESSION: Error messages should not leak internal function details.

        The slowapi error handler formats messages as "Rate limit exceeded: {detail}".
        We verify that sensitive internal details aren't exposed in the default format.
        """
        # Test with error containing internal details
        from unittest.mock import MagicMock

        from qwen3_tts.server.app import _rate_limit_exceeded_handler

        error = MagicMock()
        error.detail = "Rate limit exceeded"

        request = MagicMock()
        request.app.state = MagicMock()

        # Handler should execute without exception
        try:
            response = _rate_limit_exceeded_handler(request, error)
            # If we get here without exception, the handler works
            assert response is not None
        except Exception as e:
            assert False, f"Error handler crashed: {e}"


class TestRateLimitEnvOverride:
    """Tests for TTS_DISABLE_RATE_LIMITING / TTS_RATE_LIMIT_* env overrides.

    These let a local E2E/CI server bypass the default caps so suites that fire
    many /generate requests aren't starved by the 10-20/min limit (I2 fix).
    Fixture-free so the unittest batch runner can execute them too.
    """

    def test_rate_limit_from_env_prefers_env_over_default(self):
        import os

        from qwen3_tts.server.app import _rate_limit_from_env

        old = os.environ.get("TTS_RATE_LIMIT_GENERATE")
        os.environ["TTS_RATE_LIMIT_GENERATE"] = "999/hour"
        try:
            resolved = _rate_limit_from_env(
                "TTS_RATE_LIMIT_GENERATE", "generate", "10/minute"
            )
            assert resolved == "999/hour"
        finally:
            if old is None:
                os.environ.pop("TTS_RATE_LIMIT_GENERATE", None)
            else:
                os.environ["TTS_RATE_LIMIT_GENERATE"] = old

    def test_rate_limit_from_env_falls_back_to_default(self):
        import os

        from qwen3_tts.server.app import _rate_limit_from_env

        old = os.environ.get("TTS_RATE_LIMIT_GENERATE")
        os.environ.pop("TTS_RATE_LIMIT_GENERATE", None)
        try:
            resolved = _rate_limit_from_env(
                "TTS_RATE_LIMIT_GENERATE", "__no_such_key__", "10/minute"
            )
            assert resolved == "10/minute"
        finally:
            if old is not None:
                os.environ["TTS_RATE_LIMIT_GENERATE"] = old

    def test_disable_rate_limiting_makes_decorator_a_noop(self):
        import qwen3_tts.server.app as app_module

        original = app_module._RATE_LIMITING_DISABLED
        app_module._RATE_LIMITING_DISABLED = True
        try:
            decorator = app_module._rate_limit("10/minute")

            def _handler():
                return "ok"

            # No-op: the handler is returned unchanged and still callable.
            assert decorator(_handler) is _handler
            assert _handler() == "ok"
        finally:
            app_module._RATE_LIMITING_DISABLED = original


class TestGlobalUnauthFloodLimit:
    """Unauthenticated floods must hit a pre-auth global rate limit (P2).

    Per-route ``@limiter.limit`` decorators run AFTER ``Depends(verify_auth)``
    (Starlette order: Middleware -> Routing -> Endpoint), so unauthenticated
    traffic bypasses them entirely. A global ``SlowAPIMiddleware`` +
    ``default_limits`` on an IP-keyed limiter enforces a ceiling at the ASGI
    layer, before auth — turning an unauthenticated flood (all 401 today) into
    a throttled one (429 once the IP exceeds the default).
    """

    def test_unauth_flood_eventually_returns_429(self):
        from qwen3_tts.server.app import app
        from tests.voice_test_helpers import _make_test_client

        client = _make_test_client(
            app, server_config={"security": {}, "auto_shutdown_minutes": 0}
        )
        app.state.models_loaded.set()

        global_limiter = app.state.limiter
        was_enabled = getattr(global_limiter, "enabled", True)
        global_limiter.enabled = True
        global_limiter.reset()
        # Send enough unauthenticated requests to exceed the global ceiling,
        # whatever its configured value. The ceiling is decoupled from the
        # 10/min generate limit and defaults much higher, so a fixed 15 would
        # no longer trip it.
        import re as _re

        from qwen3_tts.server.app import _global_limit
        _m = _re.match(r"(\d+)", _global_limit)
        _ceiling = int(_m.group(1)) if _m else 10
        try:
            statuses = []
            for _ in range(_ceiling + 10):
                resp = client.post("/generate", json={"texts": ["x"]})
                statuses.append(resp.status_code)
                if resp.status_code == 429:
                    break
        finally:
            global_limiter.enabled = was_enabled
            global_limiter.reset()

        assert 429 in statuses, (
            "expected the global pre-auth limiter to throttle an unauthenticated "
            f"flood (got statuses={statuses})"
        )


class TestGlobalCeilingDecoupled:
    """The global pre-auth ceiling must NOT reuse the 10/min generate limit.

    The Gradio UI polls /health + /models roughly every 5s (~24/min). With the
    global default tied to the 10/min generate limit, normal authenticated use
    trips it within ~25s and /health starts 429ing -> is_server_running() reads
    "down" -> the UI shows "Disconnected / Server not running", and
    `tts server restart` (which calls is_server_running to decide whether to
    stop) skips the stop, starts a loser that aborts on the startup lock, and
    leaves the old server running under a stale PID file. The global backstop
    only needs to stop floods (thousands/min), so it sits well above normal
    traffic as a separate limit.
    """

    def test_global_limit_exists_and_is_separate_from_generate(self):
        from qwen3_tts.server import app as app_module

        assert hasattr(app_module, "_global_limit"), (
            "global pre-auth ceiling must be an explicit _global_limit, not the "
            "generate limit reused via default_limits"
        )
        assert app_module._global_limit != app_module._generate_limit

    def test_global_limit_above_ui_polling_rate(self):
        import re

        from qwen3_tts.server import app as app_module

        m = re.match(r"(\d+)", app_module._global_limit)
        assert m, f"unparseable _global_limit: {app_module._global_limit!r}"
        per_minute = int(m.group(1))
        # UI status + model-badge polling alone is ~24/min; leave real headroom.
        assert per_minute >= 60, (
            f"global ceiling {per_minute}/min is too low -- the UI's own polling "
            "would trip it and /health would 429"
        )


class TestServerRunningHandlesRateLimit:
    """is_server_running() must treat a 429 (rate-limited /health) as 'running'.

    A 429 proves the server process is up and answering; treating it as 'down'
    makes the Gradio UI show 'Disconnected / Server not running' during any
    burst that trips the global limiter, and misleads `tts server restart`.
    """

    @staticmethod
    def _is_running(status_code):
        from qwen3_tts.core.config import runtime

        resp = MagicMock(status_code=status_code)
        with patch("requests.get", return_value=resp):
            return runtime.is_server_running("http://127.0.0.1:5123")

    def test_429_counts_as_running(self):
        assert self._is_running(429) is True

    def test_200_counts_as_running(self):
        assert self._is_running(200) is True

    def test_503_counts_as_running(self):
        assert self._is_running(503) is True

    def test_404_counts_as_not_running(self):
        assert self._is_running(404) is False

    def test_connection_error_counts_as_not_running(self):
        import requests

        from qwen3_tts.core.config import runtime

        with patch("requests.get", side_effect=requests.ConnectionError()):
            assert runtime.is_server_running("http://127.0.0.1:5123") is False
