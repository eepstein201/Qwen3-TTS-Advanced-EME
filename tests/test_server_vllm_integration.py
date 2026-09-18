"""Tests for VLLMAdapter integration with FastAPI server (HIGH-2)."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


class TestVLLMLifecycleFunctions(unittest.IsolatedAsyncioTestCase):
    """HIGH-2: lifespan helpers for starting/stopping VLLMAdapter."""

    def test_maybe_start_vllm_adapter_exists(self):
        """_maybe_start_vllm_adapter must be importable from app_lifespan."""
        from qwen3_tts.server.app_lifespan import _maybe_start_vllm_adapter

        assert callable(_maybe_start_vllm_adapter)

    def test_maybe_stop_vllm_adapter_exists(self):
        """_maybe_stop_vllm_adapter must be importable from app_lifespan."""
        from qwen3_tts.server.app_lifespan import _maybe_stop_vllm_adapter

        assert callable(_maybe_stop_vllm_adapter)

    async def test_start_skips_when_backend_not_vllm(self):
        """_maybe_start_vllm_adapter does nothing when backend != vllm."""
        from qwen3_tts.server.app_lifespan import _maybe_start_vllm_adapter

        with patch("qwen3_tts.server.app_lifespan.get_backend", lambda: "mlx"):
            state = SimpleNamespace(vllm_adapter=None)
            await _maybe_start_vllm_adapter(state)
        assert state.vllm_adapter is None

    async def test_start_creates_adapter_when_backend_is_vllm(self):
        """_maybe_start_vllm_adapter creates and starts adapter when backend=vllm."""
        from qwen3_tts.server.app_lifespan import _maybe_start_vllm_adapter

        # Mock load_config to return config with vLLM enabled
        mock_config = {
            "vllm": {
                "enabled": True,
                "fallback_to_torch": True,
            }
        }

        mock_adapter = MagicMock()
        mock_adapter.start = AsyncMock()
        mock_adapter.port = 5124  # Mock port for client creation
        mock_cls = MagicMock(return_value=mock_adapter)
        # Patch where the lazy import resolves, not where it's referenced
        with (
            patch("qwen3_tts.server.app_lifespan.load_config", lambda: mock_config),
            patch("qwen3_tts.core.engine_vllm.VLLMAdapter", mock_cls),
        ):
            state = SimpleNamespace(vllm_adapter=None)
            await _maybe_start_vllm_adapter(state)

        mock_adapter.start.assert_awaited_once()
        assert state.vllm_adapter is mock_adapter

    async def test_stop_calls_adapter_stop(self):
        """_maybe_stop_vllm_adapter calls stop() and sets vllm_adapter to None."""
        from qwen3_tts.server.app_lifespan import _maybe_stop_vllm_adapter

        mock_adapter = MagicMock()
        state = SimpleNamespace(vllm_adapter=mock_adapter)
        await _maybe_stop_vllm_adapter(state)

        mock_adapter.stop.assert_called_once()
        assert state.vllm_adapter is None

    async def test_stop_noop_when_no_adapter(self):
        """_maybe_stop_vllm_adapter does nothing when vllm_adapter is None."""
        from qwen3_tts.server.app_lifespan import _maybe_stop_vllm_adapter

        state = SimpleNamespace(vllm_adapter=None)
        await _maybe_stop_vllm_adapter(state)  # Must not raise
        assert state.vllm_adapter is None


class TestVLLMStateInitialization(unittest.TestCase):
    """HIGH-2: app.state.vllm_adapter must be initialized in lifespan."""

    def test_lifespan_initializes_vllm_adapter_to_none(self):
        """The lifespan function sets app.state.vllm_adapter = None at startup."""
        import inspect

        from qwen3_tts.server import app_lifespan

        source = inspect.getsource(app_lifespan.lifespan)
        assert "vllm_adapter" in source, (
            "lifespan() must initialize app.state.vllm_adapter"
        )


# ===========================================================================
# /generate backend routing: the vLLM circuit-breaker decision block
# ===========================================================================

try:
    import numpy as np
    from fastapi.testclient import TestClient

    _HAS_ROUTING_DEPS = True
except ImportError:
    _HAS_ROUTING_DEPS = False

_APP_GENERATION = "qwen3_tts.server.app_generation"


@unittest.skipUnless(_HAS_ROUTING_DEPS, "requires fastapi and numpy")
class TestVLLMCircuitRouting(unittest.TestCase):
    """/generate picks vLLM or torch/MLX from config + circuit-breaker state.

    The block is pure decision logic over ``vllm_client.circuit_state`` and an
    awaitable adapter, so a fake client and adapter exercise every arm — no
    vLLM server is involved.
    """

    def setUp(self):
        from qwen3_tts.server.app import app
        from tests.conftest import _init_app_state, _restore_app_state, _save_app_state

        self._restore_app_state = _restore_app_state
        self._original_state = _save_app_state(app)
        _init_app_state(app, auth_token="test_token")
        app.state.models["clone"] = MagicMock()
        app.state.server_config = {
            "auto_shutdown_minutes": 0,
            "security": {"max_text_length": 50000, "max_batch_size": 20},
        }
        self.app = app
        self.client = TestClient(app, raise_server_exceptions=False)
        self.headers = {"Authorization": "Bearer test_token"}

    def tearDown(self):
        import os

        for entry in list(getattr(self.app.state, "gen_cache", {}).values()):
            main_file = entry.get("main_file")
            if main_file and os.path.exists(main_file):
                try:
                    os.remove(main_file)
                except OSError:
                    pass
        self._restore_app_state(self.app, self._original_state)

    def _post(
        self,
        text,
        *,
        enabled,
        fallback,
        circuit_state,
        adapter_error=None,
        with_adapter=True,
        with_client=True,
    ):
        """Drive /generate once; return (response, run_inference_mock, adapter)."""
        wav = np.zeros(4800, dtype=np.float32)

        adapter = None
        if with_adapter:
            adapter = MagicMock()
            if adapter_error is not None:
                adapter.generate = AsyncMock(side_effect=adapter_error)
            else:
                adapter.generate = AsyncMock(return_value=(wav, 24000))

        vllm_client = None
        if with_client:
            vllm_client = MagicMock()
            vllm_client.circuit_state = circuit_state

        self.app.state.server_config["vllm"] = {
            "enabled": enabled,
            "fallback_to_torch": fallback,
        }
        self.app.state.vllm_adapter = adapter
        self.app.state.vllm_client = vllm_client

        with (
            patch(
                f"{_APP_GENERATION}._check_memory_available", return_value=(True, 4000)
            ),
            patch("qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock()),
            patch(
                "qwen3_tts.core.engine.run_inference", return_value=(wav, 24000)
            ) as mock_inference,
            patch("soundfile.write"),
        ):
            resp = self.client.post(
                "/generate",
                json={"text": text, "mode": "clone", "prompt_file": "voice.wav"},
                headers=self.headers,
            )
        return resp, mock_inference, adapter

    def test_closed_circuit_routes_to_vllm_and_skips_local_inference(self):
        """A CLOSED circuit sends the work to vLLM, not to run_inference."""
        resp, mock_inference, adapter = self._post(
            "closed circuit routes to vllm",
            enabled=True,
            fallback=True,
            circuit_state="CLOSED",
        )

        self.assertEqual(resp.status_code, 200, resp.text)
        adapter.generate.assert_awaited_once()
        mock_inference.assert_not_called()

    def test_open_circuit_falls_back_to_local_inference(self):
        """An OPEN circuit bypasses vLLM entirely and still returns audio."""
        resp, mock_inference, adapter = self._post(
            "open circuit falls back",
            enabled=True,
            fallback=True,
            circuit_state="OPEN",
        )

        self.assertEqual(resp.status_code, 200, resp.text)
        adapter.generate.assert_not_awaited()
        mock_inference.assert_called_once()

    def test_open_circuit_without_fallback_fails_the_request(self):
        """fallback_to_torch=False must surface an error, not silently degrade.

        The whole point of disabling fallback is that the caller wants to know
        vLLM is unavailable rather than transparently get torch/MLX output.
        """
        resp, mock_inference, adapter = self._post(
            "open circuit no fallback",
            enabled=True,
            fallback=False,
            circuit_state="OPEN",
        )

        self.assertGreaterEqual(resp.status_code, 500)
        adapter.generate.assert_not_awaited()
        mock_inference.assert_not_called()

    def test_enabled_but_uninitialized_falls_back_to_local_inference(self):
        """vllm.enabled with no adapter/client is a fallback, not a crash."""
        resp, mock_inference, _ = self._post(
            "enabled but uninitialized",
            enabled=True,
            fallback=True,
            circuit_state="CLOSED",
            with_adapter=False,
            with_client=False,
        )

        self.assertEqual(resp.status_code, 200, resp.text)
        mock_inference.assert_called_once()

    def test_adapter_failure_falls_back_to_local_inference(self):
        """A vLLM generate() exception is absorbed when fallback is enabled."""
        resp, mock_inference, adapter = self._post(
            "adapter failure with fallback",
            enabled=True,
            fallback=True,
            circuit_state="CLOSED",
            adapter_error=RuntimeError("vllm exploded"),
        )

        self.assertEqual(resp.status_code, 200, resp.text)
        adapter.generate.assert_awaited_once()
        mock_inference.assert_called_once()

    def test_adapter_failure_without_fallback_fails_the_request(self):
        """A vLLM generate() exception propagates when fallback is disabled."""
        resp, mock_inference, adapter = self._post(
            "adapter failure no fallback",
            enabled=True,
            fallback=False,
            circuit_state="CLOSED",
            adapter_error=RuntimeError("vllm exploded"),
        )

        self.assertGreaterEqual(resp.status_code, 500)
        adapter.generate.assert_awaited_once()
        mock_inference.assert_not_called()

    def test_disabled_vllm_never_consults_the_adapter(self):
        """vllm.enabled=False ignores a present adapter entirely."""
        resp, mock_inference, adapter = self._post(
            "vllm disabled",
            enabled=False,
            fallback=True,
            circuit_state="CLOSED",
        )

        self.assertEqual(resp.status_code, 200, resp.text)
        adapter.generate.assert_not_awaited()
        mock_inference.assert_called_once()
