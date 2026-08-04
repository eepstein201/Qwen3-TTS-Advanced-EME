#!/usr/bin/env python3
"""TDD tests for Python review remediation fixes.

Each test documents which review finding it validates (C-1, C-2, C-3, H-1, H-3, H-4, H-5).
These tests are written FIRST (RED), then implementations are fixed to make them pass (GREEN).

No GPU, models, or running server required.

Run: pytest tests/test_python_review_fixes.py -v --tb=short
"""
import asyncio
import os
import signal
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

try:
    import numpy as np
    from fastapi.testclient import TestClient
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi, numpy")


# ---------------------------------------------------------------------------
# C-1: Missing asyncio.to_thread on unload_model, load_asr, unload_asr endpoints
# ---------------------------------------------------------------------------

@_skip
class TestAsyncEndpointsUseToThread(unittest.TestCase):
    """C-1: Sync handlers must be dispatched via asyncio.to_thread, not called directly."""

    def _make_state(self):
        from qwen3_tts.server.app import app
        state = app.state
        state.auth_token = "test_token"
        if not hasattr(state, "server_config"):
            state.server_config = {}
        if not hasattr(state, "models"):
            state.models = {}
        if not hasattr(state, "generation_state"):
            state.generation_state = {"active": False}
        if not hasattr(state, "activity_timer"):
            state.activity_timer = None
        return state

    def _make_client(self):
        from qwen3_tts.server.app import app
        self._make_state()
        return TestClient(app, raise_server_exceptions=False)

    def test_unload_model_dispatched_via_to_thread(self):
        """unload_model endpoint must use asyncio.to_thread for the handler call."""
        from qwen3_tts.server.app import app
        self._make_state()
        called_via_to_thread = []

        original_to_thread = asyncio.to_thread

        async def mock_to_thread(fn, *args, **kwargs):
            called_via_to_thread.append(fn.__name__)
            return await original_to_thread(fn, *args, **kwargs)

        with patch("qwen3_tts.server.app.asyncio.to_thread", side_effect=mock_to_thread):
            with patch("qwen3_tts.server.app_models.handle_unload_model", return_value={"status": "unloaded"}):
                client = TestClient(app, raise_server_exceptions=False)
                client.post(
                    "/unload-model",
                    json={"model_type": "clone"},
                    headers={"Authorization": "Bearer test_token"},
                )
        self.assertIn(
            "handle_unload_model", called_via_to_thread,
            "handle_unload_model must be dispatched via asyncio.to_thread"
        )

    def test_load_asr_dispatched_via_to_thread(self):
        """load_asr endpoint must use asyncio.to_thread for the handler call."""
        from qwen3_tts.server.app import app
        self._make_state()
        called_via_to_thread = []

        original_to_thread = asyncio.to_thread

        async def mock_to_thread(fn, *args, **kwargs):
            called_via_to_thread.append(fn.__name__)
            return await original_to_thread(fn, *args, **kwargs)

        with patch("qwen3_tts.server.app.asyncio.to_thread", side_effect=mock_to_thread):
            with patch("qwen3_tts.server.app_models.handle_load_asr", return_value={"status": "loaded"}):
                client = TestClient(app, raise_server_exceptions=False)
                client.post("/load-asr", headers={"Authorization": "Bearer test_token"})
        self.assertIn(
            "handle_load_asr", called_via_to_thread,
            "handle_load_asr must be dispatched via asyncio.to_thread"
        )

    def test_unload_asr_dispatched_via_to_thread(self):
        """unload_asr endpoint must use asyncio.to_thread for the handler call."""
        from qwen3_tts.server.app import app
        self._make_state()
        called_via_to_thread = []

        original_to_thread = asyncio.to_thread

        async def mock_to_thread(fn, *args, **kwargs):
            called_via_to_thread.append(fn.__name__)
            return await original_to_thread(fn, *args, **kwargs)

        with patch("qwen3_tts.server.app.asyncio.to_thread", side_effect=mock_to_thread):
            with patch("qwen3_tts.server.app_models.handle_unload_asr", return_value={"status": "unloaded"}):
                client = TestClient(app, raise_server_exceptions=False)
                client.post("/unload-asr", headers={"Authorization": "Bearer test_token"})
        self.assertIn(
            "handle_unload_asr", called_via_to_thread,
            "handle_unload_asr must be dispatched via asyncio.to_thread"
        )


# ---------------------------------------------------------------------------
# C-2: Missing return guard after _error_response() in handler except blocks
# ---------------------------------------------------------------------------

class TestErrorResponseReturnGuard(unittest.TestCase):
    """C-2: When _error_response doesn't raise, handler must not return success dict."""

    def _make_state(self):
        state = MagicMock()
        state.models = MagicMock()
        state.models.get.return_value = None  # not loaded by default
        state.model_load_errors = {}
        state.model_load_times = {}
        return state

    def test_handle_load_model_no_success_on_import_error(self):
        """handle_load_model must not return success dict when ImportError occurs."""
        from qwen3_tts.server.app_models import handle_load_model

        state = self._make_state()
        state.models.get.return_value = None  # not loaded

        req = MagicMock()
        req.model_type = "clone"

        # Mock _error_response to NOT raise — simulates future refactor safety
        with patch("qwen3_tts.server.app_models._error_response") as mock_err:
            mock_err.return_value = None  # doesn't raise
            with patch("qwen3_tts.core.engine.load_model", side_effect=ImportError("no backend")):
                result = handle_load_model(state, req)

        # After fix: result is None (explicit return after _error_response)
        # Before fix: result is {"status": "loaded", ...} — THIS IS THE BUG
        self.assertIsNone(
            result,
            f"handle_load_model returned {result!r} on ImportError path — "
            "expected None (explicit return guard after _error_response)"
        )

    def test_handle_load_asr_no_success_on_import_error(self):
        """handle_load_asr must not fall through to None return without explicit guard."""
        from qwen3_tts.server.app_models import handle_load_asr

        state = MagicMock()

        with patch("qwen3_tts.server.app_models._error_response") as mock_err:
            mock_err.return_value = None
            with patch("qwen3_tts.core.engine.is_asr_loaded", return_value=False):
                with patch("qwen3_tts.core.engine.load_asr_model", side_effect=ImportError("no whisper")):
                    result = handle_load_asr(state)

        # After fix: result is None (explicit return)
        # Before fix: also None BUT only because function has no return after except block —
        # the implicit None is acceptable here but we want explicit `return` for clarity
        # The critical check: no success dict was returned
        self.assertNotEqual(
            result, {"status": "loaded"},
            "handle_load_asr must not return a loaded-success dict on ImportError"
        )

    def test_handle_create_voice_prompt_no_success_on_import_error(self):
        """handle_create_voice_prompt must not return success dict when ImportError occurs."""
        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = MagicMock()
        state.models = {"clone": MagicMock()}

        req = MagicMock()
        req.name = "myvoice"
        req.no_transcript = True
        req.transcript = ""

        import base64

        # Create minimal valid WAV bytes for base64
        import io
        import struct
        buf = io.BytesIO()
        # 1-second silence at 16kHz, 16-bit mono
        n_samples = 16000
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + n_samples * 2))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16))
        buf.write(b"data")
        buf.write(struct.pack("<I", n_samples * 2))
        buf.write(b"\x00" * (n_samples * 2))
        wav_bytes = buf.getvalue()
        req.audio_base64 = base64.b64encode(wav_bytes).decode()

        with patch("qwen3_tts.server.app_prompts._error_response") as mock_err:
            mock_err.return_value = None
            with patch("qwen3_tts.core.engine.load_audio_for_cloning", side_effect=ImportError("no torchaudio")):
                result = handle_create_voice_prompt(state, req)

        self.assertIsNone(
            result,
            f"handle_create_voice_prompt returned {result!r} on ImportError — "
            "expected None (explicit return guard)"
        )


# ---------------------------------------------------------------------------
# C-3: WebSocket _stream_generation must acquire inference_lock
# ---------------------------------------------------------------------------

@_skip
class TestWebSocketInferenceLock(unittest.IsolatedAsyncioTestCase):
    """C-3: _stream_generation must acquire app_state.inference_lock during inference."""

    async def test_stream_generation_acquires_inference_lock(self):
        """inference_lock must be held while the inference thread runs."""
        from qwen3_tts.server.websocket import _stream_generation

        acquired = []
        released = []

        class TrackingLock:
            async def __aenter__(self):
                acquired.append(True)
                return self
            async def __aexit__(self, *args):
                released.append(True)

        # Minimal mock WebSocket
        ws = AsyncMock()
        sent_messages = []
        ws.send_json = AsyncMock(side_effect=lambda m: sent_messages.append(m))
        ws.send_bytes = AsyncMock()

        # Mock model
        model = MagicMock()

        app_state = MagicMock()
        app_state.models = {"clone": model}
        app_state.server_config = {"security": {"max_text_length": 10000}}
        app_state.inference_lock = TrackingLock()

        stop_event = threading.Event()

        # Mock inference to immediately return sentinel (no actual audio)
        def mock_run_inference_streaming(**kwargs):
            return iter([])  # no chunks

        # Provide a valid clone prompt so the request passes H5's pre-lock
        # prompt validation and actually reaches inference (and the lock).
        # Pre-H5 clone validation happened inside the inference thread (under
        # the lock); H5 moved it before the lock, so a missing prompt now
        # correctly short-circuits WITHOUT acquiring the lock. This test must
        # therefore supply a prompt to exercise the lock-acquisition path.
        with patch("qwen3_tts.core.engine.run_inference_streaming", mock_run_inference_streaming), \
             patch("qwen3_tts.server.validation._validate_generation_request", return_value=None), \
             patch("qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock()):
            await _stream_generation(
                websocket=ws,
                app_state=app_state,
                text="hello",
                mode="clone",
                data={"prompt_file": "voice.pt"},
                stop_event=stop_event,
            )

        self.assertTrue(
            len(acquired) > 0,
            "_stream_generation must acquire app_state.inference_lock before running inference"
        )
        self.assertEqual(
            len(acquired), len(released),
            "inference_lock must be released after inference completes"
        )


# ---------------------------------------------------------------------------
# H-1: save_config must be wrapped in asyncio.to_thread in async handlers
# ---------------------------------------------------------------------------

@_skip
class TestSaveConfigAsyncDispatch(unittest.TestCase):
    """H-1: save_config called in async handlers must be dispatched via asyncio.to_thread."""

    def _make_client_with_state(self):
        from qwen3_tts.server.app import app
        state = app.state
        state.auth_token = "test_token"
        if not hasattr(state, "server_config"):
            state.server_config = {"advanced": {}, "models": {}}
        if not hasattr(state, "models"):
            state.models = {}
        if not hasattr(state, "activity_timer"):
            state.activity_timer = None
        return TestClient(app, raise_server_exceptions=False)

    def test_update_model_config_save_config_via_to_thread(self):
        """handle_update_model_config must call save_config via asyncio.to_thread."""
        import asyncio as _asyncio
        to_thread_calls = []

        original_to_thread = _asyncio.to_thread

        async def tracking_to_thread(fn, *args, **kwargs):
            to_thread_calls.append(getattr(fn, "__name__", str(fn)))
            return await original_to_thread(fn, *args, **kwargs)

        # app_models imports asyncio at module level after fix — patch it there
        with patch("qwen3_tts.server.app_models.asyncio.to_thread",
                   side_effect=tracking_to_thread):
            with patch("qwen3_tts.server.app_models.save_config"):
                with patch("qwen3_tts.core.config.load_config",
                           return_value={"advanced": {}}):
                    client = self._make_client_with_state()
                    client.post(
                        "/update-model-config",
                        json={"model_size": "0.6B"},
                        headers={"Authorization": "Bearer test_token"},
                    )

        self.assertTrue(
            any("save_config" in name for name in to_thread_calls),
            f"save_config must be dispatched via asyncio.to_thread in handle_update_model_config; got {to_thread_calls}"
        )


# ---------------------------------------------------------------------------
# H-3: auto_shutdown must use os.kill(SIGTERM) not sys.exit(0)
# ---------------------------------------------------------------------------

class TestAutoShutdownSignal(unittest.TestCase):
    """H-3: auto_shutdown must send SIGTERM via os.kill, not sys.exit(0)."""

    def test_auto_shutdown_uses_os_kill_sigterm(self):
        """auto_shutdown must call os.kill(os.getpid(), signal.SIGTERM) not sys.exit."""
        from qwen3_tts.server.app_lifespan import auto_shutdown

        app_state = MagicMock()
        app_state.server_config = {"auto_shutdown_minutes": 30}

        os_kill_calls = []
        sys_exit_calls = []

        with patch("qwen3_tts.server.app_lifespan.os.kill",
                   side_effect=lambda pid, sig: os_kill_calls.append((pid, sig))):
            with patch("qwen3_tts.server.app_lifespan.sys.exit",
                       side_effect=lambda code: sys_exit_calls.append(code)):
                with patch("qwen3_tts.server.app_lifespan.cleanup_resources"):
                    auto_shutdown(app_state)

        self.assertEqual(len(sys_exit_calls), 0,
                         "auto_shutdown must not call sys.exit() — use os.kill(SIGTERM) instead")
        self.assertTrue(len(os_kill_calls) > 0,
                        "auto_shutdown must call os.kill(os.getpid(), signal.SIGTERM)")
        if os_kill_calls:
            pid, sig = os_kill_calls[0]
            self.assertEqual(pid, os.getpid(), "os.kill must target the current process")
            self.assertEqual(sig, signal.SIGTERM, "os.kill must send SIGTERM signal")


# ---------------------------------------------------------------------------
# H-4: audio_processing broad except must not swallow AttributeError
# ---------------------------------------------------------------------------

class TestAudioProcessingNarrowExcept(unittest.TestCase):
    """H-4: torchaudio fallback except must not catch AttributeError (programming bugs)."""

    def test_attribute_error_in_torchaudio_propagates(self):
        """AttributeError from torchaudio (a programming bug) must propagate, not be swallowed."""
        from qwen3_tts.core.engine.audio_processing import load_audio

        # torchaudio is lazily imported inside load_audio — patch via sys.modules
        mock_ta = MagicMock()
        mock_ta.load.side_effect = AttributeError("torchaudio API changed: no attribute 'load'")

        import sys as _sys
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            import numpy as np_
            import soundfile as sf
            sf.write(tmp_path, np_.zeros(1000, dtype=np_.float32), 16000)

            with patch("qwen3_tts.core.engine.audio_processing.get_audio_loader",
                       return_value="torchaudio"):
                with patch.dict(_sys.modules, {"torchaudio": mock_ta}):
                    # After fix (narrow except), AttributeError must propagate
                    # Before fix (broad except Exception), it's swallowed silently
                    with self.assertRaises(AttributeError,
                                           msg="AttributeError must propagate after except narrowing"):
                        load_audio(tmp_path)
        except ImportError:
            self.skipTest("soundfile not available")
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Modern-torchaudio compat: torchaudio.info was removed in torchaudio>=2.11
# ---------------------------------------------------------------------------

class TestLoadAudioForCloningModernTorchaudio(unittest.TestCase):
    """load_audio_for_cloning must work on modern torchaudio (>=2.11), which
    removed ``torchaudio.info``. Regression for the silent voice-prompt-build
    failure: ``module 'torchaudio' has no attribute 'info'`` aborted cloning on
    every modern-torchaudio environment (Docker, Colab, install.sh, local)."""

    def test_does_not_call_removed_torchaudio_info(self):
        """Must not call torchaudio.info (removed in torchaudio>=2.11).

        A fake torchaudio exposes load/transforms but no info attribute
        (mimicking modern torchaudio). Before the fix, torchaudio.info(...)
        raised AttributeError that escaped the narrow except and aborted the
        build; after the fix .info is never touched and the OSError from load
        falls through to the soundfile fallback.
        """
        import sys as _sys
        import tempfile
        import types

        from qwen3_tts.core.engine.audio_processing import load_audio_for_cloning

        def _raise_oserror(*_a, **_k):
            raise OSError("force soundfile fallback")

        fake_ta = types.SimpleNamespace(
            load=_raise_oserror,
            transforms=types.SimpleNamespace(
                Resample=lambda *_a, **_k: (lambda x: x)
            ),
        )
        # SimpleNamespace genuinely lacks .info -> AttributeError on access.
        with self.assertRaises(AttributeError):
            _ = fake_ta.info

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            import numpy as _np
            import soundfile as _sf
            _sf.write(tmp_path, _np.zeros(24000, dtype=_np.float32), 24000)

            with patch(
                "qwen3_tts.core.engine.audio_processing.get_audio_loader",
                return_value="torchaudio",
            ):
                with patch.dict(_sys.modules, {"torchaudio": fake_ta}):
                    audio, sr = load_audio_for_cloning(tmp_path)
            self.assertEqual(sr, 24000)
            self.assertEqual(len(audio), 24000)
        except ImportError:
            self.skipTest("soundfile not available")
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# H-5: cli_config edit must not mutate original config dict
# ---------------------------------------------------------------------------

class TestCliConfigImmutability(unittest.TestCase):
    """H-5: cli_config edit command must not mutate the loaded config dict in-place."""

    def test_config_edit_does_not_mutate_original(self):
        """The config dict returned by load_config must not be modified in-place."""
        try:
            from click.testing import CliRunner

            from qwen3_tts.cli_config import config as config_group
        except ImportError:
            self.skipTest("click or cli_config not importable")

        original_config = {
            "advanced": {"backend": "mlx", "model_size": "1.7B"},
            "server": {"host": "127.0.0.1", "port": 5123},
        }
        # Deep copy to verify mutation
        import copy
        config_snapshot = copy.deepcopy(original_config)

        saved_configs = []

        # load_config/save_config are lazily imported in cli_config — patch at source module
        with patch("qwen3_tts.core.config.load_config", return_value=original_config):
            with patch("qwen3_tts.core.config.save_config",
                       side_effect=lambda c: saved_configs.append(c)):
                runner = CliRunner()
                runner.invoke(config_group, ["edit", "--backend", "torch"])

        # The original dict must be unchanged (immutable update pattern)
        self.assertEqual(
            original_config, config_snapshot,
            "load_config() result must not be mutated in-place; use {**config, ...} pattern"
        )

        # The saved config should have the new backend
        if saved_configs:
            self.assertEqual(saved_configs[-1]["advanced"]["backend"], "torch",
                             "Saved config should have updated backend value")


# ---------------------------------------------------------------------------
# H-2: generate_dialogue numpy concatenation (behavior correctness check)
# ---------------------------------------------------------------------------

class TestGenerateDialogueNumpyConcat(unittest.TestCase):
    """H-2: generate_dialogue combines audio with np.concatenate (not list.extend)."""

    @unittest.skipUnless(HAS_DEPS, "requires numpy")
    def test_concatenate_produces_correct_shape(self):
        """np.concatenate combination produces correct total length."""
        # Directly test the concatenation pattern used after the fix.
        sr = 16000
        pause_ms = 300
        silence_samples = int(sr * pause_ms / 1000)

        chunk_a = np.ones(sr, dtype=np.float32)
        chunk_b = np.ones(sr, dtype=np.float32) * 2
        all_audio = [chunk_a, chunk_b]

        # Fixed pattern: np.concatenate
        parts = []
        for i, wav in enumerate(all_audio):
            parts.append(wav)
            if i < len(all_audio) - 1:
                parts.append(np.zeros(silence_samples, dtype=np.float32))
        combined = np.concatenate(parts)

        expected_length = sr + silence_samples + sr
        self.assertEqual(len(combined), expected_length)
        # Values must be preserved correctly
        np.testing.assert_array_equal(combined[:sr], chunk_a)
        np.testing.assert_array_equal(combined[sr:sr + silence_samples], np.zeros(silence_samples))
        np.testing.assert_array_equal(combined[sr + silence_samples:], chunk_b)

    @unittest.skipUnless(HAS_DEPS, "requires numpy")
    def test_generator_uses_concatenate_not_extend(self):
        """generate_dialogue source uses np.concatenate pattern (not list.extend)."""
        import inspect

        from qwen3_tts.server.client import generator as gen_mod
        src = inspect.getsource(gen_mod.GeneratorMixin.generate_dialogue)
        self.assertIn("np.concatenate", src, "generate_dialogue must use np.concatenate")
        self.assertNotIn("combined.extend", src, "generate_dialogue must not use list.extend")


if __name__ == "__main__":
    unittest.main()
