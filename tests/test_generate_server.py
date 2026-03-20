#!/usr/bin/env python3
"""Tests for qwen3_tts.interface.generate_server module.

Covers: ensure_server_running, load_model_on_server, generate_via_server,
_voice_param_for_log, and _run_single_generation.

Run: python -m pytest tests/test_generate_server.py -v
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

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

    class _DummyMarker:
        def __call__(self, func):
            return func
        def __getattr__(self, name):
            if name == "skipif":
                return _DummyMarkerFunc(name)
            return _DummyMarkerFunc(name)
        @property
        def unit(self):
            return self

    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()

    class _DummyPytest:
        mark = _DummyMark()

    pytest = _DummyPytest()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qwen3_tts.interface.generate_server import (
        ensure_server_running,
        load_model_on_server,
        generate_via_server,
        _voice_param_for_log,
        _run_single_generation,
    )
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires qwen3_tts.interface.generate_server")
_MOD = "qwen3_tts.interface.generate_server"
_CONFIG = {"server": {"host": "127.0.0.1", "port": 5123}}


# ---------------------------------------------------------------------------
# ensure_server_running
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
class TestEnsureServerRunning(unittest.TestCase):
    """Tests for ensure_server_running()."""

    def test_already_running_returns_true(self):
        """Return True immediately when server is already running."""
        with patch(f"{_MOD}.is_server_running", return_value=True):
            self.assertTrue(ensure_server_running(_CONFIG))

    def test_tts_cli_found_runs_successfully(self):
        """Use shutil.which to find tts, run subprocess.run, return True on success."""
        mock_result = MagicMock(returncode=0)
        with patch(f"{_MOD}.is_server_running", return_value=False), \
             patch("shutil.which", return_value="/usr/local/bin/tts"), \
             patch(f"{_MOD}.subprocess.run", return_value=mock_result) as mock_run:
            result = ensure_server_running(_CONFIG)
        self.assertTrue(result)
        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd, ["/usr/local/bin/tts", "server", "start"])

    def test_fallback_popen_starts_server(self):
        """When tts CLI not found, fall back to Popen and poll until ready."""
        mock_popen = MagicMock()
        with patch(f"{_MOD}.is_server_running", side_effect=[False, True]), \
             patch("shutil.which", return_value=None), \
             patch("builtins.open", mock_open()), \
             patch(f"{_MOD}.subprocess.Popen", mock_popen), \
             patch(f"{_MOD}.time.sleep", return_value=None):
            result = ensure_server_running(_CONFIG)
        self.assertTrue(result)
        self.assertTrue(mock_popen.called)

    def test_timeout_returns_false(self):
        """Return False after 300 poll iterations without server becoming ready."""
        mock_popen = MagicMock()
        with patch(f"{_MOD}.is_server_running", return_value=False), \
             patch("shutil.which", return_value=None), \
             patch("builtins.open", mock_open()), \
             patch(f"{_MOD}.subprocess.Popen", mock_popen), \
             patch(f"{_MOD}.time.sleep", return_value=None):
            result = ensure_server_running(_CONFIG)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# load_model_on_server
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
class TestLoadModelOnServer(unittest.TestCase):
    """Tests for load_model_on_server()."""

    def _mock_response(self, status_code, json_data):
        """Create a mock requests.Response."""
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data
        return resp

    @patch(f"{_MOD}.auth_headers", return_value={"Authorization": "Bearer tok"})
    @patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123")
    def test_success_loaded(self, _url, _auth):
        """Return True when server responds with status=loaded."""
        resp = self._mock_response(200, {"status": "loaded"})
        with patch("requests.post", return_value=resp):
            self.assertTrue(load_model_on_server(_CONFIG, "clone"))

    @patch(f"{_MOD}.auth_headers", return_value={"Authorization": "Bearer tok"})
    @patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123")
    def test_already_loaded(self, _url, _auth):
        """Return True when server responds with status=already_loaded."""
        resp = self._mock_response(200, {"status": "already_loaded"})
        with patch("requests.post", return_value=resp):
            self.assertTrue(load_model_on_server(_CONFIG, "design"))

    @patch(f"{_MOD}.auth_headers", return_value={"Authorization": "Bearer tok"})
    @patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123")
    def test_failure_returns_false(self, _url, _auth):
        """Return False when server responds with non-200 status."""
        resp = self._mock_response(500, {"error": "out of memory"})
        with patch("requests.post", return_value=resp):
            self.assertFalse(load_model_on_server(_CONFIG, "custom"))


# ---------------------------------------------------------------------------
# generate_via_server
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
class TestGenerateViaServer(unittest.TestCase):
    """Tests for generate_via_server()."""

    def _mock_response(self, status_code, json_data):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data
        return resp

    def _base_patches(self):
        """Return common patches for generate_via_server tests."""
        poller = MagicMock()
        return {
            "url": patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123"),
            "auth": patch(f"{_MOD}.auth_headers", return_value={}),
            "payload": patch(f"{_MOD}._build_generation_payload", return_value={}),
            "poller": patch(
                "qwen3_tts.interface.generate_interactive._ProgressPoller",
                return_value=poller,
            ),
        }

    def test_success_returns_results(self):
        """Return results list on HTTP 200."""
        resp = self._mock_response(200, {"results": [{"audio": "base64data"}]})
        patches = self._base_patches()
        with patches["url"], patches["auth"], patches["payload"], patches["poller"], \
             patch("requests.post", return_value=resp):
            results = generate_via_server(
                ["Hello"], "clone", _CONFIG, {},
                prompt_file="voice.pt",
            )
        self.assertEqual(results, [{"audio": "base64data"}])

    def test_503_model_not_loaded_auto_loads(self):
        """On 503 with model_not_loaded, prompt user and reload."""
        resp_503 = self._mock_response(503, {
            "error": "model_not_loaded",
            "model_type": "clone",
            "description": "Voice cloning",
        })
        resp_200 = self._mock_response(200, {"results": [{"audio": "ok"}]})
        patches = self._base_patches()
        with patches["url"], patches["auth"], patches["payload"], patches["poller"], \
             patch("requests.post", side_effect=[resp_503, resp_200]), \
             patch("builtins.input", return_value="y"), \
             patch(f"{_MOD}.load_model_on_server", return_value=True):
            results = generate_via_server(["Hi"], "clone", _CONFIG, {})
        self.assertEqual(results, [{"audio": "ok"}])

    def test_non_200_error_with_recovery(self):
        """Raise Exception with recovery suggestion on non-200 error."""
        resp = self._mock_response(500, {
            "error": "generation_failed",
            "detail": "OOM",
            "recovery": "restart",
        })
        patches = self._base_patches()
        with patches["url"], patches["auth"], patches["payload"], patches["poller"], \
             patch("requests.post", return_value=resp):
            with self.assertRaises(Exception) as ctx:
                generate_via_server(["Hello"], "clone", _CONFIG, {})
        self.assertIn("generation_failed", str(ctx.exception))
        self.assertIn("restart", str(ctx.exception))

    def test_503_non_json_raises(self):
        """Raise Exception when 503 response is not valid JSON."""
        import requests as req_mod
        resp = MagicMock()
        resp.status_code = 503
        resp.json.side_effect = req_mod.exceptions.JSONDecodeError("", "", 0)
        patches = self._base_patches()
        with patches["url"], patches["auth"], patches["payload"], patches["poller"], \
             patch("requests.post", return_value=resp):
            with self.assertRaises(Exception) as ctx:
                generate_via_server(["Hello"], "clone", _CONFIG, {})
        self.assertIn("503", str(ctx.exception))


# ---------------------------------------------------------------------------
# _voice_param_for_log
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
class TestVoiceParamForLog(unittest.TestCase):
    """Tests for _voice_param_for_log()."""

    def test_clone_mode_returns_prompt_file(self):
        """Clone mode returns the prompt_file argument."""
        result = _voice_param_for_log("clone", "my_voice.pt", None, None, None)
        self.assertEqual(result, "my_voice.pt")

    def test_design_mode_returns_voice_description(self):
        """Design mode returns the voice_description argument."""
        result = _voice_param_for_log("design", None, "A calm female voice", None, None)
        self.assertEqual(result, "A calm female voice")

    def test_custom_mode_with_instruct(self):
        """Custom mode returns speaker name with optional instruct parenthetical."""
        result = _voice_param_for_log("custom", None, None, "Ryan", "whisper softly")
        self.assertEqual(result, "Ryan (whisper softly)")

    def test_custom_mode_without_instruct(self):
        """Custom mode returns speaker name only when instruct is empty."""
        result = _voice_param_for_log("custom", None, None, "Ryan", "")
        self.assertEqual(result, "Ryan")


# ---------------------------------------------------------------------------
# _run_single_generation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
class TestRunSingleGeneration(unittest.TestCase):
    """Tests for _run_single_generation()."""

    def _make_args(self, **overrides):
        """Create a mock args namespace with generation flags."""
        args = MagicMock()
        args.stream = overrides.get("stream", False)
        args.trim_silence = overrides.get("trim_silence", False)
        args.normalize = overrides.get("normalize", False)
        args.speed = overrides.get("speed", None)
        args.pitch = overrides.get("pitch", None)
        args.play = overrides.get("play", False)
        args.no_open = overrides.get("no_open", True)
        args.no_transcript = overrides.get("no_transcript", False)
        return args

    @patch(f"{_MOD}.log_generation")
    @patch(f"{_MOD}._save_base64_result")
    @patch(f"{_MOD}.generate_via_server", return_value=[{"audio": "base64data", "sample_rate": 24000}])
    def test_server_mode_saves_audio(self, mock_gen, mock_save, mock_log):
        """Server mode calls generate_via_server and saves the result."""
        args = self._make_args()
        result = _run_single_generation(
            "Hello world", args, _CONFIG, {}, True, 500,
            "/tmp/out.wav", "clone", "English", "voice.pt",
            None, "Ryan",
        )
        self.assertTrue(result)
        mock_gen.assert_called_once()
        mock_save.assert_called_once_with(
            {"audio": "base64data", "sample_rate": 24000}, "/tmp/out.wav"
        )

    @patch(f"{_MOD}.log_generation")
    @patch(f"{_MOD}.generate_streaming")
    def test_streaming_mode_calls_generate_streaming(self, mock_stream, mock_log):
        """Streaming mode delegates to generate_streaming."""
        args = self._make_args(stream=True)
        result = _run_single_generation(
            "Hello world", args, _CONFIG, {}, True, 500,
            "/tmp/out.wav", "design", "English", None,
            "A calm voice", "Ryan",
        )
        self.assertTrue(result)
        mock_stream.assert_called_once()
        self.assertEqual(mock_stream.call_args.kwargs.get("voice_description"), "A calm voice")


if __name__ == "__main__":
    unittest.main()
