#!/usr/bin/env python3
"""Tests for qwen3_tts.interface.generate_server module.

Covers: ensure_server_running, load_model_on_server, generate_via_server,
_voice_param_for_log, and _run_single_generation.

Run: python -m pytest tests/test_generate_server.py -v
"""
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

    @patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123")
    def test_success_loaded(self, _url):
        """Return True when server responds with status=loaded."""
        resp = self._mock_response(200, {"status": "loaded"})
        with patch("qwen3_tts.core.http_client.server_request", return_value=resp):
            self.assertTrue(load_model_on_server(_CONFIG, "clone"))

    @patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123")
    def test_already_loaded(self, _url):
        """Return True when server responds with status=already_loaded."""
        resp = self._mock_response(200, {"status": "already_loaded"})
        with patch("qwen3_tts.core.http_client.server_request", return_value=resp):
            self.assertTrue(load_model_on_server(_CONFIG, "design"))

    @patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123")
    def test_failure_returns_false(self, _url):
        """Return False when server responds with non-200 status."""
        resp = self._mock_response(500, {"error": "out of memory"})
        with patch("qwen3_tts.core.http_client.server_request", return_value=resp):
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
        with patches["url"], patches["payload"], patches["poller"], \
             patch("qwen3_tts.core.http_client.server_request", return_value=resp):
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
        with patches["url"], patches["payload"], patches["poller"], \
             patch("qwen3_tts.core.http_client.server_request", side_effect=[resp_503, resp_200]), \
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
        with patches["url"], patches["payload"], patches["poller"], \
             patch("qwen3_tts.core.http_client.server_request", return_value=resp):
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
        with patches["url"], patches["payload"], patches["poller"], \
             patch("qwen3_tts.core.http_client.server_request", return_value=resp):
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


# ---------------------------------------------------------------------------
# launch_gradio_ui / build_ui_and_launch
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
class TestLaunchGradioUI(unittest.TestCase):
    """Tests for launch_gradio_ui and build_ui_and_launch."""

    def test_launch_gradio_ui_server_not_running(self):
        from qwen3_tts.interface.generate_server import launch_gradio_ui
        with patch(f"{_MOD}.ensure_server_running", return_value=False), \
             patch("builtins.print") as mock_print:
            launch_gradio_ui(_CONFIG)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Cannot launch", output)

    def test_launch_gradio_ui_success(self):
        from qwen3_tts.interface.generate_server import launch_gradio_ui
        with patch(f"{_MOD}.ensure_server_running", return_value=True), \
             patch(f"{_MOD}.build_ui_and_launch") as mock_launch, \
             patch("builtins.print"):
            launch_gradio_ui(_CONFIG)
        mock_launch.assert_called_once_with(_CONFIG)

    def test_build_ui_and_launch_no_port(self):
        from qwen3_tts.interface.generate_server import build_ui_and_launch
        with patch("qwen3_tts.interface.ui._find_available_port", return_value=None), \
             patch("qwen3_tts.interface.ui.build_ui"), \
             patch("builtins.print") as mock_print:
            build_ui_and_launch({"ui": {"port": 7860}})
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("No available port", output)

    def test_build_ui_and_launch_success(self):
        from qwen3_tts.interface.generate_server import build_ui_and_launch
        mock_demo = MagicMock()
        with patch("qwen3_tts.interface.ui._find_available_port", return_value=7861), \
             patch("qwen3_tts.interface.ui.build_ui", return_value=mock_demo), \
             patch("qwen3_tts.core.config.IN_COLAB", False):
            build_ui_and_launch({"ui": {"port": 7860}})
        mock_demo.launch.assert_called_once()
        self.assertEqual(mock_demo.launch.call_args[1]["server_port"], 7861)


# ---------------------------------------------------------------------------
# generate_streaming
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
class TestGenerateStreaming(unittest.TestCase):
    """Tests for generate_streaming()."""

    def _base_patches(self):
        return {
            "url": patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123"),
            "payload": patch(f"{_MOD}._build_generation_payload", return_value={}),
            "play": patch(f"{_MOD}.play_audio"),
        }

    def test_success_saves_combined(self):
        """Streaming collects chunks and saves combined wav."""
        import struct
        import numpy as np
        from qwen3_tts.interface.generate_server import generate_streaming
        # Build a mock streamed response with 2 chunks
        sr = 24000
        chunk = np.zeros(100, dtype="<f4")
        header = struct.pack("<II", sr, len(chunk.tobytes()))
        raw = header + chunk.tobytes() + header + chunk.tobytes()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [raw]

        patches = self._base_patches()
        with patches["url"], patches["payload"], patches["play"], \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
             patch("soundfile.write") as mock_sf, \
             patch("builtins.print"):
            result = generate_streaming(
                "Hello", "clone", _CONFIG, {}, "/tmp/out.wav",
                prompt_file="voice.pt",
            )
        self.assertEqual(result, "/tmp/out.wav")
        mock_sf.assert_called()
        # Combined should be 200 samples (2 chunks of 100)
        saved_array = mock_sf.call_args[0][1]
        self.assertEqual(len(saved_array), 200)

    def test_error_status_raises(self):
        """Non-200 response raises Exception."""
        from qwen3_tts.interface.generate_server import generate_streaming
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "OOM"}

        patches = self._base_patches()
        with patches["url"], patches["payload"], patches["play"], \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
             patch("builtins.print"):
            with self.assertRaises(Exception) as ctx:
                generate_streaming("Hello", "clone", _CONFIG, {}, "/tmp/out.wav")
        self.assertIn("OOM", str(ctx.exception))

    def test_no_chunks_returns_none(self):
        """Return None when no audio chunks received."""
        from qwen3_tts.interface.generate_server import generate_streaming
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = []

        patches = self._base_patches()
        with patches["url"], patches["payload"], patches["play"], \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
             patch("builtins.print"):
            result = generate_streaming("Hello", "clone", _CONFIG, {}, "/tmp/out.wav")
        self.assertIsNone(result)

    def test_request_error_raises(self):
        """RequestException re-raised as Exception."""
        import requests as req_mod
        from qwen3_tts.interface.generate_server import generate_streaming
        patches = self._base_patches()
        with patches["url"], patches["payload"], patches["play"], \
             patch("qwen3_tts.core.http_client.server_request", side_effect=req_mod.exceptions.ConnectionError("refused")), \
             patch("builtins.print"):
            with self.assertRaises(Exception) as ctx:
                generate_streaming("Hello", "clone", _CONFIG, {}, "/tmp/out.wav")
        self.assertIn("Streaming request failed", str(ctx.exception))


# ---------------------------------------------------------------------------
# generate_local
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
class TestGenerateLocal(unittest.TestCase):
    """Tests for generate_local()."""

    def test_clone_success(self):
        from qwen3_tts.interface.generate_server import generate_local
        mock_model = MagicMock()
        import numpy as np
        wav = np.zeros(1000, dtype=np.float32)
        with patch("qwen3_tts.core.engine.load_model", return_value=mock_model), \
             patch("qwen3_tts.core.engine.run_inference", return_value=(wav, 24000)), \
             patch("qwen3_tts.core.engine.load_voice_prompt", return_value="prompt_data"), \
             patch(f"{_MOD}.voice_prompt_exists", return_value=True), \
             patch("builtins.print"):
            result_wav, result_sr = generate_local(
                "Hello", "clone", {}, prompt_file="voice.pt",
            )
        self.assertEqual(result_sr, 24000)
        self.assertEqual(len(result_wav), 1000)

    def test_clone_no_prompt_exits(self):
        from qwen3_tts.interface.generate_server import generate_local
        with patch("qwen3_tts.core.engine.load_model"), \
             patch("builtins.print"):
            with self.assertRaises(SystemExit):
                generate_local("Hello", "clone", {}, prompt_file=None)

    def test_clone_prompt_not_found_mlx(self):
        from qwen3_tts.interface.generate_server import generate_local
        with patch("qwen3_tts.core.engine.load_model"), \
             patch(f"{_MOD}.voice_prompt_exists", return_value=False), \
             patch(f"{_MOD}.get_backend", return_value="mlx"), \
             patch("builtins.print"):
            with self.assertRaises(SystemExit):
                generate_local("Hello", "clone", {}, prompt_file="missing.pt")

    def test_clone_prompt_not_found_torch(self):
        from qwen3_tts.interface.generate_server import generate_local
        with patch("qwen3_tts.core.engine.load_model"), \
             patch(f"{_MOD}.voice_prompt_exists", return_value=False), \
             patch(f"{_MOD}.get_backend", return_value="torch"), \
             patch("builtins.print"):
            with self.assertRaises(SystemExit):
                generate_local("Hello", "clone", {}, prompt_file="missing.pt")

    def test_custom_mode_defaults_speaker(self):
        from qwen3_tts.interface.generate_server import generate_local
        import numpy as np
        wav = np.zeros(100, dtype=np.float32)
        with patch("qwen3_tts.core.engine.load_model"), \
             patch("qwen3_tts.core.engine.run_inference", return_value=(wav, 24000)) as mock_inf, \
             patch("builtins.print"):
            generate_local("Hello", "custom", {}, speaker=None)
        # Default speaker should be "Ryan"
        call_kwargs = mock_inf.call_args[1]
        self.assertEqual(call_kwargs["speaker"], "Ryan")

    def test_custom_with_instruct(self):
        from qwen3_tts.interface.generate_server import generate_local
        import numpy as np
        wav = np.zeros(100, dtype=np.float32)
        with patch("qwen3_tts.core.engine.load_model"), \
             patch("qwen3_tts.core.engine.run_inference", return_value=(wav, 24000)), \
             patch("builtins.print") as mock_print:
            generate_local("Hello", "custom", {}, speaker="Ryan", instruct="whisper")
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("whisper", output)

    def test_design_mode(self):
        from qwen3_tts.interface.generate_server import generate_local
        import numpy as np
        wav = np.zeros(100, dtype=np.float32)
        with patch("qwen3_tts.core.engine.load_model"), \
             patch("qwen3_tts.core.engine.run_inference", return_value=(wav, 24000)) as mock_inf, \
             patch("builtins.print"):
            generate_local("Hello", "design", {}, voice_description="warm female")
        call_kwargs = mock_inf.call_args[1]
        self.assertEqual(call_kwargs["voice_description"], "warm female")


# ---------------------------------------------------------------------------
# _run_single_generation — additional paths
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
class TestRunSingleGenerationExtended(unittest.TestCase):
    """Additional path coverage for _run_single_generation."""

    def _make_args(self, **overrides):
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
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, s, a: w)
    @patch(f"{_MOD}.generate_via_server", return_value=[{"audio": "base64", "sample_rate": 24000}])
    @patch(f"{_MOD}._decode_base64_result")
    def test_server_with_processing(self, mock_decode, mock_gen, mock_process, mock_log):
        """Server mode with audio processing decodes + processes + saves."""
        import numpy as np
        wav = np.zeros(100, dtype=np.float32)
        mock_decode.return_value = (wav, 24000)
        args = self._make_args(trim_silence=True)
        with patch("soundfile.write"):
            _run_single_generation(
                "Hello", args, _CONFIG, {}, True, 500,
                "/tmp/out.wav", "clone", "English", "voice.pt",
                None, "Ryan",
            )
        mock_decode.assert_called_once()
        mock_process.assert_called_once()

    @patch(f"{_MOD}.log_generation")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, s, a: w)
    @patch(f"{_MOD}.generate_local")
    def test_local_clone_mode(self, mock_gen, mock_process, mock_log):
        """Local mode calls generate_local and processes audio."""
        import numpy as np
        wav = np.zeros(100, dtype=np.float32)
        mock_gen.return_value = (wav, 24000)
        args = self._make_args()
        with patch("soundfile.write"):
            _run_single_generation(
                "Hello", args, _CONFIG, {}, False, 500,
                "/tmp/out.wav", "clone", "English", "voice.pt",
                None, "Ryan",
            )
        mock_gen.assert_called_once()
        self.assertEqual(mock_gen.call_args[1].get("prompt_file"), "voice.pt")

    @patch(f"{_MOD}.log_generation")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, s, a: w)
    @patch(f"{_MOD}.generate_local")
    def test_local_design_mode(self, mock_gen, mock_process, mock_log):
        """Local design mode passes voice_description."""
        import numpy as np
        mock_gen.return_value = (np.zeros(100, dtype=np.float32), 24000)
        args = self._make_args()
        with patch("soundfile.write"):
            _run_single_generation(
                "Hello", args, _CONFIG, {}, False, 500,
                "/tmp/out.wav", "design", "English", None,
                "warm female", "Ryan",
            )
        self.assertEqual(mock_gen.call_args[1].get("voice_description"), "warm female")

    @patch(f"{_MOD}.log_generation")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, s, a: w)
    @patch(f"{_MOD}.generate_local")
    def test_local_custom_mode(self, mock_gen, mock_process, mock_log):
        """Local custom mode passes speaker and instruct."""
        import numpy as np
        mock_gen.return_value = (np.zeros(100, dtype=np.float32), 24000)
        args = self._make_args()
        with patch("soundfile.write"):
            _run_single_generation(
                "Hello", args, _CONFIG, {}, False, 500,
                "/tmp/out.wav", "custom", "English", None,
                None, "Ryan", instruct="whisper",
            )
        self.assertEqual(mock_gen.call_args[1].get("speaker"), "Ryan")
        self.assertEqual(mock_gen.call_args[1].get("instruct"), "whisper")

    @patch(f"{_MOD}.log_generation")
    @patch(f"{_MOD}.play_audio")
    @patch(f"{_MOD}._save_base64_result")
    @patch(f"{_MOD}.generate_via_server", return_value=[{"audio": "base64"}])
    def test_play_flag_calls_play_audio(self, mock_gen, mock_save, mock_play, mock_log):
        """With play=True, play_audio is called after generation."""
        args = self._make_args(play=True)
        _run_single_generation(
            "Hello", args, _CONFIG, {}, True, 500,
            "/tmp/out.wav", "clone", "English", "voice.pt",
            None, "Ryan",
        )
        mock_play.assert_called_once_with("/tmp/out.wav")

    @patch(f"{_MOD}.log_generation")
    @patch(f"{_MOD}._save_base64_result")
    @patch(f"{_MOD}.generate_via_server", return_value=[{"audio": "base64"}])
    def test_no_play_opens_file(self, mock_gen, mock_save, mock_log):
        """With play=False and no_open=False, open_file is called."""
        args = self._make_args(play=False, no_open=False)
        with patch("qwen3_tts.interface.generate_helpers.open_file") as mock_open:
            _run_single_generation(
                "Hello", args, _CONFIG, {}, True, 500,
                "/tmp/out.wav", "clone", "English", "voice.pt",
                None, "Ryan",
            )
        mock_open.assert_called_once_with("/tmp/out.wav")

    @patch(f"{_MOD}.log_generation")
    @patch(f"{_MOD}.generate_streaming")
    def test_streaming_custom_mode(self, mock_stream, mock_log):
        """Streaming custom mode passes speaker."""
        args = self._make_args(stream=True)
        _run_single_generation(
            "Hello", args, _CONFIG, {}, True, 500,
            "/tmp/out.wav", "custom", "English", None,
            None, "Ryan", instruct="whisper",
        )
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs.get("speaker"), "Ryan")
        self.assertEqual(call_kwargs.get("instruct"), "whisper")

    @patch(f"{_MOD}.log_generation")
    @patch(f"{_MOD}.generate_streaming")
    def test_streaming_clone_mode(self, mock_stream, mock_log):
        """Streaming clone mode passes prompt_file and no_transcript."""
        args = self._make_args(stream=True, no_transcript=True)
        _run_single_generation(
            "Hello", args, _CONFIG, {}, True, 500,
            "/tmp/out.wav", "clone", "English", "voice.pt",
            None, "Ryan",
        )
        call_kwargs = mock_stream.call_args[1]
        self.assertEqual(call_kwargs.get("prompt_file"), "voice.pt")
        self.assertTrue(call_kwargs.get("x_vector_only_mode"))


# ---------------------------------------------------------------------------
# generate_via_server — additional paths
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
class TestGenerateViaServerExtended(unittest.TestCase):
    """Additional paths for generate_via_server."""

    def _mock_response(self, status_code, json_data):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data
        return resp

    def _base_patches(self):
        poller = MagicMock()
        return {
            "url": patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123"),
            "payload": patch(f"{_MOD}._build_generation_payload", return_value={}),
            "poller": patch(
                "qwen3_tts.interface.generate_interactive._ProgressPoller",
                return_value=poller,
            ),
        }

    def test_503_user_declines_load(self):
        """User says 'n' to model load prompt — raises."""
        resp_503 = self._mock_response(503, {
            "error": "model_not_loaded",
            "model_type": "clone",
            "description": "Clone mode",
        })
        patches = self._base_patches()
        with patches["url"], patches["payload"], patches["poller"], \
             patch("qwen3_tts.core.http_client.server_request", return_value=resp_503), \
             patch("builtins.input", return_value="n"), \
             patch("builtins.print"):
            with self.assertRaises(Exception) as ctx:
                generate_via_server(["Hi"], "clone", _CONFIG, {})
        self.assertIn("not loaded", str(ctx.exception))

    def test_503_load_fails(self):
        """Model load attempt fails — raises."""
        resp_503 = self._mock_response(503, {
            "error": "model_not_loaded",
            "model_type": "clone",
            "description": "Clone",
        })
        patches = self._base_patches()
        with patches["url"], patches["payload"], patches["poller"], \
             patch("qwen3_tts.core.http_client.server_request", return_value=resp_503), \
             patch("builtins.input", return_value="y"), \
             patch(f"{_MOD}.load_model_on_server", return_value=False), \
             patch("builtins.print"):
            with self.assertRaises(Exception) as ctx:
                generate_via_server(["Hi"], "clone", _CONFIG, {})
        self.assertIn("Failed to load", str(ctx.exception))

    def test_error_with_config_recovery(self):
        """Error with recovery=config shows config path suggestion."""
        resp = self._mock_response(422, {
            "error": "invalid_config",
            "detail": "bad mode",
            "recovery": "config",
        })
        patches = self._base_patches()
        with patches["url"], patches["payload"], patches["poller"], \
             patch("qwen3_tts.core.http_client.server_request", return_value=resp):
            with self.assertRaises(Exception) as ctx:
                generate_via_server(["Hi"], "clone", _CONFIG, {})
        self.assertIn("configuration", str(ctx.exception))

    def test_error_with_retry_recovery(self):
        """Error with recovery=retry shows transient suggestion."""
        resp = self._mock_response(500, {
            "error": "timeout",
            "detail": "",
            "recovery": "retry",
        })
        patches = self._base_patches()
        with patches["url"], patches["payload"], patches["poller"], \
             patch("qwen3_tts.core.http_client.server_request", return_value=resp):
            with self.assertRaises(Exception) as ctx:
                generate_via_server(["Hi"], "clone", _CONFIG, {})
        self.assertIn("transient", str(ctx.exception))

    def test_non_200_non_json_error(self):
        """Non-200 with non-JSON body still raises with status code."""
        import requests as req_mod
        resp = MagicMock()
        resp.status_code = 502
        resp.json.side_effect = req_mod.exceptions.JSONDecodeError("", "", 0)
        patches = self._base_patches()
        with patches["url"], patches["payload"], patches["poller"], \
             patch("qwen3_tts.core.http_client.server_request", return_value=resp):
            with self.assertRaises(Exception) as ctx:
                generate_via_server(["Hi"], "clone", _CONFIG, {})
        self.assertIn("502", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
