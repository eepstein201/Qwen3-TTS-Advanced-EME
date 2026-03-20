"""Extended tests for qwen3_tts/server/client/ submodules.

Covers generator.py, models.py, and voices.py with mocked HTTP calls.
No running server or GPU required.

Run with:
    python -m pytest tests/test_client_extended.py -v --tb=short
"""

import base64
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

try:
    import soundfile  # noqa: F401 — verify availability

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import numpy as np  # noqa: F401 — availability check; used in test methods

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

HAS_DEPS = HAS_SOUNDFILE and HAS_NUMPY

_skip = unittest.skipUnless(HAS_DEPS, "soundfile and numpy required")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(data=None):
    """Create a temp config file and return its path."""
    if data is None:
        data = {
            "server": {"host": "127.0.0.1", "port": 5123},
            "presets": {"consistent": {"temperature": 0.5}},
            "aliases": {
                "narrator": {"prompt": "narrator.pt", "mode": "clone"},
                "designer": {"mode": "design", "description": "warm female"},
            },
            "generation": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            "output_directory": "~/Downloads",
            "default_clone_prompt": "default.pt",
            "default_voice_description": "neutral voice",
            "default_speaker": "ryan",
            "language": "English",
            "prosody_presets": {"excited": "Speak with excitement and energy"},
        }
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return f.name


def _make_audio_wav_bytes():
    """Return minimal valid WAV bytes using soundfile."""
    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    samples = np.zeros(480, dtype=np.float32)
    sf.write(buf, samples, 24000, format="WAV")
    return buf.getvalue()


def _mock_generate_response():
    """Build a mock 200 response from /generate with valid base64 audio."""
    audio_b64 = base64.b64encode(_make_audio_wav_bytes()).decode()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"results": [{"audio_base64": audio_b64}]}
    return resp


def _mock_error_response(status=500, message="something went wrong"):
    """Build a mock error response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"error": message}
    return resp


def _client_with_server(config_path):
    """Return (client, session_mock) with is_server_running patched True."""
    from qwen3_tts.server.client import TTSClient

    client = TTSClient(config_path=config_path)
    session = MagicMock()
    client._session = session
    return client, session


# ============================================================================
# generator.py tests
# ============================================================================


@_skip
class TestGenerateVoiceAliasResolution(unittest.TestCase):
    """generate() voice alias resolution."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_voice_alias_resolves_prompt_and_mode(self):
        """Voice alias sets prompt and mode correctly."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with patch("soundfile.write"):
            client.generate("hello", voice="narrator")

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["prompt_file"], "narrator.pt")
        self.assertEqual(payload["mode"], "clone")

    def test_unknown_voice_alias_raises(self):
        """Unknown voice alias raises ValueError."""
        client, _ = _client_with_server(self.cfg)
        with self.assertRaises(ValueError) as ctx:
            client.generate("hello", voice="nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))


@_skip
class TestGenerateProsodyPreset(unittest.TestCase):
    """generate() prosody preset resolution."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    @patch("qwen3_tts.core.config.get_prosody_presets")
    def test_prosody_preset_sets_instruct(self, mock_prosody):
        """Prosody preset resolves to instruct text for custom mode."""
        mock_prosody.return_value = {"excited": "Speak with excitement and energy"}
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with patch("soundfile.write"):
            client.generate("hello", mode="custom", speaker="ryan", prosody="excited")

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["instruct"], "Speak with excitement and energy")

    @patch("qwen3_tts.core.config.get_prosody_presets")
    def test_unknown_prosody_preset_raises(self, mock_prosody):
        """Unknown prosody preset raises ValueError."""
        mock_prosody.return_value = {"excited": "Speak with excitement"}
        client, _ = _client_with_server(self.cfg)
        with self.assertRaises(ValueError) as ctx:
            client.generate("hello", mode="custom", prosody="nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))


@_skip
class TestGenerateAudioProcessing(unittest.TestCase):
    """generate() audio processing chain."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    @patch("qwen3_tts.core.engine.process_audio")
    def test_speed_triggers_audio_processing(self, mock_proc):
        """Speed != 1.0 triggers process_audio call."""
        import numpy as np

        mock_proc.return_value = np.zeros(480, dtype=np.float32)
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with patch("soundfile.write"):
            client.generate("hello", speed=1.5)

        mock_proc.assert_called_once()
        _, kwargs = mock_proc.call_args
        self.assertEqual(kwargs["speed"], 1.5)

    def test_no_processing_when_defaults(self):
        """No audio processing when all flags are default."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with patch("soundfile.write"), \
             patch("qwen3_tts.core.engine.process_audio") as mock_proc:
            client.generate("hello")

        mock_proc.assert_not_called()


@_skip
class TestGenerateOutputPath(unittest.TestCase):
    """generate() output path handling."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_explicit_output_path_returned(self):
        """Explicit output path is used and returned."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        try:
            with patch("soundfile.write"):
                result = client.generate("hello", output=out_path)
            self.assertEqual(result, out_path)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_output_without_wav_gets_extension(self):
        """Output path without .wav gets extension appended."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with patch("soundfile.write"):
            result = client.generate("hello", output="/tmp/test_out")
        self.assertTrue(result.endswith(".wav"))


# ---------------------------------------------------------------------------
# _generate_via_server tests
# ---------------------------------------------------------------------------


@_skip
class TestGenerateViaServer(unittest.TestCase):
    """_generate_via_server() internal method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_success_decodes_base64(self):
        """Successful response decodes base64 audio to numpy array."""
        import numpy as np

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        wav, sr = client._generate_via_server(
            "hello", "clone", "default.pt", None, None, None, {}
        )
        self.assertIsInstance(wav, np.ndarray)
        self.assertEqual(sr, 24000)

    def test_error_response_raises_generation_error(self):
        """Non-200 response raises GenerationError."""
        from qwen3_tts.core.config import GenerationError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(500, "model not loaded")

        with self.assertRaises(GenerationError):
            client._generate_via_server(
                "hello", "clone", "default.pt", None, None, None, {}
            )

    def test_x_vector_only_mode_in_payload(self):
        """x_vector_only_mode=True adds flag to payload."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        client._generate_via_server(
            "hello", "clone", "voice.pt", None, None, None, {},
            x_vector_only_mode=True,
        )
        payload = session.post.call_args[1]["json"]
        self.assertTrue(payload.get("x_vector_only_mode"))

    def test_clone_mode_payload_has_prompt_file(self):
        """Clone mode includes prompt_file in payload."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        client._generate_via_server(
            "hello", "clone", "my.pt", None, None, None, {}
        )
        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["prompt_file"], "my.pt")
        self.assertEqual(payload["mode"], "clone")


# ---------------------------------------------------------------------------
# generate_streaming tests
# ---------------------------------------------------------------------------


@_skip
class TestGenerateStreaming(unittest.TestCase):
    """generate_streaming() chunk iteration."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_yields_chunks(self):
        """Streaming yields (wav_chunk, sample_rate) tuples."""
        import numpy as np

        client, session = _client_with_server(self.cfg)

        # Build a valid binary chunk: header (sr=24000, length) + float32 data
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        audio_bytes = samples.tobytes()
        header = struct.pack("<II", 24000, len(audio_bytes))
        chunk_data = header + audio_bytes

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_content.return_value = [chunk_data]
        session.post.return_value = mock_resp

        chunks = list(client.generate_streaming("hello", mode="custom", speaker="ryan"))
        self.assertEqual(len(chunks), 1)
        wav_chunk, sr = chunks[0]
        self.assertEqual(sr, 24000)
        np.testing.assert_array_almost_equal(wav_chunk, samples)

    def test_error_status_raises(self):
        """Non-200 streaming response raises GenerationError."""
        from qwen3_tts.core.config import GenerationError

        client, session = _client_with_server(self.cfg)

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.json.return_value = {"error": "model not loaded"}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        session.post.return_value = mock_resp

        with self.assertRaises(GenerationError):
            list(client.generate_streaming("hello", mode="custom", speaker="ryan"))

    def test_streaming_error_includes_model_type(self):
        """Streaming error includes model_type prefix in technical_detail."""
        from qwen3_tts.core.config import GenerationError

        client, session = _client_with_server(self.cfg)

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.json.return_value = {
            "error": "not loaded",
            "model_type": "clone",
        }
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        session.post.return_value = mock_resp

        with self.assertRaises(GenerationError) as ctx:
            list(client.generate_streaming("hello", mode="clone"))
        # technical_detail contains the model_type prefix
        self.assertIn("clone", ctx.exception.technical_detail)

    def test_multiple_chunks_in_one_response(self):
        """Multiple binary chunks in a single iter_content response are parsed."""
        import numpy as np

        client, session = _client_with_server(self.cfg)

        # Two chunks concatenated
        s1 = np.array([1.0, 2.0], dtype=np.float32)
        s2 = np.array([3.0], dtype=np.float32)
        b1 = s1.tobytes()
        b2 = s2.tobytes()
        data = (
            struct.pack("<II", 24000, len(b1)) + b1
            + struct.pack("<II", 24000, len(b2)) + b2
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_content.return_value = [data]
        session.post.return_value = mock_resp

        chunks = list(client.generate_streaming("hello", mode="custom", speaker="ryan"))
        self.assertEqual(len(chunks), 2)


# ---------------------------------------------------------------------------
# generate_dialogue tests
# ---------------------------------------------------------------------------


@_skip
class TestGenerateDialogue(unittest.TestCase):
    """generate_dialogue() multi-speaker generation."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_two_lines_combined(self):
        """Two dialogue lines produce a combined audio file."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with patch.object(client, "is_server_running", return_value=True), \
             patch("soundfile.write") as mock_write:
            result = client.generate_dialogue(
                lines=[
                    {"text": "Hello", "mode": "clone", "prompt": "a.pt"},
                    {"text": "World", "mode": "clone", "prompt": "b.pt"},
                ],
                output="/tmp/dialogue_test.wav",
            )
        self.assertEqual(result, "/tmp/dialogue_test.wav")
        mock_write.assert_called_once()

    def test_speaker_config_mapping(self):
        """Speaker names map to speaker config dicts."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        speakers = {
            "alice": {"mode": "custom", "speaker": "Vivian", "instruct": ""},
        }

        with patch.object(client, "is_server_running", return_value=True), \
             patch("soundfile.write"):
            client.generate_dialogue(
                lines=[{"text": "Hi", "speaker": "alice"}],
                speakers=speakers,
                output="/tmp/dlg.wav",
            )

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["mode"], "custom")
        self.assertEqual(payload["speaker"], "vivian")

    def test_empty_lines_skipped(self):
        """Lines with empty text are skipped."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with patch.object(client, "is_server_running", return_value=True), \
             patch("soundfile.write"):
            client.generate_dialogue(
                lines=[
                    {"text": ""},
                    {"text": "Real line", "mode": "clone"},
                ],
                output="/tmp/dlg.wav",
            )

        # Only one POST call (empty line skipped)
        self.assertEqual(session.post.call_count, 1)

    def test_all_empty_lines_raises(self):
        """All empty lines raises ValueError."""
        client, session = _client_with_server(self.cfg)

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                client.generate_dialogue(
                    lines=[{"text": ""}, {"text": ""}],
                    output="/tmp/dlg.wav",
                )
            self.assertIn("No audio", str(ctx.exception))

    def test_server_not_running_raises(self):
        """generate_dialogue raises when server not running."""
        client, _ = _client_with_server(self.cfg)

        with patch.object(client, "is_server_running", return_value=False):
            with self.assertRaises(ConnectionError):
                client.generate_dialogue(
                    lines=[{"text": "Hi"}],
                    output="/tmp/dlg.wav",
                )


# ---------------------------------------------------------------------------
# cancel_generation tests
# ---------------------------------------------------------------------------


@_skip
class TestCancelGeneration(unittest.TestCase):
    """cancel_generation() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_cancel_returns_response(self):
        """cancel_generation returns response dict."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value.json.return_value = {"status": "cancellation_requested"}

        with patch.object(client, "is_server_running", return_value=True):
            result = client.cancel_generation()
        self.assertEqual(result["status"], "cancellation_requested")

    def test_cancel_requires_server(self):
        """cancel_generation raises when server is down."""
        client, _ = _client_with_server(self.cfg)

        with patch.object(client, "is_server_running", return_value=False):
            with self.assertRaises(ConnectionError):
                client.cancel_generation()


# ============================================================================
# models.py tests
# ============================================================================


@_skip
class TestLoadModel(unittest.TestCase):
    """load_model() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_load_model_success(self):
        """load_model returns response dict on success."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "loaded"}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.load_model("clone")
        self.assertEqual(result["status"], "loaded")

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["model_type"], "clone")

    def test_load_model_error_raises(self):
        """load_model raises ModelError on failure."""
        from qwen3_tts.core.config import ModelError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(500, "OOM")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ModelError):
                client.load_model("clone")

    def test_load_model_requires_server(self):
        """load_model raises when server is down."""
        client, _ = _client_with_server(self.cfg)

        with patch.object(client, "is_server_running", return_value=False):
            with self.assertRaises(ConnectionError):
                client.load_model("clone")


@_skip
class TestUnloadModel(unittest.TestCase):
    """unload_model() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_unload_model_success(self):
        """unload_model returns response dict on success."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "unloaded"}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.unload_model("design")
        self.assertEqual(result["status"], "unloaded")

    def test_unload_model_409_accepted(self):
        """unload_model accepts 409 (already unloaded) without raising."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=409,
            json=MagicMock(return_value={"status": "already_unloaded"}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.unload_model("design")
        self.assertEqual(result["status"], "already_unloaded")

    def test_unload_model_error_raises(self):
        """unload_model raises ModelError on 500."""
        from qwen3_tts.core.config import ModelError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(500, "internal error")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ModelError):
                client.unload_model("design")


@_skip
class TestGetModels(unittest.TestCase):
    """get_models() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_get_models_returns_data(self):
        """get_models returns model info dict."""
        client, session = _client_with_server(self.cfg)
        expected = {"models": {"clone": {"loaded": True}}, "backend": "mlx"}
        session.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=expected),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.get_models()
        self.assertEqual(result["backend"], "mlx")

    def test_get_models_requires_server(self):
        """get_models raises when server is down."""
        client, _ = _client_with_server(self.cfg)

        with patch.object(client, "is_server_running", return_value=False):
            with self.assertRaises(ConnectionError):
                client.get_models()


@_skip
class TestUpdateModelConfig(unittest.TestCase):
    """update_model_config() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_update_model_config_success(self):
        """update_model_config sends correct payload and returns result."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "updated", "changes": {}}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.update_model_config(model_size="0.6B", mlx_quantization="4bit")

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["model_size"], "0.6B")
        self.assertEqual(payload["mlx_quantization"], "4bit")
        self.assertEqual(result["status"], "updated")

    def test_update_model_config_no_args_raises(self):
        """update_model_config raises ValueError with no arguments."""
        client, _ = _client_with_server(self.cfg)

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                client.update_model_config()
            self.assertIn("At least one", str(ctx.exception))

    def test_update_model_config_error_raises(self):
        """update_model_config raises ModelError on failure."""
        from qwen3_tts.core.config import ModelError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(400, "invalid size")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ModelError):
                client.update_model_config(model_size="99B")


@_skip
class TestUpdateStartupConfig(unittest.TestCase):
    """update_startup_config() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_update_startup_config_success(self):
        """update_startup_config sends correct payload."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "updated", "changes": {}}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.update_startup_config(clone=True, design=False)

        payload = session.post.call_args[1]["json"]
        self.assertTrue(payload["clone"])
        self.assertFalse(payload["design"])
        self.assertNotIn("custom", payload)
        self.assertEqual(result["status"], "updated")

    def test_update_startup_config_no_args_raises(self):
        """update_startup_config raises ValueError with no arguments."""
        client, _ = _client_with_server(self.cfg)

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                client.update_startup_config()
            self.assertIn("At least one", str(ctx.exception))

    def test_update_startup_config_error_raises(self):
        """update_startup_config raises ModelError on failure."""
        from qwen3_tts.core.config import ModelError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(500, "server error")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ModelError):
                client.update_startup_config(clone=True)


# ============================================================================
# voices.py tests
# ============================================================================


@_skip
class TestListPrompts(unittest.TestCase):
    """list_prompts() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_list_prompts_via_server(self):
        """list_prompts uses server endpoint when available."""
        client, session = _client_with_server(self.cfg)
        session.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"prompts": ["voice1.pt", "voice2.wav"]}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.list_prompts()
        self.assertEqual(result, ["voice1.pt", "voice2.wav"])

    def test_list_prompts_filesystem_fallback(self):
        """list_prompts falls back to filesystem when server is down."""
        client, _ = _client_with_server(self.cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            client.voice_prompts_dir = tmpdir
            # Create .pt file
            open(os.path.join(tmpdir, "torch_voice.pt"), "w").close()
            # Create .wav + .txt pair (MLX format)
            open(os.path.join(tmpdir, "mlx_voice.wav"), "w").close()
            open(os.path.join(tmpdir, "mlx_voice.txt"), "w").close()
            # Create .wav without .txt (should be excluded)
            open(os.path.join(tmpdir, "orphan.wav"), "w").close()

            with patch.object(client, "is_server_running", return_value=False):
                result = client.list_prompts()

        self.assertIn("torch_voice.pt", result)
        self.assertIn("mlx_voice.wav", result)
        self.assertNotIn("orphan.wav", result)

    def test_list_prompts_missing_dir_returns_empty(self):
        """list_prompts returns empty list when voice dir doesn't exist."""
        client, _ = _client_with_server(self.cfg)
        client.voice_prompts_dir = "/nonexistent/path"

        with patch.object(client, "is_server_running", return_value=False):
            result = client.list_prompts()
        self.assertEqual(result, [])


@_skip
class TestDeletePrompt(unittest.TestCase):
    """delete_prompt() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_delete_prompt_success(self):
        """delete_prompt returns response on success."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "deleted", "files_removed": ["v.pt"]}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.delete_prompt("v.pt")
        self.assertEqual(result["status"], "deleted")

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["name"], "v.pt")

    def test_delete_prompt_error_raises(self):
        """delete_prompt raises VoicePromptError on failure."""
        from qwen3_tts.core.config import VoicePromptError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(404, "not found")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(VoicePromptError):
                client.delete_prompt("missing.pt")


@_skip
class TestRenamePrompt(unittest.TestCase):
    """rename_prompt() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_rename_prompt_success(self):
        """rename_prompt sends old_name and new_name."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "renamed"}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.rename_prompt("old.pt", "new.pt")
        self.assertEqual(result["status"], "renamed")

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["old_name"], "old.pt")
        self.assertEqual(payload["new_name"], "new.pt")

    def test_rename_prompt_error_raises(self):
        """rename_prompt raises VoicePromptError on failure."""
        from qwen3_tts.core.config import VoicePromptError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(404, "prompt not found")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(VoicePromptError):
                client.rename_prompt("missing.pt", "new.pt")


@_skip
class TestPreviewPrompt(unittest.TestCase):
    """preview_prompt() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_preview_prompt_returns_bytes(self):
        """preview_prompt returns raw audio bytes."""
        client, session = _client_with_server(self.cfg)
        wav_bytes = b"\x00\x01\x02\x03"
        session.get.return_value = MagicMock(
            status_code=200,
            content=wav_bytes,
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.preview_prompt("voice.pt")
        self.assertEqual(result, wav_bytes)

        params = session.get.call_args[1]["params"]
        self.assertEqual(params["name"], "voice.pt")

    def test_preview_prompt_error_raises(self):
        """preview_prompt raises VoicePromptError on failure."""
        from qwen3_tts.core.config import VoicePromptError

        client, session = _client_with_server(self.cfg)
        session.get.return_value = _mock_error_response(404, "not found")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(VoicePromptError):
                client.preview_prompt("missing.pt")


@_skip
class TestGetPromptDetails(unittest.TestCase):
    """get_prompt_details() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_get_single_prompt_details(self):
        """get_prompt_details with name returns single prompt metadata."""
        client, session = _client_with_server(self.cfg)
        expected = {"name": "v.pt", "size": 1024, "duration": 3.5}
        session.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=expected),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.get_prompt_details("v.pt")
        self.assertEqual(result["duration"], 3.5)

        params = session.get.call_args[1]["params"]
        self.assertEqual(params["name"], "v.pt")

    def test_get_all_prompt_details(self):
        """get_prompt_details without name returns all prompts."""
        client, session = _client_with_server(self.cfg)
        expected = {"prompts": [{"name": "a.pt"}, {"name": "b.pt"}]}
        session.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=expected),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.get_prompt_details()
        self.assertEqual(len(result["prompts"]), 2)

        # No name param when querying all
        params = session.get.call_args[1]["params"]
        self.assertEqual(params, {})

    def test_get_prompt_details_error_raises(self):
        """get_prompt_details raises VoicePromptError on failure."""
        from qwen3_tts.core.config import VoicePromptError

        client, session = _client_with_server(self.cfg)
        session.get.return_value = _mock_error_response(500, "internal")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(VoicePromptError):
                client.get_prompt_details("v.pt")


if __name__ == "__main__":
    unittest.main()
