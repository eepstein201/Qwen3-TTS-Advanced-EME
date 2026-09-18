"""Tests for qwen3_tts/server/client/generator.py.

Covers: generate(), _generate_via_server(), generate_streaming(),
generate_dialogue(), cancel_generation() — all with mocked HTTP calls.
No running server or GPU required.

Run with:
    python -m pytest tests/test_client_generator.py -v --tb=short
"""

import base64
import io
import json
import os
import struct
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
# generate() — voice alias resolution
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

        with (
            patch("soundfile.write"),
            patch("qwen3_tts.core.engine.process_audio") as mock_proc,
        ):
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


# ============================================================================
# _generate_via_server()
# ============================================================================


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

    def test_empty_results_raises_generation_error_not_indexerror(self):
        """A 200 carrying zero results must raise a clear GenerationError.

        H3: the client indexed ``resp.json()["results"][0]`` unconditionally.
        The server can legitimately return a short or empty ``results`` list —
        a batch cancelled before its first item does exactly that — and the
        caller then saw a bare ``IndexError: list index out of range`` with no
        hint that the generation was cancelled or that the server was healthy.
        """
        from qwen3_tts.core.config import GenerationError

        client, session = _client_with_server(self.cfg)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": []}
        session.post.return_value = resp

        with self.assertRaises(GenerationError) as ctx:
            client._generate_via_server(
                "hello", "clone", "default.pt", None, None, None, {}
            )
        self.assertNotIsInstance(ctx.exception, IndexError)
        # technical_detail is what format_cli/format_gradio surface to users.
        self.assertIn("no audio", (ctx.exception.technical_detail or "").lower())

    def test_cancelled_empty_results_says_cancelled(self):
        """An empty result set flagged ``cancelled`` must name the cause."""
        from qwen3_tts.core.config import GenerationError

        client, session = _client_with_server(self.cfg)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": [], "cancelled": True}
        session.post.return_value = resp

        with self.assertRaises(GenerationError) as ctx:
            client._generate_via_server(
                "hello", "clone", "default.pt", None, None, None, {}
            )
        self.assertIn("cancel", (ctx.exception.technical_detail or "").lower())

    def test_error_response_raises_generation_error(self):
        """Non-200 response raises GenerationError."""
        from qwen3_tts.core.config import GenerationError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(500, "model not loaded")

        with self.assertRaises(GenerationError):
            client._generate_via_server(
                "hello", "clone", "default.pt", None, None, None, {}
            )

    def test_model_not_loaded_raises_model_not_loaded_error(self):
        """A 503 model_not_loaded response raises ModelNotLoadedError (carrying
        model_type), not a generic GenerationError, so callers can present an
        actionable recovery path instead of a bare "not loaded" string.

        The server (app_generation.handle_generate) returns this exact envelope
        when the requested mode's model is absent. Previously the client
        collapsed it to a plain string and lost the structured fields.
        """
        from qwen3_tts.core.config import ModelNotLoadedError

        client, session = _client_with_server(self.cfg)
        # Mirror the FastAPI HTTPException envelope: {"detail": { ... }}
        resp = MagicMock()
        resp.status_code = 503
        resp.json.return_value = {
            "detail": {
                "error": "model_not_loaded",
                "detail": "The 'design' model is not loaded.",
                "recovery": "restart",
                "model_type": "design",
            }
        }
        session.post.return_value = resp

        with self.assertRaises(ModelNotLoadedError) as ctx:
            client._generate_via_server(
                "hello", "design", None, "a warm voice", None, None, {}
            )
        self.assertEqual(ctx.exception.model_type, "design")

    def test_x_vector_only_mode_in_payload(self):
        """x_vector_only_mode=True adds flag to payload."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        client._generate_via_server(
            "hello",
            "clone",
            "voice.pt",
            None,
            None,
            None,
            {},
            x_vector_only_mode=True,
        )
        payload = session.post.call_args[1]["json"]
        self.assertTrue(payload.get("x_vector_only_mode"))

    def test_clone_mode_payload_has_prompt_file(self):
        """Clone mode includes prompt_file in payload."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        client._generate_via_server("hello", "clone", "my.pt", None, None, None, {})
        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["prompt_file"], "my.pt")
        self.assertEqual(payload["mode"], "clone")


# ============================================================================
# generate_streaming()
# ============================================================================


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

    def test_terminal_error_frame_raises_instead_of_yielding_garbage(self):
        """U3: the sentinel frame (sample_rate 0) must raise, not be decoded.

        Starlette commits the 200 headers before the body is iterated, so a
        mid-stream failure arrives in band as a frame with sample_rate 0 whose
        payload is JSON. The CLI path has handled this since WS2 2.5; the
        public client (README:303) did not — it ran the JSON bytes through
        np.frombuffer and yielded garbage samples with sr=0, so a failed
        generation looked like a successful one with odd audio.
        """
        from qwen3_tts.core.config import GenerationError
        from qwen3_tts.core.stream_protocol import encode_stream_error_frame

        client, session = _client_with_server(self.cfg)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_content.return_value = [
            encode_stream_error_frame("model exploded")
        ]
        session.post.return_value = mock_resp

        with self.assertRaises(GenerationError) as ctx:
            list(client.generate_streaming("hello", mode="custom", speaker="ryan"))
        self.assertIn("model exploded", (ctx.exception.technical_detail or "").lower())

    def test_partial_audio_before_error_frame_is_not_a_success(self):
        """Audio already yielded must not be treated as a complete generation.

        The failure can land after chunks have streamed. The consumer gets the
        chunks, then the raise — it must never end cleanly and let a caller
        save truncated audio as a successful generation.
        """
        import numpy as np

        from qwen3_tts.core.config import GenerationError
        from qwen3_tts.core.stream_protocol import encode_stream_error_frame

        client, session = _client_with_server(self.cfg)

        samples = np.array([0.1, 0.2], dtype=np.float32)
        audio = samples.tobytes()
        body = (
            struct.pack("<II", 24000, len(audio))
            + audio
            + encode_stream_error_frame("died mid-stream")
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_content.return_value = [body]
        session.post.return_value = mock_resp

        received = []
        with self.assertRaises(GenerationError):
            for chunk in client.generate_streaming(
                "hello", mode="custom", speaker="ryan"
            ):
                received.append(chunk)

        self.assertEqual(
            len(received), 1, "the good chunk before the error should still arrive"
        )
        self.assertNotEqual(
            received[0][1], 0, "a real chunk must never carry sample_rate 0"
        )

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
            struct.pack("<II", 24000, len(b1))
            + b1
            + struct.pack("<II", 24000, len(b2))
            + b2
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_content.return_value = [data]
        session.post.return_value = mock_resp

        chunks = list(client.generate_streaming("hello", mode="custom", speaker="ryan"))
        self.assertEqual(len(chunks), 2)


# ============================================================================
# generate_dialogue()
# ============================================================================


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

        with (
            patch.object(client, "is_server_running", return_value=True),
            patch("soundfile.write") as mock_write,
        ):
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

        with (
            patch.object(client, "is_server_running", return_value=True),
            patch("soundfile.write"),
        ):
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

        with (
            patch.object(client, "is_server_running", return_value=True),
            patch("soundfile.write"),
        ):
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

    def test_empty_results_raises_generation_error_not_indexerror(self):
        """The dialogue path must guard ``results[0]`` like _generate_via_server.

        generate_dialogue posts to /generate once per line and indexed
        ``resp.json()["results"][0]`` unconditionally — the exact defect fixed
        in _generate_via_server, in the same module. cancel_generation() is the
        next method in this class, so cancelling mid-dialogue is the natural
        trigger and produced a bare IndexError.
        """
        from qwen3_tts.core.config import GenerationError

        client, session = _client_with_server(self.cfg)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": []}
        session.post.return_value = resp

        with (
            patch.object(client, "is_server_running", return_value=True),
            patch("soundfile.write"),
        ):
            with self.assertRaises(GenerationError) as ctx:
                client.generate_dialogue(
                    lines=[{"text": "Hello", "mode": "clone", "prompt": "a.pt"}],
                    output="/tmp/dlg.wav",
                )
        self.assertNotIsInstance(ctx.exception, IndexError)
        self.assertIn("no audio", (ctx.exception.technical_detail or "").lower())

    def test_cancelled_empty_results_says_cancelled(self):
        """A dialogue line cancelled mid-flight must name cancellation."""
        from qwen3_tts.core.config import GenerationError

        client, session = _client_with_server(self.cfg)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": [], "cancelled": True}
        session.post.return_value = resp

        with (
            patch.object(client, "is_server_running", return_value=True),
            patch("soundfile.write"),
        ):
            with self.assertRaises(GenerationError) as ctx:
                client.generate_dialogue(
                    lines=[{"text": "Hello", "mode": "clone", "prompt": "a.pt"}],
                    output="/tmp/dlg.wav",
                )
        self.assertIn("cancel", (ctx.exception.technical_detail or "").lower())


# ============================================================================
# cancel_generation()
# ============================================================================


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
        session.post.return_value.json.return_value = {
            "status": "cancellation_requested"
        }

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
# Preset merge — the same three-line block in generate(), generate_streaming()
# and generate_dialogue()
# ============================================================================


def _mock_stream_response(chunk_bytes=b"", headers=None):
    """Build a mock streaming response with real (non-Mock) headers.

    headers must be a genuine dict: a MagicMock's .get("X-Seed") returns a
    MagicMock, which int() happily converts to 1 — the seed branch would then
    pass hollowly.
    """
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = headers if headers is not None else {}
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    resp.iter_content.return_value = [chunk_bytes] if chunk_bytes else []
    return resp


@_skip
class TestPresetMerge(unittest.TestCase):
    """A named preset's params override the config generation defaults.

    The config fixture sets generation.temperature 0.7 and a "consistent"
    preset of temperature 0.5, so asserting 0.5 in the payload distinguishes
    a real merge from the default that would be there anyway.
    """

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_generate_merges_named_preset_into_payload(self):
        """generate() applies the preset over the generation defaults."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with patch("soundfile.write"):
            client.generate(
                "hello",
                mode="custom",
                speaker="ryan",
                output="/tmp/preset.wav",
                preset="consistent",
            )

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["temperature"], 0.5)

    def test_streaming_merges_named_preset_into_payload(self):
        """generate_streaming() applies the preset over the defaults."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_stream_response()

        list(
            client.generate_streaming(
                "hello", mode="custom", speaker="ryan", preset="consistent"
            )
        )

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["temperature"], 0.5)

    def test_dialogue_merges_named_preset_into_payload(self):
        """generate_dialogue() applies the preset over the defaults."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with (
            patch.object(client, "is_server_running", return_value=True),
            patch("soundfile.write"),
        ):
            client.generate_dialogue(
                lines=[{"text": "Hi", "mode": "clone", "prompt": "a.pt"}],
                output="/tmp/dlg_preset.wav",
                preset="consistent",
            )

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["temperature"], 0.5)

    def test_unknown_preset_name_is_ignored_not_an_error(self):
        """An unknown preset leaves the generation defaults untouched."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with patch("soundfile.write"):
            client.generate(
                "hello",
                mode="custom",
                speaker="ryan",
                output="/tmp/preset_unknown.wav",
                preset="does-not-exist",
            )

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["temperature"], 0.7)


# ============================================================================
# Design-mode description fallback
# ============================================================================


@_skip
class TestDefaultDescriptionFallback(unittest.TestCase):
    """Design mode with no description falls back to the configured one."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_generate_design_falls_back_to_config_description(self):
        """generate() sends default_voice_description when none is passed."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with patch("soundfile.write"):
            client.generate("hello", mode="design", output="/tmp/design.wav")

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["voice_description"], "neutral voice")

    def test_streaming_design_falls_back_to_config_description(self):
        """generate_streaming() applies the same fallback."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_stream_response()

        list(client.generate_streaming("hello", mode="design"))

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["voice_description"], "neutral voice")


# ============================================================================
# generate_streaming() — voice alias resolution
# ============================================================================


@_skip
class TestStreamingVoiceAliasResolution(unittest.TestCase):
    """generate_streaming() resolves voice aliases like generate() does."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_clone_alias_resolves_prompt_and_mode(self):
        """A clone alias sets both the mode and the prompt file."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_stream_response()

        list(client.generate_streaming("hello", voice="narrator"))

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["mode"], "clone")
        self.assertEqual(payload["prompt_file"], "narrator.pt")

    def test_design_alias_resolves_mode_and_description(self):
        """A design alias switches the mode away from the clone default."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_stream_response()

        list(client.generate_streaming("hello", voice="designer"))

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["mode"], "design")
        self.assertEqual(payload["voice_description"], "warm female")

    def test_unknown_alias_raises_value_error(self):
        """An unresolvable alias raises rather than silently generating."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_stream_response()

        with self.assertRaises(ValueError):
            list(client.generate_streaming("hello", voice="no-such-alias"))
        session.post.assert_not_called()


# ============================================================================
# generate_streaming() — X-Seed header and non-JSON error bodies
# ============================================================================


@_skip
class TestStreamingSeedHeader(unittest.TestCase):
    """last_seed mirrors the server's X-Seed header on the streaming path."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_valid_x_seed_header_is_recorded(self):
        """A numeric X-Seed lands on last_seed as an int."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_stream_response(headers={"X-Seed": "4242"})

        list(client.generate_streaming("hello", mode="custom", speaker="ryan"))

        self.assertEqual(client.last_seed, 4242)

    def test_malformed_x_seed_header_clears_last_seed(self):
        """A non-numeric X-Seed clears last_seed rather than leaving a stale one.

        last_seed is pre-set here so the assertion proves the branch actively
        clears it — an untouched attribute would still hold 999.
        """
        client, session = _client_with_server(self.cfg)
        client.last_seed = 999
        session.post.return_value = _mock_stream_response(
            headers={"X-Seed": "not-a-number"}
        )

        list(client.generate_streaming("hello", mode="custom", speaker="ryan"))

        self.assertIsNone(client.last_seed)


@_skip
class TestStreamingNonJsonErrorBody(unittest.TestCase):
    """A non-JSON error body still produces a GenerationError."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_unparseable_error_body_still_raises_generation_error(self):
        """resp.json() blowing up must not mask the failure as a success."""
        from qwen3_tts.core.config import GenerationError

        client, session = _client_with_server(self.cfg)

        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json.side_effect = ValueError("not JSON")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        session.post.return_value = mock_resp

        with self.assertRaises(GenerationError):
            list(client.generate_streaming("hello", mode="custom", speaker="ryan"))


# ============================================================================
# generate_dialogue() — error arm, audio processing, output path
# ============================================================================


@_skip
class TestDialogueErrorAndOutputPaths(unittest.TestCase):
    """generate_dialogue() per-line failure and output-path handling."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_error_response_raises_with_the_servers_message(self):
        """A non-200 on any line aborts, carrying the server's own message.

        Asserting on technical_detail is what makes this test discriminating:
        dropping the status check does NOT make the call succeed — the error
        body has no "results", so _first_result raises the same exception type
        a few lines later. Only the server's message ("boom") separates the
        real status-code arm from that accidental downstream failure.
        """
        from qwen3_tts.core.config import GenerationError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(500, "boom")

        with (
            patch.object(client, "is_server_running", return_value=True),
            patch("soundfile.write") as mock_write,
        ):
            with self.assertRaises(GenerationError) as ctx:
                client.generate_dialogue(
                    lines=[{"text": "Hi", "mode": "clone", "prompt": "a.pt"}],
                    output="/tmp/dlg_err.wav",
                )
        self.assertEqual(ctx.exception.technical_detail, "boom")
        mock_write.assert_not_called()

    def test_speed_routes_audio_through_process_audio(self):
        """A non-default speed sends each line's audio through process_audio."""
        import numpy as np

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()
        processed = np.array([0.25, -0.25], dtype=np.float32)

        with (
            patch.object(client, "is_server_running", return_value=True),
            patch(
                "qwen3_tts.core.engine.process_audio", return_value=processed
            ) as mock_proc,
            patch("soundfile.write") as mock_write,
        ):
            client.generate_dialogue(
                lines=[{"text": "Hi", "mode": "clone", "prompt": "a.pt"}],
                output="/tmp/dlg_speed.wav",
                speed=1.5,
            )

        mock_proc.assert_called_once()
        self.assertEqual(mock_proc.call_args[1]["speed"], 1.5)
        np.testing.assert_array_equal(mock_write.call_args[0][1], processed)

    def test_default_output_path_uses_config_directory(self):
        """output=None writes dialogue_output.wav under the configured dir."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with (
            patch.object(client, "is_server_running", return_value=True),
            patch("soundfile.write"),
        ):
            result = client.generate_dialogue(
                lines=[{"text": "Hi", "mode": "clone", "prompt": "a.pt"}],
            )

        expected = os.path.join(
            os.path.expanduser("~/Downloads"), "dialogue_output.wav"
        )
        self.assertEqual(result, expected)

    def test_output_without_wav_gets_extension(self):
        """An extensionless output path gains .wav before it is written."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_generate_response()

        with (
            patch.object(client, "is_server_running", return_value=True),
            patch("soundfile.write") as mock_write,
        ):
            result = client.generate_dialogue(
                lines=[{"text": "Hi", "mode": "clone", "prompt": "a.pt"}],
                output="/tmp/dlg_noext",
            )

        self.assertEqual(result, "/tmp/dlg_noext.wav")
        self.assertEqual(mock_write.call_args[0][0], "/tmp/dlg_noext.wav")


if __name__ == "__main__":
    unittest.main()
