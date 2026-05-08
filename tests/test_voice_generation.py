"""Generation-related tests extracted from test_voice.py."""

import os
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from tests.voice_test_helpers import (
    _skip_server, _skip_client, _skip_generate, _make_test_client,
)


class TestStability(unittest.TestCase):
    """Test stability hardening features."""

    def test_retry_delays_constant_exists(self):
        """_RETRY_DELAYS constant is defined."""
        from qwen3_tts.core.engine.model_loader import _RETRY_DELAYS
        self.assertEqual(len(_RETRY_DELAYS), 3)
        self.assertEqual(_RETRY_DELAYS, (5, 15, 45))

    def test_retry_delays_is_exponential(self):
        """_RETRY_DELAYS uses exponential backoff pattern."""
        from qwen3_tts.core.engine.model_loader import _RETRY_DELAYS
        # Each delay should be roughly 3x the previous (5 -> 15 -> 45)
        self.assertEqual(_RETRY_DELAYS[1], _RETRY_DELAYS[0] * 3)
        self.assertEqual(_RETRY_DELAYS[2], _RETRY_DELAYS[1] * 3)

    def test_max_chunk_chars_helper_exists(self):
        """_get_max_chunk_chars helper function exists."""
        from qwen3_tts.core.engine.inference import _get_max_chunk_chars
        self.assertTrue(callable(_get_max_chunk_chars))

    def test_max_chunk_chars_default(self):
        """_get_max_chunk_chars returns default 500."""
        from qwen3_tts.core.engine.inference import _get_max_chunk_chars
        with patch("qwen3_tts.core.engine.inference.load_config", return_value={}):
            result = _get_max_chunk_chars()
        self.assertEqual(result, 500)

    def test_max_chunk_chars_from_config(self):
        """_get_max_chunk_chars reads from config."""
        from qwen3_tts.core.engine.inference import _get_max_chunk_chars
        config = {"generation": {"max_chunk_chars": 300}}
        with patch("qwen3_tts.core.engine.inference.load_config", return_value=config):
            result = _get_max_chunk_chars()
        self.assertEqual(result, 300)


@_skip_server
class TestETACache(unittest.TestCase):
    """Test ETA estimation cache in FastAPI app."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={"security": {}, "auto_shutdown_minutes": 0})
        app.state.models_loaded.set()

    def test_eta_cache_exists(self):
        """FastAPI app has eta_cache in app.state."""
        from qwen3_tts.server.app import app
        self.assertTrue(hasattr(app.state, "eta_cache"))
        self.assertIn("median_rate", app.state.eta_cache)
        self.assertIn("last_updated", app.state.eta_cache)

    def test_eta_cache_ttl_function(self):
        """get_eta_cache_ttl returns the configured default value."""
        from qwen3_tts.core.config import get_eta_cache_ttl
        result = get_eta_cache_ttl()
        self.assertEqual(result, 30)

    def test_estimate_eta_uses_cache(self):
        """_estimate_eta reads from cache when fresh."""
        from qwen3_tts.server.app import app, _estimate_eta
        # Pre-populate cache with a known rate
        app.state.eta_cache["median_rate"] = 10.0  # 10 chars/sec
        app.state.eta_cache["last_updated"] = time.time()  # fresh

        result = _estimate_eta(app.state, 100, 5.0)
        # 100 chars / 10 chars/sec = 10s total, 10 - 5 = 5s remaining
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 5.0, delta=0.5)

    def test_estimate_eta_returns_none_without_data(self):
        """_estimate_eta returns None when no history data."""
        from qwen3_tts.server.app import app, _estimate_eta
        app.state.eta_cache["median_rate"] = None
        app.state.eta_cache["last_updated"] = time.time()

        result = _estimate_eta(app.state, 100, 5.0)
        self.assertIsNone(result)


@_skip_server
class TestGenerationCache(unittest.TestCase):
    """Test generation result cache in FastAPI app."""

    def setUp(self):
        from qwen3_tts.server.app import app
        app.state.gen_cache.clear()

    def test_gen_cache_key_deterministic(self):
        """Same inputs produce same cache key."""
        from qwen3_tts.server.app import _gen_cache_key
        key1 = _gen_cache_key("hello", "clone", {"temperature": 0.7}, prompt_file="voice.pt")
        key2 = _gen_cache_key("hello", "clone", {"temperature": 0.7}, prompt_file="voice.pt")
        self.assertEqual(key1, key2)

    def test_gen_cache_key_varies_by_text(self):
        """Different text produces different cache key."""
        from qwen3_tts.server.app import _gen_cache_key
        key1 = _gen_cache_key("hello", "clone", {})
        key2 = _gen_cache_key("world", "clone", {})
        self.assertNotEqual(key1, key2)

    def test_gen_cache_key_varies_by_mode(self):
        """Different mode produces different cache key."""
        from qwen3_tts.server.app import _gen_cache_key
        key1 = _gen_cache_key("hello", "clone", {})
        key2 = _gen_cache_key("hello", "design", {})
        self.assertNotEqual(key1, key2)

    def test_gen_cache_dict_exists(self):
        """FastAPI app has gen_cache in app.state."""
        from qwen3_tts.server.app import app
        self.assertTrue(hasattr(app.state, "gen_cache"))
        self.assertIsInstance(app.state.gen_cache, dict)

    def test_gen_cache_max_size_function(self):
        """get_generation_cache_max returns the configured default value."""
        from qwen3_tts.core.config import get_generation_cache_max
        result = get_generation_cache_max()
        self.assertEqual(result, 5)


@_skip_client
class TestClientUpdateModelConfig(unittest.TestCase):
    """Test TTSClient.update_model_config method."""

    def test_update_model_config_method_exists(self):
        """TTSClient has update_model_config method."""
        from qwen3_tts.server.client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "update_model_config"))
        self.assertTrue(callable(client.update_model_config))


@_skip_client
class TestClientModelMethods(unittest.TestCase):
    """Test that TTSClient has unload_model, update_startup_config, get_models methods."""

    def test_unload_model_exists(self):
        from qwen3_tts.server.client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "unload_model"))
        self.assertTrue(callable(getattr(client, "unload_model")))

    def test_update_startup_config_exists(self):
        from qwen3_tts.server.client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "update_startup_config"))
        self.assertTrue(callable(getattr(client, "update_startup_config")))

    def test_get_models_exists(self):
        from qwen3_tts.server.client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "get_models"))
        self.assertTrue(callable(getattr(client, "get_models")))


class TestReturnValueCounts(unittest.TestCase):
    """Tests that UI functions return correct number of values for Gradio wiring."""

    @patch("qwen3_tts.interface.ui.voice_management.get_voice_prompts", return_value=["v.wav"])
    @patch("qwen3_tts.interface.ui.voice_management.get_prompt_table_data", return_value=[])
    @patch("qwen3_tts.interface.ui.voice_management.is_server_running", return_value=True)
    @patch("qwen3_tts.interface.ui.voice_management.load_config", return_value={})
    @patch("qwen3_tts.interface.ui.voice_management.get_server_url", return_value="http://127.0.0.1:5123")
    def test_rename_voice_returns_3_values(self, *mocks):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        with patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
            from qwen3_tts.interface.ui.voice_management import rename_voice
            result = rename_voice("old", "new_name")
            self.assertEqual(len(result), 3, f"Expected 3 return values, got {len(result)}")

    @patch("qwen3_tts.interface.ui.voice_management.get_voice_prompts", return_value=["v.wav"])
    @patch("qwen3_tts.interface.ui.voice_management.get_prompt_table_data", return_value=[])
    @patch("qwen3_tts.interface.ui.voice_management.is_server_running", return_value=True)
    @patch("qwen3_tts.interface.ui.voice_management.load_config", return_value={})
    @patch("qwen3_tts.interface.ui.voice_management.get_server_url", return_value="http://127.0.0.1:5123")
    def test_delete_voice_returns_3_values(self, *mocks):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        with patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
            from qwen3_tts.interface.ui.voice_management import delete_voice
            result = delete_voice("voice1")
            self.assertEqual(len(result), 3, f"Expected 3 return values, got {len(result)}")

    def test_prompt_table_rows_match_3_columns(self):
        """Table data rows must have 3 elements to match headers [Name, Format, Default]."""
        with patch("qwen3_tts.interface.ui.voice_management.get_voice_prompts", return_value=["v.pt"]):
            with patch("qwen3_tts.interface.ui.voice_management.load_config", return_value={}):
                from qwen3_tts.interface.ui.voice_management import get_prompt_table_data
                rows = get_prompt_table_data()
                if rows:
                    self.assertEqual(len(rows[0]), 3)


class TestTextChunking(unittest.TestCase):
    """Test text chunking for long-form generation."""

    def test_split_text_short(self):
        """Short text is not split."""
        from qwen3_tts.core.engine.text_processing import _split_text
        chunks = _split_text("Hello world.", max_chars=500)
        self.assertEqual(chunks, ["Hello world."])

    def test_split_text_sentences(self):
        """Text is split on sentence boundaries."""
        from qwen3_tts.core.engine.text_processing import _split_text
        text = "First sentence. Second sentence. Third sentence."
        chunks = _split_text(text, max_chars=30)
        self.assertGreater(len(chunks), 1)
        # Each chunk should be <= max_chars
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 30)

    def test_split_text_preserves_content(self):
        """All content is preserved after splitting."""
        from qwen3_tts.core.engine.text_processing import _split_text
        text = "The quick brown fox jumps over the lazy dog. A second sentence follows."
        chunks = _split_text(text, max_chars=50)
        combined = " ".join(chunks)
        # All words should be present
        for word in text.split():
            self.assertIn(word.rstrip(".,"), combined)

    def test_split_text_question_mark(self):
        """Text splits on question marks."""
        from qwen3_tts.core.engine.text_processing import _split_text
        text = "Is this a question? Yes it is."
        chunks = _split_text(text, max_chars=25)
        self.assertGreater(len(chunks), 1)

    def test_split_text_exclamation(self):
        """Text splits on exclamation marks."""
        from qwen3_tts.core.engine.text_processing import _split_text
        text = "Hello! How are you today?"
        chunks = _split_text(text, max_chars=15)
        self.assertGreater(len(chunks), 1)

    def test_split_text_newlines(self):
        """Text splits on newlines."""
        from qwen3_tts.core.engine.text_processing import _split_text
        text = "First paragraph.\n\nSecond paragraph."
        chunks = _split_text(text, max_chars=20)
        self.assertGreater(len(chunks), 1)

    def test_split_text_comma_fallback(self):
        """Very long sentence falls back to clause boundaries."""
        from qwen3_tts.core.engine.text_processing import _split_text
        # A single long sentence with commas but no periods
        text = "This is a very long sentence, with several clauses, that should be split at commas when needed"
        chunks = _split_text(text, max_chars=40)
        # Should split due to length
        self.assertGreater(len(chunks), 1)


@_skip_generate
class TestSSMLParsing(unittest.TestCase):
    """Test SSML parsing in voice_generate."""

    def test_no_ssml(self):
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml("Hello world")
        self.assertEqual(text, "Hello world")
        self.assertFalse(meta["has_ssml"])

    def test_break_tag(self):
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('Hello <break time="500ms"/> world')
        self.assertTrue(meta["has_ssml"])
        self.assertNotIn("<break", text)

    def test_sub_tag(self):
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('<sub alias="World Wide Web">WWW</sub>')
        self.assertIn("World Wide Web", text)
        self.assertNotIn("WWW", text)

    def test_say_as_characters(self):
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('<say-as interpret-as="characters">ABC</say-as>')
        self.assertIn("A B C", text)

    def test_prosody_speed(self):
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('<prosody rate="fast">Quick text</prosody>')
        self.assertTrue(meta["has_ssml"])
        self.assertEqual(meta["prosody"]["speed"], 1.2)


@_skip_generate
class TestSRTParsing(unittest.TestCase):
    """Test SRT parsing."""

    def test_parse_srt(self):
        from qwen3_tts.interface.generate import parse_srt
        srt_content = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,500
Second subtitle
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            srt_path = f.name
        try:
            entries = parse_srt(srt_path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0][3], "Hello world")
            self.assertEqual(entries[1][3], "Second subtitle")
        finally:
            os.unlink(srt_path)

    def test_srt_time_to_ms(self):
        from qwen3_tts.interface.generate import srt_time_to_ms
        self.assertEqual(srt_time_to_ms("00:01:30,500"), 90500)
        self.assertEqual(srt_time_to_ms("01:00:00,000"), 3600000)


@_skip_generate
class TestAutoIncrementFilename(unittest.TestCase):
    """Test auto_increment_filename helper."""

    def test_no_conflict(self):
        from qwen3_tts.interface.generate import auto_increment_filename
        # Non-existent file should return as-is
        _tmp = os.path.join(tempfile.gettempdir(), "nonexistent_test_xyz.wav")
        result = auto_increment_filename(_tmp)
        self.assertEqual(result, _tmp)

    def test_conflict_increments(self):
        from qwen3_tts.interface.generate import auto_increment_filename
        # Create a temp file that definitely exists
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
            f.write(b"test")  # Write some content to ensure file exists
        try:
            # Verify file exists before calling
            self.assertTrue(os.path.exists(path), "Temp file should exist")
            result = auto_increment_filename(path)
            self.assertNotEqual(result, path)
            # Check that result has _2 before the extension
            base, ext = os.path.splitext(result)
            self.assertTrue(base.endswith("_2"), f"Expected '_2' suffix in {base}")
        finally:
            os.unlink(path)
            # Also clean up the incremented file if it was created
            if os.path.exists(result):
                os.unlink(result)

    def test_already_numbered(self):
        from qwen3_tts.interface.generate import auto_increment_filename
        # Create files with _2 suffix
        with tempfile.NamedTemporaryFile(suffix="_2.wav", delete=False, dir=tempfile.gettempdir(), prefix="test_") as f:
            path = f.name
        try:
            result = auto_increment_filename(path)
            self.assertNotEqual(result, path)
            self.assertIn("_3", result)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
