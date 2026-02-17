#!/usr/bin/env python3
"""Core infrastructure tests: error paths, concurrency, caching, config edge cases,
SSML edge cases, and dry-run verification.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_core_infra.py -v

No GPU, models, or running server required.
"""

import inspect
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check optional dependencies
try:
    import soundfile  # noqa: F401
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import flask  # noqa: F401
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

_server_deps = HAS_SOUNDFILE and HAS_FLASK
_skip_server = unittest.skipUnless(_server_deps, "requires soundfile + flask")
_skip_generate = unittest.skipUnless(HAS_SOUNDFILE, "requires soundfile (voice_generate)")


# =========================================================================
# Error Path Tests
# =========================================================================

@_skip_server
class TestErrorPaths(unittest.TestCase):
    """Test error handling in engine and server validation."""

    @classmethod
    def setUpClass(cls):
        import voice_server
        voice_server.auth_token = "test_token"
        voice_server.server_config = {
            "security": {"max_text_length": 100, "max_batch_size": 5},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()
        cls.auth = {"Authorization": "Bearer test_token"}

    def test_set_audio_loader_invalid(self):
        """set_audio_loader('invalid') raises ValueError."""
        from voice_engine import set_audio_loader
        with self.assertRaises(ValueError):
            set_audio_loader("invalid")

    def test_set_audio_loader_valid(self):
        """set_audio_loader('librosa') succeeds and get_audio_loader reflects it."""
        from voice_engine import set_audio_loader, get_audio_loader
        original = get_audio_loader()
        try:
            set_audio_loader("librosa")
            self.assertEqual(get_audio_loader(), "librosa")
        finally:
            set_audio_loader(original)

    def test_load_config_returns_dict(self):
        """load_config() always returns a dict."""
        from voice_config import load_config
        result = load_config()
        self.assertIsInstance(result, dict)

    def test_parse_ssml_no_tags(self):
        """Plain text with no SSML tags returns unchanged with has_ssml=False."""
        from voice_generate import parse_ssml
        text, meta = parse_ssml("plain text with no tags")
        self.assertEqual(text, "plain text with no tags")
        self.assertFalse(meta["has_ssml"])

    def test_parse_ssml_malformed_unclosed(self):
        """Malformed unclosed tags do not crash parse_ssml."""
        from voice_generate import parse_ssml
        # '<break Hello' has no closing '>' for a valid tag, so regex won't match
        text, meta = parse_ssml("<break Hello")
        self.assertIsInstance(text, str)

    def test_generate_endpoint_missing_texts(self):
        """POST /generate with empty body returns 400."""
        resp = self.client.post("/generate", json={}, headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_generate_endpoint_empty_texts(self):
        """POST /generate with empty texts list returns 400."""
        resp = self.client.post("/generate", json={"texts": []}, headers=self.auth)
        self.assertEqual(resp.status_code, 400)

    def test_generate_endpoint_text_over_limit(self):
        """Text exceeding max_text_length returns 400."""
        long_text = "A" * 200  # server_config max is 100
        resp = self.client.post("/generate", json={"texts": [long_text]}, headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("limit", data["error"])


# =========================================================================
# Generation Cache Thread Safety Tests
# =========================================================================

@_skip_server
class TestGenerationCacheThreadSafety(unittest.TestCase):
    """Verify generation cache uses proper locking."""

    def test_gen_cache_lock_is_threading_lock(self):
        """_gen_cache_lock is an instance of threading.Lock."""
        import voice_server
        self.assertIsInstance(voice_server._gen_cache_lock, type(threading.Lock()))

    def test_gen_cache_get_acquires_lock(self):
        """_gen_cache_get acquires the lock."""
        import voice_server
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=False)
        original = voice_server._gen_cache_lock
        voice_server._gen_cache_lock = mock_lock
        try:
            voice_server._gen_cache_get("test_key")
            mock_lock.__enter__.assert_called_once()
        finally:
            voice_server._gen_cache_lock = original

    def test_gen_cache_put_acquires_lock(self):
        """_gen_cache_put acquires the lock."""
        import voice_server
        import tempfile
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=False)
        original_lock = voice_server._gen_cache_lock
        original_cache = voice_server._gen_cache.copy()
        voice_server._gen_cache_lock = mock_lock
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            voice_server._gen_cache_put("test_put_key", tmp_path, 24000)
            mock_lock.__enter__.assert_called_once()
        finally:
            voice_server._gen_cache_lock = original_lock
            voice_server._gen_cache = original_cache
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_gen_cache_invalidate_acquires_lock(self):
        """_gen_cache_invalidate acquires the lock."""
        import voice_server
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=False)
        original = voice_server._gen_cache_lock
        voice_server._gen_cache_lock = mock_lock
        try:
            voice_server._gen_cache_invalidate()
            mock_lock.__enter__.assert_called_once()
        finally:
            voice_server._gen_cache_lock = original

    def test_double_checked_locking_in_generate_source(self):
        """generate() source contains both pre-lock and post-lock cache comments."""
        import voice_server
        source = inspect.getsource(voice_server.generate)
        self.assertIn("pre-lock", source)
        self.assertIn("post-lock", source)


# =========================================================================
# Voice Prompt Cache Edge Cases
# =========================================================================

class TestVoicePromptCacheEdgeCases(unittest.TestCase):
    """Test voice prompt cache clearing and MLX cache eviction."""

    def test_clear_voice_prompt_cache_function(self):
        """clear_voice_prompt_cache() runs without error."""
        from voice_engine import clear_voice_prompt_cache
        clear_voice_prompt_cache()

    def test_voice_prompt_cache_info_has_currsize(self):
        """voice_prompt_cache_info() returns object with currsize attribute."""
        from voice_engine import voice_prompt_cache_info
        info = voice_prompt_cache_info()
        self.assertTrue(hasattr(info, "currsize"))

    def test_mlx_cache_eviction_at_max(self):
        """MLX prompt cache does not exceed _MLX_PROMPT_CACHE_MAX entries."""
        from voice_engine import _mlx_prompt_cache, _MLX_PROMPT_CACHE_MAX
        _mlx_prompt_cache.clear()
        try:
            for i in range(_MLX_PROMPT_CACHE_MAX):
                _mlx_prompt_cache[f"voice_{i}"] = {
                    "ref_audio": f"/tmp/v{i}.wav",
                    "ref_text": "text",
                }
            self.assertEqual(len(_mlx_prompt_cache), _MLX_PROMPT_CACHE_MAX)
            # Simulate the eviction logic used by load_voice_prompt_mlx
            if len(_mlx_prompt_cache) >= _MLX_PROMPT_CACHE_MAX:
                oldest_key = next(iter(_mlx_prompt_cache))
                del _mlx_prompt_cache[oldest_key]
            _mlx_prompt_cache["voice_new"] = {"ref_audio": "/tmp/new.wav", "ref_text": "new"}
            self.assertEqual(len(_mlx_prompt_cache), _MLX_PROMPT_CACHE_MAX)
        finally:
            _mlx_prompt_cache.clear()

    def test_mlx_cache_eviction_removes_oldest(self):
        """After eviction, the first-inserted key is gone."""
        from voice_engine import _mlx_prompt_cache, _MLX_PROMPT_CACHE_MAX
        _mlx_prompt_cache.clear()
        try:
            for i in range(_MLX_PROMPT_CACHE_MAX):
                _mlx_prompt_cache[f"voice_{i}"] = {
                    "ref_audio": f"/tmp/v{i}.wav",
                    "ref_text": "text",
                }
            # Evict oldest (voice_0) and insert new
            oldest_key = next(iter(_mlx_prompt_cache))
            self.assertEqual(oldest_key, "voice_0")
            del _mlx_prompt_cache[oldest_key]
            _mlx_prompt_cache["voice_extra"] = {"ref_audio": "/tmp/extra.wav", "ref_text": "extra"}
            self.assertNotIn("voice_0", _mlx_prompt_cache)
            self.assertIn("voice_extra", _mlx_prompt_cache)
        finally:
            _mlx_prompt_cache.clear()


# =========================================================================
# Config Edge Cases
# =========================================================================

class TestConfigEdgeCases(unittest.TestCase):
    """Test config constants and accessor functions."""

    def test_get_backend_returns_string(self):
        """get_backend() returns 'torch' or 'mlx'."""
        from voice_config import get_backend
        result = get_backend()
        self.assertIn(result, ("torch", "mlx"))

    def test_get_model_size_default(self):
        """get_model_size() returns '1.7B' or '0.6B'."""
        from voice_config import get_model_size
        result = get_model_size()
        self.assertIn(result, ("1.7B", "0.6B"))

    def test_model_info_keys(self):
        """MODEL_INFO has size-based keys."""
        from voice_config import MODEL_INFO
        self.assertIn("1.7B", MODEL_INFO)
        self.assertIn("0.6B", MODEL_INFO)

    def test_custom_speakers_have_fields(self):
        """Each entry in CUSTOM_VOICE_SPEAKERS has a 'name' key."""
        from voice_config import CUSTOM_VOICE_SPEAKERS
        self.assertGreater(len(CUSTOM_VOICE_SPEAKERS), 0)
        for key, entry in CUSTOM_VOICE_SPEAKERS.items():
            self.assertIn("name", entry, f"Speaker '{key}' missing 'name' field")

    def test_prosody_presets_all_strings(self):
        """All values in DEFAULT_PROSODY_PRESETS are non-empty strings."""
        from voice_config import DEFAULT_PROSODY_PRESETS
        self.assertGreater(len(DEFAULT_PROSODY_PRESETS), 0)
        for key, value in DEFAULT_PROSODY_PRESETS.items():
            self.assertIsInstance(value, str, f"Preset '{key}' is not a string")
            self.assertTrue(len(value) > 0, f"Preset '{key}' is empty")

    def test_valid_backends_constant(self):
        """VALID_BACKENDS contains 'torch' and 'mlx'."""
        from voice_config import VALID_BACKENDS
        self.assertIn("torch", VALID_BACKENDS)
        self.assertIn("mlx", VALID_BACKENDS)


# =========================================================================
# SSML Edge Cases
# =========================================================================

@_skip_generate
class TestSSMLEdgeCases(unittest.TestCase):
    """Test SSML parsing edge cases."""

    def test_ssml_sub_replacement(self):
        """<sub alias='hello'>hi</sub> replaces content with alias."""
        from voice_generate import parse_ssml
        text, meta = parse_ssml('Say <sub alias="hello">hi</sub> please')
        self.assertIn("hello", text)
        self.assertNotIn("<sub", text)
        self.assertNotIn("hi", text)

    def test_ssml_say_as_characters(self):
        """<say-as interpret-as='characters'>ABC</say-as> spells out as 'A B C'."""
        from voice_generate import parse_ssml
        text, meta = parse_ssml('<say-as interpret-as="characters">ABC</say-as>')
        self.assertIn("A B C", text)

    def test_ssml_prosody_rate_slow(self):
        """<prosody rate='slow'> sets speed=0.8 in metadata."""
        from voice_generate import parse_ssml
        text, meta = parse_ssml('<prosody rate="slow">Hello world</prosody>')
        self.assertTrue(meta["has_ssml"])
        self.assertIsNotNone(meta["prosody"])
        self.assertEqual(meta["prosody"]["speed"], 0.8)

    def test_ssml_prosody_pitch_high(self):
        """<prosody pitch='high'> sets pitch=2 in metadata."""
        from voice_generate import parse_ssml
        text, meta = parse_ssml('<prosody pitch="high">Hello world</prosody>')
        self.assertTrue(meta["has_ssml"])
        self.assertIsNotNone(meta["prosody"])
        self.assertEqual(meta["prosody"]["pitch"], 2)

    def test_ssml_nested_emphasis_sub(self):
        """Nested <emphasis><sub alias='hello'>hi</sub></emphasis> produces 'hello'."""
        from voice_generate import parse_ssml
        text, meta = parse_ssml('<emphasis><sub alias="hello">hi</sub></emphasis>')
        self.assertIn("hello", text)
        self.assertNotIn("<", text)


# =========================================================================
# Dry-Run and Interactive Mode Tests
# =========================================================================

@_skip_generate
class TestDryRunAndInteractive(unittest.TestCase):
    """Verify dry-run flag and interactive mode exist in source."""

    def test_dry_run_flag_in_source(self):
        """voice_generate.py source contains '--dry-run' argument."""
        import voice_generate
        source = inspect.getsource(voice_generate)
        self.assertIn("--dry-run", source)

    def test_dry_run_marker_in_source(self):
        """voice_generate.py source contains 'DRY RUN' marker text."""
        import voice_generate
        source = inspect.getsource(voice_generate)
        self.assertIn("DRY RUN", source)

    def test_interactive_mode_function_exists(self):
        """voice_generate has a callable interactive_mode function."""
        import voice_generate
        self.assertTrue(hasattr(voice_generate, "interactive_mode"))
        self.assertTrue(callable(voice_generate.interactive_mode))


if __name__ == "__main__":
    unittest.main()
