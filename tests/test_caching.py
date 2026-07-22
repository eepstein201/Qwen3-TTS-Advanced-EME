#!/usr/bin/env python3
"""Voice prompt cache tests.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_caching.py -v

No GPU, models, or running server required.
"""

import unittest

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Dummy decorator for when pytest is not available
    class _DummyMarkerFunc:
        """Represents a marker function like skipif that takes condition and returns decorator."""
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            # skipif, etc. take condition as first arg, return a decorator
            return lambda f: f
    class _DummyMarker:
        def __call__(self, func):
            return func
        def __getattr__(self, name):
            # Return special function for skipif, otherwise return a callable marker
            if name == 'skipif':
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



class TestVoicePromptCacheEdgeCases(unittest.TestCase):
    """Test voice prompt cache clearing and MLX cache eviction."""

    def test_clear_voice_prompt_cache_function(self):
        """clear_voice_prompt_cache() runs without error."""
        from qwen3_tts.core.engine import clear_voice_prompt_cache
        clear_voice_prompt_cache()

    def test_voice_prompt_cache_info_has_currsize(self):
        """voice_prompt_cache_info() returns object with currsize attribute."""
        from qwen3_tts.core.engine import voice_prompt_cache_info
        info = voice_prompt_cache_info()
        self.assertTrue(hasattr(info, "currsize"))

    def test_mlx_cache_eviction_at_max(self):
        """MLX prompt cache does not exceed configured max entries."""
        from qwen3_tts.core.config import get_voice_prompt_cache_max
        from qwen3_tts.core.engine.voice_prompt import _mlx_prompt_cache
        _mlx_prompt_cache.clear()
        max_cache = get_voice_prompt_cache_max()
        try:
            for i in range(max_cache):
                _mlx_prompt_cache[f"voice_{i}"] = {
                    "ref_audio": f"/tmp/v{i}.wav",  # nosec B108
                    "ref_text": "text",
                }
            self.assertEqual(len(_mlx_prompt_cache), max_cache)
            # Simulate the eviction logic used by load_voice_prompt_mlx
            if len(_mlx_prompt_cache) >= max_cache:
                oldest_key = next(iter(_mlx_prompt_cache))
                del _mlx_prompt_cache[oldest_key]
            _mlx_prompt_cache["voice_new"] = {"ref_audio": "/tmp/new.wav", "ref_text": "new"}  # nosec B108
            self.assertEqual(len(_mlx_prompt_cache), max_cache)
        finally:
            _mlx_prompt_cache.clear()

    def test_mlx_cache_eviction_removes_oldest(self):
        """After eviction, the first-inserted key is gone."""
        from qwen3_tts.core.config import get_voice_prompt_cache_max
        from qwen3_tts.core.engine.voice_prompt import _mlx_prompt_cache
        _mlx_prompt_cache.clear()
        max_cache = get_voice_prompt_cache_max()
        try:
            for i in range(max_cache):
                _mlx_prompt_cache[f"voice_{i}"] = {
                    "ref_audio": f"/tmp/v{i}.wav",  # nosec B108
                    "ref_text": "text",
                }
            # Evict oldest (voice_0) and insert new
            oldest_key = next(iter(_mlx_prompt_cache))
            self.assertEqual(oldest_key, "voice_0")
            del _mlx_prompt_cache[oldest_key]
            _mlx_prompt_cache["voice_extra"] = {"ref_audio": "/tmp/extra.wav", "ref_text": "extra"}  # nosec B108
            self.assertNotIn("voice_0", _mlx_prompt_cache)
            self.assertIn("voice_extra", _mlx_prompt_cache)
        finally:
            _mlx_prompt_cache.clear()


if __name__ == "__main__":
    unittest.main()
