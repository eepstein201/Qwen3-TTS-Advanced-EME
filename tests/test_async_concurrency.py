#!/usr/bin/env python3
"""Tests for thread safety in concurrent scenarios."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestMPSPatchThreadSafety(unittest.TestCase):
    """Test that _install_mps_patch is thread-safe."""

    def test_mps_patch_lock_exists(self):
        """Verify that a lock exists for MPS patch synchronization."""
        from qwen3_tts.core.engine import model_loader

        # Check for lock attribute
        self.assertTrue(
            hasattr(model_loader, '_mps_patch_lock'),
            "_mps_patch_lock should exist for thread safety"
        )
        self.assertIsInstance(
            model_loader._mps_patch_lock,
            type(threading.Lock()),
            "_mps_patch_lock should be a threading.Lock"
        )

    def test_mps_patch_thread_safety(self):
        """Concurrent calls should only install patch once.

        The old assertion `mock_torch.multinomial != MagicMock()` is a
        tautology (two distinct mocks are always !=), so it passed even
        when the wrapper was never assigned. The identity check below is
        the real detector: verified RED when production skips the
        ``torch.multinomial = _safe_multinomial`` assignment.
        """
        from qwen3_tts.core.engine import model_loader

        # Reset state; restore afterwards so the installed flag set under
        # a mock torch does not leak into later tests in the same process.
        saved_installed = model_loader._mps_patch_installed
        model_loader._mps_patch_installed = False

        def _restore():
            model_loader._mps_patch_installed = saved_installed

        self.addCleanup(_restore)

        # Mock torch module
        mock_torch = MagicMock()
        original_multinomial = MagicMock()
        mock_torch.multinomial = original_multinomial
        mock_torch.nan_to_num = MagicMock(return_value=MagicMock())
        mock_torch.cuda.is_available.return_value = False

        with patch.dict('sys.modules', {'torch': mock_torch}):
            with patch('qwen3_tts.core.config.IS_MACOS', True):
                threads = []
                for _ in range(10):
                    t = threading.Thread(target=model_loader._install_mps_patch)
                    threads.append(t)

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                # With proper locking, patch should be installed exactly once:
                # the module attribute must be replaced with the wrapper...
                self.assertIsNot(
                    mock_torch.multinomial,
                    original_multinomial,
                    "torch.multinomial was never replaced — patch not installed"
                )
                # ...and the wrapper must delegate to the original exactly
                # once per call (no double-wrapping).
                mock_torch.multinomial(MagicMock(), 2)
                self.assertEqual(
                    original_multinomial.call_count, 1,
                    "patched multinomial must delegate to the original exactly once"
                )

        # And state should be True
        self.assertTrue(model_loader._mps_patch_installed, "Flag should be set")


class TestMLXPromptCacheThreadSafety(unittest.TestCase):
    """Test that MLX voice prompt cache is thread-safe."""

    def test_mlx_prompt_cache_lock_exists(self):
        """Verify that a lock exists for MLX prompt cache synchronization."""
        from qwen3_tts.core.engine import voice_prompt

        # Check for lock attribute
        self.assertTrue(
            hasattr(voice_prompt, '_mlx_prompt_cache_lock'),
            "_mlx_prompt_cache_lock should exist for thread safety"
        )
        self.assertIsInstance(
            voice_prompt._mlx_prompt_cache_lock,
            type(threading.Lock()),
            "_mlx_prompt_cache_lock should be a threading.Lock"
        )

    def test_mlx_prompt_cache_concurrent_access(self):
        """Concurrent load_voice_prompt_mlx() calls must hit the real cache.

        The old version exercised a LOCAL OrderedDict that mimicked the
        cache — production code was never called, so disabling the real
        cache (or its lock) left the test green. This version runs the
        real loader against a temp prompt directory: a cache hit must
        return the SAME object the first (miss) call produced, and
        concurrent callers must all observe that one entry. Verified RED
        when production stops populating _mlx_prompt_cache.
        """
        import qwen3_tts.core.engine.voice_prompt as voice_prompt

        # Snapshot the module cache; restore after the test so a prompt
        # cached against the temp dir does not leak into other tests.
        saved_cache = voice_prompt._mlx_prompt_cache.copy()
        voice_prompt._mlx_prompt_cache.clear()

        def _restore():
            voice_prompt._mlx_prompt_cache.clear()
            voice_prompt._mlx_prompt_cache.update(saved_cache)

        self.addCleanup(_restore)

        with tempfile.TemporaryDirectory() as tmp:
            prompt_name = "concurrency_probe"
            Path(tmp, f"{prompt_name}.wav").write_bytes(b"RIFF-fake-wav")
            Path(tmp, f"{prompt_name}.txt").write_text("probe transcript")

            with patch.object(voice_prompt, "VOICE_PROMPTS_DIR", tmp):
                # First call: cache miss, loads from disk.
                first = voice_prompt.load_voice_prompt_mlx(prompt_name)
                self.assertEqual(first["ref_text"], "probe transcript")

                # Second call: must be a cache HIT — same object, not a
                # freshly loaded dict.
                second = voice_prompt.load_voice_prompt_mlx(prompt_name)
                self.assertIs(
                    second, first,
                    "repeat load must be served from the cache (same object)"
                )

                # Concurrent callers: all succeed, all see the cached entry,
                # exactly one cache entry for the prompt.
                errors = []

                def load_many():
                    try:
                        for _ in range(20):
                            voice_prompt.load_voice_prompt_mlx(prompt_name)
                    except Exception as e:  # noqa: BLE001 — recorded, asserted below
                        errors.append(str(e))

                threads = [threading.Thread(target=load_many) for _ in range(10)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                self.assertEqual(errors, [], f"concurrent loads raised: {errors}")
                self.assertEqual(
                    len(voice_prompt._mlx_prompt_cache), 1,
                    "concurrent loads must leave exactly one cache entry"
                )



def _is_mlx_backend():
    """Check if current backend is MLX."""
    try:
        from qwen3_tts.core.config import load_config
        config = load_config()
        return config.get("advanced", {}).get("backend") == "mlx"
    except Exception:
        return False


class TestASRThreadSafety(unittest.TestCase):
    def _restore_asr_globals(self, asr):
        """Snapshot the ASR module globals and restore them after the test.

        The thread-safety tests below deliberately null ``_asr_model_mlx`` and
        then race real ``load_asr_model()`` calls into it, which leaves a mock
        model in the module global. Left set, ``is_asr_loaded()`` returns True
        for the remainder of the process, so any later test expecting an
        unloaded ASR sees "already_loaded" instead of "loaded" —
        tests/test_asr_endpoints.py fails exactly that way inside the batch
        runner while passing when run alone.
        """
        saved_mlx = asr._asr_model_mlx
        saved_torch = asr._asr_model_torch

        def _restore():
            asr._asr_model_mlx = saved_mlx
            asr._asr_model_torch = saved_torch

        self.addCleanup(_restore)

    """Test that ASR model loading is thread-safe."""

    def test_asr_lock_exists(self):
        """Verify that a lock exists for ASR model synchronization."""
        from qwen3_tts.core.engine import asr

        # Check for lock attribute
        self.assertTrue(
            hasattr(asr, '_asr_lock'),
            "_asr_lock should exist for thread safety"
        )
        self.assertIsInstance(
            asr._asr_lock,
            type(threading.Lock()),
            "_asr_lock should be a threading.Lock"
        )

    @unittest.skipUnless(_is_mlx_backend(), "requires MLX backend")
    def test_asr_mlx_thread_safety(self):
        """Concurrent load_asr_model() calls should only load model once for MLX backend."""
        from qwen3_tts.core.engine import asr

        self._restore_asr_globals(asr)

        # Track how many times the MLX model loader is called
        load_count = {"count": 0}
        errors = []

        # Mock the backend to return "mlx"
        with patch('qwen3_tts.core.engine.asr.get_backend', return_value='mlx'):
            # Mock mlx_audio.stt.load_model
            mock_model = MagicMock()
            mock_load_fn = MagicMock(return_value=mock_model)

            def track_load(*args, **kwargs):
                load_count["count"] += 1
                # Simulate slow load to increase race condition chance
                import time
                time.sleep(0.01)
                return mock_model

            mock_load_fn.side_effect = track_load

            with patch.dict('sys.modules', {'mlx_audio.stt': MagicMock(load_model=mock_load_fn)}):
                # Reset ASR model state
                asr._asr_model_mlx = None

                # Spawn concurrent load_asr_model() calls
                def load_in_thread():
                    try:
                        asr.load_asr_model()
                    except Exception as e:
                        errors.append(str(e))

                threads = [threading.Thread(target=load_in_thread) for _ in range(10)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        # With proper locking, model should only be loaded once
        self.assertEqual(
            load_count["count"], 1,
            f"MLX ASR model should be loaded exactly once, but was loaded {load_count['count']} times"
        )
        self.assertEqual(len(errors), 0, f"Concurrent loading caused errors: {errors}")

    @unittest.skipUnless(_is_mlx_backend(), "requires MLX backend")
    def test_asr_mlx_transcribe_thread_safety(self):
        """Concurrent _transcribe_mlx calls should only load model once."""
        from qwen3_tts.core.engine import asr

        self._restore_asr_globals(asr)

        # Track how many times the MLX model loader is called
        load_count = {"count": 0}
        errors = []

        # Mock mlx_audio.stt.load_model
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=MagicMock(text="test transcript"))
        mock_load_fn = MagicMock(return_value=mock_model)

        def track_load(*args, **kwargs):
            load_count["count"] += 1
            # Simulate slow load to increase race condition chance
            import time
            time.sleep(0.01)
            return mock_model

        mock_load_fn.side_effect = track_load

        with patch.dict('sys.modules', {'mlx_audio.stt': MagicMock(load_model=mock_load_fn)}):
            # Reset ASR model state
            asr._asr_model_mlx = None

            # Spawn concurrent _transcribe_mlx calls (which load model on first use)
            def transcribe_in_thread():
                try:
                    asr._transcribe_mlx("/fake/audio.wav")
                except Exception as e:
                    # We expect some errors from the fake path, but not from loading
                    if "Loading" in str(e) or "load" in str(e).lower():
                        errors.append(str(e))

            threads = [threading.Thread(target=transcribe_in_thread) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # With proper locking, model should only be loaded once
        self.assertEqual(
            load_count["count"], 1,
            f"MLX ASR model should be loaded exactly once during transcription, but was loaded {load_count['count']} times"
        )


if __name__ == "__main__":
    unittest.main()
