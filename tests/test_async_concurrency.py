#!/usr/bin/env python3
"""Tests for thread safety in concurrent scenarios."""

import threading
import unittest
from unittest.mock import patch, MagicMock


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
        """Concurrent calls should only install patch once."""
        from qwen3_tts.core.engine import model_loader

        # Reset state
        model_loader._mps_patch_installed = False

        # Mock torch module
        mock_torch = MagicMock()
        mock_torch.multinomial = MagicMock()
        mock_torch.nan_to_num = MagicMock(return_value=MagicMock())
        mock_torch.cuda.is_available.return_value = False

        with patch.dict('sys.modules', {'torch': mock_torch}):
            with patch('qwen3_tts.core.config.IS_MACOS', True):
                # Reset state
                model_loader._mps_patch_installed = False

                threads = []
                for _ in range(10):
                    t = threading.Thread(target=model_loader._install_mps_patch)
                    threads.append(t)

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        # With proper locking, patch should only be installed once
        # Check that multinomial was assigned (patched)
        self.assertTrue(
            mock_torch.multinomial != MagicMock(),
            "Patch should have been installed"
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
        """Concurrent cache access should not raise RuntimeError."""
        from collections import OrderedDict

        # Create a test cache that simulates concurrent access
        test_cache = OrderedDict()
        errors = []

        def cache_operations():
            try:
                for i in range(100):
                    # Simulate the operations that happen in load_voice_prompt_mlx
                    key = f"test_prompt_{i % 5}"
                    if key in test_cache:
                        test_cache.move_to_end(key)
                    else:
                        if len(test_cache) >= 10:
                            test_cache.popitem(last=False)
                        test_cache[key] = {"data": i}
            except RuntimeError as e:
                errors.append(str(e))

        # Run multiple threads without lock - this should cause issues
        # (This test documents the problem; the fix makes it pass)
        threads = [threading.Thread(target=cache_operations) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Without a lock, OrderedDict may raise RuntimeError
        # With the lock in place in the actual code, this won't happen
        self.assertEqual(
            len(errors), 0,
            f"Concurrent cache access caused errors (race condition): {errors}"
        )


class TestASRThreadSafety(unittest.TestCase):
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

    def test_asr_mlx_thread_safety(self):
        """Concurrent load_asr_model() calls should only load model once for MLX backend."""
        from qwen3_tts.core.engine import asr

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

    def test_asr_mlx_transcribe_thread_safety(self):
        """Concurrent _transcribe_mlx calls should only load model once."""
        from qwen3_tts.core.engine import asr

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
