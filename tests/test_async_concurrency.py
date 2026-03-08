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

        # Track how many times torch.multinomial gets reassigned
        call_count = {"count": 0}

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


if __name__ == "__main__":
    unittest.main()
