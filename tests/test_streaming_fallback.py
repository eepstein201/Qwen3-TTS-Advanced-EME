"""Tests for streaming fallback functionality."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from qwen3_tts.interface.ui import _generate_streaming_impl


class TestStreamingFallback(unittest.TestCase):
    """Test the fallback behavior when streaming fails."""

    @patch('qwen3_tts.interface.ui.TTSClient')
    def test_streaming_fallback_on_exception(self, mock_client_class):
        """Should fall back to non-streaming when streaming raises exception."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.is_server_running.return_value = True

        # First chunk succeeds, then exception
        call_count = [0]
        def failing_stream(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                yield (24000, [0.1] * 1000)
            raise Exception("Streaming connection lost")

        mock_client.generate_streaming.side_effect = failing_stream
        mock_client.generate.return_value = "/tmp/test_output.wav"

        # Mock model loading
        with patch('qwen3_tts.interface.ui._ensure_model_loaded'):
            results = list(_generate_streaming_impl(
                mode="clone",
                text="test",
                preset="default",
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                rep_penalty=1.05,
                seed=None
            ))

        # Should have streaming chunk, then fallback status
        self.assertGreater(len(results), 1)
        # Check for fallback indicators
        statuses = [r[1] for r in results if isinstance(r, tuple) and len(r) > 1]
        has_fallback = any("file mode" in str(s).lower() or "trying file" in str(s).lower() for s in statuses)
        self.assertTrue(has_fallback, f"Expected fallback status in: {statuses}")

    @patch('qwen3_tts.interface.ui.TTSClient')
    def test_normal_streaming_completes(self, mock_client_class):
        """Normal streaming should complete without fallback."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.is_server_running.return_value = True

        # Normal streaming
        def normal_stream(**kwargs):
            for i in range(3):
                yield (24000, [0.1] * 1000)

        mock_client.generate_streaming.side_effect = normal_stream

        with patch('qwen3_tts.interface.ui._ensure_model_loaded'), \
             patch('qwen3_tts.interface.ui._save_streaming_audio') as mock_save:
            mock_save.return_value = "/tmp/test.wav"

            results = list(_generate_streaming_impl(
                mode="clone",
                text="test",
                preset="default",
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                rep_penalty=1.05,
                seed=None
            ))

        # Should complete normally with "Complete" status
        final_results = [r for r in results if isinstance(r, tuple) and len(r) > 1]
        self.assertGreater(len(final_results), 0)
        final_status = str(final_results[-1][1])
        self.assertIn("Complete", final_status)


if __name__ == "__main__":
    unittest.main()
