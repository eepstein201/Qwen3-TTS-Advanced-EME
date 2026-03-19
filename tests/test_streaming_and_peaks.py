"""Tests for Phase 3: Streaming response, waveform peaks, and WebSocket endpoint.

Task 3.1: Validate /generate-stream returns StreamingResponse
Task 3.3: Backend waveform peak calculation
Task 3.2: WebSocket endpoint (when implemented)
"""

import ast
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStreamingResponseType(unittest.TestCase):
    """Task 3.1: Verify /generate-stream returns StreamingResponse."""

    def test_generate_stream_returns_streaming_response(self):
        """The generate_stream function should return a StreamingResponse."""
        # Parse app.py AST to verify the return type
        with open("qwen3_tts/server/app.py") as f:
            tree = ast.parse(f.read())

        # Find the generate_stream function
        found_streaming = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "generate_stream":
                    # Check for StreamingResponse in function body
                    source = ast.dump(node)
                    found_streaming = "StreamingResponse" in source
                    break

        self.assertTrue(
            found_streaming,
            "/generate-stream endpoint must return StreamingResponse",
        )

    def test_streaming_response_imported(self):
        """StreamingResponse must be imported in app.py."""
        with open("qwen3_tts/server/app.py") as f:
            content = f.read()
        self.assertIn("StreamingResponse", content)

    def test_streaming_media_type_is_octet_stream(self):
        """Streaming response should use application/octet-stream media type."""
        with open("qwen3_tts/server/app.py") as f:
            content = f.read()
        self.assertIn("application/octet-stream", content)


class TestCalculateWaveformPeaks(unittest.TestCase):
    """Task 3.3: Backend waveform peak calculation."""

    def test_returns_correct_length(self):
        from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks

        audio = np.random.randn(24000).astype(np.float32)
        peaks = calculate_waveform_peaks(audio, num_peaks=100)
        self.assertEqual(len(peaks), 100)

    def test_peaks_in_valid_range(self):
        from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks

        audio = np.random.randn(24000).astype(np.float32)
        peaks = calculate_waveform_peaks(audio, num_peaks=100)
        for p in peaks:
            self.assertGreaterEqual(p, -1.0)
            self.assertLessEqual(p, 1.0)

    def test_all_zeros_returns_zeros(self):
        from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks

        audio = np.zeros(24000, dtype=np.float32)
        peaks = calculate_waveform_peaks(audio, num_peaks=50)
        self.assertEqual(len(peaks), 50)
        for p in peaks:
            self.assertEqual(p, 0.0)

    def test_empty_audio_returns_zeros(self):
        from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks

        audio = np.array([], dtype=np.float32)
        peaks = calculate_waveform_peaks(audio, num_peaks=50)
        self.assertEqual(len(peaks), 50)
        for p in peaks:
            self.assertEqual(p, 0.0)

    def test_stereo_audio_handled(self):
        """Should flatten stereo to mono before computing peaks."""
        from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks

        audio = np.random.randn(24000, 2).astype(np.float32)
        peaks = calculate_waveform_peaks(audio, num_peaks=100)
        self.assertEqual(len(peaks), 100)

    def test_num_peaks_capped_at_audio_length(self):
        """If audio is shorter than num_peaks, clamp to audio length."""
        from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks

        audio = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        peaks = calculate_waveform_peaks(audio, num_peaks=100)
        self.assertEqual(len(peaks), 3)  # capped at audio size

    def test_known_signal(self):
        """Test with a known signal to verify peak detection."""
        from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks

        # Create signal: first half silence, second half full amplitude
        audio = np.zeros(1000, dtype=np.float32)
        audio[500:] = 0.9
        peaks = calculate_waveform_peaks(audio, num_peaks=2)
        self.assertEqual(len(peaks), 2)
        self.assertAlmostEqual(peaks[0], 0.0, places=5)
        self.assertAlmostEqual(peaks[1], 0.9, places=5)

    def test_default_num_peaks(self):
        """Default num_peaks should be 500."""
        import inspect
        from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks

        sig = inspect.signature(calculate_waveform_peaks)
        self.assertEqual(sig.parameters["num_peaks"].default, 500)

    def test_exported_from_engine_facade(self):
        """calculate_waveform_peaks should be accessible from engine package."""
        from qwen3_tts.core.engine import calculate_waveform_peaks

        self.assertTrue(callable(calculate_waveform_peaks))


class TestWebSocketEndpointExists(unittest.TestCase):
    """Task 3.2: WebSocket endpoint structure test."""

    def test_websocket_module_exists(self):
        """WebSocket module should exist for bidirectional audio."""
        self.assertTrue(
            os.path.exists("qwen3_tts/server/websocket.py"),
            "qwen3_tts/server/websocket.py should exist for bidirectional audio streaming",
        )

    def test_websocket_has_handler(self):
        """WebSocket module should define a handler function."""
        with open("qwen3_tts/server/websocket.py") as f:
            content = f.read()
        self.assertIn("async def", content,
                       "WebSocket handler must be an async function")
        self.assertIn("WebSocket", content,
                       "WebSocket module must reference WebSocket type")

    def test_websocket_route_registered(self):
        """The /ws WebSocket route must be registered in the FastAPI app."""
        from qwen3_tts.server.app import app
        routes = [r.path for r in app.routes]
        self.assertIn("/ws", routes, "/ws WebSocket route must be registered in app")


if __name__ == "__main__":
    unittest.main()
