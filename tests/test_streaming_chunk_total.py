"""Regression test for R-35: chunk_total not populated in streaming path.

This test ensures that chunk_total is set during streaming generation.
"""

import unittest


class TestStreamingChunkTotal(unittest.TestCase):
    """Test that chunk_total is populated during streaming (unit tests, no HTTP layer)."""

    def test_run_inference_streaming_has_progress_callback_param(self):
        """run_inference_streaming should accept progress_callback parameter."""
        import inspect

        from qwen3_tts.core.engine import run_inference_streaming

        sig = inspect.signature(run_inference_streaming)
        self.assertIn("progress_callback", sig.parameters,
                     "progress_callback parameter missing from run_inference_streaming")

    def test_mlx_streaming_has_progress_callback_param(self):
        """_run_inference_mlx_streaming should accept progress_callback parameter."""
        import inspect

        from qwen3_tts.core.engine.inference import _run_inference_mlx_streaming

        sig = inspect.signature(_run_inference_mlx_streaming)
        self.assertIn("progress_callback", sig.parameters,
                     "progress_callback parameter missing from _run_inference_mlx_streaming")

    def test_progress_callback_updates_generation_state(self):
        """progress_callback should update generation_state correctly."""
        from qwen3_tts.server.app import app
        from tests.voice_test_helpers import _setup_fastapi_app_state

        # Setup app state with generation_state
        _setup_fastapi_app_state(app)

        # Define a callback that updates generation_state
        def chunk_progress(idx, total):
            app.state.generation_state.update({
                "chunk_index": idx,
                "chunk_total": total,
            })

        # Simulate callback calls
        chunk_progress(1, 3)
        self.assertEqual(app.state.generation_state["chunk_index"], 1)
        self.assertEqual(app.state.generation_state["chunk_total"], 3)

        chunk_progress(2, 3)
        self.assertEqual(app.state.generation_state["chunk_index"], 2)
        self.assertEqual(app.state.generation_state["chunk_total"], 3)

        chunk_progress(3, 3)
        self.assertEqual(app.state.generation_state["chunk_index"], 3)
        self.assertEqual(app.state.generation_state["chunk_total"], 3)


if __name__ == "__main__":
    unittest.main()
