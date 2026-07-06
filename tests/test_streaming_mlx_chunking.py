"""Regression test: MLX streaming path must apply text chunking for long texts.

Root cause (fixed): run_inference_streaming passed the full text to
_run_inference_mlx_streaming without splitting it. max_new_tokens=2048 at
12 Hz ≈ 170 s of audio, so texts longer than ~2500 chars were silently
truncated. The fix applies _prepare_text_chunks before streaming, matching
what the batch path (run_inference) already does.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _make_fake_mlx_audio_result(sample_rate=24000, n_samples=240):
    """Return a minimal result object as yielded by model.generate(stream=True)."""
    import numpy as np

    result = MagicMock()
    result.audio = np.zeros(n_samples, dtype=np.float32)
    result.sample_rate = sample_rate
    return result


class TestMLXStreamingChunking(unittest.TestCase):
    """MLX streaming path splits long text into chunks before calling the model."""

    def _patch_backend_mlx(self):
        return patch("qwen3_tts.core.engine.inference.get_backend", return_value="mlx")

    def _patch_mlx_streaming(self, yields_per_call=1):
        import numpy as np

        fake_result = _make_fake_mlx_audio_result()

        def _fake_streaming(*args, **kwargs):
            for _ in range(yields_per_call):
                yield np.zeros(240, dtype=np.float32), 24000

        return patch(
            "qwen3_tts.core.engine.inference._run_inference_mlx_streaming",
            side_effect=_fake_streaming,
        )

    # ------------------------------------------------------------------
    # Short text — should result in exactly one model call
    # ------------------------------------------------------------------

    def test_short_text_single_chunk(self):
        """A short text (≤500 chars) must produce exactly one streaming call."""
        from qwen3_tts.core.engine.inference import run_inference_streaming

        short_text = "Hello world."
        model = MagicMock()
        model.tokenizer = None  # force char-based chunking path

        with self._patch_backend_mlx(), self._patch_mlx_streaming() as mock_stream:
            list(
                run_inference_streaming(
                    model=model,
                    text=short_text,
                    mode="custom",
                    gen_params={},
                )
            )
        self.assertEqual(mock_stream.call_count, 1)
        called_text = mock_stream.call_args[0][1]
        self.assertEqual(called_text, short_text.strip())

    # ------------------------------------------------------------------
    # Long text — must be split into multiple model calls
    # ------------------------------------------------------------------

    def test_long_text_multiple_chunks(self):
        """A text that exceeds max_chunk_chars must be split into multiple streaming calls."""
        from qwen3_tts.core.engine.inference import run_inference_streaming

        # Build a text that is clearly longer than the default 500-char chunk limit.
        sentence = "This is a sentence that takes up space in the text. "
        long_text = sentence * 20  # ~1040 chars → at least 2 chunks

        model = MagicMock()
        model.tokenizer = None  # force char-based chunking path

        with self._patch_backend_mlx(), self._patch_mlx_streaming() as mock_stream:
            list(
                run_inference_streaming(
                    model=model,
                    text=long_text,
                    mode="custom",
                    gen_params={},
                    max_chunk_chars=500,
                )
            )
        self.assertGreater(
            mock_stream.call_count,
            1,
            "Long text should produce more than one _run_inference_mlx_streaming call",
        )

    # ------------------------------------------------------------------
    # All audio chunks from all text chunks are yielded
    # ------------------------------------------------------------------

    def test_all_audio_chunks_yielded(self):
        """Audio from every text chunk must be forwarded to the caller."""
        import numpy as np
        from qwen3_tts.core.engine.inference import run_inference_streaming

        sentence = "This is a sentence that takes up space in the text. "
        long_text = sentence * 20  # >500 chars → multiple text chunks

        model = MagicMock()
        model.tokenizer = None  # force char-based chunking path

        with self._patch_backend_mlx(), self._patch_mlx_streaming(
            yields_per_call=2
        ) as mock_stream:
            results = list(
                run_inference_streaming(
                    model=model,
                    text=long_text,
                    mode="custom",
                    gen_params={},
                    max_chunk_chars=500,
                )
            )

        # Each text chunk yields 2 audio frames; total = call_count × 2
        expected = mock_stream.call_count * 2
        self.assertEqual(len(results), expected)

    # ------------------------------------------------------------------
    # Progress callback receives correct text-chunk totals
    # ------------------------------------------------------------------

    def test_progress_callback_reports_text_chunks(self):
        """progress_callback should be called with (chunk_idx, chunk_total) per text chunk."""
        from qwen3_tts.core.engine.inference import run_inference_streaming

        sentence = "This is a sentence that takes up space in the text. "
        long_text = sentence * 20

        model = MagicMock()
        model.tokenizer = None  # force char-based chunking path
        progress_calls = []

        with self._patch_backend_mlx(), self._patch_mlx_streaming():
            list(
                run_inference_streaming(
                    model=model,
                    text=long_text,
                    mode="custom",
                    gen_params={},
                    max_chunk_chars=500,
                    progress_callback=lambda idx, total: progress_calls.append(
                        (idx, total)
                    ),
                )
            )

        self.assertTrue(
            len(progress_calls) > 0, "progress_callback should have been called"
        )
        # All reported totals should be equal and > 1
        totals = {t for _, t in progress_calls}
        self.assertEqual(len(totals), 1, "chunk_total should be consistent across calls")
        chunk_total = totals.pop()
        self.assertGreater(chunk_total, 1, "chunk_total should reflect multiple text chunks")
        # Indices should be 1..chunk_total
        indices = [i for i, _ in progress_calls]
        self.assertEqual(indices, list(range(1, chunk_total + 1)))


if __name__ == "__main__":
    unittest.main()
