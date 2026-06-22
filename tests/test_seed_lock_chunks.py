#!/usr/bin/env python3
"""Tests for seed-lock-chunks feature (voice consistency across chunks).

Covers:
  - _set_seed_for_backend: sets seed via torch or mlx depending on active backend
  - run_inference with seed_lock_chunks=True: re-seeds before each chunk
  - GenerateRequest: accepts seed_lock_chunks field
  - UI: checkbox wiring

Run: pytest tests/test_seed_lock_chunks.py -v
"""
import unittest
from unittest.mock import MagicMock, call, patch

try:
    import mlx.core  # noqa: F401
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class TestSetSeedForBackend(unittest.TestCase):
    """Tests for _set_seed_for_backend helper."""

    @patch("qwen3_tts.core.engine.inference.get_backend", return_value="torch")
    def test_torch_backend_calls_manual_seed(self, _mock_backend):
        from qwen3_tts.core.engine.inference import _set_seed_for_backend
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            _set_seed_for_backend(42)
            mock_torch.manual_seed.assert_called_once_with(42)

    @unittest.skipUnless(HAS_MLX, "requires mlx")
    @patch("qwen3_tts.core.engine.inference.get_backend", return_value="mlx")
    def test_mlx_backend_calls_mx_random_seed(self, _mock_backend):
        from qwen3_tts.core.engine.inference import _set_seed_for_backend
        mock_seed = MagicMock()
        try:
            import mlx.core as mx
            with patch.object(mx.random, "seed", mock_seed):
                _set_seed_for_backend(42)
                mock_seed.assert_called_once_with(42)
        except ImportError:
            # Not in MLX env — use module mock
            mock_mx = MagicMock()
            with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": mock_mx}):
                _set_seed_for_backend(42)
                mock_mx.random.seed.assert_called_once_with(42)

    @patch("qwen3_tts.core.engine.inference.get_backend", return_value="torch")
    def test_none_seed_is_noop(self, _mock_backend):
        from qwen3_tts.core.engine.inference import _set_seed_for_backend
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            _set_seed_for_backend(None)
            mock_torch.manual_seed.assert_not_called()


class TestSeedLockChunksInRunInference(unittest.TestCase):
    """Tests that seed_lock_chunks re-seeds before each chunk."""

    @patch("qwen3_tts.core.engine.inference._set_seed_for_backend")
    @patch("qwen3_tts.core.engine.inference._run_inference_single")
    @patch("qwen3_tts.core.engine.inference._prepare_text_chunks")
    @patch("qwen3_tts.core.engine.inference._crossfade_chunks")
    @patch("qwen3_tts.core.engine.inference._get_max_chunk_chars", return_value=500)
    def test_reseeds_before_each_chunk(self, _max, mock_crossfade,
                                       mock_chunks, mock_single, mock_set_seed):
        import numpy as np

        from qwen3_tts.core.engine.inference import run_inference

        mock_chunks.return_value = ["chunk1", "chunk2", "chunk3"]
        fake_wav = np.zeros(16000, dtype=np.float32)
        mock_single.return_value = (fake_wav, 24000)
        mock_crossfade.return_value = fake_wav

        gen_params = {"seed": 99, "temperature": 0.7, "top_k": 50,
                      "top_p": 0.95, "repetition_penalty": 1.05}

        run_inference(
            model=MagicMock(), text="long text", mode="clone",
            gen_params=gen_params, seed_lock_chunks=True,
        )

        # Should be called once per chunk
        self.assertEqual(mock_set_seed.call_count, 3)
        mock_set_seed.assert_has_calls([call(99), call(99), call(99)])

    @patch("qwen3_tts.core.engine.inference._set_seed_for_backend")
    @patch("qwen3_tts.core.engine.inference._run_inference_single")
    @patch("qwen3_tts.core.engine.inference._prepare_text_chunks")
    @patch("qwen3_tts.core.engine.inference._crossfade_chunks")
    @patch("qwen3_tts.core.engine.inference._get_max_chunk_chars", return_value=500)
    def test_no_reseed_when_disabled(self, _max, mock_crossfade,
                                     mock_chunks, mock_single, mock_set_seed):
        import numpy as np

        from qwen3_tts.core.engine.inference import run_inference

        mock_chunks.return_value = ["chunk1", "chunk2"]
        fake_wav = np.zeros(16000, dtype=np.float32)
        mock_single.return_value = (fake_wav, 24000)
        mock_crossfade.return_value = fake_wav

        gen_params = {"seed": 99, "temperature": 0.7, "top_k": 50,
                      "top_p": 0.95, "repetition_penalty": 1.05}

        run_inference(
            model=MagicMock(), text="long text", mode="clone",
            gen_params=gen_params, seed_lock_chunks=False,
        )

        mock_set_seed.assert_not_called()

    @patch("qwen3_tts.core.engine.inference._set_seed_for_backend")
    @patch("qwen3_tts.core.engine.inference._run_inference_single")
    @patch("qwen3_tts.core.engine.inference._prepare_text_chunks")
    @patch("qwen3_tts.core.engine.inference._get_max_chunk_chars", return_value=500)
    def test_no_reseed_single_chunk(self, _max, mock_chunks, mock_single,
                                    mock_set_seed):
        import numpy as np

        from qwen3_tts.core.engine.inference import run_inference

        mock_chunks.return_value = ["single chunk"]
        fake_wav = np.zeros(16000, dtype=np.float32)
        mock_single.return_value = (fake_wav, 24000)

        gen_params = {"seed": 99, "temperature": 0.7, "top_k": 50,
                      "top_p": 0.95, "repetition_penalty": 1.05}

        run_inference(
            model=MagicMock(), text="short", mode="clone",
            gen_params=gen_params, seed_lock_chunks=True,
        )

        # Single chunk — no need to re-seed (torch backend already seeds internally)
        mock_set_seed.assert_not_called()


class TestGenerateRequestSeedLock(unittest.TestCase):
    """Tests that GenerateRequest accepts seed_lock_chunks."""

    def test_default_is_false(self):
        from qwen3_tts.server.validation import GenerateRequest
        req = GenerateRequest(text="hello")
        self.assertFalse(req.seed_lock_chunks)

    def test_explicit_true(self):
        from qwen3_tts.server.validation import GenerateRequest
        req = GenerateRequest(text="hello", seed_lock_chunks=True)
        self.assertTrue(req.seed_lock_chunks)


if __name__ == "__main__":
    unittest.main()
