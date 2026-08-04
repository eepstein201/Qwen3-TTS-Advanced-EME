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


class TestMLXSeedApplication(unittest.TestCase):
    """Tests that MLX backend applies seed for single-chunk generations (H7).

    The default Apple-Silicon backend (MLX) previously ignored the seed for
    single-chunk generations while torch honored it. These tests verify the
    inline seed call in _run_inference_mlx mirrors torch's behavior.
    """

    def _make_mock_model(self):
        """Build a mock MLX model whose generate_custom_voice returns one result."""
        import numpy as np

        mock_model = MagicMock()
        fake_result = MagicMock()
        fake_result.audio = np.zeros(1000, dtype=np.float32)
        fake_result.sample_rate = 24000
        mock_model.generate_custom_voice.return_value = [fake_result]
        return mock_model

    @patch("qwen3_tts.core.engine.inference._set_seed_for_backend")
    @patch("qwen3_tts.core.engine.inference.load_config", return_value={})
    @patch("qwen3_tts.core.engine.inference.get_backend", return_value="mlx")
    @patch("qwen3_tts.core.engine.inference._prepare_text_chunks")
    def test_mlx_single_chunk_seeds(
        self, mock_chunks, _backend, _cfg, mock_set_seed
    ):
        """MLX single-chunk generation calls _set_seed_for_backend with the seed."""
        from qwen3_tts.core.engine.inference import run_inference

        mock_chunks.return_value = ["test text"]
        mock_model = self._make_mock_model()
        config_provider = MagicMock()
        config_provider.load.return_value = {}

        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": MagicMock()}):
            run_inference(
                model=mock_model,
                text="test text",
                mode="custom",
                gen_params={
                    "seed": 42,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95,
                    "max_new_tokens": 2048,
                },
                max_chunk_chars=500,
                config_provider=config_provider,
            )

        mock_set_seed.assert_called_once_with(42)

    @patch("qwen3_tts.core.engine.inference._set_seed_for_backend")
    @patch("qwen3_tts.core.engine.inference.load_config", return_value={})
    @patch("qwen3_tts.core.engine.inference.get_backend", return_value="mlx")
    @patch("qwen3_tts.core.engine.inference._prepare_text_chunks")
    def test_no_seed_no_call(self, mock_chunks, _backend, _cfg, mock_set_seed):
        """When seed is None, _set_seed_for_backend is not called."""
        from qwen3_tts.core.engine.inference import run_inference

        mock_chunks.return_value = ["test text"]
        mock_model = self._make_mock_model()
        config_provider = MagicMock()
        config_provider.load.return_value = {}

        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": MagicMock()}):
            run_inference(
                model=mock_model,
                text="test text",
                mode="custom",
                gen_params={
                    "seed": None,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95,
                    "max_new_tokens": 2048,
                },
                max_chunk_chars=500,
                config_provider=config_provider,
            )

        mock_set_seed.assert_not_called()

    def test_mlx_gen_params_still_drops_seed(self):
        """Guard: _get_mlx_gen_params must not pass seed to model.generate."""
        from qwen3_tts.core.engine.inference import _get_mlx_gen_params

        params = _get_mlx_gen_params({"seed": 42, "temperature": 0.7}, {})
        self.assertNotIn("seed", params)


class TestMLXStreamingSeedApplication(unittest.TestCase):
    """Tests that MLX streaming applies seed per text chunk (H7 streaming)."""

    def _make_mock_model(self):
        """Build a mock MLX model whose generate_custom_voice returns one result.

        Using a list (not an iterator) as return_value so it is re-iterable
        across multiple text-chunk calls in the multi-chunk test.
        """
        import numpy as np

        mock_model = MagicMock()
        fake_result = MagicMock()
        fake_result.audio = np.zeros(1000, dtype=np.float32)
        fake_result.sample_rate = 24000
        mock_model.generate_custom_voice.return_value = [fake_result]
        return mock_model

    @patch("qwen3_tts.core.engine.inference._set_seed_for_backend")
    @patch("qwen3_tts.core.engine.inference.get_backend", return_value="mlx")
    @patch("qwen3_tts.core.engine.inference._prepare_text_chunks")
    def test_mlx_streaming_seeds(
        self, mock_chunks, _backend, mock_set_seed
    ):
        """MLX streaming with a single text-chunk calls _set_seed_for_backend once."""
        from qwen3_tts.core.engine.inference import run_inference_streaming

        mock_chunks.return_value = ["test text"]
        mock_model = self._make_mock_model()
        config_provider = MagicMock()
        config_provider.load.return_value = {}

        list(
            run_inference_streaming(
                mock_model,
                "test text",
                "custom",
                {
                    "seed": 42,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95,
                    "max_new_tokens": 2048,
                },
                max_chunk_chars=500,
                config_provider=config_provider,
            )
        )

        mock_set_seed.assert_called_once_with(42)

    @patch("qwen3_tts.core.engine.inference._set_seed_for_backend")
    @patch("qwen3_tts.core.engine.inference.get_backend", return_value="mlx")
    @patch("qwen3_tts.core.engine.inference._prepare_text_chunks")
    def test_mlx_streaming_multi_chunk_seeds_each(
        self, mock_chunks, _backend, mock_set_seed
    ):
        """MLX streaming with multiple text-chunks seeds once per chunk."""
        from qwen3_tts.core.engine.inference import run_inference_streaming

        mock_chunks.return_value = ["chunk1", "chunk2", "chunk3"]
        mock_model = self._make_mock_model()
        config_provider = MagicMock()
        config_provider.load.return_value = {}

        list(
            run_inference_streaming(
                mock_model,
                "long text that splits into multiple chunks",
                "custom",
                {
                    "seed": 42,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95,
                    "max_new_tokens": 2048,
                },
                max_chunk_chars=500,
                config_provider=config_provider,
            )
        )

        self.assertEqual(mock_set_seed.call_count, 3)
        mock_set_seed.assert_has_calls([call(42), call(42), call(42)])


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
