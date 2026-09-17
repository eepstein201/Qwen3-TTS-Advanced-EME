#!/usr/bin/env python3
"""Tests for seed-lock-chunks feature (voice consistency across chunks).

Covers:
  - _set_seed_for_backend: sets seed via torch or mlx depending on active backend
  - _chunk_seed: the effective-seed table per text chunk (Step 3A)
  - run_inference: seed_lock_chunks=True re-seeds the base seed before each
    chunk; seed_lock_chunks=False re-seeds the DERIVED seed (base + chunk
    index); base_seed=None means no seeding at all, under both flag values
  - run_inference_streaming: the same table on the MLX branch and the
    torch-fallback branch (Step 3A added the seed_lock_chunks parameter)
  - Server call sites: /generate-stream and /ws forward req.seed_lock_chunks
  - GenerateRequest: accepts seed_lock_chunks field
  - UI: checkbox wiring

Run: pytest tests/test_seed_lock_chunks.py -v
"""
import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

try:
    import mlx.core  # noqa: F401
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _gen_params_of(invoke):
    """gen_params dict from a recorded _run_inference_single call.

    The engine passes gen_params positionally (4th argument); reading the
    keyword form first keeps the assertion valid if that convention changes.
    """
    if "gen_params" in invoke.kwargs:
        return invoke.kwargs["gen_params"]
    return invoke.args[3]


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


class TestChunkSeedHelper(unittest.TestCase):
    """Step 3A — the effective-seed table for text-chunk index i.

    | base_seed | seed_lock_chunks | effective seed for chunk i |
    |-----------|------------------|----------------------------|
    | None      | either           | None — no seeding at all   |
    | set       | True             | base_seed                  |
    | set       | False            | base_seed + i              |
    """

    def test_none_base_seed_returns_none_with_lock(self):
        from qwen3_tts.core.engine.inference import _chunk_seed

        self.assertIsNone(_chunk_seed(None, chunk_index=2, seed_lock_chunks=True))

    def test_none_base_seed_returns_none_without_lock(self):
        from qwen3_tts.core.engine.inference import _chunk_seed

        self.assertIsNone(_chunk_seed(None, chunk_index=2, seed_lock_chunks=False))

    def test_lock_true_returns_base_seed_unchanged(self):
        from qwen3_tts.core.engine.inference import _chunk_seed

        self.assertEqual(_chunk_seed(99, chunk_index=2, seed_lock_chunks=True), 99)

    def test_lock_false_returns_base_plus_chunk_index(self):
        from qwen3_tts.core.engine.inference import _chunk_seed

        self.assertEqual(_chunk_seed(99, chunk_index=2, seed_lock_chunks=False), 101)

    def test_chunk_zero_returns_base_for_both_flag_values(self):
        """Single-chunk generations are i=0, so both flag values give base."""
        from qwen3_tts.core.engine.inference import _chunk_seed

        self.assertEqual(_chunk_seed(99, chunk_index=0, seed_lock_chunks=True), 99)
        self.assertEqual(_chunk_seed(99, chunk_index=0, seed_lock_chunks=False), 99)


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
    def test_derived_seeds_when_disabled(self, _max, mock_crossfade,
                                         mock_chunks, mock_single, mock_set_seed):
        """seed_lock_chunks=False: per-chunk DERIVED seed base + index.

        Rewritten for Step 3A — the pre-3A behaviour this test asserted
        (never re-seed when the flag is False, i.e. the flag was a no-op)
        was the defect: the spec table requires base_seed + chunk_index.
        """
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

        # Called once per chunk with the derived seed 99 + i
        self.assertEqual(mock_set_seed.call_count, 2)
        mock_set_seed.assert_has_calls([call(99), call(100)])

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


class TestBatchPerChunkSeedDelivery(unittest.TestCase):
    """Step 3A — the batch path must DELIVER the effective seed per chunk.

    The loop-level _set_seed_for_backend call alone is not enough: the inner
    runners seed from gen_params["seed"] themselves, so the per-chunk copy
    handed to _run_inference_single must carry the same effective seed or the
    inner call re-seeds the base value and undoes the loop-level seeding.
    """

    @patch("qwen3_tts.core.engine.inference._set_seed_for_backend")
    @patch("qwen3_tts.core.engine.inference._run_inference_single")
    @patch("qwen3_tts.core.engine.inference._prepare_text_chunks")
    @patch("qwen3_tts.core.engine.inference._crossfade_chunks")
    @patch("qwen3_tts.core.engine.inference._get_max_chunk_chars", return_value=500)
    def test_false_flag_seeds_and_delivers_per_chunk_seeds(
        self, _max, mock_crossfade, mock_chunks, mock_single, mock_set_seed
    ):
        import numpy as np

        from qwen3_tts.core.engine.inference import run_inference

        mock_chunks.return_value = ["chunk1", "chunk2", "chunk3"]
        fake_wav = np.zeros(16000, dtype=np.float32)
        mock_single.return_value = (fake_wav, 24000)
        mock_crossfade.return_value = fake_wav

        gen_params = {
            "seed": 99,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
        }

        run_inference(
            model=MagicMock(),
            text="long text",
            mode="clone",
            gen_params=gen_params,
            seed_lock_chunks=False,
        )

        self.assertEqual(mock_set_seed.call_count, 3)
        mock_set_seed.assert_has_calls([call(99), call(100), call(101)])

        delivered = [_gen_params_of(c) for c in mock_single.call_args_list]
        self.assertEqual([p["seed"] for p in delivered], [99, 100, 101])

        # Identity pin: the per-chunk seeds must travel as NEW dicts. Handing
        # the caller's object through makes value equality invisible once the
        # flag is True (every chunk would carry the same seed).
        for params in delivered:
            self.assertIsNot(params, gen_params)

    @patch("qwen3_tts.core.engine.inference._set_seed_for_backend")
    @patch("qwen3_tts.core.engine.inference._run_inference_single")
    @patch("qwen3_tts.core.engine.inference._prepare_text_chunks")
    @patch("qwen3_tts.core.engine.inference._crossfade_chunks")
    @patch("qwen3_tts.core.engine.inference._get_max_chunk_chars", return_value=500)
    def test_caller_gen_params_not_mutated(
        self, _max, mock_crossfade, mock_chunks, mock_single, mock_set_seed
    ):
        """Immutability pin: per-chunk seeds travel as NEW dicts.

        The caller's gen_params must keep its original contents — dict
        equality also fails if the loop adds or replaces keys in place.
        """
        import numpy as np

        from qwen3_tts.core.engine.inference import run_inference

        mock_chunks.return_value = ["chunk1", "chunk2"]
        fake_wav = np.zeros(16000, dtype=np.float32)
        mock_single.return_value = (fake_wav, 24000)
        mock_crossfade.return_value = fake_wav

        original = {
            "seed": 99,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
        }
        gen_params = dict(original)

        run_inference(
            model=MagicMock(),
            text="long text",
            mode="clone",
            gen_params=gen_params,
            seed_lock_chunks=False,
        )

        self.assertEqual(gen_params, original)

    @patch("qwen3_tts.core.engine.inference._set_seed_for_backend")
    @patch("qwen3_tts.core.engine.inference._run_inference_single")
    @patch("qwen3_tts.core.engine.inference._prepare_text_chunks")
    @patch("qwen3_tts.core.engine.inference._crossfade_chunks")
    @patch("qwen3_tts.core.engine.inference._get_max_chunk_chars", return_value=500)
    def test_none_base_seed_no_seeding_flag_true(
        self, _max, mock_crossfade, mock_chunks, mock_single, mock_set_seed
    ):
        import numpy as np

        from qwen3_tts.core.engine.inference import run_inference

        mock_chunks.return_value = ["chunk1", "chunk2"]
        fake_wav = np.zeros(16000, dtype=np.float32)
        mock_single.return_value = (fake_wav, 24000)
        mock_crossfade.return_value = fake_wav

        run_inference(
            model=MagicMock(),
            text="long text",
            mode="clone",
            gen_params={
                "seed": None,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
            },
            seed_lock_chunks=True,
        )

        mock_set_seed.assert_not_called()

        # Delivery pin: the None guard must reach the gen_params copies too —
        # an impl whose guard lives only at the seeding site could deliver
        # {"seed": 0} (None + chunk index).
        delivered = [_gen_params_of(c)["seed"] for c in mock_single.call_args_list]
        self.assertEqual(delivered, [None, None])

    @patch("qwen3_tts.core.engine.inference._set_seed_for_backend")
    @patch("qwen3_tts.core.engine.inference._run_inference_single")
    @patch("qwen3_tts.core.engine.inference._prepare_text_chunks")
    @patch("qwen3_tts.core.engine.inference._crossfade_chunks")
    @patch("qwen3_tts.core.engine.inference._get_max_chunk_chars", return_value=500)
    def test_none_base_seed_no_seeding_flag_false(
        self, _max, mock_crossfade, mock_chunks, mock_single, mock_set_seed
    ):
        import numpy as np

        from qwen3_tts.core.engine.inference import run_inference

        mock_chunks.return_value = ["chunk1", "chunk2"]
        fake_wav = np.zeros(16000, dtype=np.float32)
        mock_single.return_value = (fake_wav, 24000)
        mock_crossfade.return_value = fake_wav

        run_inference(
            model=MagicMock(),
            text="long text",
            mode="clone",
            gen_params={
                "seed": None,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
            },
            seed_lock_chunks=False,
        )

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
        """MLX streaming with multiple text-chunks seeds once per chunk.

        Step 3A: streaming defaults to seed_lock_chunks=False, so each chunk
        receives the DERIVED seed base + index (previously every chunk got the
        identical base seed, making the flag meaningless).
        """
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
        mock_set_seed.assert_has_calls([call(42), call(43), call(44)])

    @patch("qwen3_tts.core.engine.inference._set_seed_for_backend")
    @patch("qwen3_tts.core.engine.inference.get_backend", return_value="mlx")
    @patch("qwen3_tts.core.engine.inference._prepare_text_chunks")
    def test_mlx_streaming_seed_lock_true_pins_base_seed(
        self, mock_chunks, _backend, mock_set_seed
    ):
        """seed_lock_chunks=True: every text chunk gets the identical base seed."""
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
                seed_lock_chunks=True,
            )
        )

        self.assertEqual(mock_set_seed.call_count, 3)
        mock_set_seed.assert_has_calls([call(42), call(42), call(42)])


class TestStreamingTorchFallbackPerChunkSeeds(unittest.TestCase):
    """Step 3A — the torch-fallback streaming branch delivers per-chunk seeds.

    The fallback chunk loop calls _run_inference_single directly, so the
    effective seed must arrive via a per-chunk gen_params copy there too.
    """

    def _drive(self, seed_lock_chunks=None):
        """Run the torch-fallback stream over 3 chunks; return mock_single."""
        import numpy as np

        from qwen3_tts.core.engine import inference

        fake_wav = np.zeros(16000, dtype=np.float32)
        config_provider = MagicMock()
        config_provider.load.return_value = {}

        kwargs = {}
        if seed_lock_chunks is not None:
            kwargs["seed_lock_chunks"] = seed_lock_chunks

        with (
            patch.object(inference, "get_backend", return_value="torch"),
            patch.object(inference, "_prepare_text_chunks",
                         return_value=["chunk1", "chunk2", "chunk3"]),
            patch.object(inference, "_run_inference_single",
                         return_value=(fake_wav, 24000)) as mock_single,
        ):
            list(
                inference.run_inference_streaming(
                    model=MagicMock(),
                    text="long text",
                    mode="custom",
                    gen_params={
                        "seed": 42,
                        "temperature": 0.7,
                        "top_k": 50,
                        "top_p": 0.95,
                        "repetition_penalty": 1.05,
                    },
                    max_chunk_chars=500,
                    config_provider=config_provider,
                    **kwargs,
                )
            )
        return mock_single

    def test_default_flag_delivers_derived_seeds(self):
        """Default seed_lock_chunks=False: derived seeds 42, 43, 44 per chunk."""
        mock_single = self._drive()

        delivered = [_gen_params_of(c)["seed"] for c in mock_single.call_args_list]
        self.assertEqual(delivered, [42, 43, 44])

    def test_lock_true_delivers_base_seed_every_chunk(self):
        """seed_lock_chunks=True: identical base seed 42 for every chunk."""
        mock_single = self._drive(seed_lock_chunks=True)

        delivered = [_gen_params_of(c)["seed"] for c in mock_single.call_args_list]
        self.assertEqual(delivered, [42, 42, 42])


def _make_streaming_state():
    """Create a minimal app.state with the attributes the streaming paths need.

    Mirrors the module-local helper in tests/test_streaming_thread_lifecycle.py
    (conftest autouse fixtures do NOT fire under the batch runner).
    """
    state = SimpleNamespace()
    state.auth_token = "test_token"  # nosec B105
    state.models = {"clone": None, "design": None, "custom": MagicMock()}
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    state.generation_state = {
        "active": False,
        "start_time": 0.0,
        "text_length": 0,
        "mode": "",
        "batch_index": 0,
        "batch_total": 0,
        "chunk_index": 0,
        "chunk_total": 0,
        "generation_id": None,
        "cancelled": False,
    }
    state.request_queue = set()
    state.request_queue_lock = threading.Lock()
    state.pending_requests = []
    state.pending_lock = asyncio.Lock()
    state.last_activity = 0
    state.models_loaded = threading.Event()
    state.models_loaded.set()
    state.gen_cache = {}
    state.gen_cache_lock = threading.Lock()
    state.inference_lock = asyncio.Lock()
    state.generation_lock = asyncio.Lock()
    state.eta_cache = {"median_rate": None, "last_updated": 0}
    state.eta_cache_lock = threading.Lock()
    state.shutdown_timer = None
    state.server_config = {
        "security": {"max_text_length": 50000, "max_batch_size": 20},
        "auto_shutdown_minutes": 0,
    }
    return state


class TestStreamingCallSitesForwardFlag(unittest.IsolatedAsyncioTestCase):
    """Step 3A — both server streaming call sites forward req.seed_lock_chunks.

    Harness follows tests/test_streaming_thread_lifecycle.py: mock the engine's
    run_inference_streaming, drive the real handler, assert on the recorded
    call kwargs.
    """

    async def _forwarded_via_http(self, seed_lock_chunks):
        """Drive handle_generate_stream; return the forwarded flag value."""
        import numpy as np

        from qwen3_tts.server.app_generation import handle_generate_stream
        from qwen3_tts.server.validation import GenerateRequest

        state = _make_streaming_state()

        def one_chunk_inference(**kwargs):
            chunk = np.zeros(100, dtype=np.float32)
            yield (chunk, 24000)

        req = GenerateRequest(
            text="Hello world",
            mode="custom",
            seed=1234,
            seed_lock_chunks=seed_lock_chunks,
        )

        with (
            patch(
                "qwen3_tts.core.engine.run_inference_streaming",
                side_effect=one_chunk_inference,
            ) as mock_run,
            patch("qwen3_tts.server.validation._validate_generation_request"),
            patch(
                "qwen3_tts.server.app_generation._check_memory_available",
                return_value=(True, 4096),
            ),
        ):
            response = await handle_generate_stream(
                request=MagicMock(),
                state=state,
                req=req,
                security={"max_text_length": 50000},
                config_provider=None,
            )
            async for _ in response.body_iterator:
                pass

        return mock_run.call_args.kwargs.get("seed_lock_chunks")

    async def _forwarded_via_ws(self, seed_lock_chunks):
        """Drive websocket._stream_generation; return the forwarded flag value."""
        import numpy as np

        from qwen3_tts.server.websocket import _stream_generation

        state = _make_streaming_state()

        def one_chunk_inference(**kwargs):
            chunk = np.zeros(100, dtype=np.float32)
            yield (chunk, 24000)

        ws = MagicMock()
        ws.send_bytes = AsyncMock()
        ws.send_json = AsyncMock()

        with (
            patch(
                "qwen3_tts.core.engine.run_inference_streaming",
                side_effect=one_chunk_inference,
            ) as mock_run,
            patch("qwen3_tts.server.validation._validate_generation_request"),
            patch(
                "qwen3_tts.server.app_lifespan._check_memory_available",
                return_value=(True, 4096),
            ),
        ):
            await _stream_generation(
                websocket=ws,
                app_state=state,
                text="Hello world",
                mode="custom",
                data={
                    "text": "Hello world",
                    "mode": "custom",
                    "seed": 1234,
                    "seed_lock_chunks": seed_lock_chunks,
                },
                stop_event=threading.Event(),
                disconnect_event=threading.Event(),
            )

        return mock_run.call_args.kwargs.get("seed_lock_chunks")

    async def test_http_forwards_request_value_true(self):
        forwarded = await self._forwarded_via_http(True)
        self.assertIs(
            forwarded, True, "handle_generate_stream must forward True verbatim"
        )

    async def test_http_forwards_request_value_false(self):
        forwarded = await self._forwarded_via_http(False)
        self.assertIs(
            forwarded, False, "handle_generate_stream must forward False verbatim"
        )

    async def test_ws_forwards_request_value_true(self):
        forwarded = await self._forwarded_via_ws(True)
        self.assertIs(
            forwarded, True, "websocket must forward True verbatim"
        )

    async def test_ws_forwards_request_value_false(self):
        forwarded = await self._forwarded_via_ws(False)
        self.assertIs(
            forwarded, False, "websocket must forward False verbatim"
        )


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
