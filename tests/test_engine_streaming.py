"""WS2: batch/streaming post-processing equivalence.

The batch path (`run_inference`) has always applied per-chunk post-processing —
ICL echo trim, clone rate control, audio validation — while the streaming paths
(`run_inference_streaming`, consumed by `/generate-stream` and `/ws`) applied
none of it. Identical requests therefore produced different audio depending on
whether the caller streamed, which is the H2 finding behind WS2.

These tests pin the intended contract:

* per-chunk-feasible steps run in BOTH paths (`_postprocess_chunk`), and
* LUFS remains batch-only **by design** — EBU R128 integrated loudness applies a
  relative gate computed over every block of the whole signal, so it is not
  computable incrementally per chunk. That divergence is architectural, not a
  bug, and the test below exists to keep someone from "fixing" it.

The wiring tests are AST-based: they assert the server call sites actually
forward `max_chunk_chars` / `config_provider`, which end-to-end mocks would not
catch if a keyword were silently dropped.
"""

import ast
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from qwen3_tts.core.engine import inference

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Distinct marker so a test can prove post-processing actually touched the audio
# rather than merely being called (a mock assert_called would pass hollowly).
_PROCESSED = np.array([0.5, 0.25, -0.5, -0.25], dtype=np.float32)
_RAW = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
_SR = 24000


class _FakeConfigProvider:
    def __init__(self, cfg):
        self._cfg = cfg

    def load(self):
        return self._cfg


def _clone_cfg(**generation):
    return _FakeConfigProvider({"generation": generation})


class TestPostprocessChunk(unittest.TestCase):
    """Task 2.2 — the extracted helper, tested directly."""

    def test_applies_clone_speed_and_returns_new_array(self):
        with patch.object(inference, "process_audio", return_value=_PROCESSED):
            out, sr = inference._postprocess_chunk(
                _RAW,
                _SR,
                gen_params={"speed": 1.5},
                mode="clone",
                config={"generation": {}},
            )
        np.testing.assert_array_equal(out, _PROCESSED)
        self.assertEqual(sr, _SR)

    def test_leaves_non_clone_modes_alone(self):
        """design/custom keep the model's native instruct rate control."""
        with patch.object(inference, "process_audio", return_value=_PROCESSED):
            out, _ = inference._postprocess_chunk(
                _RAW,
                _SR,
                gen_params={"speed": 1.5},
                mode="design",
                config={"generation": {}},
            )
        np.testing.assert_array_equal(out, _RAW)

    def test_validates_audio(self):
        """NaN must be corrected, not propagated to the caller."""
        dirty = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        out, _ = inference._postprocess_chunk(
            dirty, _SR, gen_params={}, mode="custom", config={"generation": {}}
        )
        self.assertFalse(np.isnan(out).any(), "NaN survived _postprocess_chunk")


class TestBatchStreamingEquivalence(unittest.TestCase):
    """Tasks 2.1 + 2.3 — the same per-chunk steps in both paths."""

    def _batch(self):
        with (
            patch.object(inference, "_run_inference_single", return_value=(_RAW, _SR)),
            patch.object(inference, "process_audio", return_value=_PROCESSED),
        ):
            return inference.run_inference(
                model=MagicMock(),
                text="hello",
                mode="clone",
                gen_params={"speed": 1.5},
                config_provider=_clone_cfg(),
            )

    def _stream(self):
        with (
            patch.object(inference, "get_backend", return_value="mlx"),
            patch.object(
                inference,
                "_run_inference_mlx_streaming",
                return_value=iter([(_RAW, _SR)]),
            ),
            patch.object(inference, "process_audio", return_value=_PROCESSED),
        ):
            return list(
                inference.run_inference_streaming(
                    model=MagicMock(),
                    text="hello",
                    mode="clone",
                    gen_params={"speed": 1.5},
                    config_provider=_clone_cfg(),
                )
            )

    def test_batch_applies_clone_speed(self):
        wav, _ = self._batch()
        np.testing.assert_array_equal(wav, _PROCESSED)

    def test_streaming_applies_clone_speed_per_chunk(self):
        chunks = self._stream()
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(chunks[0][0], _PROCESSED)

    def test_streaming_output_matches_batch_for_per_chunk_steps(self):
        batch_wav, batch_sr = self._batch()
        chunks = self._stream()
        np.testing.assert_array_equal(chunks[0][0], batch_wav)
        self.assertEqual(chunks[0][1], batch_sr)

    def test_lufs_stays_batch_only(self):
        """Intentional divergence — see module docstring. Do not "fix" this."""
        with patch.object(inference, "_maybe_apply_lufs") as lufs:
            with (
                patch.object(inference, "get_backend", return_value="mlx"),
                patch.object(
                    inference,
                    "_run_inference_mlx_streaming",
                    return_value=iter([(_RAW, _SR)]),
                ),
                patch.object(inference, "process_audio", return_value=_PROCESSED),
            ):
                list(
                    inference.run_inference_streaming(
                        model=MagicMock(),
                        text="hello",
                        mode="clone",
                        gen_params={},
                        config_provider=_clone_cfg(lufs_normalize=True),
                    )
                )
        lufs.assert_not_called()


def _streaming_call_keywords(relative_path):
    """Keywords passed to run_inference_streaming(...) at a server call site."""
    tree = ast.parse((_REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_inference_streaming"
        ):
            return {kw.arg for kw in node.keywords}
    raise AssertionError(f"No run_inference_streaming(...) call in {relative_path}")


class TestStreamingCallSitesForwardConfig(unittest.TestCase):
    """Task 2.3 — a dropped keyword silently falls back to config defaults.

    Both call sites parse max_chunk_chars off the request and then never passed
    it on, so a streaming client's chunk size was ignored with no error.
    """

    def test_http_stream_forwards_max_chunk_chars_and_config_provider(self):
        kwargs = _streaming_call_keywords("qwen3_tts/server/app_generation.py")
        self.assertIn("max_chunk_chars", kwargs)
        self.assertIn("config_provider", kwargs)

    def test_websocket_forwards_max_chunk_chars_and_config_provider(self):
        kwargs = _streaming_call_keywords("qwen3_tts/server/websocket.py")
        self.assertIn("max_chunk_chars", kwargs)
        self.assertIn("config_provider", kwargs)


# Deltas, not totals: mlx-audio's streaming branch returns before it can yield
# the cumulative `token_count = len(generated_codes)`, so every streamed
# GenerationResult carries `token_count=new_tokens` — bounded by
# `streaming_chunk_size = int(streaming_interval * 12.5)` = 25 at the default
# 2.0 s interval.
_STREAM_CHUNK_TOKENS = 25


def _fake_chunk(tokens, segment_idx=0):
    """One streamed GenerationResult carrying a per-chunk token DELTA."""
    return SimpleNamespace(
        audio=np.array([0.1, -0.1], dtype=np.float32),
        sample_rate=_SR,
        token_count=tokens,
        segment_idx=segment_idx,
    )


def _chunks_totalling(tokens, segment_idx=0):
    """Split a per-segment token total into the deltas mlx-audio would emit."""
    full, remainder = divmod(tokens, _STREAM_CHUNK_TOKENS)
    chunks = [_fake_chunk(_STREAM_CHUNK_TOKENS, segment_idx) for _ in range(full)]
    if remainder:
        chunks.append(_fake_chunk(remainder, segment_idx))
    return chunks


class _CountingGenerator:
    """Yields the given chunks and records how many were actually consumed.

    Guard-the-guard: a cap test that silently drove an EMPTY generator would
    report "no warning" and pass for the wrong reason.
    """

    def __init__(self, chunks):
        self._chunks = chunks
        self.consumed = 0

    def __call__(self, *args, **kwargs):
        for chunk in self._chunks:
            self.consumed += 1
            yield chunk


class TestStreamingCapWarning(unittest.TestCase):
    """#192 — the streaming token-cap warning must be able to fire at all.

    `_warn_if_cap_reached` was handed the LAST streamed result, whose
    `token_count` is a per-chunk delta capped at 25. Comparing that against
    `max_tokens` (2048) is never true, so a runaway that never emits EOS was
    silent on `/generate-stream` and `/ws` — exactly the failure #192 reports,
    on the two paths where truncation is least visible.

    The cap is applied PER SEGMENT upstream (`generate()` resets
    `generated_codes` and `decoded_tokens` inside
    `for segment_idx, segment_text in enumerate(segments)`), so the total must
    be tracked per segment rather than summed across the whole stream.
    """

    MAX_TOKENS = 2048

    def _drive(self, chunks):
        """Run the streaming path over `chunks`; return (records, generator)."""
        gen = _CountingGenerator(chunks)
        model = SimpleNamespace(generate_custom_voice=gen)

        with patch.object(inference, "_mlx_lang_code", return_value="en"):
            with self.assertLogs(inference.logger, level="WARNING") as captured:
                # A no-op warning keeps assertLogs from failing when the
                # code under test correctly stays silent; the assertions
                # below filter for the cap message specifically.
                inference.logger.warning("sentinel")
                emitted = list(
                    inference._run_inference_mlx_streaming(
                        model,
                        "Token test three",
                        "custom",
                        {"max_new_tokens": self.MAX_TOKENS},
                        config={"generation": {}},
                    )
                )

        cap_warnings = [r for r in captured.output if "token cap" in r]
        return cap_warnings, gen, emitted

    def test_warns_when_a_segment_runs_to_the_cap(self):
        """82 deltas of 25 tokens reach 2050 >= 2048 — this must warn."""
        chunks = _chunks_totalling(self.MAX_TOKENS + 2)

        cap_warnings, gen, emitted = self._drive(chunks)

        # Guard-the-guard: prove the stream was actually driven.
        self.assertEqual(gen.consumed, len(chunks))
        self.assertEqual(len(emitted), len(chunks))
        self.assertTrue(
            all(c.token_count <= _STREAM_CHUNK_TOKENS for c in chunks),
            "fixture must carry deltas, not a cumulative count",
        )
        self.assertEqual(
            len(cap_warnings),
            1,
            "a stream totalling >= max_tokens in one segment must warn",
        )
        self.assertIn("mode=custom", cap_warnings[0])

    def test_does_not_warn_below_the_cap(self):
        chunks = _chunks_totalling(self.MAX_TOKENS - 100)

        cap_warnings, gen, _ = self._drive(chunks)

        self.assertEqual(gen.consumed, len(chunks))
        self.assertEqual(cap_warnings, [])

    def test_does_not_sum_across_segments(self):
        """Two 1200-token segments total 2400 but neither hit the 2048 cap.

        Naively summing every delta in the stream would report a runaway that
        did not happen — the cap resets per segment upstream.
        """
        chunks = _chunks_totalling(1200, segment_idx=0) + _chunks_totalling(
            1200, segment_idx=1
        )

        cap_warnings, gen, _ = self._drive(chunks)

        self.assertEqual(gen.consumed, len(chunks))
        self.assertEqual(
            cap_warnings,
            [],
            "per-segment totals stayed under the cap; summing them is a false positive",
        )


class _Prompt:
    """Minimal stand-in for a voice prompt carrying a reference transcript."""

    def __init__(self, ref_text):
        self.ref_text = ref_text


class TestTorchStreamingFallback(unittest.TestCase):
    """The torch branch of run_inference_streaming (coverage item 6).

    MLX streams natively; torch has no streaming API, so it falls back to
    chunking the text and yielding per-chunk audio. That whole loop — the
    per-chunk seed, the progress callback, the first-chunk-only echo trim, and
    the _postprocess_chunk call that keeps streaming equal to batch — had no
    test coverage at all.

    `_prepare_text_chunks` is stubbed so the chunk boundaries are explicit:
    chunking itself is covered elsewhere, and what matters here is what the
    loop does with each chunk. No torch import happens on this path — the only
    heavy call, `_run_inference_single`, is patched.
    """

    CHUNKS = ["first chunk.", "second chunk.", "third chunk."]

    def _stream(self, chunks=None, postprocess=None, **kwargs):
        """Drive the torch branch, returning (yields, single_calls)."""
        chunks = self.CHUNKS if chunks is None else chunks
        single_calls = []

        def _fake_single(model, chunk, mode, chunk_params, *args, **kw):
            single_calls.append({"chunk": chunk, "params": chunk_params})
            return _RAW, _SR

        stack = [
            patch.object(inference, "get_backend", return_value="torch"),
            patch.object(inference, "_prepare_text_chunks", return_value=list(chunks)),
            patch.object(inference, "_run_inference_single", side_effect=_fake_single),
            patch.object(inference, "process_audio", return_value=_PROCESSED),
        ]
        if postprocess is not None:
            stack.append(patch.object(inference, "_postprocess_chunk", postprocess))

        with ExitStack() as es:
            for cm in stack:
                es.enter_context(cm)
            params = {"speed": 1.5, **kwargs.pop("gen_params", {})}
            yields = list(
                inference.run_inference_streaming(
                    model=MagicMock(),
                    text=" ".join(chunks),
                    mode="clone",
                    gen_params=params,
                    config_provider=_clone_cfg(),
                    **kwargs,
                )
            )
        return yields, single_calls

    def test_yields_one_audio_chunk_per_text_chunk(self):
        yields, calls = self._stream()
        self.assertEqual(len(yields), 3)
        self.assertEqual([c["chunk"] for c in calls], self.CHUNKS)

    def test_every_chunk_is_postprocessed(self):
        yields, _ = self._stream()
        for wav, sr in yields:
            np.testing.assert_array_equal(wav, _PROCESSED)
            self.assertEqual(sr, _SR)

    def test_progress_callback_reports_a_known_total_upfront(self):
        """Unlike MLX, torch knows the chunk count before generating."""
        seen = []
        self._stream(progress_callback=lambda i, total: seen.append((i, total)))
        self.assertEqual(seen, [(1, 3), (2, 3), (3, 3)])

    def test_reference_text_reaches_only_the_first_chunk(self):
        """The ICL echo sits at the head of the generation, nowhere else."""
        seen = []

        def _spy(audio, sr, *args, **kw):
            seen.append(kw.get("reference_text"))
            return audio, sr

        self._stream(postprocess=_spy, voice_prompt=_Prompt("my reference line"))
        self.assertEqual(seen, ["my reference line", None, None])

    def test_seed_derives_per_chunk_by_default(self):
        _, calls = self._stream(gen_params={"seed": 100})
        self.assertEqual([c["params"]["seed"] for c in calls], [100, 101, 102])

    def test_seed_lock_pins_every_chunk_to_the_base_seed(self):
        _, calls = self._stream(gen_params={"seed": 100}, seed_lock_chunks=True)
        self.assertEqual([c["params"]["seed"] for c in calls], [100, 100, 100])

    def test_no_base_seed_means_no_seeding(self):
        _, calls = self._stream()
        self.assertEqual([c["params"]["seed"] for c in calls], [None, None, None])

    def test_falls_back_to_configured_max_chunk_chars(self):
        """max_chunk_chars=None must consult config, not stay None."""
        with patch.object(
            inference, "_get_max_chunk_chars", return_value=321
        ) as get_cap:
            with (
                patch.object(inference, "get_backend", return_value="torch"),
                patch.object(
                    inference, "_prepare_text_chunks", return_value=["only"]
                ) as prep,
                patch.object(
                    inference, "_run_inference_single", return_value=(_RAW, _SR)
                ),
                patch.object(inference, "process_audio", return_value=_PROCESSED),
            ):
                list(
                    inference.run_inference_streaming(
                        model=MagicMock(),
                        text="only",
                        mode="clone",
                        gen_params={},
                        config_provider=_clone_cfg(),
                    )
                )
        get_cap.assert_called_once()
        self.assertEqual(prep.call_args[0][3], 321)

    def test_lufs_stays_batch_only_on_torch_too(self):
        with patch.object(inference, "_maybe_apply_lufs") as lufs:
            self._stream(chunks=["one"])
        lufs.assert_not_called()


class TestTorchStreamingMatchesBatch(unittest.TestCase):
    """The parity invariant, pinned end-to-end (coverage item 6 exit criterion).

    CLAUDE.md states as a design guarantee that `_postprocess_chunk` is called
    by BOTH run_inference and run_inference_streaming on BOTH backends, so an
    identical request produces identical audio however it is consumed. The
    per-chunk steps run for real here — only `process_audio` (the rubberband/
    librosa stretch) is stubbed — so dropping the _postprocess_chunk call from
    either path makes this test fail rather than silently diverging the audio.
    """

    TEXT = "one chunk of text."

    def _batch(self):
        with (
            patch.object(inference, "get_backend", return_value="torch"),
            patch.object(inference, "_prepare_text_chunks", return_value=[self.TEXT]),
            patch.object(inference, "_run_inference_single", return_value=(_RAW, _SR)),
            patch.object(inference, "process_audio", return_value=_PROCESSED),
        ):
            return inference.run_inference(
                model=MagicMock(),
                text=self.TEXT,
                mode="clone",
                gen_params={"speed": 1.5},
                config_provider=_clone_cfg(),
            )

    def _stream(self):
        with (
            patch.object(inference, "get_backend", return_value="torch"),
            patch.object(inference, "_prepare_text_chunks", return_value=[self.TEXT]),
            patch.object(inference, "_run_inference_single", return_value=(_RAW, _SR)),
            patch.object(inference, "process_audio", return_value=_PROCESSED),
        ):
            return list(
                inference.run_inference_streaming(
                    model=MagicMock(),
                    text=self.TEXT,
                    mode="clone",
                    gen_params={"speed": 1.5},
                    config_provider=_clone_cfg(),
                )
            )

    def test_torch_streaming_output_matches_torch_batch(self):
        batch_wav, batch_sr = self._batch()
        chunks = self._stream()
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(
            chunks[0][0],
            batch_wav,
            "torch streaming and batch drifted — _postprocess_chunk must run on both",
        )
        self.assertEqual(chunks[0][1], batch_sr)

    def test_both_paths_actually_transformed_the_audio(self):
        """Guards the assertion above from passing on two untouched arrays."""
        batch_wav, _ = self._batch()
        np.testing.assert_array_equal(batch_wav, _PROCESSED)
        self.assertFalse(
            np.array_equal(batch_wav, _RAW),
            "post-processing was a no-op, so equality proves nothing",
        )


if __name__ == "__main__":
    unittest.main()
