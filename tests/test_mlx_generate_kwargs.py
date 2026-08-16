"""Tests that MLX generation kwargs actually reach mlx-audio.

Regression guard for the PRF-9 follow-ups (see
``docs/reviews/prf9-max-new-tokens-measurement-2026-08-15.md``): our
``max_new_tokens`` and ``language`` kwargs were silently swallowed by
mlx-audio's ``**kwargs`` — its parameters are named ``max_tokens`` and
``lang_code`` — which made the generation cap and the language selection
dead knobs on the MLX backend.

Why the fake model below is written the way it is
-------------------------------------------------
A bare ``MagicMock()`` accepts *any* keyword, so a test built on one would
pass just as happily with the old, broken key names — a hollow green.
``FakeMLXModel`` therefore mirrors the real mlx-audio 0.4.8 signatures:

* ``generate()`` has ``lang_code`` / ``max_tokens`` **and** ``**kwargs``
  (matching upstream), so wrong names are caught by asserting that nothing
  landed in ``**kwargs``.
* ``generate_custom_voice()`` / ``generate_voice_design()`` have **no**
  ``**kwargs`` (also matching upstream), so a wrong name raises
  ``TypeError`` outright.

``test_fake_model_rejects_legacy_kwarg_names`` asserts that strictness
directly, so the harness itself cannot silently soften over time.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from qwen3_tts.core.engine import inference

MOD = "qwen3_tts.core.engine.inference"
LOGGER = "tts.engine"

# mlx-audio's own defaults, mirrored so the fakes drift visibly if upstream moves.
MLX_DEFAULT_MAX_TOKENS = 4096
MLX_DEFAULT_LANG = "auto"


def _fake_result(n_samples=1000, sample_rate=24000):
    """One mlx-audio GenerationResult-alike carrying non-silent audio."""
    result = MagicMock()
    # Non-silent so _validate_audio() doesn't emit an unrelated warning that
    # could pollute assertLogs() in the clamp tests.
    result.audio = np.full(n_samples, 0.1, dtype=np.float32)
    result.sample_rate = sample_rate
    return result


class FakeMLXModel:
    """Signature-faithful stand-in for the mlx-audio 0.4.8 Qwen3-TTS model.

    Signatures verified live against mlx-audio 0.4.8 in the qwen3-tts-mlx
    conda env. Do NOT relax these into ``**kwargs`` catch-alls — the
    strictness is the entire point.
    """

    supported_languages = ["auto", "english", "chinese"]

    def __init__(self):
        self.calls = {}

    def generate(
        self,
        *,
        text,
        ref_audio=None,
        ref_text=None,
        lang_code=MLX_DEFAULT_LANG,
        max_tokens=MLX_DEFAULT_MAX_TOKENS,
        stream=False,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
        **kwargs,
    ):
        self.calls["generate"] = {
            "text": text,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "lang_code": lang_code,
            "max_tokens": max_tokens,
            "stream": stream,
            "swallowed": dict(kwargs),
        }
        return iter([_fake_result()]) if stream else [_fake_result()]

    def generate_custom_voice(
        self,
        *,
        text,
        speaker,
        language=MLX_DEFAULT_LANG,
        instruct="",
        max_tokens=MLX_DEFAULT_MAX_TOKENS,
        stream=False,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
    ):
        self.calls["generate_custom_voice"] = {
            "text": text,
            "speaker": speaker,
            "language": language,
            "instruct": instruct,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        return iter([_fake_result()]) if stream else [_fake_result()]

    def generate_voice_design(
        self,
        *,
        text,
        instruct="",
        language=MLX_DEFAULT_LANG,
        max_tokens=MLX_DEFAULT_MAX_TOKENS,
        stream=False,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
    ):
        self.calls["generate_voice_design"] = {
            "text": text,
            "instruct": instruct,
            "language": language,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        return iter([_fake_result()]) if stream else [_fake_result()]


CLONE_PROMPT = {"ref_audio": "/tmp/ref.wav", "ref_text": "reference transcript"}


def _run_batch(model, mode, gen_params, language="English", **kwargs):
    """Drive _run_inference_mlx with mlx.core stubbed out.

    ``gen_params`` is copied so a caller-supplied constant (BASE_PARAMS) can
    never be mutated by the implementation and leak across tests.
    """
    with patch.dict(
        sys.modules, {"mlx": MagicMock(), "mlx.core": MagicMock()}
    ), patch(f"{MOD}.load_config", return_value={}):
        return inference._run_inference_mlx(
            model=model,
            text="hello world",
            mode=mode,
            gen_params=dict(gen_params),
            language=language,
            **kwargs,
        )


def _run_streaming(model, mode, gen_params, language="English", **kwargs):
    """Drive _run_inference_mlx_streaming to exhaustion.

    The streaming path imports only numpy today, but mlx is stubbed anyway so
    the harness stays environment-independent if that ever changes. Params are
    copied for the same reason as in _run_batch.
    """
    with patch.dict(sys.modules, {"mlx": MagicMock(), "mlx.core": MagicMock()}):
        return list(
            inference._run_inference_mlx_streaming(
                model=model,
                text="hello world",
                mode=mode,
                gen_params=dict(gen_params),
                language=language,
                config={},
                **kwargs,
            )
        )


BASE_PARAMS = {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "max_new_tokens": 2048,
}


class TestHarnessStrictness(unittest.TestCase):
    """Meta-tests: prove the fake would catch a revert to the old names."""

    def test_fake_model_rejects_legacy_kwarg_names(self):
        model = FakeMLXModel()

        # custom/design have no **kwargs upstream — legacy names must raise.
        with self.assertRaises(TypeError):
            model.generate_custom_voice(
                text="t", speaker="Ryan", max_new_tokens=2048
            )
        with self.assertRaises(TypeError):
            model.generate_voice_design(text="t", max_new_tokens=2048)

        # clone's generate() does have **kwargs upstream, so a legacy name is
        # swallowed rather than raised — which is exactly the bug. Assert the
        # fake records it, so the clone tests can detect it.
        model.generate(text="t", language="English", max_new_tokens=2048)
        swallowed = model.calls["generate"]["swallowed"]
        self.assertEqual(
            swallowed, {"language": "English", "max_new_tokens": 2048}
        )
        self.assertEqual(
            model.calls["generate"]["max_tokens"], MLX_DEFAULT_MAX_TOKENS
        )


class TestCloneKwargs(unittest.TestCase):
    """Clone mode is the only mode taking lang_code."""

    def test_batch_clone_forwards_lang_code_and_max_tokens(self):
        model = FakeMLXModel()
        _run_batch(model, "clone", BASE_PARAMS, voice_prompt=CLONE_PROMPT)

        call = model.calls["generate"]
        self.assertEqual(call["lang_code"], "english")
        self.assertEqual(call["max_tokens"], 2048)
        # Nothing may land in **kwargs — that is where the bug used to hide.
        self.assertEqual(call["swallowed"], {})

    def test_streaming_clone_forwards_lang_code_and_max_tokens(self):
        model = FakeMLXModel()
        _run_streaming(model, "clone", BASE_PARAMS, voice_prompt=CLONE_PROMPT)

        call = model.calls["generate"]
        self.assertEqual(call["lang_code"], "english")
        self.assertEqual(call["max_tokens"], 2048)
        self.assertEqual(call["swallowed"], {})
        self.assertTrue(call["stream"])


class TestCustomDesignKwargs(unittest.TestCase):
    """custom/design keep language= but need the cap renamed to max_tokens."""

    def test_batch_custom_forwards_max_tokens_and_keeps_language(self):
        model = FakeMLXModel()
        _run_batch(model, "custom", BASE_PARAMS, speaker="Ryan")

        call = model.calls["generate_custom_voice"]
        self.assertEqual(call["max_tokens"], 2048)
        self.assertEqual(call["language"], "English")

    def test_batch_design_forwards_max_tokens_and_keeps_language(self):
        model = FakeMLXModel()
        _run_batch(model, "design", BASE_PARAMS, voice_description="calm voice")

        call = model.calls["generate_voice_design"]
        self.assertEqual(call["max_tokens"], 2048)
        self.assertEqual(call["language"], "English")

    def test_streaming_custom_forwards_max_tokens_and_keeps_language(self):
        model = FakeMLXModel()
        _run_streaming(model, "custom", BASE_PARAMS, speaker="Ryan")

        call = model.calls["generate_custom_voice"]
        self.assertEqual(call["max_tokens"], 2048)
        # Asserted explicitly: dropping language= would silently fall back to
        # the fake's "auto" default and pass unnoticed.
        self.assertEqual(call["language"], "English")
        self.assertTrue(call["stream"])

    def test_streaming_design_forwards_max_tokens_and_keeps_language(self):
        model = FakeMLXModel()
        _run_streaming(
            model, "design", BASE_PARAMS, voice_description="calm voice"
        )

        call = model.calls["generate_voice_design"]
        self.assertEqual(call["max_tokens"], 2048)
        self.assertEqual(call["language"], "English")
        self.assertTrue(call["stream"])


class TestMaxTokensClamp(unittest.TestCase):
    """PRF-9 measured >=8192 as unstable on 16 GB; the MLX path must clamp."""

    def test_over_ceiling_is_clamped_and_warned(self):
        model = FakeMLXModel()
        params = {**BASE_PARAMS, "max_new_tokens": 8192}

        with self.assertLogs(LOGGER, level="WARNING") as logs:
            _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

        self.assertEqual(
            model.calls["generate"]["max_tokens"],
            inference._MLX_MAX_TOKENS_CEILING,
        )
        self.assertTrue(
            any("8192" in line and "clamp" in line.lower() for line in logs.output),
            f"expected a clamp warning naming 8192, got: {logs.output}",
        )

    def test_exactly_at_ceiling_is_not_clamped(self):
        """Boundary: the ceiling itself is allowed (`>` not `>=`)."""
        model = FakeMLXModel()
        ceiling = inference._MLX_MAX_TOKENS_CEILING
        params = {**BASE_PARAMS, "max_new_tokens": ceiling}

        _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

        self.assertEqual(model.calls["generate"]["max_tokens"], ceiling)

    def test_one_over_ceiling_is_clamped(self):
        model = FakeMLXModel()
        ceiling = inference._MLX_MAX_TOKENS_CEILING
        params = {**BASE_PARAMS, "max_new_tokens": ceiling + 1}

        _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

        self.assertEqual(model.calls["generate"]["max_tokens"], ceiling)

    def test_under_ceiling_passes_through_unclamped(self):
        model = FakeMLXModel()
        params = {**BASE_PARAMS, "max_new_tokens": 128}

        _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

        self.assertEqual(model.calls["generate"]["max_tokens"], 128)

    def test_clamp_applies_to_custom_and_design(self):
        for mode, method, extra in (
            ("custom", "generate_custom_voice", {"speaker": "Ryan"}),
            ("design", "generate_voice_design", {"voice_description": "calm"}),
        ):
            with self.subTest(mode=mode):
                model = FakeMLXModel()
                params = {**BASE_PARAMS, "max_new_tokens": 16384}

                _run_batch(model, mode, params, **extra)

                self.assertEqual(
                    model.calls[method]["max_tokens"],
                    inference._MLX_MAX_TOKENS_CEILING,
                )

    def test_clamp_applies_on_every_streaming_path(self):
        """The runaway guard must hold on streaming too, not just batch.

        Clamping only in _run_inference_mlx would leave a streaming caller
        sending max_new_tokens=16384 exposed to the exact runaway + OOM that
        PRF-9 measured.
        """
        for mode, method, extra in (
            ("clone", "generate", {"voice_prompt": CLONE_PROMPT}),
            ("custom", "generate_custom_voice", {"speaker": "Ryan"}),
            ("design", "generate_voice_design", {"voice_description": "calm"}),
        ):
            with self.subTest(mode=mode):
                model = FakeMLXModel()
                params = {**BASE_PARAMS, "max_new_tokens": 16384}

                _run_streaming(model, mode, params, **extra)

                self.assertEqual(
                    model.calls[method]["max_tokens"],
                    inference._MLX_MAX_TOKENS_CEILING,
                )


class TestNoneAndMissingCap(unittest.TestCase):
    """An explicit None must not crash the comparison.

    _get_mlx_gen_params uses gen_params.get(...), so a caller passing
    max_new_tokens=None explicitly gets None back — and a naive
    `if requested > ceiling` raises TypeError on it.
    """

    def test_explicit_none_falls_back_to_ceiling_batch(self):
        model = FakeMLXModel()
        params = {**BASE_PARAMS, "max_new_tokens": None}

        _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

        self.assertEqual(
            model.calls["generate"]["max_tokens"],
            inference._MLX_MAX_TOKENS_CEILING,
        )

    def test_explicit_none_falls_back_to_ceiling_streaming(self):
        model = FakeMLXModel()
        params = {**BASE_PARAMS, "max_new_tokens": None}

        _run_streaming(model, "clone", params, voice_prompt=CLONE_PROMPT)

        self.assertEqual(
            model.calls["generate"]["max_tokens"],
            inference._MLX_MAX_TOKENS_CEILING,
        )

    def test_explicit_none_on_custom_and_design(self):
        for mode, method, extra in (
            ("custom", "generate_custom_voice", {"speaker": "Ryan"}),
            ("design", "generate_voice_design", {"voice_description": "calm"}),
        ):
            with self.subTest(mode=mode):
                model = FakeMLXModel()
                params = {**BASE_PARAMS, "max_new_tokens": None}

                _run_batch(model, mode, params, **extra)

                self.assertEqual(
                    model.calls[method]["max_tokens"],
                    inference._MLX_MAX_TOKENS_CEILING,
                )

    def test_split_helper_handles_none(self):
        _, max_tokens = inference._split_mlx_params({"max_new_tokens": None})

        self.assertEqual(max_tokens, inference._MLX_MAX_TOKENS_CEILING)


class TestLangCodeMapping(unittest.TestCase):
    """mlx-audio matches lang_code case-insensitively and never raises."""

    def test_language_is_lowercased(self):
        for language, expected in (
            ("English", "english"),
            ("Chinese", "chinese"),
            ("auto", "auto"),
            ("  English  ", "english"),
        ):
            with self.subTest(language=language):
                model = FakeMLXModel()
                _run_batch(
                    model,
                    "clone",
                    BASE_PARAMS,
                    language=language,
                    voice_prompt=CLONE_PROMPT,
                )
                self.assertEqual(model.calls["generate"]["lang_code"], expected)

    def test_unsupported_language_still_generates_but_warns(self):
        model = FakeMLXModel()

        with self.assertLogs(LOGGER, level="WARNING") as logs:
            wav, sr = _run_batch(
                model,
                "clone",
                BASE_PARAMS,
                language="Klingon",
                voice_prompt=CLONE_PROMPT,
            )

        self.assertEqual(model.calls["generate"]["lang_code"], "klingon")
        self.assertIsNotNone(wav)
        # Case-insensitive: the implementation may log the original or the
        # normalized form; either is acceptable, silence is not.
        self.assertTrue(
            any("klingon" in line.lower() for line in logs.output),
            f"expected a warning naming the language, got: {logs.output}",
        )

    def test_streaming_also_maps_and_validates_language(self):
        """Language mapping must hold on the streaming clone path too."""
        model = FakeMLXModel()

        _run_streaming(
            model,
            "clone",
            BASE_PARAMS,
            language="Chinese",
            voice_prompt=CLONE_PROMPT,
        )

        self.assertEqual(model.calls["generate"]["lang_code"], "chinese")
        self.assertEqual(model.calls["generate"]["swallowed"], {})

    def test_empty_language_falls_back_to_auto(self):
        model = FakeMLXModel()
        _run_batch(
            model, "clone", BASE_PARAMS, language="", voice_prompt=CLONE_PROMPT
        )
        self.assertEqual(model.calls["generate"]["lang_code"], "auto")


class TestSplitMlxParams(unittest.TestCase):
    """Direct unit tests for the helper, incl. the no-mutation contract."""

    def test_does_not_mutate_caller_params(self):
        params = dict(BASE_PARAMS)
        before = dict(params)

        inference._split_mlx_params(params)

        self.assertEqual(params, before)

    def test_strips_max_new_tokens_from_sampling_kwargs(self):
        sampling, _ = inference._split_mlx_params(dict(BASE_PARAMS))

        self.assertNotIn("max_new_tokens", sampling)
        self.assertNotIn("max_tokens", sampling)
        self.assertEqual(sampling["temperature"], 0.7)

    def test_absent_max_new_tokens_uses_ceiling(self):
        params = {k: v for k, v in BASE_PARAMS.items() if k != "max_new_tokens"}

        _, max_tokens = inference._split_mlx_params(params)

        self.assertEqual(max_tokens, inference._MLX_MAX_TOKENS_CEILING)


if __name__ == "__main__":
    unittest.main()
