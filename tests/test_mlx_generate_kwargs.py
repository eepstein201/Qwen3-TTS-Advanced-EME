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


class _Unset:
    """Sentinel for 'this argument was never passed'.

    Using mlx-audio's real defaults here would make assertions hollow: a test
    asserting ``max_tokens == 4096`` cannot tell "the implementation forwarded
    4096" from "the implementation forwarded nothing and the default applied".
    That is exactly the dead-knob bug under repair, so the fake must be able to
    distinguish the two.
    """

    def __repr__(self):
        return "<UNSET>"


UNSET = _Unset()


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

    Parameter names, ordering and the ``**kwargs`` asymmetry were verified live
    against mlx-audio 0.4.8 in the qwen3-tts-mlx conda env:

        generate(text, voice, instruct, temperature, speed, lang_code,
                 ref_audio, ref_text, split_pattern, max_tokens, verbose,
                 stream, streaming_interval, streaming_context_size, top_k,
                 top_p, repetition_penalty, **kwargs)          <-- HAS **kwargs
        generate_custom_voice(text, speaker, language, instruct, temperature,
                 max_tokens, top_k, top_p, repetition_penalty, verbose,
                 stream, streaming_interval)                   <-- NO **kwargs
        generate_voice_design(text, instruct, language, ...)   <-- NO **kwargs

    Two deliberate deviations from upstream, both to make the gate stricter:
    every parameter is keyword-only, and defaults are ``UNSET`` sentinels
    rather than upstream's real defaults. ``instruct`` on
    ``generate_voice_design`` is REQUIRED, matching upstream — giving it a
    default would hide a production TypeError.
    """

    def __init__(self):
        self.calls = {}
        # Instance attribute, matching upstream: mlx-audio builds this per
        # checkpoint from config.talker_config.codec_language_id. Tests mutate
        # it to prove the implementation reads the model rather than a
        # hardcoded constant.
        self.supported_languages = ["auto", "english", "chinese"]

    def generate(
        self,
        *,
        text,
        voice=UNSET,
        instruct=UNSET,
        temperature=UNSET,
        speed=UNSET,
        lang_code=UNSET,
        ref_audio=UNSET,
        ref_text=UNSET,
        split_pattern=UNSET,
        max_tokens=UNSET,
        verbose=UNSET,
        stream=False,
        streaming_interval=UNSET,
        streaming_context_size=UNSET,
        top_k=UNSET,
        top_p=UNSET,
        repetition_penalty=UNSET,
        **kwargs,
    ):
        self.calls["generate"] = {
            "text": text,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "lang_code": lang_code,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "swallowed": dict(kwargs),
        }
        return iter([_fake_result()]) if stream else [_fake_result()]

    def generate_custom_voice(
        self,
        *,
        text,
        speaker,
        language=UNSET,
        instruct=UNSET,
        temperature=UNSET,
        max_tokens=UNSET,
        top_k=UNSET,
        top_p=UNSET,
        repetition_penalty=UNSET,
        verbose=UNSET,
        stream=False,
        streaming_interval=UNSET,
    ):
        self.calls["generate_custom_voice"] = {
            "text": text,
            "speaker": speaker,
            "language": language,
            "instruct": instruct,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        return iter([_fake_result()]) if stream else [_fake_result()]

    def generate_voice_design(
        self,
        *,
        text,
        instruct,
        language=UNSET,
        temperature=UNSET,
        max_tokens=UNSET,
        top_k=UNSET,
        top_p=UNSET,
        repetition_penalty=UNSET,
        verbose=UNSET,
        stream=False,
        streaming_interval=UNSET,
    ):
        self.calls["generate_voice_design"] = {
            "text": text,
            "instruct": instruct,
            "language": language,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        return iter([_fake_result()]) if stream else [_fake_result()]


#: Sampling values carried by BASE_PARAMS, asserted at every call site so that
#: dropping ``**sampling`` cannot pass (lens-A probe 2 proved it otherwise did).
EXPECTED_SAMPLING = {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
}


def assert_sampling_forwarded(testcase, call):
    """Every sampling knob must survive the trip to mlx-audio."""
    for key, expected in EXPECTED_SAMPLING.items():
        testcase.assertEqual(
            call[key],
            expected,
            f"{key} did not reach mlx-audio (got {call[key]!r}) — a dropped "
            f"**sampling would silently revert it to mlx-audio's default",
        )


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
            model.generate_voice_design(
                text="t", instruct="i", max_new_tokens=2048
            )

        # instruct is REQUIRED on generate_voice_design upstream. If the fake
        # ever gains a default, an implementation that drops instruct would
        # pass here and TypeError in production.
        with self.assertRaises(TypeError):
            model.generate_voice_design(text="t")

        # clone's generate() does have **kwargs upstream, so a legacy name is
        # swallowed rather than raised — which is exactly the bug. Assert the
        # fake records it, so the clone tests can detect it.
        model.generate(text="t", language="English", max_new_tokens=2048)
        swallowed = model.calls["generate"]["swallowed"]
        self.assertEqual(
            swallowed, {"language": "English", "max_new_tokens": 2048}
        )
        # Sentinel, not 4096: "never passed" must be distinguishable from
        # "passed the default value".
        self.assertIs(model.calls["generate"]["max_tokens"], UNSET)
        self.assertIs(model.calls["generate"]["lang_code"], UNSET)


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
        # The batch path must NOT stream: list(generate(stream=True)) would
        # return partial chunks and take sample_rate from a partial result.
        self.assertFalse(call["stream"])
        assert_sampling_forwarded(self, call)

    def test_batch_clone_forwards_the_reference_audio_and_transcript(self):
        """Dropping ref_audio/ref_text silently degrades cloning to a generic voice."""
        model = FakeMLXModel()
        _run_batch(model, "clone", BASE_PARAMS, voice_prompt=CLONE_PROMPT)

        call = model.calls["generate"]
        self.assertEqual(call["ref_audio"], CLONE_PROMPT["ref_audio"])
        self.assertEqual(call["ref_text"], CLONE_PROMPT["ref_text"])

    def test_streaming_clone_forwards_lang_code_and_max_tokens(self):
        model = FakeMLXModel()
        _run_streaming(model, "clone", BASE_PARAMS, voice_prompt=CLONE_PROMPT)

        call = model.calls["generate"]
        self.assertEqual(call["lang_code"], "english")
        self.assertEqual(call["max_tokens"], 2048)
        self.assertEqual(call["swallowed"], {})
        self.assertTrue(call["stream"])
        self.assertEqual(call["ref_audio"], CLONE_PROMPT["ref_audio"])
        self.assertEqual(call["ref_text"], CLONE_PROMPT["ref_text"])
        assert_sampling_forwarded(self, call)


class TestCustomDesignKwargs(unittest.TestCase):
    """custom/design keep language= but need the cap renamed to max_tokens."""

    def test_batch_custom_forwards_max_tokens_and_keeps_language(self):
        model = FakeMLXModel()
        _run_batch(model, "custom", BASE_PARAMS, speaker="Ryan", instruct="calm")

        call = model.calls["generate_custom_voice"]
        self.assertEqual(call["max_tokens"], 2048)
        self.assertEqual(call["language"], "english")
        self.assertEqual(call["speaker"], "Ryan")
        self.assertEqual(call["instruct"], "calm")
        self.assertFalse(call["stream"])
        assert_sampling_forwarded(self, call)

    def test_batch_design_forwards_max_tokens_and_keeps_language(self):
        model = FakeMLXModel()
        _run_batch(model, "design", BASE_PARAMS, voice_description="calm voice")

        call = model.calls["generate_voice_design"]
        self.assertEqual(call["max_tokens"], 2048)
        self.assertEqual(call["language"], "english")
        # instruct carries the voice description; dropping it is a production
        # TypeError because upstream makes it required.
        self.assertEqual(call["instruct"], "calm voice")
        self.assertFalse(call["stream"])
        assert_sampling_forwarded(self, call)

    def test_streaming_custom_forwards_max_tokens_and_keeps_language(self):
        model = FakeMLXModel()
        _run_streaming(model, "custom", BASE_PARAMS, speaker="Ryan")

        call = model.calls["generate_custom_voice"]
        self.assertEqual(call["max_tokens"], 2048)
        # Asserted explicitly: dropping language= would land on the UNSET
        # sentinel and pass unnoticed if we only checked max_tokens.
        self.assertEqual(call["language"], "english")
        self.assertTrue(call["stream"])
        assert_sampling_forwarded(self, call)

    def test_streaming_design_forwards_max_tokens_and_keeps_language(self):
        model = FakeMLXModel()
        _run_streaming(
            model, "design", BASE_PARAMS, voice_description="calm voice"
        )

        call = model.calls["generate_voice_design"]
        self.assertEqual(call["max_tokens"], 2048)
        self.assertEqual(call["language"], "english")
        self.assertEqual(call["instruct"], "calm voice")
        self.assertTrue(call["stream"])
        assert_sampling_forwarded(self, call)


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

    def test_explicit_none_falls_back_to_product_default_batch(self):
        model = FakeMLXModel()
        params = {**BASE_PARAMS, "max_new_tokens": None}

        _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

        self.assertEqual(
            model.calls["generate"]["max_tokens"],
            inference._DEFAULT_MAX_NEW_TOKENS,
        )

    def test_explicit_none_falls_back_to_default_streaming(self):
        model = FakeMLXModel()
        params = {**BASE_PARAMS, "max_new_tokens": None}

        _run_streaming(model, "clone", params, voice_prompt=CLONE_PROMPT)

        self.assertEqual(
            model.calls["generate"]["max_tokens"],
            inference._DEFAULT_MAX_NEW_TOKENS,
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
                    inference._DEFAULT_MAX_NEW_TOKENS,
                )

    def test_non_positive_cap_falls_back_to_default(self):
        """<=0 would make mlx-audio's range(max_tokens) loop generate nothing.

        The request schema enforces ge=1, but direct engine callers are not
        bound by it, so the engine must not emit a silently empty generation.
        """
        for bad in (0, -1):
            with self.subTest(value=bad):
                model = FakeMLXModel()
                params = {**BASE_PARAMS, "max_new_tokens": bad}

                _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

                self.assertEqual(
                    model.calls["generate"]["max_tokens"],
                    inference._DEFAULT_MAX_NEW_TOKENS,
                )

    def test_non_numeric_cap_falls_back_instead_of_crashing(self):
        """A hand-edited config.json string must not 500 every generation.

        validate_config() type-checks only generation.temperature, so
        "max_new_tokens": "4096" reaches the engine as a str and would raise
        TypeError on the ceiling comparison.
        """
        for bad in ("4096", "not-a-number", [], {}):
            with self.subTest(value=bad):
                model = FakeMLXModel()
                params = {**BASE_PARAMS, "max_new_tokens": bad}

                _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

                self.assertEqual(
                    model.calls["generate"]["max_tokens"],
                    inference._DEFAULT_MAX_NEW_TOKENS,
                )

    def test_bool_cap_is_rejected_not_treated_as_int(self):
        """bool is an int subclass; range(True) would emit ~1 token silently."""
        model = FakeMLXModel()
        params = {**BASE_PARAMS, "max_new_tokens": True}

        _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

        self.assertEqual(
            model.calls["generate"]["max_tokens"],
            inference._DEFAULT_MAX_NEW_TOKENS,
        )

    def test_float_cap_is_coerced_to_int(self):
        """mlx-audio does range(max_tokens); a float would TypeError there."""
        model = FakeMLXModel()
        params = {**BASE_PARAMS, "max_new_tokens": 2048.7}

        _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

        forwarded = model.calls["generate"]["max_tokens"]
        self.assertIsInstance(forwarded, int)
        self.assertEqual(forwarded, 2048)

    def test_split_helper_handles_none(self):
        _, max_tokens = inference._split_mlx_params({"max_new_tokens": None})

        self.assertEqual(max_tokens, inference._DEFAULT_MAX_NEW_TOKENS)


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

    def test_supported_languages_comes_from_the_model_not_a_constant(self):
        """The allow-list is per-checkpoint, so it must be read off the model.

        Upstream builds ``supported_languages`` from
        ``config.talker_config.codec_language_id``. A hardcoded tuple would
        warn spuriously on every generation for a language the loaded
        checkpoint genuinely supports (the repo ships a Japanese premium
        speaker, so this is reachable).
        """
        model = FakeMLXModel()
        model.supported_languages = ["auto", "english", "chinese", "japanese"]

        with patch.object(inference.logger, "warning") as mock_warning:
            _run_batch(
                model,
                "clone",
                BASE_PARAMS,
                language="Japanese",
                voice_prompt=CLONE_PROMPT,
            )

        self.assertEqual(model.calls["generate"]["lang_code"], "japanese")
        lang_warnings = [
            c for c in mock_warning.call_args_list if "language" in str(c).lower()
        ]
        self.assertEqual(
            lang_warnings,
            [],
            f"warned about a language the model supports: {lang_warnings}",
        )

    def test_dialect_codes_do_not_false_warn(self):
        """mlx-audio omits *_dialect from supported_languages but still honors
        them at generation time, so warning would be actively misleading."""
        model = FakeMLXModel()  # supported_languages has no dialect entries

        with patch.object(inference.logger, "warning") as mock_warning:
            _run_batch(
                model,
                "clone",
                BASE_PARAMS,
                language="beijing_dialect",
                voice_prompt=CLONE_PROMPT,
            )

        self.assertEqual(model.calls["generate"]["lang_code"], "beijing_dialect")
        lang_warnings = [
            c for c in mock_warning.call_args_list if "mlx-audio match" in str(c)
        ]
        self.assertEqual(lang_warnings, [])

    def test_model_without_supported_languages_does_not_crash(self):
        """getattr fallback: not every model object exposes the attribute."""
        model = FakeMLXModel()
        del model.supported_languages

        _run_batch(model, "clone", BASE_PARAMS, voice_prompt=CLONE_PROMPT)

        self.assertEqual(model.calls["generate"]["lang_code"], "english")

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

    def test_absent_max_new_tokens_uses_product_default(self):
        """Absent must resolve to the documented default (2048), NOT the
        ceiling. PRF-9 established that higher is the risk direction, so
        "unspecified means maximum permitted" is the wrong semantics."""
        params = {k: v for k, v in BASE_PARAMS.items() if k != "max_new_tokens"}

        _, max_tokens = inference._split_mlx_params(params)

        self.assertEqual(max_tokens, inference._DEFAULT_MAX_NEW_TOKENS)
        self.assertLess(
            inference._DEFAULT_MAX_NEW_TOKENS,
            inference._MLX_MAX_TOKENS_CEILING,
        )


class TestConstantsArePinned(unittest.TestCase):
    """Pin the constants themselves.

    Every other clamp test compares against ``_MLX_MAX_TOKENS_CEILING``, so a
    wrong constant would move the goalposts and keep those tests green. These
    assertions are deliberately literal.
    """

    def test_ceiling_is_4096(self):
        self.assertEqual(inference._MLX_MAX_TOKENS_CEILING, 4096)

    def test_default_is_2048_and_matches_config(self):
        from qwen3_tts.core.config import get_default_config

        self.assertEqual(inference._DEFAULT_MAX_NEW_TOKENS, 2048)
        self.assertEqual(
            get_default_config()["generation"]["max_new_tokens"],
            inference._DEFAULT_MAX_NEW_TOKENS,
            "engine fallback drifted from the documented config default",
        )


class TestClampWarningPrecision(unittest.TestCase):
    """The warning must fire when clamping happens — and only then."""

    def test_no_warning_at_exactly_the_ceiling(self):
        """Guards `>` vs `>=`: at the ceiling nothing is clamped, so nothing
        should be logged."""
        model = FakeMLXModel()
        params = {
            **BASE_PARAMS,
            "max_new_tokens": inference._MLX_MAX_TOKENS_CEILING,
        }

        with patch.object(inference.logger, "warning") as mock_warning:
            _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

        clamp_warnings = [
            c for c in mock_warning.call_args_list if "clamp" in str(c).lower()
        ]
        self.assertEqual(clamp_warnings, [], f"spurious clamp: {clamp_warnings}")

    def test_no_warning_below_the_ceiling(self):
        model = FakeMLXModel()
        params = {**BASE_PARAMS, "max_new_tokens": 128}

        with patch.object(inference.logger, "warning") as mock_warning:
            _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

        clamp_warnings = [
            c for c in mock_warning.call_args_list if "clamp" in str(c).lower()
        ]
        self.assertEqual(clamp_warnings, [])

    def test_warning_fires_on_the_streaming_path_too(self):
        model = FakeMLXModel()
        params = {**BASE_PARAMS, "max_new_tokens": 16384}

        with self.assertLogs(LOGGER, level="WARNING") as logs:
            _run_streaming(model, "clone", params, voice_prompt=CLONE_PROMPT)

        self.assertTrue(
            any("clamp" in line.lower() for line in logs.output),
            f"streaming clamp was silent: {logs.output}",
        )


class TestLanguageNormalizationEverywhere(unittest.TestCase):
    """custom/design take `language=`, but it still needs normalizing.

    mlx-audio lowercases the value but does NOT strip it, so a padded
    "  English  " silently misses the codec_language_id lookup and drops
    language conditioning with no error.
    """

    def test_whitespace_is_stripped_for_custom_and_design(self):
        for mode, method, extra in (
            ("custom", "generate_custom_voice", {"speaker": "Ryan"}),
            ("design", "generate_voice_design", {"voice_description": "calm"}),
        ):
            with self.subTest(mode=mode):
                model = FakeMLXModel()

                _run_batch(
                    model, mode, BASE_PARAMS, language="  English  ", **extra
                )

                self.assertEqual(model.calls[method]["language"], "english")

    def test_language_none_does_not_crash(self):
        model = FakeMLXModel()

        _run_batch(
            model, "clone", BASE_PARAMS, language=None, voice_prompt=CLONE_PROMPT
        )

        self.assertEqual(model.calls["generate"]["lang_code"], "auto")


class TestAbsentCapThroughPublicApi(unittest.TestCase):
    """The absent-key case must be exercised through the real entry point."""

    def test_absent_key_batch_uses_product_default(self):
        model = FakeMLXModel()
        params = {k: v for k, v in BASE_PARAMS.items() if k != "max_new_tokens"}

        _run_batch(model, "clone", params, voice_prompt=CLONE_PROMPT)

        self.assertEqual(
            model.calls["generate"]["max_tokens"],
            inference._DEFAULT_MAX_NEW_TOKENS,
        )

    def test_absent_key_streaming_uses_product_default(self):
        model = FakeMLXModel()
        params = {k: v for k, v in BASE_PARAMS.items() if k != "max_new_tokens"}

        _run_streaming(model, "clone", params, voice_prompt=CLONE_PROMPT)

        self.assertEqual(
            model.calls["generate"]["max_tokens"],
            inference._DEFAULT_MAX_NEW_TOKENS,
        )


if __name__ == "__main__":
    unittest.main()
