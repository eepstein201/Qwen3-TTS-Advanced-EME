#!/usr/bin/env python3
"""Extended tests for model_loader.py uncovered paths.

Covers:
  - _install_mps_patch(): macOS vs non-macOS, double-call idempotency
  - _resolve_load_kwargs(): none/4bit/8bit quantization, auto-8bit on Turing
  - _is_model_cached(): cached vs not cached
  - _apply_torch_compile(): success and failure
  - _patch_tokenizer(): success and unsupported
  - _warmup_model(): design vs non-design, exception handling
  - load_model(): dispatch to mlx vs torch

Run: pytest tests/test_model_loader_extended.py -v
"""

import sys
import unittest

try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

    class _DummyMarkerFunc:
        def __init__(self, name=None):
            self._name = name

        def __call__(self, condition, **kwargs):
            return lambda f: f

    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()

    class _DummyPytest:
        mark = _DummyMark()

    pytest = _DummyPytest()

from unittest.mock import MagicMock, patch

# ---- _install_mps_patch ----


@pytest.mark.unit
def test_install_mps_patch_non_macos():
    """_install_mps_patch marks as done but doesn't patch on non-macOS."""
    from qwen3_tts.core.engine import model_loader

    orig = model_loader._mps_patch_installed
    try:
        model_loader._mps_patch_installed = False
        with patch("qwen3_tts.core.config.IS_MACOS", False):
            model_loader._install_mps_patch()
        assert model_loader._mps_patch_installed is True
    finally:
        model_loader._mps_patch_installed = orig


@pytest.mark.unit
def test_install_mps_patch_already_installed():
    """_install_mps_patch is a no-op when already installed."""
    from qwen3_tts.core.engine import model_loader

    orig = model_loader._mps_patch_installed
    try:
        model_loader._mps_patch_installed = True
        # Should return immediately without touching anything
        model_loader._install_mps_patch()
        assert model_loader._mps_patch_installed is True
    finally:
        model_loader._mps_patch_installed = orig


@pytest.mark.unit
def test_install_mps_patch_on_macos():
    """_install_mps_patch patches torch.multinomial on macOS."""
    from qwen3_tts.core.engine import model_loader

    orig = model_loader._mps_patch_installed
    try:
        model_loader._mps_patch_installed = False

        mock_torch = MagicMock()
        mock_torch.multinomial = MagicMock(name="original_multinomial")

        with (
            patch("qwen3_tts.core.config.IS_MACOS", True),
            patch.dict(sys.modules, {"torch": mock_torch}),
        ):
            model_loader._install_mps_patch()

        assert model_loader._mps_patch_installed is True
        # torch.multinomial should have been replaced
        assert mock_torch.multinomial != mock_torch.multinomial.__class__
    finally:
        model_loader._mps_patch_installed = orig


# ---- _resolve_load_kwargs ----


@pytest.mark.unit
def test_resolve_load_kwargs_none_quant():
    """_resolve_load_kwargs with no quantization sets dtype."""
    from qwen3_tts.core.engine.model_loader import _resolve_load_kwargs

    mock_torch = MagicMock()
    mock_torch.float32 = "float32_sentinel"
    mock_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = _resolve_load_kwargs("none", "float32_sentinel", "mps", "sdpa", "auto")

    assert result["dtype"] == "float32_sentinel"
    assert result["attn_implementation"] == "sdpa"
    assert "load_in_8bit" not in result


@pytest.mark.unit
def test_resolve_load_kwargs_8bit_no_cuda():
    """_resolve_load_kwargs raises for 8bit without CUDA."""
    from qwen3_tts.core.engine.model_loader import _resolve_load_kwargs

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with (
        patch.dict(sys.modules, {"torch": mock_torch}),
        pytest.raises(RuntimeError, match="8-bit quantization requires CUDA"),
    ):
        _resolve_load_kwargs("8bit", "float16", "cpu", "sdpa", "auto")


@pytest.mark.unit
def test_resolve_load_kwargs_4bit_no_linux():
    """_resolve_load_kwargs raises for 4bit without CUDA+Linux."""
    from qwen3_tts.core.engine.model_loader import _resolve_load_kwargs

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with (
        patch.dict(sys.modules, {"torch": mock_torch}),
        patch("sys.platform", "darwin"),
        pytest.raises(RuntimeError, match="4-bit quantization requires CUDA on Linux"),
    ):
        _resolve_load_kwargs("4bit", "float16", "cuda", "sdpa", "auto")


@pytest.mark.unit
def test_resolve_load_kwargs_auto_8bit_turing():
    """_resolve_load_kwargs auto-enables 8bit on Turing GPUs when not explicitly set."""
    from qwen3_tts.core.engine.model_loader import _resolve_load_kwargs

    mock_torch = MagicMock()
    mock_torch.float16 = "float16_sentinel"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_capability.return_value = (7, 5)

    with (
        patch.dict(sys.modules, {"torch": mock_torch}),
        patch(
            "qwen3_tts.core.engine.model_loader.load_config",
            return_value={"advanced": {}},
        ),
    ):
        result = _resolve_load_kwargs(
            "none", "float16_sentinel", "cuda", "sdpa", "auto"
        )

    assert result.get("load_in_8bit") is True


@pytest.mark.unit
def test_resolve_load_kwargs_turing_explicit_none():
    """_resolve_load_kwargs respects explicit 'none' quant on Turing."""
    from qwen3_tts.core.engine.model_loader import _resolve_load_kwargs

    mock_torch = MagicMock()
    mock_torch.float16 = "float16_sentinel"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_capability.return_value = (7, 5)

    config = {"advanced": {"torch_quantization": "none"}}
    with (
        patch.dict(sys.modules, {"torch": mock_torch}),
        patch("qwen3_tts.core.engine.model_loader.load_config", return_value=config),
    ):
        result = _resolve_load_kwargs(
            "none", "float16_sentinel", "cuda", "sdpa", "auto"
        )

    assert "load_in_8bit" not in result


# ---- _is_model_cached ----


@pytest.mark.unit
def test_is_model_cached_true():
    """_is_model_cached returns True when model is cached."""
    from qwen3_tts.core.engine.model_loader import _is_model_cached

    mock_hf = MagicMock()
    mock_hf.snapshot_download.return_value = "/path"
    with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
        assert _is_model_cached("repo/model") is True


@pytest.mark.unit
def test_is_model_cached_false():
    """_is_model_cached returns False when model is not cached."""
    from qwen3_tts.core.engine.model_loader import _is_model_cached

    mock_hf = MagicMock()
    mock_hf.snapshot_download.side_effect = Exception("not found")
    with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
        assert _is_model_cached("repo/model") is False


# ---- _apply_torch_compile ----


@pytest.mark.unit
def test_apply_torch_compile_success():
    """_apply_torch_compile wraps model.model with torch.compile."""
    from qwen3_tts.core.engine.model_loader import _apply_torch_compile

    mock_model = MagicMock()
    mock_torch = MagicMock()
    compiled = MagicMock(name="compiled")
    mock_torch.compile.return_value = compiled

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = _apply_torch_compile(mock_model, "clone", "cuda", True)

    mock_torch.compile.assert_called_once()
    assert result.model == compiled


@pytest.mark.unit
def test_apply_torch_compile_skips_non_cuda():
    """_apply_torch_compile skips compilation on non-CUDA."""
    from qwen3_tts.core.engine.model_loader import _apply_torch_compile

    mock_model = MagicMock()
    original_inner = mock_model.model

    mock_torch = MagicMock()
    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = _apply_torch_compile(mock_model, "clone", "mps", True)

    mock_torch.compile.assert_not_called()
    assert result.model == original_inner


@pytest.mark.unit
def test_apply_torch_compile_failure():
    """_apply_torch_compile handles compile failure gracefully."""
    from qwen3_tts.core.engine.model_loader import _apply_torch_compile

    mock_model = MagicMock()
    mock_torch = MagicMock()
    mock_torch.compile.side_effect = RuntimeError("compile failed")

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = _apply_torch_compile(mock_model, "clone", "cuda", True)

    # Model should be returned as-is
    assert result == mock_model


@pytest.mark.unit
def test_apply_torch_compile_disabled():
    """_apply_torch_compile does nothing when should_compile=False."""
    from qwen3_tts.core.engine.model_loader import _apply_torch_compile

    mock_model = MagicMock()
    mock_torch = MagicMock()

    with patch.dict(sys.modules, {"torch": mock_torch}):
        _apply_torch_compile(mock_model, "clone", "cuda", False)

    mock_torch.compile.assert_not_called()


# ---- _patch_tokenizer ----


@pytest.mark.unit
def test_patch_tokenizer_success():
    """_patch_tokenizer reloads tokenizer with fix_mistral_regex."""
    from qwen3_tts.core.engine.model_loader import _patch_tokenizer

    mock_model = MagicMock()
    mock_tokenizer = MagicMock(name="new_tokenizer")

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        result = _patch_tokenizer(mock_model, "repo/model")

    assert result.tokenizer == mock_tokenizer


@pytest.mark.unit
def test_patch_tokenizer_unsupported():
    """_patch_tokenizer handles TypeError when fix_mistral_regex not supported."""
    from qwen3_tts.core.engine.model_loader import _patch_tokenizer

    mock_model = MagicMock()
    original_tokenizer = mock_model.tokenizer

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.side_effect = TypeError(
        "unexpected keyword"
    )

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        result = _patch_tokenizer(mock_model, "repo/model")

    # Tokenizer should remain unchanged
    assert result.tokenizer == original_tokenizer


# ---- _warmup_model ----


@pytest.mark.unit
def test_warmup_model_design_mlx():
    """_warmup_model runs warm-up inference for design model on MLX."""
    from qwen3_tts.core.engine.model_loader import _warmup_model

    mock_model = MagicMock()
    mock_model.generate_voice_design.return_value = iter([b"audio"])

    _warmup_model(mock_model, "design", "mlx")
    mock_model.generate_voice_design.assert_called_once()


@pytest.mark.unit
def test_warmup_model_design_torch():
    """_warmup_model runs warm-up inference for design model on torch."""
    from qwen3_tts.core.engine.model_loader import _warmup_model

    mock_model = MagicMock()
    mock_torch = MagicMock()

    with patch.dict(sys.modules, {"torch": mock_torch}):
        _warmup_model(mock_model, "design", "torch")

    mock_model.generate_voice_design.assert_called_once()


@pytest.mark.unit
def test_warmup_model_skips_non_design():
    """_warmup_model skips warm-up for clone and custom models."""
    from qwen3_tts.core.engine.model_loader import _warmup_model

    mock_model = MagicMock()
    _warmup_model(mock_model, "clone", "mlx")
    mock_model.generate_voice_design.assert_not_called()

    _warmup_model(mock_model, "custom", "torch")
    mock_model.generate_voice_design.assert_not_called()


@pytest.mark.unit
def test_warmup_model_handles_exception():
    """_warmup_model handles exceptions gracefully."""
    from qwen3_tts.core.engine.model_loader import _warmup_model

    mock_model = MagicMock()
    mock_model.generate_voice_design.side_effect = RuntimeError("warmup failed")

    # Should not raise
    _warmup_model(mock_model, "design", "mlx")


@pytest.mark.unit
def test_warmup_model_skipped_when_env_knob_set(monkeypatch):
    """TTS_SKIP_WARMUP=1 skips the design warm-up (issue #192 mitigation)."""
    from qwen3_tts.core.engine.model_loader import _warmup_model

    monkeypatch.setenv("TTS_SKIP_WARMUP", "1")
    mock_model = MagicMock()
    _warmup_model(mock_model, "design", "mlx")
    mock_model.generate_voice_design.assert_not_called()

    # Quoted/exported values can carry padding — strip before matching.
    monkeypatch.setenv("TTS_SKIP_WARMUP", " true ")
    _warmup_model(mock_model, "design", "mlx")
    mock_model.generate_voice_design.assert_not_called()


@pytest.mark.unit
def test_warmup_model_skip_logs_knob_positively(monkeypatch, caplog):
    """The skip is logged — absence of warm-up lines alone proves nothing."""
    import logging

    from qwen3_tts.core.engine.model_loader import _warmup_model

    monkeypatch.setenv("TTS_SKIP_WARMUP", "true")
    with caplog.at_level(logging.INFO, logger="tts.engine"):
        _warmup_model(MagicMock(), "design", "mlx")
    assert "Skipping design warm-up" in caplog.text
    assert "TTS_SKIP_WARMUP" in caplog.text


@pytest.mark.unit
def test_warmup_model_runs_when_env_knob_falsy(monkeypatch):
    """Unset or falsy knob values leave the warm-up enabled."""
    from qwen3_tts.core.engine.model_loader import _warmup_model

    for value in (None, "0", ""):
        if value is None:
            monkeypatch.delenv("TTS_SKIP_WARMUP", raising=False)
        else:
            monkeypatch.setenv("TTS_SKIP_WARMUP", value)
        mock_model = MagicMock()
        mock_model.generate_voice_design.return_value = iter([b"audio"])
        _warmup_model(mock_model, "design", "mlx")
        mock_model.generate_voice_design.assert_called_once()


# ---- load_model dispatch ----


@pytest.mark.unit
def test_load_model_dispatches_mlx():
    """load_model dispatches to _load_model_mlx when backend is mlx."""
    from qwen3_tts.core.engine.model_loader import load_model

    mock_model = MagicMock()
    with (
        patch("qwen3_tts.core.engine.model_loader.get_backend", return_value="mlx"),
        patch(
            "qwen3_tts.core.engine.model_loader._load_model_mlx",
            return_value=mock_model,
        ) as mock_mlx,
        patch("qwen3_tts.core.engine.model_loader._warmup_model"),
    ):
        result = load_model("clone")

    mock_mlx.assert_called_once_with("clone")
    assert result == mock_model


@pytest.mark.unit
def test_load_model_dispatches_torch():
    """load_model dispatches to _load_model_torch when backend is torch."""
    from qwen3_tts.core.engine.model_loader import load_model

    mock_model = MagicMock()
    with (
        patch("qwen3_tts.core.engine.model_loader.get_backend", return_value="torch"),
        patch(
            "qwen3_tts.core.engine.model_loader._load_model_torch",
            return_value=mock_model,
        ) as mock_torch,
        patch("qwen3_tts.core.engine.model_loader._warmup_model"),
    ):
        result = load_model("design")

    mock_torch.assert_called_once_with("design")
    assert result == mock_model


@pytest.mark.unit
def test_load_model_warmup_false_skips_warmup():
    """load_model(warmup=False) defers warm-up to the caller (issue #192).

    The server layer loads unlocked, then runs the warm-up inference under
    inference_lock — the engine must not run the warm-up itself when the
    caller has taken over that responsibility.
    """
    from qwen3_tts.core.engine.model_loader import load_model

    mock_model = MagicMock()
    with (
        patch("qwen3_tts.core.engine.model_loader.get_backend", return_value="mlx"),
        patch(
            "qwen3_tts.core.engine.model_loader._load_model_mlx",
            return_value=mock_model,
        ),
        patch("qwen3_tts.core.engine.model_loader._warmup_model") as mock_warm,
    ):
        result = load_model("design", warmup=False)

    mock_warm.assert_not_called()
    assert result == mock_model


@pytest.mark.unit
def test_load_model_warmup_kwarg_is_keyword_only():
    """warmup is keyword-only — call sites cannot silently pass it positionally."""
    from qwen3_tts.core.engine.model_loader import load_model

    with pytest.raises(TypeError):
        load_model("design", False)


# ---- _load_model_mlx revision wiring (backlog Step 3B / master-plan M7) ----


class TestMlxRevisionWiring(unittest.TestCase):
    """models.<type>.revision must reach the mlx_audio load call.

    The torch path threads it end-to-end (_load_model_torch ->
    from_pretrained/_is_model_cached/_patch_tokenizer); the MLX path
    dropped it, so a pinned revision was silently ignored on the MLX
    backend — the model always tracked the moving default branch.
    mlx_audio's load_model forwards ``revision`` into its hub download
    (base_load_model kwargs), so wiring it in is supported upstream.
    """

    def _load_mlx_with_revision(self, model_type, revision):
        """Run _load_model_mlx with a stubbed mlx_audio and patched resolvers."""
        from qwen3_tts.core.engine.model_loader import _load_model_mlx

        mlx_utils = MagicMock()
        with (
            patch.dict(
                sys.modules,
                {
                    "mlx_audio": MagicMock(),
                    "mlx_audio.tts": MagicMock(),
                    "mlx_audio.tts.utils": mlx_utils,
                },
            ),
            patch(
                "qwen3_tts.core.engine.model_loader.get_mlx_model_name",
                return_value="mlx-community/test-repo",
            ),
            patch(
                "qwen3_tts.core.engine.model_loader.get_model_size",
                return_value="1.7B",
            ),
            patch(
                "qwen3_tts.core.engine.model_loader.get_model_revision",
                return_value=revision,
            ),
        ):
            _load_model_mlx(model_type)
        return mlx_utils.load_model

    def test_pinned_revision_reaches_mlx_load_call(self):
        """All three model types forward the resolved revision to mlx_audio."""
        for model_type in ("clone", "design", "custom"):
            with self.subTest(model_type=model_type):
                load = self._load_mlx_with_revision(model_type, "deadbeef")
                load.assert_called_once()
                self.assertEqual(load.call_args.args[0], "mlx-community/test-repo")
                self.assertEqual(load.call_args.kwargs.get("revision"), "deadbeef")

    def test_default_main_revision_is_passed_through(self):
        """Unpinned resolution passes revision='main' explicitly, never None."""
        load = self._load_mlx_with_revision("clone", "main")
        self.assertEqual(load.call_args.kwargs.get("revision"), "main")


# ---- _apply_cuda_optimizations additional coverage ----


@pytest.mark.unit
def test_cuda_optimizations_no_cuda():
    """_apply_cuda_optimizations returns defaults when no CUDA."""
    from qwen3_tts.core.engine.model_loader import _apply_cuda_optimizations

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float32 = "float32"

    with patch.dict(sys.modules, {"torch": mock_torch}):
        attn, dtype, compile_ = _apply_cuda_optimizations({})

    assert attn == "sdpa"
    assert dtype == "float32"
    assert compile_ is False


@pytest.mark.unit
def test_cuda_optimizations_turing():
    """_apply_cuda_optimizations on Turing uses float16 and no compile by default."""
    from qwen3_tts.core.engine.model_loader import _apply_cuda_optimizations

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_capability.return_value = (7, 5)
    mock_torch.float16 = "float16"
    mock_torch.backends.cudnn = MagicMock()

    with patch.dict(sys.modules, {"torch": mock_torch}):
        attn, dtype, compile_ = _apply_cuda_optimizations({})

    assert attn == "sdpa"
    assert dtype == "float16"
    assert compile_ is False


@pytest.mark.unit
def test_cuda_optimizations_ampere_defaults_to_sdpa():
    """PRF-4: Ampere+ defaults to sdpa even when flash_attn is installed."""
    from qwen3_tts.core.engine.model_loader import _apply_cuda_optimizations

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_capability.return_value = (8, 0)
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.backends.cudnn = MagicMock()

    with patch.dict(sys.modules, {"torch": mock_torch}):
        with patch("qwen3_tts.core.config._has_flash_attn", return_value=True):
            attn, dtype, compile_ = _apply_cuda_optimizations({})

    assert attn == "sdpa"
    assert dtype == "bfloat16"


@pytest.mark.unit
def test_cuda_optimizations_ampere_fa2_optin():
    """PRF-4: flash_attention_2 only when advanced.attn_implementation opts in."""
    from qwen3_tts.core.engine.model_loader import _apply_cuda_optimizations

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_capability.return_value = (8, 9)
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.backends.cudnn = MagicMock()

    config = {"advanced": {"attn_implementation": "flash_attention_2"}}
    with patch.dict(sys.modules, {"torch": mock_torch}):
        with patch("qwen3_tts.core.config._has_flash_attn", return_value=True):
            attn, dtype, compile_ = _apply_cuda_optimizations(config)

    assert attn == "flash_attention_2"
    assert dtype == "bfloat16"


# ---- _retry_model_load ----


@pytest.mark.unit
def test_retry_model_load_immediate_success():
    """_retry_model_load returns on first success without retrying."""
    from qwen3_tts.core.engine.model_loader import _retry_model_load

    mock_fn = MagicMock(return_value="model")
    result = _retry_model_load(mock_fn, "clone", "repo/model")

    assert result == "model"
    mock_fn.assert_called_once()


@pytest.mark.unit
def test_retry_model_load_retries_on_oserror():
    """_retry_model_load retries on OSError."""
    from qwen3_tts.core.engine.model_loader import _retry_model_load

    mock_fn = MagicMock(side_effect=[OSError("disk"), OSError("net"), "model"])

    with patch("time.sleep"):
        result = _retry_model_load(mock_fn, "clone", "repo/model")

    assert result == "model"
    assert mock_fn.call_count == 3


@pytest.mark.unit
def test_retry_model_load_raises_after_exhaustion():
    """_retry_model_load raises after max retries."""
    from qwen3_tts.core.engine.model_loader import _retry_model_load

    mock_fn = MagicMock(side_effect=OSError("persistent failure"))

    with patch("time.sleep"), pytest.raises(OSError, match="persistent failure"):
        _retry_model_load(mock_fn, "clone", "repo/model")


# ---- R-18: Turing GPU torch_quantization (already implemented) ----


@pytest.mark.unit
def test_turing_respects_explicit_torch_quantization():
    """Turing GPUs should respect explicit torch_quantization setting (R-18).

    When torch_quantization is explicitly set in config (e.g., "none"),
    the auto-8-bit override on Turing GPUs should NOT be applied.
    """
    from qwen3_tts.core.engine.model_loader import _resolve_load_kwargs

    mock_torch = MagicMock()
    mock_torch.float16 = "float16_sentinel"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_capability.return_value = (7, 5)  # Turing T4

    # Test 1: Explicitly set to "none" should NOT get 8-bit override
    mock_config_none = {"advanced": {"torch_quantization": "none"}}
    with (
        patch.dict(sys.modules, {"torch": mock_torch}),
        patch(
            "qwen3_tts.core.engine.model_loader.load_config",
            return_value=mock_config_none,
        ),
    ):
        result_none = _resolve_load_kwargs(
            "none", "float16_sentinel", "cuda", "sdpa", "auto"
        )
        assert "load_in_8bit" not in result_none, (
            "Should not auto-override when explicitly set to none"
        )

    # Test 2: NOT explicitly set (key missing) SHOULD get 8-bit override
    mock_config_missing = {"advanced": {}}  # torch_quantization key missing
    with (
        patch.dict(sys.modules, {"torch": mock_torch}),
        patch(
            "qwen3_tts.core.engine.model_loader.load_config",
            return_value=mock_config_missing,
        ),
    ):
        result_missing = _resolve_load_kwargs(
            "none", "float16_sentinel", "cuda", "sdpa", "auto"
        )
        assert result_missing.get("load_in_8bit") is True, (
            "Should auto-enable 8-bit when not explicitly set"
        )


# ---- Step 4B.3 item 12: torch-path success arms ----
# TestCase classes so the batch runner collects them. The function-style
# tests above predate the batched-TestCase rule; converting them is
# finding-7's sweep, deliberately out of scope here.


class _RowSums:
    """Stand-in whose __eq__ returns an object with a controllable .any()."""

    def __init__(self, any_result):
        self._zero_rows = MagicMock()
        self._zero_rows.any.return_value = any_result

    def __eq__(self, other):
        return self._zero_rows


class TestResolveLoadKwargsQuantSuccessArms(unittest.TestCase):
    """4bit/8bit SUCCESS arms + bitsandbytes-missing arms (missed 242-267)."""

    def _resolve(self, quant, *, inject_none_bnb=False):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_transformers = MagicMock()
        modules = {"torch": fake_torch, "transformers": fake_transformers}
        if inject_none_bnb:
            modules["bitsandbytes"] = None  # sys.modules None -> ImportError
        else:
            modules["bitsandbytes"] = MagicMock()
        with patch.dict(sys.modules, modules), patch("sys.platform", "linux"):
            from qwen3_tts.core.engine.model_loader import _resolve_load_kwargs

            kwargs = _resolve_load_kwargs(
                quant, "dtype_sentinel", "cuda", "sdpa", "auto"
            )
        return kwargs, fake_transformers

    def test_4bit_success_builds_bnb_config(self):
        kwargs, fake_transformers = self._resolve("4bit")
        self.assertIs(
            kwargs["quantization_config"],
            fake_transformers.BitsAndBytesConfig.return_value,
        )
        ctor_kwargs = fake_transformers.BitsAndBytesConfig.call_args.kwargs
        self.assertTrue(ctor_kwargs["load_in_4bit"])
        self.assertIs(ctor_kwargs["bnb_4bit_compute_dtype"], "dtype_sentinel")
        self.assertTrue(ctor_kwargs["bnb_4bit_use_double_quant"])
        self.assertEqual(kwargs["attn_implementation"], "sdpa")
        self.assertEqual(kwargs["device_map"], "auto")

    def test_4bit_bitsandbytes_missing_raises(self):
        with self.assertRaisesRegex(RuntimeError, "bitsandbytes"):
            self._resolve("4bit", inject_none_bnb=True)

    def test_8bit_success_sets_load_in_8bit(self):
        kwargs, _ = self._resolve("8bit")
        self.assertIs(kwargs["load_in_8bit"], True)
        self.assertNotIn("quantization_config", kwargs)
        self.assertNotIn("dtype", kwargs)

    def test_8bit_bitsandbytes_missing_raises(self):
        with self.assertRaisesRegex(RuntimeError, "bitsandbytes"):
            self._resolve("8bit", inject_none_bnb=True)


class TestPatchDeepcopyForBnb(unittest.TestCase):
    """The bitsandbytes deepcopy workaround generator (missed 182-218)."""

    def test_yields_clean_when_no_candidate_module(self):
        from qwen3_tts.core.engine.model_loader import _patch_deepcopy_for_bnb

        with patch("importlib.import_module", side_effect=ImportError("nope")):
            with _patch_deepcopy_for_bnb():
                pass  # enters and exits cleanly, nothing patched

    def test_patches_and_restores_get_keys(self):
        from qwen3_tts.core.engine.model_loader import _patch_deepcopy_for_bnb

        fake_mod = MagicMock()
        original = MagicMock(name="original_get_keys")
        fake_mod.get_keys_to_not_convert = original
        with patch("importlib.import_module", return_value=fake_mod):
            with _patch_deepcopy_for_bnb():
                self.assertIsNot(fake_mod.get_keys_to_not_convert, original)
            self.assertIs(fake_mod.get_keys_to_not_convert, original)

    def test_safe_get_keys_converts_dict_views_to_lists(self):
        from qwen3_tts.core.engine.model_loader import _patch_deepcopy_for_bnb

        class _Model:
            def modules(self):
                return []

        model = _Model()
        model.keys_view = {"a": 1}.keys()
        model.values_view = {"b": 2}.values()
        original = MagicMock(return_value="orig-result")
        fake_mod = MagicMock()
        fake_mod.get_keys_to_not_convert = original
        with patch("importlib.import_module", return_value=fake_mod):
            with _patch_deepcopy_for_bnb():
                result = fake_mod.get_keys_to_not_convert(model)
        self.assertEqual(result, "orig-result")
        self.assertEqual(model.keys_view, ["a"])
        self.assertEqual(model.values_view, [2])

    def test_safe_get_keys_walks_submodules(self):
        from qwen3_tts.core.engine.model_loader import _patch_deepcopy_for_bnb

        child = MagicMock()
        child.attached_keys = {"c": 3}.keys()
        parent = MagicMock()
        parent.modules.return_value = [child]
        original = MagicMock(return_value="ok")
        fake_mod = MagicMock()
        fake_mod.get_keys_to_not_convert = original
        with patch("importlib.import_module", return_value=fake_mod):
            with _patch_deepcopy_for_bnb():
                fake_mod.get_keys_to_not_convert(parent)
        self.assertEqual(child.attached_keys, ["c"])


class TestSafeMultinomialArms(unittest.TestCase):
    """The _safe_multinomial interior installed by _install_mps_patch."""

    def setUp(self):
        from qwen3_tts.core.engine import model_loader

        self._model_loader = model_loader
        self._orig_installed = model_loader._mps_patch_installed
        model_loader._mps_patch_installed = False
        self.fake_torch = MagicMock()
        self.original = MagicMock(name="original_multinomial")
        self.fake_torch.multinomial = self.original
        with (
            patch.dict(sys.modules, {"torch": self.fake_torch}),
            patch("qwen3_tts.core.config.IS_MACOS", True),
        ):
            model_loader._install_mps_patch()
        self.wrapper = self.fake_torch.multinomial
        self.assertIsNot(self.wrapper, self.original)

    def tearDown(self):
        self._model_loader._mps_patch_installed = self._orig_installed

    def _mps_input(self, dtype):
        inp = MagicMock()
        inp.device.type = "mps"
        inp.is_floating_point.return_value = True
        inp.dtype = dtype
        return inp

    def _sanitized_chain(self, inp, any_result):
        sanitized = MagicMock(name="sanitized")
        sanitized.device.type = "mps"
        inp.float.return_value = sanitized
        self.fake_torch.nan_to_num.return_value = sanitized
        sanitized.clamp.return_value = sanitized
        sanitized.sum.return_value = _RowSums(any_result=any_result)
        return sanitized

    def test_mps_non_f32_casts_and_sanitizes(self):
        inp = self._mps_input(dtype=object())
        sanitized = self._sanitized_chain(inp, any_result=False)
        result = self.wrapper(inp, 7)
        self.assertIs(result, self.original.return_value)
        inp.float.assert_called_once()
        self.fake_torch.nan_to_num.assert_called_once_with(
            sanitized, nan=0.0, posinf=1.0, neginf=0.0
        )
        sanitized.clamp.assert_called_once_with(min=0.0)
        self.original.assert_called_once_with(
            sanitized, 7, replacement=False, generator=None
        )

    def test_mps_f32_skips_cast_still_sanitizes(self):
        inp = self._mps_input(dtype=self.fake_torch.float32)
        self._sanitized_chain(inp, any_result=False)
        self.wrapper(inp, 7)
        inp.float.assert_not_called()
        self.fake_torch.nan_to_num.assert_called_once()

    def test_mps_zero_rows_refilled_uniformly(self):
        inp = self._mps_input(dtype=self.fake_torch.float32)
        sanitized = self._sanitized_chain(inp, any_result=True)
        self.wrapper(inp, 7)
        sanitized.masked_fill.assert_called_once()
        self.original.assert_called_once()

    def test_non_mps_passthrough_untouched(self):
        inp = MagicMock()
        inp.device.type = "cpu"
        inp.is_floating_point.return_value = True
        result = self.wrapper(inp, 3)
        self.assertIs(result, self.original.return_value)
        inp.float.assert_not_called()
        self.fake_torch.nan_to_num.assert_not_called()
        self.original.assert_called_once_with(inp, 3, replacement=False, generator=None)


class TestLoadModelTorch(unittest.TestCase):
    """_load_model_torch end-to-end on fakes (missed 346-423)."""

    _ML = "qwen3_tts.core.engine.model_loader."

    def _run(
        self,
        *,
        quant="none",
        device="mps",
        cached=True,
        side_effect=None,
        extra_modules=None,
        assert_download_log=False,
    ):
        import contextlib
        from types import SimpleNamespace

        from qwen3_tts.core.engine import model_loader

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = device == "cuda"
        fake_qwen = MagicMock()
        model = MagicMock(name="loaded_model")
        fake_qwen.Qwen3TTSModel.from_pretrained.side_effect = (
            side_effect if side_effect is not None else [model]
        )
        modules = {"torch": fake_torch, "qwen_tts": fake_qwen}
        if extra_modules:
            modules.update(extra_modules)
        bnb_ctx = MagicMock()
        ml = self._ML
        cm = (
            self.assertLogs("tts.engine", level="INFO")
            if assert_download_log
            else contextlib.nullcontext()
        )
        with (
            cm as logs,
            patch.dict(sys.modules, modules),
            patch(ml + "get_torch_model_name", return_value="fake/repo"),
            patch(ml + "get_model_revision", return_value="rev1"),
            patch(ml + "get_model_size", return_value="1.7B"),
            patch(ml + "get_torch_dtype_name", return_value="float32"),
            patch(ml + "get_torch_quantization", return_value=quant),
            patch(ml + "load_config", return_value={}),
            patch(
                ml + "_apply_cuda_optimizations",
                return_value=("sdpa", "optimal_dtype", False),
            ),
            patch(ml + "_is_model_cached", return_value=cached),
            patch(ml + "_install_mps_patch") as mock_mps,
            patch(ml + "_apply_torch_compile", side_effect=lambda m, *a, **k: m),
            patch(ml + "_patch_tokenizer", side_effect=lambda m, *a, **k: m),
            patch(ml + "_retry_model_load", side_effect=lambda fn, *a, **k: fn()),
            patch(ml + "_patch_deepcopy_for_bnb", return_value=bnb_ctx) as mock_bnb,
            patch("qwen3_tts.core.config.get_device", return_value=device),
        ):
            result = model_loader._load_model_torch("clone")
        return SimpleNamespace(
            result=result,
            model=model,
            calls=fake_qwen.Qwen3TTSModel.from_pretrained.call_args_list,
            fake_torch=fake_torch,
            mock_mps=mock_mps,
            mock_bnb=mock_bnb,
            logs=logs,
        )

    def test_mps_happy_path(self):
        out = self._run()
        self.assertIs(out.result, out.model)
        out.mock_mps.assert_called_once()
        self.assertEqual(len(out.calls), 1)
        self.assertEqual(out.calls[0].args, ("fake/repo",))
        self.assertEqual(out.calls[0].kwargs["revision"], "rev1")
        self.assertIs(out.calls[0].kwargs["dtype"], out.fake_torch.float32)
        self.assertEqual(out.calls[0].kwargs["device_map"], "mps")
        self.assertEqual(out.calls[0].kwargs["attn_implementation"], "sdpa")

    def test_not_cached_logs_download(self):
        out = self._run(cached=False, assert_download_log=True)
        self.assertIs(out.result, out.model)
        self.assertTrue(
            any("Downloading" in line for line in out.logs.output),
            out.logs.output,
        )

    def test_typeerror_dict_keys_retries_without_quant(self):
        model2 = MagicMock(name="reloaded_model")
        out = self._run(
            side_effect=[TypeError("cannot pickle 'dict_keys' object"), model2]
        )
        self.assertIs(out.result, model2)
        self.assertEqual(len(out.calls), 2)
        # Retry keeps the plain dtype path (quant kwargs dropped if present).
        self.assertIn("dtype", out.calls[1].kwargs)
        self.assertNotIn("load_in_8bit", out.calls[1].kwargs)
        self.assertNotIn("quantization_config", out.calls[1].kwargs)

    def test_8bit_cuda_uses_bnb_ctx(self):
        out = self._run(
            quant="8bit",
            device="cuda",
            extra_modules={
                "bitsandbytes": MagicMock(),
                "transformers": MagicMock(),
            },
        )
        self.assertIs(out.result, out.model)
        out.mock_bnb.assert_called_once()
        self.assertIs(out.calls[0].kwargs["load_in_8bit"], True)
        self.assertEqual(out.calls[0].kwargs["device_map"], "auto")
        self.assertNotIn("dtype", out.calls[0].kwargs)
