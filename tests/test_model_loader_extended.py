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

from unittest.mock import patch, MagicMock


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

        with patch("qwen3_tts.core.config.IS_MACOS", True), \
             patch.dict(sys.modules, {"torch": mock_torch}):
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

    with patch.dict(sys.modules, {"torch": mock_torch}), \
         pytest.raises(RuntimeError, match="8-bit quantization requires CUDA"):
        _resolve_load_kwargs("8bit", "float16", "cpu", "sdpa", "auto")


@pytest.mark.unit
def test_resolve_load_kwargs_4bit_no_linux():
    """_resolve_load_kwargs raises for 4bit without CUDA+Linux."""
    from qwen3_tts.core.engine.model_loader import _resolve_load_kwargs

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with patch.dict(sys.modules, {"torch": mock_torch}), \
         patch("sys.platform", "darwin"), \
         pytest.raises(RuntimeError, match="4-bit quantization requires CUDA on Linux"):
        _resolve_load_kwargs("4bit", "float16", "cuda", "sdpa", "auto")


@pytest.mark.unit
def test_resolve_load_kwargs_auto_8bit_turing():
    """_resolve_load_kwargs auto-enables 8bit on Turing GPUs when not explicitly set."""
    from qwen3_tts.core.engine.model_loader import _resolve_load_kwargs

    mock_torch = MagicMock()
    mock_torch.float16 = "float16_sentinel"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_capability.return_value = (7, 5)

    with patch.dict(sys.modules, {"torch": mock_torch}), \
         patch("qwen3_tts.core.engine.model_loader.load_config", return_value={"advanced": {}}):
        result = _resolve_load_kwargs("none", "float16_sentinel", "cuda", "sdpa", "auto")

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
    with patch.dict(sys.modules, {"torch": mock_torch}), \
         patch("qwen3_tts.core.engine.model_loader.load_config", return_value=config):
        result = _resolve_load_kwargs("none", "float16_sentinel", "cuda", "sdpa", "auto")

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
    mock_transformers.AutoTokenizer.from_pretrained.side_effect = TypeError("unexpected keyword")

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


# ---- load_model dispatch ----

@pytest.mark.unit
def test_load_model_dispatches_mlx():
    """load_model dispatches to _load_model_mlx when backend is mlx."""
    from qwen3_tts.core.engine.model_loader import load_model

    mock_model = MagicMock()
    with patch("qwen3_tts.core.engine.model_loader.get_backend", return_value="mlx"), \
         patch("qwen3_tts.core.engine.model_loader._load_model_mlx", return_value=mock_model) as mock_mlx, \
         patch("qwen3_tts.core.engine.model_loader._warmup_model"):
        result = load_model("clone")

    mock_mlx.assert_called_once_with("clone")
    assert result == mock_model


@pytest.mark.unit
def test_load_model_dispatches_torch():
    """load_model dispatches to _load_model_torch when backend is torch."""
    from qwen3_tts.core.engine.model_loader import load_model

    mock_model = MagicMock()
    with patch("qwen3_tts.core.engine.model_loader.get_backend", return_value="torch"), \
         patch("qwen3_tts.core.engine.model_loader._load_model_torch", return_value=mock_model) as mock_torch, \
         patch("qwen3_tts.core.engine.model_loader._warmup_model"):
        result = load_model("design")

    mock_torch.assert_called_once_with("design")
    assert result == mock_model


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

    assert mock_fn.call_count == 4  # 1 initial + 3 retries
