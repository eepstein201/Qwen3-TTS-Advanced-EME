#!/usr/bin/env python3
"""Model loading for torch and MLX backends.

Imports from: config only. Does NOT import from inference or voice_prompt.
"""

import contextlib
import logging
import os
import threading
import time

from qwen3_tts.core.config import (
    get_backend,
    get_mlx_model_name,
    get_model_revision,
    get_model_size,
    get_torch_dtype_name,
    get_torch_model_name,
    get_torch_quantization,
    load_config,
    sanitize_log,
)

logger = logging.getLogger("tts.engine")

_RETRY_DELAYS = (5, 15, 45)  # seconds between retry attempts


# ---------------------------------------------------------------------------
# MPS bfloat16 safety patch (installed once on first torch backend use)
# ---------------------------------------------------------------------------

_mps_patch_installed = False
_mps_patch_lock = threading.Lock()


def _install_mps_patch():
    """Install the MPS-safe multinomial patch.

    Called once on first torch backend use. Patches torch.multinomial to
    cast to float32 and sanitize NaN/Inf before sampling on MPS devices.
    Only runs on macOS — skipped on Linux/Colab where MPS is not available.
    """
    global _mps_patch_installed
    if _mps_patch_installed:
        return

    with _mps_patch_lock:
        # Double-check inside lock
        if _mps_patch_installed:
            return

        from qwen3_tts.core.config import IS_MACOS

        if not IS_MACOS:
            _mps_patch_installed = True  # Mark as done, no patch needed
            return

        import torch

        _original_multinomial = torch.multinomial

        def _safe_multinomial(input, num_samples, replacement=False, *, generator=None):
            if (
                input.device.type == "mps"
                and input.is_floating_point()
                and input.dtype != torch.float32
            ):
                input = input.float()
            if input.device.type == "mps":
                input = torch.nan_to_num(input, nan=0.0, posinf=1.0, neginf=0.0)
                input = input.clamp(min=0.0)
                row_sums = input.sum(dim=-1, keepdim=True)
                zero_rows = row_sums == 0
                if zero_rows.any():
                    input = input.masked_fill(
                        zero_rows.expand_as(input), 1.0 / input.shape[-1]
                    )
            return _original_multinomial(
                input, num_samples, replacement=replacement, generator=generator
            )

        torch.multinomial = _safe_multinomial
        _mps_patch_installed = True
        logger.debug("Installed MPS-safe multinomial patch")


# ---------------------------------------------------------------------------
# CUDA optimizations
# ---------------------------------------------------------------------------


def _apply_cuda_optimizations(config):
    """Detect CUDA hardware and return optimal settings for model loading.

    Returns:
        (attn_impl, optimal_dtype, should_compile) tuple.
    """
    import torch

    if not torch.cuda.is_available():
        return ("sdpa", torch.float32, False)

    capability = torch.cuda.get_device_capability()

    # Always apply these on CUDA
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    if capability[0] >= 8:
        # Ampere+ (A100, A10G, RTX 30xx, etc.)
        from qwen3_tts.core.config import (
            _has_flash_attn,
            _resolve_attn_implementation,
        )

        # SDPA is the safe default (upstream #333: NaN logits with
        # flash_attention_2 for Qwen3-TTS on L4/A100). FA2 is opt-in via
        # advanced.attn_implementation and only when flash_attn is installed.
        preference = config.get("advanced", {}).get("attn_implementation", "auto")
        has_flash = _has_flash_attn()
        attn_impl = _resolve_attn_implementation(preference, has_flash)
        if attn_impl != "flash_attention_2":
            logger.info(
                "Using SDPA attention on Ampere+ (FA2 is opt-in via "
                "advanced.attn_implementation); upstream #333 reports NaN "
                "logits with flash_attention_2 for Qwen3-TTS."
            )
        should_compile = config.get("generation", {}).get("compile_model", True)
        return (attn_impl, torch.bfloat16, should_compile)
    else:
        # Turing / T4 (capability 7.x)
        should_compile = config.get("generation", {}).get("compile_model", False)
        return ("sdpa", torch.float16, should_compile)


# ---------------------------------------------------------------------------
# Shared retry helper
# ---------------------------------------------------------------------------


def _retry_model_load(loader_fn, model_type: str, model_name: str):
    """Invoke loader_fn(), retrying on transient IO/network errors with exponential backoff."""
    last_error = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            return loader_fn()
        except (OSError, ConnectionError, TimeoutError) as e:
            last_error = e
            if attempt < len(_RETRY_DELAYS):
                delay = _RETRY_DELAYS[attempt]
                logger.warning(
                    "Model load attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1,
                    len(_RETRY_DELAYS) + 1,
                    e,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Model download failed after %d attempts. "
                    "Check your internet connection.",
                    len(_RETRY_DELAYS) + 1,
                )
                raise last_error


# ---------------------------------------------------------------------------
# bitsandbytes deepcopy workaround (qwen-tts dict_keys pickling bug)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _patch_deepcopy_for_bnb():
    """Temporarily patch get_keys_to_not_convert to handle dict_keys attributes.

    qwen-tts stores dict_keys views as model attributes which causes
    TypeError during deepcopy in transformers <= 4.x bitsandbytes path.
    """
    import importlib

    target_module = None
    original_fn = None
    for module_path in (
        "transformers.integrations.bitsandbytes",
        "transformers.quantizers.base",
    ):
        try:
            mod = importlib.import_module(module_path)
            if hasattr(mod, "get_keys_to_not_convert"):
                target_module = mod
                original_fn = mod.get_keys_to_not_convert
                break
        except ImportError:
            continue

    if original_fn is None:
        yield
        return

    _dict_keys_type = type({}.keys())
    _dict_values_type = type({}.values())

    def _safe_get_keys(model):
        for obj in [model, *model.modules()]:
            for attr_name in list(vars(obj)):
                val = getattr(obj, attr_name, None)
                if isinstance(val, (_dict_keys_type, _dict_values_type)):
                    setattr(obj, attr_name, list(val))
        return original_fn(model)

    target_module.get_keys_to_not_convert = _safe_get_keys
    try:
        yield
    finally:
        target_module.get_keys_to_not_convert = original_fn


# ---------------------------------------------------------------------------
# Torch backend — model loading helpers (H5)
# ---------------------------------------------------------------------------


def _resolve_load_kwargs(
    torch_quant: str, torch_dtype, device: str, attn_impl: str, device_map: str
) -> dict:
    """Build the from_pretrained kwargs dict based on quantization setting."""
    import sys

    import torch

    load_kwargs: dict = dict(attn_implementation=attn_impl, device_map=device_map)

    if torch_quant == "4bit":
        if not (torch.cuda.is_available() and sys.platform.startswith("linux")):
            raise RuntimeError(
                "4-bit quantization requires CUDA on Linux. "
                "Set torch_quantization to 'none' or '8bit', or use a different backend."
            )
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise RuntimeError(
                f"4-bit quantization requires bitsandbytes. Install with: pip install bitsandbytes. Error: {e}"
            )
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using 4-bit quantization (bitsandbytes NF4)")
    elif torch_quant == "8bit":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "8-bit quantization requires CUDA. Set torch_quantization to 'none' or use a different backend."
            )
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                f"8-bit quantization requires bitsandbytes. Install with: pip install bitsandbytes. Error: {e}"
            )
        load_kwargs["load_in_8bit"] = True
        logger.info("Using 8-bit quantization (bitsandbytes)")
    else:
        load_kwargs["dtype"] = torch_dtype
        if device == "cuda":
            cap = torch.cuda.get_device_capability()
            explicitly_set = "torch_quantization" in load_config().get("advanced", {})
            if cap[0] < 8 and not explicitly_set:
                load_kwargs["load_in_8bit"] = True
                logger.info(
                    "Auto-enabled 8-bit quantization for compute capability %s "
                    "(override: set advanced.torch_quantization in config)",
                    cap,
                )
            elif cap[0] < 8 and explicitly_set:
                logger.info(
                    "Turing GPU (compute %s) but torch_quantization explicitly set to '%s' "
                    "— respecting user config",
                    cap,
                    torch_quant,
                )

    return load_kwargs


def _is_model_cached(repo_id: str, revision: str = "main") -> bool:
    """Return True if the model snapshot is already in the local HF cache."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id,
            revision=revision,
            local_files_only=True,
            allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"],
        )
        return True
    except Exception as e:
        logger.debug("Cache check failed for %s: %s", repo_id, e)
        return False


def _apply_torch_compile(model, model_type: str, device: str, should_compile: bool):
    """Apply torch.compile to the model's inner nn.Module if conditions are met."""
    import torch

    if should_compile and device == "cuda":
        try:
            logger.info(
                "Applying torch.compile (reduce-overhead) to %s model", model_type
            )
            model.model = torch.compile(model.model, mode="reduce-overhead")
        except Exception as e:
            logger.warning("torch.compile failed (%s) — running without compilation", e)
    return model


def _patch_tokenizer(model, repo_id: str, revision: str = "main"):
    """Reload tokenizer with fix_mistral_regex=True if the transformers version supports it."""
    try:
        from transformers import AutoTokenizer

        model.tokenizer = AutoTokenizer.from_pretrained(
            repo_id, revision=revision, fix_mistral_regex=True
        )
    except TypeError:
        pass  # Older transformers doesn't support fix_mistral_regex
    return model


# ---------------------------------------------------------------------------
# Torch backend — model loading
# ---------------------------------------------------------------------------


def _load_model_torch(model_type):
    """Load a TTS model using the PyTorch/MPS backend.

    Retries up to 3 times with exponential backoff on download/load failures.
    """
    import torch
    from qwen_tts import Qwen3TTSModel

    _install_mps_patch()

    repo_id = get_torch_model_name(model_type)
    revision = get_model_revision(model_type)
    model_size = get_model_size()
    dtype_name = get_torch_dtype_name()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    attn_impl, optimal_dtype, should_compile = _apply_cuda_optimizations(load_config())
    torch_quant = get_torch_quantization()

    logger.info(
        "Loading %s (%s) with dtype=%s, quant=%s, size=%s [torch backend]...",
        sanitize_log(model_type),
        sanitize_log(repo_id),
        sanitize_log(dtype_name),
        sanitize_log(torch_quant),
        sanitize_log(model_size),
    )
    t0 = time.time()

    def _do_load():
        from qwen3_tts.core.config import get_device

        device = get_device()
        torch_dtype = optimal_dtype if device == "cuda" else dtype_map[dtype_name]
        device_map = "auto" if device == "cuda" else device

        load_kwargs = _resolve_load_kwargs(
            torch_quant, torch_dtype, device, attn_impl, device_map
        )

        if not _is_model_cached(repo_id, revision):
            logger.info(
                "Downloading %s model (this may take several minutes on first run)...",
                model_type,
            )
            logger.info(
                "Model size: ~%s — ensure stable internet connection",
                "3.5GB" if model_size == "1.7B" else "2GB",
            )

        uses_bnb = "load_in_8bit" in load_kwargs or "quantization_config" in load_kwargs
        ctx = _patch_deepcopy_for_bnb() if uses_bnb else contextlib.nullcontext()

        try:
            with ctx:
                model = Qwen3TTSModel.from_pretrained(
                    repo_id, revision=revision, **load_kwargs
                )
        except TypeError as e:
            if "pickle" not in str(e) and "dict_keys" not in str(e):
                raise
            logger.warning(
                "Quantization loading failed (TypeError: %s). "
                "Retrying without quantization — model will use more VRAM.",
                e,
            )
            load_kwargs.pop("load_in_8bit", None)
            load_kwargs.pop("quantization_config", None)
            if "dtype" not in load_kwargs:
                load_kwargs["dtype"] = torch_dtype
            model = Qwen3TTSModel.from_pretrained(
                repo_id, revision=revision, **load_kwargs
            )

        model = _apply_torch_compile(model, model_type, device, should_compile)
        model = _patch_tokenizer(model, repo_id, revision)
        logger.info("Loaded %s model in %.1fs", model_type, time.time() - t0)
        return model

    return _retry_model_load(_do_load, model_type, repo_id)


# ---------------------------------------------------------------------------
# MLX backend — model loading
# ---------------------------------------------------------------------------


def _load_model_mlx(model_type):
    """Load a TTS model using the MLX backend.

    Uses mlx-community quantized models via mlx_audio.tts.utils.load_model.
    Retries up to 3 times with exponential backoff on download/load failures.
    """
    import warnings

    try:
        from mlx_audio.tts.utils import load_model as mlx_load_model
    except ImportError:
        raise ImportError(
            "MLX backend selected but mlx-audio is not installed. "
            "Activate the MLX environment: conda activate qwen3-tts-mlx\n"
            "Or install dependencies: pip install -e .[mlx]"
        )

    repo_id = get_mlx_model_name(model_type)
    model_size = get_model_size()
    logger.info(
        "Loading %s (%s) size=%s [mlx backend]...",
        sanitize_log(model_type),
        sanitize_log(repo_id),
        sanitize_log(model_size),
    )
    t0 = time.time()

    def _do_load():
        # Suppress spurious Mistral tokenizer regex warning for Qwen3-TTS models.
        # This warning is triggered by transformers for non-Mistral models that share
        # similar tokenizer patterns. Qwen3-TTS is NOT a Mistral model and does not
        # need the fix_mistral_regex flag. See: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*incorrect regex pattern.*fix_mistral_regex.*",
                category=UserWarning,
            )
            model = mlx_load_model(repo_id)
        elapsed = time.time() - t0
        logger.info("Loaded %s model in %.1fs [mlx]", model_type, elapsed)
        return model

    return _retry_model_load(_do_load, model_type, repo_id)


# ---------------------------------------------------------------------------
# Model warm-up
# ---------------------------------------------------------------------------


def _warmup_disabled() -> bool:
    """True when the TTS_SKIP_WARMUP ablation knob is set (#192).

    Read at call time (not import time) so per-process env changes apply.
    The server's warm-up serialization guards call this BEFORE acquiring
    inference_lock, so ablation runs don't queue loads behind generations
    just to no-op.
    """
    return os.environ.get("TTS_SKIP_WARMUP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _warmup_model(model, model_type, backend):
    """Run a short warm-up inference to compile kernels. Non-fatal.

    Only warms up design models — generate_voice_design() is only supported
    by VoiceDesign model weights. Clone and Custom models use Base weights
    that do not support this method and require a voice prompt to generate.
    """
    if model_type != "design":
        return  # Base/Clone and Custom weights don't support generate_voice_design

    if _warmup_disabled():
        # Issue #192 ablation/mitigation knob: skip the load-time warm-up
        # inference. The server normally runs this warm-up serialized on
        # inference_lock (app_models/app_lifespan); /transcribe and
        # /create-voice-prompt inference serialize on it too — all MLX
        # inference reachable through the API is covered. Logged
        # positively — a run's log must record the
        # knob was active, because the absence of warm-up lines alone
        # proves nothing.
        logger.info("Skipping %s warm-up (TTS_SKIP_WARMUP set)", model_type)
        return

    try:
        t0 = time.time()
        logger.info("Running warm-up inference for %s model...", model_type)
        if backend == "mlx":
            list(
                model.generate_voice_design(
                    text="Hello.",
                    instruct="Speak normally.",
                    language="English",
                    temperature=0.5,
                )
            )
        else:
            import torch

            with torch.inference_mode():
                model.generate_voice_design(
                    text="Hello.",
                    instruct="Speak normally.",
                    language="English",
                    temperature=0.5,
                    max_new_tokens=50,
                )
        logger.info("Warm-up complete for %s in %.1fs", model_type, time.time() - t0)
    except Exception as e:
        logger.warning("Warm-up failed for %s (non-fatal): %s", model_type, e)


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------


def load_model(model_type, *, warmup: bool = True):
    """Load a TTS model by type, dispatching to the configured backend.

    After loading, runs a short warm-up inference to pre-compile kernels
    and reduce cold-start latency on the first real generation.

    Args:
        model_type: One of "clone", "design", "custom".
        warmup: Run the warm-up inference here. The server passes False and
            runs the warm-up itself under inference_lock — the warm-up is
            real MLX inference and must not execute concurrently with a
            generation (issue #192).

    Returns:
        The loaded model instance (type depends on backend).
    """
    backend = get_backend()
    if backend == "mlx":
        model = _load_model_mlx(model_type)
    else:
        model = _load_model_torch(model_type)
    if warmup:
        _warmup_model(model, model_type, backend)
    return model
