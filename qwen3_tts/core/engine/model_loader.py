#!/usr/bin/env python3
"""Model loading for torch and MLX backends.

Imports from: config only. Does NOT import from inference or voice_prompt.
"""

import logging
import threading
import time

from qwen3_tts.core.config import (
    get_backend,
    get_mlx_model_name,
    get_model_size,
    get_torch_dtype_name,
    get_torch_model_name,
    get_torch_quantization,
    load_config,
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
            if input.device.type == "mps" and input.is_floating_point() and input.dtype != torch.float32:
                input = input.float()
            if input.device.type == "mps":
                input = torch.nan_to_num(input, nan=0.0, posinf=1.0, neginf=0.0)
                input = input.clamp(min=0.0)
                row_sums = input.sum(dim=-1, keepdim=True)
                zero_rows = (row_sums == 0)
                if zero_rows.any():
                    input = input.masked_fill(zero_rows.expand_as(input), 1.0 / input.shape[-1])
            return _original_multinomial(input, num_samples, replacement=replacement, generator=generator)

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
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    if capability[0] >= 8:
        # Ampere+ (A100, A10G, RTX 30xx, etc.)
        from qwen3_tts.core.config import _has_flash_attn
        if _has_flash_attn():
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "sdpa"
            logger.info("flash_attn not installed — using SDPA attention (still fast on Ampere+)")
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
                    attempt + 1, len(_RETRY_DELAYS) + 1, e, delay,
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
# Torch backend — model loading
# ---------------------------------------------------------------------------

def _load_model_torch(model_type):
    """Load a TTS model using the PyTorch/MPS backend.

    Retries up to 3 times with exponential backoff on download/load failures.
    """
    import torch
    import sys
    from qwen_tts import Qwen3TTSModel

    _install_mps_patch()

    repo_id = get_torch_model_name(model_type)
    model_size = get_model_size()

    dtype_name = get_torch_dtype_name()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    attn_impl, optimal_dtype, should_compile = _apply_cuda_optimizations(load_config())

    # Read torch_quantization setting
    torch_quant = get_torch_quantization()

    logger.info("Loading %s (%s) with dtype=%s, quant=%s, size=%s [torch backend]...",
                model_type, repo_id, dtype_name, torch_quant, model_size)
    t0 = time.time()

    def _do_load():
        from qwen3_tts.core.config import get_device
        device = get_device()
        # Override dtype with CUDA-optimal dtype when on CUDA
        torch_dtype = optimal_dtype if device == "cuda" else dtype_map[dtype_name]
        # CUDA uses "auto" for multi-GPU support; MPS/CPU use device name directly
        device_map = "auto" if device == "cuda" else device

        # Build load_kwargs based on quantization setting
        load_kwargs = dict(
            attn_implementation=attn_impl,
            device_map=device_map,
        )

        # Handle 4-bit quantization (requires bitsandbytes on CUDA/Linux)
        if torch_quant == "4bit":
            if not (torch.cuda.is_available() and sys.platform.startswith("linux")):
                raise RuntimeError(
                    "4-bit quantization requires CUDA on Linux. "
                    "Set torch_quantization to 'none' or '8bit', or use a different backend."
                )
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401 - Verify import
            except ImportError as e:
                raise RuntimeError(
                    "4-bit quantization requires bitsandbytes. "
                    f"Install with: pip install bitsandbytes. Error: {e}"
                )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization (bitsandbytes NF4)")
        # Handle 8-bit quantization
        elif torch_quant == "8bit":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "8-bit quantization requires CUDA. "
                    "Set torch_quantization to 'none' or use a different backend."
                )
            try:
                import bitsandbytes  # noqa: F401 - Verify import
            except ImportError as e:
                raise RuntimeError(
                    "8-bit quantization requires bitsandbytes. "
                    f"Install with: pip install bitsandbytes. Error: {e}"
                )
            load_kwargs["load_in_8bit"] = True
            logger.info("Using 8-bit quantization (bitsandbytes)")
        # No quantization (torch_quant == "none")
        else:
            load_kwargs["dtype"] = torch_dtype
            # Auto-enable 8-bit on older CUDA GPUs (Turing/T4) for memory efficiency,
            # but only if user hasn't explicitly set torch_quantization (R-18)
            if device == "cuda":
                cap = torch.cuda.get_device_capability()
                user_config = load_config()
                explicitly_set = "torch_quantization" in user_config.get("advanced", {})
                if cap[0] < 8 and not explicitly_set:
                    load_kwargs["load_in_8bit"] = True
                    logger.info(
                        "Auto-enabled 8-bit quantization for compute capability %s "
                        "(override: set advanced.torch_quantization in config)", cap,
                    )
                elif cap[0] < 8 and explicitly_set:
                    logger.info(
                        "Turing GPU (compute %s) but torch_quantization explicitly "
                        "set to '%s' — respecting user config", cap, torch_quant,
                    )

        # Check if model is already cached (for better user feedback)
        from huggingface_hub import snapshot_download
        try:
            # Try to get snapshot info without downloading
            snapshot_download(repo_id, local_files_only=True, allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"])
            model_cached = True
        except Exception:
            model_cached = False

        if not model_cached:
            logger.info("Downloading %s model (this may take several minutes on first run)...", model_type)
            logger.info("Model size: ~%s — ensure stable internet connection",
                        "3.5GB" if model_size == "1.7B" else "2GB")

        model = Qwen3TTSModel.from_pretrained(repo_id, **load_kwargs)
        # Apply torch.compile to inner nn.Module for supported CUDA hardware
        if should_compile and device == "cuda":
            try:
                logger.info("Applying torch.compile (reduce-overhead) to %s model", model_type)
                model.model = torch.compile(model.model, mode="reduce-overhead")
            except Exception as e:
                logger.warning("torch.compile failed (%s) — running without compilation", e)
        # Fix tokenizer regex if supported
        try:
            from transformers import AutoTokenizer
            model.tokenizer = AutoTokenizer.from_pretrained(repo_id, fix_mistral_regex=True)  # nosec B615
        except TypeError:
            pass  # Older transformers doesn't support fix_mistral_regex
        elapsed = time.time() - t0
        logger.info("Loaded %s model in %.1fs", model_type, elapsed)
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
    logger.info("Loading %s (%s) size=%s [mlx backend]...", model_type, repo_id, model_size)
    t0 = time.time()

    def _do_load():
        model = mlx_load_model(repo_id)
        elapsed = time.time() - t0
        logger.info("Loaded %s model in %.1fs [mlx]", model_type, elapsed)
        return model

    return _retry_model_load(_do_load, model_type, repo_id)


# ---------------------------------------------------------------------------
# Model warm-up
# ---------------------------------------------------------------------------

def _warmup_model(model, model_type, backend):
    """Run a short warm-up inference to compile kernels. Non-fatal.

    Uses generate_voice_design which doesn't require a voice prompt,
    making it work for all model types. The warm-up output is discarded.
    """
    try:
        t0 = time.time()
        logger.info("Running warm-up inference for %s model...", model_type)
        if backend == "mlx":
            list(model.generate_voice_design(
                text="Hello.",
                instruct="Speak normally.",
                language="English",
                temperature=0.5,
                max_new_tokens=50,
            ))
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

def load_model(model_type):
    """Load a TTS model by type, dispatching to the configured backend.

    After loading, runs a short warm-up inference to pre-compile kernels
    and reduce cold-start latency on the first real generation.

    Args:
        model_type: One of "clone", "design", "custom".

    Returns:
        The loaded model instance (type depends on backend).
    """
    backend = get_backend()
    if backend == "mlx":
        model = _load_model_mlx(model_type)
    else:
        model = _load_model_torch(model_type)
    _warmup_model(model, model_type, backend)
    return model
