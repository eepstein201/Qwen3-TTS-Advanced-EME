#!/usr/bin/env python3
"""TTS inference engine with backend dispatch (torch or mlx).

This module owns:
- Model loading (clone / design / custom) dispatched by backend
- Core inference: run_inference() with backend dispatch
- Voice-prompt creation from reference audio
- Audio post-processing (trim, normalize, speed, pitch)
- Voice prompt loading with LRU cache

IMPORTANT: Neither torch nor mlx is imported at module scope.
All backend-specific imports are local to _*_torch() or _*_mlx() functions.
"""

import logging
import os
import threading
import time
from functools import lru_cache

import numpy as np

from voice_config import (
    VOICE_PROMPTS_DIR,
    get_torch_dtype_name,
    get_backend,
    get_mlx_model_name,
    get_torch_model_name,
    get_model_size,
    get_model_info,
    CONFIG_PATH,
    load_config,
)

logger = logging.getLogger("tts.engine")

# Audio loader preference — read once at import, updated only via set_audio_loader()
_AUDIO_LOADER = load_config().get("advanced", {}).get("audio_loader", "torchaudio")


def get_audio_loader():
    """Return cached audio loader preference. No disk I/O."""
    return _AUDIO_LOADER


def set_audio_loader(loader):
    """Update audio loader preference in memory (called by config update endpoints)."""
    global _AUDIO_LOADER
    if loader not in ("torchaudio", "librosa"):
        raise ValueError(f"Invalid audio loader: {loader}")
    _AUDIO_LOADER = loader


# ---------------------------------------------------------------------------
# Text chunking for long-form reliability
# ---------------------------------------------------------------------------

def _split_text(text, max_chars=500):
    """Split text into chunks at sentence boundaries.

    Splits on sentence-ending punctuation (. ! ?) followed by whitespace,
    or on newlines. If a single sentence exceeds max_chars, falls back to
    clause boundaries (, ; — :). Never splits mid-word.

    Args:
        text: Input text to split.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks. Returns [text] unchanged if len(text) <= max_chars.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    import re

    # Split on sentence boundaries: . ! ? followed by whitespace, or newlines
    # Keep the delimiter attached to the preceding segment
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)

    # Also split on paragraph breaks (multiple newlines)
    expanded = []
    for s in sentences:
        parts = re.split(r'\n+', s)
        expanded.extend(p.strip() for p in parts if p.strip())
    sentences = expanded

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed max_chars
        if current_chunk and len(current_chunk) + 1 + len(sentence) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        # If a single sentence exceeds max_chars, split on clause boundaries
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            clause_pattern = r'(?<=[,;:—])\s+'
            clauses = re.split(clause_pattern, sentence)

            for clause in clauses:
                if current_chunk and len(current_chunk) + 1 + len(clause) > max_chars:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # If even a single clause exceeds max_chars, force-split on word boundaries
                if len(clause) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    words = clause.split()
                    for word in words:
                        if current_chunk and len(current_chunk) + 1 + len(word) > max_chars:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                        current_chunk = (current_chunk + " " + word).strip() if current_chunk else word
                else:
                    current_chunk = (current_chunk + " " + clause).strip() if current_chunk else clause
        else:
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# MPS bfloat16 safety patch (installed once on first torch backend use)
# ---------------------------------------------------------------------------

_mps_patch_installed = False


def _install_mps_patch():
    """Install the MPS-safe multinomial patch.

    Called once on first torch backend use. Patches torch.multinomial to
    cast to float32 and sanitize NaN/Inf before sampling on MPS devices.
    Only runs on macOS — skipped on Linux/Colab where MPS is not available.
    """
    global _mps_patch_installed
    if _mps_patch_installed:
        return

    from voice_config import IS_MACOS
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
# Voice prompt cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=10)
def _load_voice_prompt_torch(prompt_file):
    """Load and cache a .pt voice prompt (torch backend).

    If the .pt file doesn't exist but .wav + .txt files do, auto-creates
    the .pt using the already-loaded clone model (avoids loading a second model).
    """
    import torch

    prompt_path = os.path.join(VOICE_PROMPTS_DIR, prompt_file)
    if not os.path.exists(prompt_path):
        # Try to auto-create .pt from .wav + .txt
        base_name = prompt_file[:-3] if prompt_file.endswith('.pt') else prompt_file
        wav_path = os.path.join(VOICE_PROMPTS_DIR, f"{base_name}.wav")
        txt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base_name}.txt")
        if os.path.exists(wav_path):
            logger.info("Auto-creating .pt from .wav for %s", base_name)
            ref_audio, ref_sr = load_audio_for_cloning(wav_path)
            transcript = ""
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    transcript = f.read().strip()
            if not transcript:
                logger.warning("No transcript for %s, using empty string", base_name)
            model = load_model("clone")
            voice_prompt = create_voice_prompt(model, ref_audio, ref_sr, transcript)
            torch.save(voice_prompt, prompt_path)
            logger.info("Auto-created and saved %s", prompt_path)
            return voice_prompt
        return None
    from voice_config import get_device
    device = get_device()
    return torch.load(prompt_path, weights_only=False, map_location=device)


def load_voice_prompt(prompt_file):
    """Load a voice prompt, dispatching to the correct format for the backend.

    - torch backend: loads .pt tensor file
    - mlx backend: loads .wav + .txt file pair as a dict
    """
    backend = get_backend()
    if backend == "mlx":
        return load_voice_prompt_mlx(prompt_file)
    return _load_voice_prompt_torch(prompt_file)


def clear_voice_prompt_cache():
    """Clear both torch and MLX voice prompt caches."""
    _load_voice_prompt_torch.cache_clear()
    _mlx_prompt_cache.clear()


def voice_prompt_cache_info():
    """Return cache statistics for the active backend.

    For torch: returns lru_cache info (named tuple with hits, misses, etc.)
    For mlx: returns a simple namespace with hits/currsize.
    """
    backend = get_backend()
    if backend == "mlx":
        from types import SimpleNamespace
        return SimpleNamespace(
            currsize=len(_mlx_prompt_cache),
            hits=0,  # dict cache doesn't track hits
            misses=0,
            maxsize=_MLX_PROMPT_CACHE_MAX,
        )
    return _load_voice_prompt_torch.cache_info()


# ---------------------------------------------------------------------------
# Torch backend — model loading
# ---------------------------------------------------------------------------

_RETRY_DELAYS = (5, 15, 45)  # seconds between retry attempts


def _load_model_torch(model_type):
    """Load a TTS model using the PyTorch/MPS backend.

    Retries up to 3 times with exponential backoff on download/load failures.
    """
    import torch
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
    torch_dtype = dtype_map[dtype_name]

    logger.info("Loading %s (%s) with dtype=%s, size=%s [torch backend]...",
                model_type, repo_id, dtype_name, model_size)
    t0 = time.time()

    last_error = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            from voice_config import get_device
            device = get_device()
            # CUDA uses "auto" for multi-GPU support; MPS/CPU use device name directly
            device_map = "auto" if device == "cuda" else device
            model = Qwen3TTSModel.from_pretrained(
                repo_id,
                attn_implementation="sdpa",
                device_map=device_map,
                dtype=torch_dtype,
            )
            elapsed = time.time() - t0
            logger.info("Loaded %s model in %.1fs", model_type, elapsed)
            return model
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
# Torch backend — inference
# ---------------------------------------------------------------------------

def _run_inference_torch(model, text, mode, gen_params, language="English",
                         voice_prompt=None, voice_description=None,
                         speaker=None, instruct=None):
    """Run TTS inference using the PyTorch/MPS backend."""
    import torch

    t0 = time.time()

    # Float32 guard: clone mode on MPS requires float32 to avoid NaN/Inf errors.
    # If a non-float32 dtype is configured, override for this call and warn.
    if mode == "clone" and torch.backends.mps.is_available():
        dtype_name = get_torch_dtype_name()
        if dtype_name != "float32":
            logger.warning(
                "Clone mode on MPS requires float32 (configured: %s). "
                "Overriding to float32 for this generation. "
                "Set advanced.dtype to 'float32' in %s to silence this warning.",
                dtype_name, CONFIG_PATH,
            )
            # Cast model to float32 for this call, restore after
            model.float()

    params = {
        "temperature": gen_params.get("temperature", 0.7),
        "top_k": gen_params.get("top_k", 50),
        "top_p": gen_params.get("top_p", 0.95),
        "repetition_penalty": gen_params.get("repetition_penalty", 1.05),
    }

    seed = gen_params.get("seed")
    if seed is not None:
        torch.manual_seed(seed)

    try:
        with torch.inference_mode():
            if mode == "clone":
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_prompt,
                    **params,
                )
            elif mode == "custom":
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    speaker=speaker,
                    instruct=instruct or "",
                    language=language,
                    **params,
                )
            else:  # design
                wavs, sr = model.generate_voice_design(
                    text=text,
                    instruct=voice_description or "",
                    language=language,
                    **params,
                )
    except RuntimeError as e:
        if "inf" in str(e) or "nan" in str(e):
            dtype_name = get_torch_dtype_name()
            if dtype_name != "float32":
                logger.error(
                    "Generation produced NaN/Inf with dtype=%s. "
                    "Switch to float32 in %s under advanced.dtype for stability.",
                    dtype_name, CONFIG_PATH,
                )
            raise
        raise

    # Device memory management
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            peak = torch.mps.current_allocated_memory()
            logger.debug(
                "MPS memory after generation: %.1f MB",
                peak / (1024 * 1024),
            )
        except Exception:
            pass
    elif torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated()
            logger.debug(
                "CUDA memory after generation: %.1f MB",
                peak / (1024 * 1024),
            )
        except Exception:
            pass

    elapsed = time.time() - t0
    logger.info(
        "Inference complete: %d chars, %.1fs, mode=%s [torch]",
        len(text), elapsed, mode,
    )

    return wavs[0], sr


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
            "Or install dependencies: pip install -r requirements-mlx.txt"
        )

    repo_id = get_mlx_model_name(model_type)
    model_size = get_model_size()
    logger.info("Loading %s (%s) size=%s [mlx backend]...", model_type, repo_id, model_size)
    t0 = time.time()

    last_error = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            model = mlx_load_model(repo_id)
            elapsed = time.time() - t0
            logger.info("Loaded %s model in %.1fs [mlx]", model_type, elapsed)
            return model
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
# MLX backend — inference
# ---------------------------------------------------------------------------

def _run_inference_mlx(model, text, mode, gen_params, language="English",
                       voice_prompt=None, voice_description=None,
                       speaker=None, instruct=None):
    """Run TTS inference using the MLX backend.

    Returns (wav_array, sample_rate) where wav_array is a float32 numpy array,
    matching the torch backend's output contract.

    Args:
        model: Loaded MLX model (from _load_model_mlx).
        text: Text to synthesize.
        mode: "clone", "design", or "custom".
        gen_params: Dict with temperature, top_k, top_p, repetition_penalty.
        language: Language string.
        voice_prompt: For clone mode — dict with "ref_audio" (path) and
                      "ref_text" (str), or a .pt prompt name (will error).
        voice_description: Voice description string (design mode).
        speaker: Speaker name string (custom mode).
        instruct: Style instruction string (custom mode).
    """
    t0 = time.time()

    params = {
        "temperature": gen_params.get("temperature", 0.9),
        "top_k": gen_params.get("top_k", 50),
        "top_p": gen_params.get("top_p", 1.0),
        "repetition_penalty": gen_params.get("repetition_penalty", 1.05),
    }

    speed = gen_params.get("speed", 1.0)

    if mode == "clone":
        # MLX clone mode uses ref_audio (wav path) + ref_text directly.
        # voice_prompt should be a dict {"ref_audio": path, "ref_text": str}
        # set up by the caller or by load_voice_prompt_mlx().
        if voice_prompt is None:
            raise ValueError("voice_prompt is required for clone mode")

        if isinstance(voice_prompt, dict):
            ref_audio_path = voice_prompt["ref_audio"]
            ref_text = voice_prompt["ref_text"]
        else:
            raise TypeError(
                "MLX clone mode requires a voice prompt dict with 'ref_audio' "
                "and 'ref_text' keys. Torch .pt prompts are not compatible "
                "with the MLX backend. Re-create the prompt with 'createVoice' "
                "to generate MLX-compatible files (.wav + .txt)."
            )

        results = list(model.generate(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            language=language,
            speed=speed,
            **params,
        ))

    elif mode == "custom":
        results = list(model.generate_custom_voice(
            text=text,
            speaker=speaker or "Ryan",
            language=language,
            instruct=instruct or "",
            **params,
        ))

    else:  # design
        results = list(model.generate_voice_design(
            text=text,
            instruct=voice_description or "",
            language=language,
            **params,
        ))

    if not results:
        raise RuntimeError("MLX generation returned no results")

    # Collect audio from all segments and concatenate
    import mlx.core as mx

    audio_segments = [r.audio for r in results]
    if len(audio_segments) == 1:
        audio_mx = audio_segments[0]
    else:
        audio_mx = mx.concatenate(audio_segments)

    # Convert mx.array → numpy float32 (matches torch backend output contract)
    wav = np.array(audio_mx, dtype=np.float32)

    # Flatten to 1-D if needed
    if wav.ndim > 1:
        wav = wav.squeeze()

    sr = results[0].sample_rate

    elapsed = time.time() - t0
    logger.info(
        "Inference complete: %d chars, %.1fs, mode=%s [mlx]",
        len(text), elapsed, mode,
    )

    return wav, sr


def _run_inference_mlx_streaming(model, text, mode, gen_params, language="English",
                                  voice_prompt=None, voice_description=None,
                                  speaker=None, instruct=None):
    """Run TTS inference using the MLX backend, yielding audio chunks as they generate.

    This is a generator that yields (audio_chunk, sample_rate) tuples as they
    become available, enabling streaming playback.

    Args:
        model: Loaded MLX model (from _load_model_mlx).
        text: Text to synthesize.
        mode: "clone", "design", or "custom".
        gen_params: Dict with temperature, top_k, top_p, repetition_penalty.
        language: Language string.
        voice_prompt: For clone mode — dict with "ref_audio" and "ref_text".
        voice_description: Voice description string (design mode).
        speaker: Speaker name string (custom mode).
        instruct: Style instruction string (custom mode).

    Yields:
        (audio_chunk, sample_rate) tuples where audio_chunk is a float32 numpy array.
    """
    import mlx.core as mx

    params = {
        "temperature": gen_params.get("temperature", 0.9),
        "top_k": gen_params.get("top_k", 50),
        "top_p": gen_params.get("top_p", 1.0),
        "repetition_penalty": gen_params.get("repetition_penalty", 1.05),
    }

    speed = gen_params.get("speed", 1.0)

    if mode == "clone":
        if voice_prompt is None:
            raise ValueError("voice_prompt is required for clone mode")
        if isinstance(voice_prompt, dict):
            ref_audio_path = voice_prompt["ref_audio"]
            ref_text = voice_prompt["ref_text"]
        else:
            raise TypeError(
                "MLX clone mode requires a voice prompt dict with 'ref_audio' "
                "and 'ref_text' keys."
            )

        generator = model.generate(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            language=language,
            speed=speed,
            stream=True,  # Enable streaming
            **params,
        )

    elif mode == "custom":
        generator = model.generate_custom_voice(
            text=text,
            speaker=speaker or "Ryan",
            language=language,
            instruct=instruct or "",
            stream=True,  # Enable streaming
            **params,
        )

    else:  # design
        generator = model.generate_voice_design(
            text=text,
            instruct=voice_description or "",
            language=language,
            stream=True,  # Enable streaming
            **params,
        )

    chunk_count = 0
    for result in generator:
        audio_mx = result.audio
        wav = np.array(audio_mx, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.squeeze()
        sr = result.sample_rate
        chunk_count += 1
        logger.debug("Streaming chunk %d: %d samples", chunk_count, len(wav))
        yield wav, sr

    logger.info("Streaming complete: %d chunks, mode=%s [mlx]", chunk_count, mode)


# ---------------------------------------------------------------------------
# MLX voice prompt loading
# ---------------------------------------------------------------------------

_mlx_prompt_cache = {}
_MLX_PROMPT_CACHE_MAX = 10


def load_voice_prompt_mlx(prompt_name):
    """Load an MLX-compatible voice prompt (wav + txt file pair).

    Looks for <prompt_name>.wav and <prompt_name>.txt in VOICE_PROMPTS_DIR.
    Returns a dict with 'ref_audio' (path) and 'ref_text' (string) keys.
    Results are cached (up to 10 entries) for repeated lookups.

    Args:
        prompt_name: Base name with or without .pt extension.
                     E.g. "my_voice" or "my_voice.pt" — the .pt is stripped.
    """
    # Check cache first
    if prompt_name in _mlx_prompt_cache:
        return _mlx_prompt_cache[prompt_name]

    # Strip known extensions to get the base name
    base = prompt_name
    if base.endswith(".pt"):
        base = base[:-3]
    elif base.endswith(".wav"):
        base = base[:-4]

    wav_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.wav")
    txt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.txt")

    if not os.path.exists(wav_path) or not os.path.exists(txt_path):
        # Check if a .pt file exists (torch-only prompt)
        pt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.pt")
        if os.path.exists(pt_path):
            raise FileNotFoundError(
                f"Voice prompt '{base}' only has a .pt file (torch format). "
                f"The MLX backend requires .wav and .txt files. "
                f"Re-create the prompt with 'createVoice' to generate "
                f"MLX-compatible files."
            )
        raise FileNotFoundError(
            f"Voice prompt not found: looked for {wav_path} and {txt_path}"
        )

    with open(txt_path, "r") as f:
        ref_text = f.read().strip()

    result = {"ref_audio": wav_path, "ref_text": ref_text}

    # Evict oldest entry if at capacity
    if len(_mlx_prompt_cache) >= _MLX_PROMPT_CACHE_MAX:
        oldest_key = next(iter(_mlx_prompt_cache))
        del _mlx_prompt_cache[oldest_key]

    _mlx_prompt_cache[prompt_name] = result
    return result


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def load_model(model_type):
    """Load a TTS model by type, dispatching to the configured backend.

    Args:
        model_type: One of "clone", "design", "custom".

    Returns:
        The loaded model instance (type depends on backend).
    """
    backend = get_backend()
    if backend == "mlx":
        return _load_model_mlx(model_type)
    return _load_model_torch(model_type)


def _get_max_chunk_chars():
    """Read max_chunk_chars from config, defaulting to 500."""
    try:
        config = load_config()
        return config.get("generation", {}).get("max_chunk_chars", 500)
    except Exception:
        return 500


def run_inference(model, text, mode, gen_params, language="English",
                  voice_prompt=None, voice_description=None,
                  speaker=None, instruct=None,
                  max_chunk_chars=None, progress_callback=None):
    """Run TTS inference, dispatching to the configured backend.

    For long texts, automatically splits into chunks at sentence boundaries
    and concatenates the results with short silence gaps.

    Args:
        model: Loaded model (from load_model).
        text: Text to synthesize.
        mode: "clone", "design", or "custom".
        gen_params: Dict with temperature, top_k, top_p, repetition_penalty.
        language: Language string.
        voice_prompt: Loaded voice prompt (clone mode).
        voice_description: Voice description string (design mode).
        speaker: Speaker name string (custom mode).
        instruct: Style instruction string (custom mode).
        max_chunk_chars: Max chars per chunk (None = read from config, 0 = disable).
        progress_callback: Optional callable(chunk_index, chunk_total) for progress.

    Returns:
        (wav_array, sample_rate) tuple.
    """
    if max_chunk_chars is None:
        max_chunk_chars = _get_max_chunk_chars()

    # Split into chunks if text is long enough
    if max_chunk_chars > 0 and len(text) > max_chunk_chars:
        chunks = _split_text(text, max_chars=max_chunk_chars)
    else:
        chunks = [text]

    if len(chunks) == 1:
        # Single chunk — no overhead
        if progress_callback:
            progress_callback(0, 1)
        return _run_inference_single(
            model, chunks[0], mode, gen_params, language,
            voice_prompt, voice_description, speaker, instruct,
        )

    # Multi-chunk: generate each, concatenate with silence gaps
    logger.info("Splitting text (%d chars) into %d chunks (max %d chars each)",
                len(text), len(chunks), max_chunk_chars)

    all_audio = []
    sample_rate = None
    silence_gap_ms = 100  # 100ms silence between chunks

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, len(chunks))

        preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
        logger.info("Chunk %d/%d: '%s' (%d chars)", i + 1, len(chunks), preview, len(chunk))

        wav, sr = _run_inference_single(
            model, chunk, mode, gen_params, language,
            voice_prompt, voice_description, speaker, instruct,
        )

        if sample_rate is None:
            sample_rate = sr

        all_audio.append(wav)

    # Concatenate with silence gaps
    silence_samples = int(sample_rate * silence_gap_ms / 1000)
    combined = []
    for i, wav in enumerate(all_audio):
        combined.append(wav)
        if i < len(all_audio) - 1:
            combined.append(np.zeros(silence_samples, dtype=np.float32))

    result = np.concatenate(combined)
    logger.info("Combined %d chunks into %.1fs audio", len(chunks), len(result) / sample_rate)
    return result, sample_rate


def _run_inference_single(model, text, mode, gen_params, language="English",
                          voice_prompt=None, voice_description=None,
                          speaker=None, instruct=None,
                          _metal_retry=False):
    """Run TTS inference for a single text chunk.

    For MLX backend, includes Metal crash recovery: on certain Metal kernel
    errors, retries once with smaller sub-chunks.
    """
    backend = get_backend()
    if backend == "mlx":
        try:
            return _run_inference_mlx(
                model, text, mode, gen_params, language,
                voice_prompt, voice_description, speaker, instruct,
            )
        except RuntimeError as e:
            # Metal kernel crashes often contain "command buffer" or "GPU" in the message
            error_str = str(e).lower()
            is_metal_crash = any(
                keyword in error_str
                for keyword in ("command buffer", "gpu", "metal", "kernel")
            )
            if is_metal_crash and not _metal_retry and len(text) > 100:
                logger.warning(
                    "Metal kernel issue detected, retrying with smaller sub-chunks: %s",
                    str(e)[:100],
                )
                # Split the chunk in half and process each sub-chunk
                mid = len(text) // 2
                # Find a space near the midpoint to avoid splitting mid-word
                split_idx = text.rfind(" ", mid - 50, mid + 50)
                if split_idx == -1:
                    split_idx = mid
                chunk1, chunk2 = text[:split_idx].strip(), text[split_idx:].strip()

                wav1, sr = _run_inference_single(
                    model, chunk1, mode, gen_params, language,
                    voice_prompt, voice_description, speaker, instruct,
                    _metal_retry=True,
                )
                wav2, _ = _run_inference_single(
                    model, chunk2, mode, gen_params, language,
                    voice_prompt, voice_description, speaker, instruct,
                    _metal_retry=True,
                )
                # Concatenate with short silence
                silence = np.zeros(int(sr * 0.1), dtype=np.float32)
                return np.concatenate([wav1, silence, wav2]), sr
            raise
    return _run_inference_torch(
        model, text, mode, gen_params, language,
        voice_prompt, voice_description, speaker, instruct,
    )


def run_inference_streaming(model, text, mode, gen_params, language="English",
                            voice_prompt=None, voice_description=None,
                            speaker=None, instruct=None,
                            max_chunk_chars=None):
    """Run TTS inference in streaming mode, yielding audio chunks as they generate.

    For MLX backend, uses native streaming from model.generate().
    For torch backend, falls back to text chunking — yields per-chunk audio.

    Args:
        model: Loaded model (from load_model).
        text: Text to synthesize.
        mode: "clone", "design", or "custom".
        gen_params: Dict with temperature, top_k, top_p, repetition_penalty.
        language: Language string.
        voice_prompt: Loaded voice prompt (clone mode).
        voice_description: Voice description string (design mode).
        speaker: Speaker name string (custom mode).
        instruct: Style instruction string (custom mode).
        max_chunk_chars: Max chars per chunk for torch fallback (None = config default).

    Yields:
        (audio_chunk, sample_rate) tuples where audio_chunk is float32 numpy array.
    """
    backend = get_backend()

    if backend == "mlx":
        # MLX has native streaming — yield chunks as they generate
        logger.info("Starting streaming inference [mlx]")
        yield from _run_inference_mlx_streaming(
            model, text, mode, gen_params, language,
            voice_prompt, voice_description, speaker, instruct,
        )
    else:
        # Torch fallback: chunk the text and yield per-chunk audio
        logger.info("Starting chunked streaming [torch fallback]")
        if max_chunk_chars is None:
            max_chunk_chars = _get_max_chunk_chars()

        if max_chunk_chars > 0 and len(text) > max_chunk_chars:
            chunks = _split_text(text, max_chars=max_chunk_chars)
        else:
            chunks = [text]

        for i, chunk in enumerate(chunks):
            preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
            logger.info("Streaming chunk %d/%d: '%s'", i + 1, len(chunks), preview)

            wav, sr = _run_inference_single(
                model, chunk, mode, gen_params, language,
                voice_prompt, voice_description, speaker, instruct,
            )
            yield wav, sr


def create_voice_prompt(model, ref_audio, ref_sr, transcript):
    """Create a reusable voice-clone prompt from reference audio.

    Args:
        model: Loaded clone (Base) model.
        ref_audio: numpy array of reference audio (mono).
        ref_sr: Sample rate of reference audio.
        transcript: Text transcript of the reference audio.

    Returns:
        Voice prompt tensor (suitable for torch.save / generate_voice_clone).
    """
    # Convert to mono if stereo
    if ref_audio.ndim > 1:
        ref_audio = np.mean(ref_audio, axis=-1).astype(np.float32)

    logger.info(
        "Creating voice prompt: %.1fs audio, %d char transcript",
        len(ref_audio) / ref_sr, len(transcript),
    )

    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=(ref_audio, ref_sr),
        ref_text=transcript,
    )
    return voice_prompt


# ---------------------------------------------------------------------------
# Smart Audio Loader (torchaudio primary, soundfile/librosa fallback)
# ---------------------------------------------------------------------------

def load_audio(file_path, target_sr=16000):
    """Load audio file, resample to target_sr. Uses cached loader preference."""
    if _AUDIO_LOADER == "torchaudio":
        try:
            import torchaudio
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            return waveform.squeeze(0).numpy(), target_sr
        except Exception as e:
            logger.warning("torchaudio failed, falling back to soundfile: %s", e)
    import soundfile as sf
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr


def load_audio_for_cloning(file_path, max_duration=30, target_sr=16000):
    """Load audio truncated to max_duration seconds. For voice embedding only."""
    if _AUDIO_LOADER == "torchaudio":
        try:
            import torchaudio
            info = torchaudio.info(file_path)
            max_frames = int(max_duration * info.sample_rate)
            waveform, sr = torchaudio.load(file_path, num_frames=max_frames)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            return waveform.squeeze(0).numpy(), target_sr
        except Exception as e:
            logger.warning("torchaudio failed, falling back to soundfile: %s", e)
    import soundfile as sf
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)
    max_samples = int(max_duration * sr)
    audio = audio[:max_samples]
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr


# ---------------------------------------------------------------------------
# Audio processing (backend-agnostic — uses only numpy)
# ---------------------------------------------------------------------------

def trim_silence(audio, sample_rate, threshold_db=-40, min_silence_ms=100):
    """Trim leading and trailing silence from audio."""
    threshold = 10 ** (threshold_db / 20)
    min_samples = int(sample_rate * min_silence_ms / 1000)

    abs_audio = np.abs(audio)
    non_silent = abs_audio > threshold

    if not np.any(non_silent):
        return audio

    start_idx = np.argmax(non_silent)
    start_idx = max(0, start_idx - min_samples)

    end_idx = len(audio) - np.argmax(non_silent[::-1])
    end_idx = min(len(audio), end_idx + min_samples)

    return audio[start_idx:end_idx]


def normalize_audio(audio, target_db=-3.0):
    """Normalize audio to target peak dB level."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    target_peak = 10 ** (target_db / 20)
    return audio * (target_peak / peak)


def adjust_speed(audio, sample_rate, speed_factor):
    """Adjust audio speed without changing pitch.

    Args:
        speed_factor: >1.0 = faster, <1.0 = slower.
    """
    if speed_factor == 1.0:
        return audio
    import librosa
    return librosa.effects.time_stretch(audio, rate=speed_factor)


def adjust_pitch(audio, sample_rate, semitones):
    """Adjust audio pitch without changing speed.

    Args:
        semitones: positive = higher pitch, negative = lower.
    """
    if semitones == 0:
        return audio
    import librosa
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)


def process_audio(audio, sample_rate, trim=False, normalize=False,
                  speed=None, pitch=None):
    """Apply all audio processing in canonical order.

    Args:
        audio: numpy audio array.
        sample_rate: Audio sample rate.
        trim: Trim leading/trailing silence.
        normalize: Normalize to -3dB peak.
        speed: Speed factor (None or 1.0 = unchanged).
        pitch: Pitch shift in semitones (None or 0 = unchanged).
    """
    if trim:
        audio = trim_silence(audio, sample_rate)

    if speed and speed != 1.0:
        audio = adjust_speed(audio, sample_rate, speed)

    if pitch and pitch != 0:
        audio = adjust_pitch(audio, sample_rate, pitch)

    if normalize:
        audio = normalize_audio(audio, target_db=-3.0)

    return audio


# ---------------------------------------------------------------------------
# Audio transcription (ASR) — lazy loading, MLX backend only
# ---------------------------------------------------------------------------

# ASR model caches — NOT loaded at startup, preloaded in background by UI
_asr_model_mlx = None
_asr_model_torch = None
_asr_lock = threading.Lock()


def _ensure_asr_torch_loaded():
    """Thread-safe loading of the torch ASR pipeline. Blocks until ready."""
    global _asr_model_torch
    with _asr_lock:
        if _asr_model_torch is not None:
            return
        logger.info("Loading torch ASR model...")
        t0 = time.time()
        try:
            from transformers import pipeline as hf_pipeline
            from voice_config import get_device
            device_name = get_device()
            if device_name == "cuda":
                device = 0
            elif device_name == "mps":
                device = "mps"
            else:
                device = -1
            _asr_model_torch = hf_pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=device,
                chunk_length_s=30,
            )
            logger.info("Torch ASR model loaded in %.1fs", time.time() - t0)
        except ImportError as e:
            raise ImportError(
                f"ASR transcription requires transformers: {e}\n"
                "Install with: pip install transformers"
            )


def preload_asr_model():
    """Preload ASR model in a background thread. Non-blocking, non-fatal."""
    def _load():
        try:
            backend = get_backend()
            if backend != "mlx":
                _ensure_asr_torch_loaded()
        except Exception as e:
            logger.warning("ASR preload failed (will retry on first use): %s", e)

    threading.Thread(target=_load, daemon=True).start()


def _transcribe_mlx(audio_path, language="en"):
    """Transcribe using MLX ASR (Apple Silicon)."""
    global _asr_model_mlx

    if _asr_model_mlx is None:
        logger.info("Loading MLX ASR model for transcription (first use)...")
        t0 = time.time()
        try:
            from mlx_audio.stt import load_model as load_stt_model
            _asr_model_mlx = load_stt_model("mlx-community/whisper-large-v3-turbo")
            logger.info("MLX ASR model loaded in %.1fs", time.time() - t0)
        except ImportError as e:
            raise ImportError(
                f"ASR transcription requires mlx-audio with STT support: {e}\n"
                "Update mlx-audio: pip install --upgrade mlx-audio"
            )

    logger.info("Transcribing (MLX): %s", audio_path)
    t0 = time.time()
    try:
        result = _asr_model_mlx.generate(audio_path, language=language)
        transcript = result.text.strip() if result.text else ""
        logger.info("Transcription complete: %d chars in %.1fs", len(transcript), time.time() - t0)
        return transcript
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")


def _transcribe_torch(audio_path, language="en"):
    """Transcribe using transformers Whisper pipeline (CUDA/CPU)."""
    _ensure_asr_torch_loaded()

    logger.info("Transcribing (torch): %s", audio_path)
    t0 = time.time()
    try:
        kwargs = {}
        if language:
            kwargs["generate_kwargs"] = {"language": language}
        result = _asr_model_torch(audio_path, **kwargs)
        transcript = result["text"].strip() if result.get("text") else ""
        logger.info("Transcription complete: %d chars in %.1fs", len(transcript), time.time() - t0)
        return transcript
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")


def transcribe_audio(audio_path, language="en"):
    """Transcribe audio file to text.

    Dispatches to MLX ASR (Apple Silicon) or torch Whisper (CUDA/CPU)
    based on configured backend. Lazily loads the ASR model on first call.

    Args:
        audio_path: Path to audio file (.wav, .mp3, etc.).
        language: Language code for transcription (default: "en").

    Returns:
        Transcribed text string.

    Raises:
        ImportError: If required ASR library is not installed.
        RuntimeError: If transcription fails.
    """
    backend = get_backend()
    if backend == "mlx":
        return _transcribe_mlx(audio_path, language)
    else:
        return _transcribe_torch(audio_path, language)


def unload_asr_model():
    """Free ASR model from memory to reclaim VRAM/RAM for TTS generation."""
    global _asr_model_mlx, _asr_model_torch
    if _asr_model_mlx is not None:
        _asr_model_mlx = None
        logger.info("Unloaded MLX ASR model")
    if _asr_model_torch is not None:
        _asr_model_torch = None
        logger.info("Unloaded torch ASR model")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    import gc
    gc.collect()


def is_asr_available():
    """Check if ASR transcription is available.

    Returns True if the current backend has a compatible ASR library.
    Does NOT load the ASR model — just checks availability.
    """
    if get_backend() == "mlx":
        try:
            from mlx_audio.stt import load_model  # noqa: F401
            return True
        except ImportError:
            return False
    else:
        try:
            from transformers import pipeline  # noqa: F401
            return True
        except ImportError:
            return False


def is_asr_loaded():
    """Check if an ASR model is currently loaded in memory."""
    return _asr_model_mlx is not None or _asr_model_torch is not None


def get_asr_model_info():
    """Return info dict about loaded ASR model (or None if not loaded)."""
    if _asr_model_mlx is not None:
        return {
            "loaded": True,
            "backend": "mlx",
            "model_name": "whisper-large-v3-turbo",
        }
    if _asr_model_torch is not None:
        return {
            "loaded": True,
            "backend": "torch",
            "model_name": "whisper-base",
        }
    return {"loaded": False, "backend": None, "model_name": None}


def unload_model_cleanup():
    """Backend-specific memory cleanup after setting a model to None."""
    import gc
    gc.collect()
    backend = get_backend()
    if backend == "torch":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
