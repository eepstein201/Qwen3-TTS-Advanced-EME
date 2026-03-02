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
import re
import threading
import time
from collections import OrderedDict
from functools import lru_cache

import numpy as np

from qwen3_tts.core.config import (
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

# Audio loader preference — lazy init on first access, thread-safe updates
_AUDIO_LOADER = None
_AUDIO_LOADER_LOCK = threading.Lock()


def get_audio_loader():
    """Return cached audio loader preference. Lazy-loads from config on first call."""
    global _AUDIO_LOADER
    if _AUDIO_LOADER is None:
        with _AUDIO_LOADER_LOCK:
            if _AUDIO_LOADER is None:
                _AUDIO_LOADER = load_config().get("advanced", {}).get("audio_loader", "torchaudio")
    return _AUDIO_LOADER


def set_audio_loader(loader):
    """Update audio loader preference in memory (called by config update endpoints)."""
    global _AUDIO_LOADER
    if loader not in ("torchaudio", "librosa"):
        raise ValueError(f"Invalid audio loader: {loader}")
    with _AUDIO_LOADER_LOCK:
        _AUDIO_LOADER = loader


# ---------------------------------------------------------------------------
# Pre-compiled regex patterns for text chunking
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_PARAGRAPH_SPLIT_RE = re.compile(r'\n+')
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[,;:\u2014])\s+')

# ---------------------------------------------------------------------------
# Language mapping helpers
# ---------------------------------------------------------------------------

_PYSBD_LANG_MAP = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "chinese": "zh",
    "russian": "ru",
    "dutch": "nl",
    "polish": "pl",
}


def _map_language(language):
    """Map a full language name to a pySBD/num2words 2-letter ISO code.

    Returns 'en' for any unrecognized or missing language.
    """
    return _PYSBD_LANG_MAP.get((language or "english").lower(), "en")


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Abbreviation table — longer patterns first to avoid partial matches
_ABBREV_TABLE = [
    (r'\bProf\.', "Professor"),
    (r'\bDr\.', "Doctor"),
    (r'\bMrs\.', "Missus"),
    (r'\bMr\.', "Mister"),
    (r'\be\.g\.', "for example"),
    (r'\bi\.e\.', "that is"),
    (r'\bvs\.', "versus"),
    (r'\betc\.', "et cetera"),
    (r'\bapprox\.', "approximately"),
    (r'\byrs\.', "years"),
    (r'\byrs\b', "years"),
]

# Currency symbols → (singular, plural)
_CURRENCY_MAP = {
    "$": ("dollar", "dollars"),
    "€": ("euro", "euros"),
    "£": ("pound", "pounds"),
    "¥": ("yen", "yen"),
}


def _normalize_text(text, language="English"):
    """Normalize text for TTS: expand numbers, dates, abbreviations, and URLs.

    Called before chunking in run_inference(). All steps are wrapped in
    try/except so a failure never blocks generation.

    Args:
        text: Raw input text.
        language: Language name string (e.g. "English").

    Returns:
        Normalized text string.
    """
    import re as _re

    if not text:
        return text

    lang = _map_language(language)

    try:
        from num2words import num2words as _n2w
    except ImportError:
        _n2w = None

    # 1. Emails: user@example.com → "user at example dot com"
    try:
        def _expand_email(m):
            addr = m.group()
            local, _, domain = addr.partition("@")
            domain_parts = domain.split(".")
            return local + " at " + " dot ".join(domain_parts)
        text = _re.sub(
            r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
            _expand_email, text)
    except Exception:
        pass

    # 2. URLs: https://example.com → "example dot com"
    try:
        def _expand_url(m):
            url = m.group()
            url = _re.sub(r'^https?://', '', url)
            url = _re.sub(r'^www\.', '', url)
            url = url.replace(".", " dot ").rstrip()
            return url
        text = _re.sub(r'https?://\S+', _expand_url, text)
    except Exception:
        pass

    # 3. Phone numbers: (800) 555-1234 or 555-1234 → "8 0 0 5 5 5 1 2 3 4"
    try:
        def _expand_phone(m):
            digits = _re.sub(r'\D', '', m.group())
            return " ".join(digits)
        text = _re.sub(r'(?:\(\d{3}\)\s*|\d{3}[-.])\d{3}[-.]?\d{4}', _expand_phone, text)
    except Exception:
        pass

    # 4. Currencies: $5.00 → "five dollars"
    try:
        symbols_pat = "[" + _re.escape("".join(_CURRENCY_MAP.keys())) + "]"

        def _expand_currency(m):
            symbol = m.group(1)
            amount_str = m.group(2)
            singular, plural = _CURRENCY_MAP.get(symbol, ("unit", "units"))
            try:
                amount = float(amount_str)
                whole = int(amount)
                if _n2w:
                    words = _n2w(whole, lang=lang)
                else:
                    words = str(whole)
                label = singular if whole == 1 else plural
                return f"{words} {label}"
            except Exception:
                return m.group()
        text = _re.sub(rf'({symbols_pat})(\d+(?:\.\d+)?)', _expand_currency, text)
    except Exception:
        pass

    # 5. Ordinals: 3rd, 21st, etc.
    try:
        def _expand_ordinal(m):
            try:
                n = int(m.group(1))
                return _n2w(n, lang=lang, to="ordinal") if _n2w else m.group()
            except Exception:
                return m.group()
        text = _re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', _expand_ordinal, text)
    except Exception:
        pass

    # 6. ISO dates: YYYY-MM-DD
    try:
        def _expand_iso_date(m):
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                import calendar
                month_name = calendar.month_name[month]
                if _n2w:
                    day_word = _n2w(day, lang=lang, to="ordinal")
                    year_word = _n2w(year, lang=lang)
                else:
                    day_word = str(day)
                    year_word = str(year)
                return f"{month_name} {day_word}, {year_word}"
            except Exception:
                return m.group()
        text = _re.sub(r'\b(\d{4})-(\d{2})-(\d{2})\b', _expand_iso_date, text)
    except Exception:
        pass

    # 7. US dates: MM/DD/YYYY
    try:
        def _expand_us_date(m):
            month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                import calendar
                month_name = calendar.month_name[month]
                if _n2w:
                    day_word = _n2w(day, lang=lang, to="ordinal")
                    year_word = _n2w(year, lang=lang)
                else:
                    day_word = str(day)
                    year_word = str(year)
                return f"{month_name} {day_word}, {year_word}"
            except Exception:
                return m.group()
        text = _re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', _expand_us_date, text)
    except Exception:
        pass

    # 8. Abbreviations
    try:
        for pattern, replacement in _ABBREV_TABLE:
            text = _re.sub(pattern, replacement, text)
    except Exception:
        pass

    # 9. Cardinal numbers (standalone integers)
    if _n2w:
        try:
            def _expand_cardinal(m):
                try:
                    return _n2w(int(m.group()), lang=lang)
                except Exception:
                    return m.group()
            text = _re.sub(r'(?<![.\w])\b\d+\b(?![.\w])', _expand_cardinal, text)
        except Exception:
            pass

    return text


# ---------------------------------------------------------------------------
# Text chunking for long-form reliability
# ---------------------------------------------------------------------------

def _split_text(text, max_chars=500, language="English", tokenizer=None, max_tokens=None):
    """Split text into chunks at sentence boundaries.

    Splits on sentence-ending punctuation (. ! ?) followed by whitespace,
    or on newlines. If a single sentence exceeds the limit, falls back to
    clause boundaries (, ; — :). Never splits mid-word.

    When tokenizer and max_tokens are provided, uses token counts instead of
    character counts to measure chunk sizes (torch backend only).

    Args:
        text: Input text to split.
        max_chars: Maximum characters per chunk (used when tokenizer is None).
        language: Language name string for pySBD segmenter.
        tokenizer: Optional tokenizer for token-aware chunking.
        max_tokens: Maximum tokens per chunk (used when tokenizer is provided).

    Returns:
        List of text chunks. Returns [text] unchanged if it fits in one chunk.
    """
    text = text.strip()

    def _measure(chunk):
        if tokenizer is not None and max_tokens is not None:
            return len(tokenizer.encode(chunk, add_special_tokens=False))
        return len(chunk)

    limit = max_tokens if (tokenizer is not None and max_tokens is not None) else max_chars

    if _measure(text) <= limit:
        return [text]

    # Sentence splitting: pySBD when available, regex fallback
    try:
        import pysbd
        segmenter = pysbd.Segmenter(language=_map_language(language), clean=False)
        sentences = segmenter.segment(text)
    except ImportError:
        sentences = _SENTENCE_SPLIT_RE.split(text)

    # Also split on paragraph breaks (multiple newlines)
    expanded = []
    for s in sentences:
        parts = _PARAGRAPH_SPLIT_RE.split(s)
        expanded.extend(p.strip() for p in parts if p.strip())
    sentences = expanded

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if current_chunk and _measure(current_chunk) + 1 + _measure(sentence) > limit:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        if _measure(sentence) > limit:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            clauses = _CLAUSE_SPLIT_RE.split(sentence)

            for clause in clauses:
                if current_chunk and _measure(current_chunk) + 1 + _measure(clause) > limit:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                if _measure(clause) > limit:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    words = clause.split()
                    for word in words:
                        if current_chunk and _measure(current_chunk) + 1 + _measure(word) > limit:
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
    from qwen3_tts.core.config import get_device
    device = get_device()
    # Register VoiceClonePromptItem as a safe global so torch.load(weights_only=True)
    # can deserialize .pt files containing this class (PyTorch 2.6+ requirement).
    try:
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
        torch.serialization.add_safe_globals([VoiceClonePromptItem])
    except ImportError:
        pass  # qwen_tts not installed — fall through to exception handler
    try:
        return torch.load(prompt_path, weights_only=True, map_location=device)
    except Exception:
        allow_unsafe = os.environ.get("TTS_ALLOW_UNSAFE_PICKLE") == "1"
        real_prompt = os.path.realpath(prompt_path)
        real_prompts_dir = os.path.realpath(VOICE_PROMPTS_DIR)
        if not real_prompt.startswith(real_prompts_dir + os.sep):
            raise ValueError(
                f"Refusing to load {prompt_file}: outside voice_prompts/ directory"
            )
        if not allow_unsafe:
            raise RuntimeError(
                f"Cannot load {prompt_file} with weights_only=True. "
                f"If this is a trusted file, set TTS_ALLOW_UNSAFE_PICKLE=1"
            )
        logger.warning(
            "Loading %s with weights_only=False — only load trusted .pt files",
            prompt_file,
        )
        return torch.load(prompt_path, weights_only=False, map_location=device)  # nosec B614


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


def migrate_orphan_mlx_prompts(clone_model=None):
    """Scan voice_prompts/ for .wav+.txt without .pt and auto-create .pt files.

    Supports users migrating from Mac (MLX) to Colab (PyTorch).

    Args:
        clone_model: Optional pre-loaded clone model. If None, loads on demand.
    """
    import glob

    wav_files = glob.glob(os.path.join(VOICE_PROMPTS_DIR, "*.wav"))
    migrated = 0
    model = None  # Lazy: only load if migration needed and no model passed
    for wav_path in wav_files:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        pt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.pt")
        if not os.path.exists(pt_path):
            logger.info("Migrating orphan MLX prompt: %s", base)
            try:
                if model is None:
                    model = clone_model or load_model("clone")
                ref_audio, ref_sr = load_audio_for_cloning(wav_path)
                transcript = ""
                txt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, "r") as f:
                        transcript = f.read().strip()
                if not transcript:
                    logger.warning("No transcript for %s, using empty string", base)
                voice_prompt = create_voice_prompt(model, ref_audio, ref_sr, transcript)
                import torch  # lazy: only needed when saving .pt
                torch.save(voice_prompt, pt_path)
                logger.info("Auto-created and saved %s", pt_path)
                migrated += 1
            except Exception as e:
                logger.warning("Failed to migrate prompt '%s': %s", base, e)
    if migrated:
        logger.info("Migrated %d orphan MLX prompt(s) to .pt format", migrated)
    return migrated


# ---------------------------------------------------------------------------
# Torch backend — model loading
# ---------------------------------------------------------------------------

_RETRY_DELAYS = (5, 15, 45)  # seconds between retry attempts


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

    attn_impl, optimal_dtype, should_compile = _apply_cuda_optimizations(load_config())

    logger.info("Loading %s (%s) with dtype=%s, size=%s [torch backend]...",
                model_type, repo_id, dtype_name, model_size)
    t0 = time.time()

    last_error = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            from qwen3_tts.core.config import get_device
            device = get_device()
            # Override dtype with CUDA-optimal dtype when on CUDA
            if device == "cuda":
                torch_dtype = optimal_dtype
            # CUDA uses "auto" for multi-GPU support; MPS/CPU use device name directly
            device_map = "auto" if device == "cuda" else device
            # Use 8-bit quantization on older CUDA GPUs (Turing/T4)
            load_in_8bit = False
            if device == "cuda":
                cap = torch.cuda.get_device_capability()
                if cap[0] < 8:
                    load_in_8bit = True
            load_kwargs = dict(
                attn_implementation=attn_impl,
                device_map=device_map,
                dtype=torch_dtype,
            )
            if load_in_8bit:
                load_kwargs["load_in_8bit"] = True
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
                         speaker=None, instruct=None,
                         x_vector_only_mode=False):
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
        "max_new_tokens": gen_params.get("max_new_tokens", 2048),
    }

    seed = gen_params.get("seed")
    if seed is not None:
        torch.manual_seed(seed)

    try:
        with torch.inference_mode():
            if mode == "clone":
                clone_kwargs = dict(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_prompt,
                    **params,
                )
                if x_vector_only_mode:
                    clone_kwargs["x_vector_only_mode"] = True
                wavs, sr = model.generate_voice_clone(**clone_kwargs)
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
        except Exception:  # nosec B110
            pass
    elif torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated()
            logger.debug(
                "CUDA memory after generation: %.1f MB",
                peak / (1024 * 1024),
            )
        except Exception:  # nosec B110
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
            "Or install dependencies: pip install -e .[mlx]"
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
                       speaker=None, instruct=None,
                       x_vector_only_mode=False):
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
        x_vector_only_mode: If True, use empty ref_text for speaker-embedding-only clone.
    """
    t0 = time.time()

    params = {
        "temperature": gen_params.get("temperature", 0.9),
        "top_k": gen_params.get("top_k", 50),
        "top_p": gen_params.get("top_p", 1.0),
        "repetition_penalty": gen_params.get("repetition_penalty", 1.05),
        "max_new_tokens": gen_params.get("max_new_tokens", 2048),
    }

    if mode == "clone":
        # MLX clone mode uses ref_audio (wav path) + ref_text directly.
        # voice_prompt should be a dict {"ref_audio": path, "ref_text": str}
        # set up by the caller or by load_voice_prompt_mlx().
        if voice_prompt is None:
            raise ValueError("voice_prompt is required for clone mode")

        if isinstance(voice_prompt, dict):
            ref_audio_path = voice_prompt["ref_audio"]
            ref_text = "" if x_vector_only_mode else voice_prompt["ref_text"]
        else:
            raise TypeError(
                "MLX clone mode requires a voice prompt dict with 'ref_audio' "
                "and 'ref_text' keys. Torch .pt prompts are not compatible "
                "with the MLX backend. Re-create the prompt with 'tts voice create' "
                "to generate MLX-compatible files (.wav + .txt)."
            )

        results = list(model.generate(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            language=language,
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
                                  speaker=None, instruct=None,
                                  x_vector_only_mode=False):
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
        x_vector_only_mode: If True, use empty ref_text for speaker-embedding-only clone.

    Yields:
        (audio_chunk, sample_rate) tuples where audio_chunk is a float32 numpy array.
    """
    import mlx.core as mx

    params = {
        "temperature": gen_params.get("temperature", 0.9),
        "top_k": gen_params.get("top_k", 50),
        "top_p": gen_params.get("top_p", 1.0),
        "repetition_penalty": gen_params.get("repetition_penalty", 1.05),
        "max_new_tokens": gen_params.get("max_new_tokens", 2048),
    }

    if mode == "clone":
        if voice_prompt is None:
            raise ValueError("voice_prompt is required for clone mode")
        if isinstance(voice_prompt, dict):
            ref_audio_path = voice_prompt["ref_audio"]
            ref_text = "" if x_vector_only_mode else voice_prompt["ref_text"]
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

_mlx_prompt_cache = OrderedDict()
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
    # Check cache first (move to end on hit for LRU eviction)
    if prompt_name in _mlx_prompt_cache:
        _mlx_prompt_cache.move_to_end(prompt_name)
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
                f"Re-create the prompt with 'tts voice create' to generate "
                f"MLX-compatible files."
            )
        raise FileNotFoundError(
            f"Voice prompt not found: looked for {wav_path} and {txt_path}"
        )

    with open(txt_path, "r") as f:
        ref_text = f.read().strip()

    result = {"ref_audio": wav_path, "ref_text": ref_text}

    # Evict least-recently-used entry if at capacity
    if len(_mlx_prompt_cache) >= _MLX_PROMPT_CACHE_MAX:
        _mlx_prompt_cache.popitem(last=False)

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


def _get_max_chunk_tokens():
    """Read max_chunk_tokens from config, defaulting to 200."""
    try:
        config = load_config()
        return config.get("generation", {}).get("max_chunk_tokens", 200)
    except Exception:
        return 200


def run_inference(model, text, mode, gen_params, language="English",
                  voice_prompt=None, voice_description=None,
                  speaker=None, instruct=None,
                  max_chunk_chars=None, progress_callback=None,
                  x_vector_only_mode=False):
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

    # Normalize text (expand numbers, dates, abbreviations) before chunking
    text = _normalize_text(text, language)

    # Resolve tokenizer for token-aware chunking (torch backend only)
    tokenizer = getattr(model, "tokenizer", None)
    max_tokens = _get_max_chunk_tokens() if tokenizer is not None else None

    # Split into chunks
    if tokenizer is not None and max_tokens is not None:
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count > max_tokens:
            chunks = _split_text(text, max_chars=max_chunk_chars,
                                 language=language, tokenizer=tokenizer,
                                 max_tokens=max_tokens)
        else:
            chunks = [text]
    elif max_chunk_chars > 0 and len(text) > max_chunk_chars:
        chunks = _split_text(text, max_chars=max_chunk_chars, language=language)
    else:
        chunks = [text]

    if len(chunks) == 1:
        # Single chunk — no overhead
        if progress_callback:
            progress_callback(0, 1)
        return _run_inference_single(
            model, chunks[0], mode, gen_params, language,
            voice_prompt, voice_description, speaker, instruct,
            x_vector_only_mode=x_vector_only_mode,
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
            x_vector_only_mode=x_vector_only_mode,
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
                          _metal_retry=False, x_vector_only_mode=False):
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
                x_vector_only_mode=x_vector_only_mode,
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
                    _metal_retry=True, x_vector_only_mode=x_vector_only_mode,
                )
                wav2, _ = _run_inference_single(
                    model, chunk2, mode, gen_params, language,
                    voice_prompt, voice_description, speaker, instruct,
                    _metal_retry=True, x_vector_only_mode=x_vector_only_mode,
                )
                # Concatenate with short silence
                silence = np.zeros(int(sr * 0.1), dtype=np.float32)
                return np.concatenate([wav1, silence, wav2]), sr
            raise
    return _run_inference_torch(
        model, text, mode, gen_params, language,
        voice_prompt, voice_description, speaker, instruct,
        x_vector_only_mode=x_vector_only_mode,
    )


def run_inference_streaming(model, text, mode, gen_params, language="English",
                            voice_prompt=None, voice_description=None,
                            speaker=None, instruct=None,
                            max_chunk_chars=None, x_vector_only_mode=False):
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
            x_vector_only_mode=x_vector_only_mode,
        )
    else:
        # Torch fallback: chunk the text and yield per-chunk audio
        logger.info("Starting chunked streaming [torch fallback]")
        if max_chunk_chars is None:
            max_chunk_chars = _get_max_chunk_chars()

        text = _normalize_text(text, language)
        tokenizer = getattr(model, "tokenizer", None)
        max_tokens = _get_max_chunk_tokens() if tokenizer is not None else None

        if tokenizer is not None and max_tokens is not None:
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            if token_count > max_tokens:
                chunks = _split_text(text, max_chars=max_chunk_chars,
                                     language=language, tokenizer=tokenizer,
                                     max_tokens=max_tokens)
            else:
                chunks = [text]
        elif max_chunk_chars > 0 and len(text) > max_chunk_chars:
            chunks = _split_text(text, max_chars=max_chunk_chars, language=language)
        else:
            chunks = [text]

        for i, chunk in enumerate(chunks):
            preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
            logger.info("Streaming chunk %d/%d: '%s'", i + 1, len(chunks), preview)

            wav, sr = _run_inference_single(
                model, chunk, mode, gen_params, language,
                voice_prompt, voice_description, speaker, instruct,
                x_vector_only_mode=x_vector_only_mode,
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
    if get_audio_loader() == "torchaudio":
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


def load_audio_for_cloning(file_path, max_duration=15, target_sr=16000):
    """Load audio truncated to max_duration seconds. For voice embedding only."""
    if get_audio_loader() == "torchaudio":
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

    end_idx = len(audio) - np.argmax(np.flip(non_silent))
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

    Uses pyrubberband (Rubber Band library) for professional-quality
    time-stretching. Falls back to librosa's phase vocoder if
    pyrubberband or the rubberband CLI tool is not installed.

    Args:
        speed_factor: >1.0 = faster, <1.0 = slower.
    """
    if speed_factor == 1.0:
        return audio
    try:
        import pyrubberband as pyrb
        return pyrb.time_stretch(audio, sample_rate, speed_factor)
    except (ImportError, FileNotFoundError):
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed_factor)


def adjust_pitch(audio, sample_rate, semitones):
    """Adjust audio pitch without changing speed.

    Uses pyrubberband (Rubber Band library) for professional-quality
    pitch-shifting. Falls back to librosa's phase vocoder if
    pyrubberband or the rubberband CLI tool is not installed.

    Args:
        semitones: positive = higher pitch, negative = lower.
    """
    if semitones == 0:
        return audio
    try:
        import pyrubberband as pyrb
        return pyrb.pitch_shift(audio, sample_rate, semitones)
    except (ImportError, FileNotFoundError):
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
            from qwen3_tts.core.config import get_device
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


def load_asr_model():
    """Load ASR model synchronously for the current backend. Returns True on success."""
    global _asr_model_mlx
    backend = get_backend()
    if backend == "mlx":
        if _asr_model_mlx is not None:
            return True
        try:
            from mlx_audio.stt import load_model as load_stt_model
            _asr_model_mlx = load_stt_model("mlx-community/whisper-large-v3-turbo")
            logger.info("Loaded MLX ASR model")
            return True
        except ImportError as e:
            raise ImportError(f"ASR requires mlx-audio with STT support: {e}")
    else:
        _ensure_asr_torch_loaded()
        return True


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
