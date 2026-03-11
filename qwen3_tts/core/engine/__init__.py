#!/usr/bin/env python3
"""TTS inference engine package — facade re-exporting all public names.

This package replaces the monolithic engine.py with 6 submodules arranged
in a strict DAG (no circular imports). This __init__.py re-exports every
public function so that all existing consumers (app.py, generate.py, ui.py,
create_voice.py, tests) continue to work unchanged.

Submodule DAG (arrows = "imports from"):
    config.py (external)
        ↑
    text_processing.py  (base utility — config only)
        ↑
    audio_processing.py (base utility — config only)
        ↑
    voice_prompt.py     (config, audio_processing)
        ↑
    model_loader.py     (config only)
        ↑
    inference.py        (config, text_processing, audio_processing)
        ↑
    asr.py              (config, audio_processing)

Internal symbols (prefixed with _) are NOT exported from this facade.
Tests that need access to internal symbols should import directly from
submodules, e.g.:
    from qwen3_tts.core.engine.text_processing import _normalize_text
    from qwen3_tts.core.engine.inference import _run_inference_single
"""

# Import submodules to make them accessible for tests that need internal symbols
# e.g., from qwen3_tts.core.engine.text_processing import _normalize_text
from . import text_processing
from . import audio_processing
from . import voice_prompt
from . import model_loader
from . import inference
from . import asr

# --- audio_processing (base utility) ---
from qwen3_tts.core.engine.audio_processing import (
    get_audio_loader,
    set_audio_loader,
    load_audio,
    load_audio_for_cloning,
    trim_silence,
    normalize_audio,
    normalize_lufs,
    adjust_speed,
    adjust_pitch,
    process_audio,
)

# --- voice_prompt ---
from qwen3_tts.core.engine.voice_prompt import (
    load_voice_prompt,
    clear_voice_prompt_cache,
    voice_prompt_cache_info,
    load_voice_prompt_mlx,
    migrate_orphan_mlx_prompts,
)

# --- model_loader ---
from qwen3_tts.core.engine.model_loader import (
    load_model,
)

# --- inference ---
from qwen3_tts.core.engine.inference import (
    run_inference,
    run_inference_streaming,
    create_voice_prompt,
    _validate_audio,
    _run_inference_torch,
    _run_inference_mlx,
    _run_inference_mlx_streaming,
    _run_inference_single,
    _get_max_chunk_chars,
    _get_max_chunk_tokens,
    _crossfade_chunks,
    _INFERENCE_STRATEGIES,
    _MODE_STRATEGIES_TORCH,
    register_backend,
    _get_backend_strategy,
)

# --- asr ---
from qwen3_tts.core.engine.asr import (
    preload_asr_model,
    load_asr_model,
    transcribe_audio,
    unload_asr_model,
    is_asr_available,
    is_asr_loaded,
    get_asr_model_info,
    unload_model_cleanup,
)

# Public API — all symbols without underscore prefix
__all__ = [
    # submodules
    "text_processing",
    "audio_processing",
    "voice_prompt",
    "model_loader",
    "inference",
    "asr",
    # audio_processing
    "get_audio_loader",
    "set_audio_loader",
    "load_audio",
    "load_audio_for_cloning",
    "trim_silence",
    "normalize_audio",
    "normalize_lufs",
    "adjust_speed",
    "adjust_pitch",
    "process_audio",
    # voice_prompt
    "load_voice_prompt",
    "clear_voice_prompt_cache",
    "voice_prompt_cache_info",
    "load_voice_prompt_mlx",
    "migrate_orphan_mlx_prompts",
    # model_loader
    "load_model",
    # inference
    "run_inference",
    "run_inference_streaming",
    "create_voice_prompt",
    # asr
    "preload_asr_model",
    "load_asr_model",
    "transcribe_audio",
    "unload_asr_model",
    "is_asr_available",
    "is_asr_loaded",
    "get_asr_model_info",
    "unload_model_cleanup",
]
