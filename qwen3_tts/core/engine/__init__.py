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
# --- asr ---
from qwen3_tts.core.engine.asr import (
    get_asr_model_info,
    is_asr_available,
    is_asr_loaded,
    load_asr_model,
    preload_asr_model,
    transcribe_audio,
    unload_asr_model,
    unload_model_cleanup,
)

# --- audio_processing (base utility) ---
from qwen3_tts.core.engine.audio_processing import (
    adjust_pitch,
    adjust_speed,
    calculate_waveform_peaks,  # noqa: F401
    get_audio_loader,
    load_audio,
    load_audio_for_cloning,
    normalize_audio,
    normalize_lufs,
    process_audio,
    set_audio_loader,
    trim_silence,
)

# --- inference ---
from qwen3_tts.core.engine.inference import (
    _INFERENCE_STRATEGIES,  # noqa: F401
    _MODE_STRATEGIES_TORCH,  # noqa: F401
    _crossfade_chunks,  # noqa: F401
    _get_backend_strategy,  # noqa: F401
    _get_max_chunk_chars,  # noqa: F401
    _get_max_chunk_tokens,  # noqa: F401
    _run_inference_mlx,  # noqa: F401
    _run_inference_mlx_streaming,  # noqa: F401
    _run_inference_single,  # noqa: F401
    _run_inference_torch,  # noqa: F401
    _validate_audio,  # noqa: F401
    create_voice_prompt,
    register_backend,  # noqa: F401
    run_inference,
    run_inference_streaming,
)

# --- model_loader ---
from qwen3_tts.core.engine.model_loader import (
    load_model,
)

# --- voice_prompt ---
from qwen3_tts.core.engine.voice_prompt import (
    UnsupportedReferenceAudioError,
    VoicePromptCreateRequired,
    clear_voice_prompt_cache,
    load_voice_prompt,
    load_voice_prompt_mlx,
    migrate_orphan_mlx_prompts,
    save_voice_prompt_mlx,
    voice_prompt_cache_info,
)

from . import (
    asr,
    audio_processing,
    inference,
    model_loader,
    text_processing,
    voice_prompt,
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
    "save_voice_prompt_mlx",
    "UnsupportedReferenceAudioError",
    "VoicePromptCreateRequired",
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
