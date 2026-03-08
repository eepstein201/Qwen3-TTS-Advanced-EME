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
"""

# --- text_processing (base utility) ---
from qwen3_tts.core.engine.text_processing import (
    _normalize_text,
    _split_text,
    _map_language,
    _SENTENCE_SPLIT_RE,
    _PARAGRAPH_SPLIT_RE,
    _CLAUSE_SPLIT_RE,
    _PYSBD_LANG_MAP,
    _ABBREV_TABLE,
    _ABBREV_TABLE_COMPILED,
    _CURRENCY_MAP,
    _EMAIL_RE,
    _URL_RE,
    _URL_PROTO_RE,
    _URL_WWW_RE,
    _PHONE_RE,
    _PHONE_NONDIGIT_RE,
    _ORDINAL_RE,
    _ISO_DATE_RE,
    _US_DATE_RE,
    _CARDINAL_RE,
    _CURRENCY_RE,
    _n2w_cached,
    _n2w_loaded,
    _SEGMENTER_CACHE,
)

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
    _load_voice_prompt_torch,
    clear_voice_prompt_cache,
    voice_prompt_cache_info,
    load_voice_prompt_mlx,
    migrate_orphan_mlx_prompts,
    _mlx_prompt_cache,
    _torch_prompt_cache,
    _torch_prompt_cache_lock,
    _torch_prompt_cache_hits,
    _torch_prompt_cache_misses,
)

# --- model_loader ---
from qwen3_tts.core.engine.model_loader import (
    load_model,
    _load_model_torch,
    _load_model_mlx,
    _install_mps_patch,
    _apply_cuda_optimizations,
    _warmup_model,
    _RETRY_DELAYS,
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
    _asr_model_mlx,
    _asr_model_torch,
    _asr_lock,
)
