#!/usr/bin/env python3
"""UI package for Qwen3-TTS.

This package provides the Gradio web interface for TTS generation.
The main entry point is build_ui() which creates the Gradio interface.

Modules:
- _facade: Main entry points (build_ui, main, stop_server)
- shared: Shared utilities, constants, status helpers
- generation: Generation tab logic and wiring
- voice_management: Voice prompt creation and management
- model_management: Model load/unload and status

Usage:
    from qwen3_tts.interface.ui import build_ui, main

    demo = build_ui()
    demo.launch()

Or from command line:
    tts ui
"""

import importlib

# Use lazy imports to avoid importing gradio when tests just need to patch
# module-level attributes or import utility functions

__all__ = [
    # Main entry points
    "build_ui",
    "main",
    "stop_server",
    "_find_available_port",
    "on_history_select",
    # Shared utilities
    "SPEAKER_CHOICES",
    "MAX_HISTORY_SIZE",
    "enhance_description_with_ai",
    "is_enhancer_available",
    "get_current_model_settings",
    "apply_model_settings",
    "update_text_info",
    "get_server_status",
    "format_status_display",
    "get_voice_prompts",
    "get_presets",
    "add_to_history",
    "get_history_data",
    # Generation
    "get_prosody_choices",
    "apply_prosody_preset",
    "cancel_streaming_generation",
    "_prepare_streaming_config",
    "_generate_server_side",
    "_build_common_controls",
    "_build_generate_buttons_and_output",
    "_wire_generation_tab",
    # Voice management
    "create_voice_prompt",
    "auto_transcribe_audio",
    "get_prompt_table_data",
    "preview_voice",
    "rename_voice",
    "delete_voice",
    "set_voice_default",
    # Model management
    "get_model_table_data",
    "toggle_model",
    "toggle_asr",
    "update_startup_defaults",
    "get_model_status_html",
    "get_audio_loader_setting",
    "set_audio_loader_setting",
    # Backward compatibility
    "TTSClient",
    "VALID_MODEL_SIZES",
    "VALID_MLX_QUANTIZATIONS",
    "get_backend",
    "get_model_size",
    "get_mlx_quantization",
]

# Map symbol names to their source modules
_LAZY_IMPORTS = {
    # Main entry points from _facade
    "build_ui": "._facade",
    "main": "._facade",
    "stop_server": "._facade",
    "_find_available_port": "._facade",
    "on_history_select": "._facade",
    # Shared utilities
    "SPEAKER_CHOICES": ".shared",
    "MAX_HISTORY_SIZE": ".shared",
    "enhance_description_with_ai": ".shared",
    "is_enhancer_available": ".shared",
    "get_current_model_settings": ".shared",
    "apply_model_settings": ".shared",
    "update_text_info": ".shared",
    "get_server_status": ".shared",
    "format_status_display": ".shared",
    "get_voice_prompts": ".shared",
    "get_presets": ".shared",
    "add_to_history": ".shared",
    "get_history_data": ".shared",
    # Generation (includes re-exports from voice_helpers)
    "get_prosody_choices": ".generation",
    "apply_prosody_preset": ".generation",
    "cancel_streaming_generation": ".generation",
    "_prepare_streaming_config": ".generation",
    "_generate_server_side": ".generation",
    "_build_common_controls": ".generation",
    "_build_generate_buttons_and_output": ".generation",
    "_wire_generation_tab": ".generation",
    # Voice management
    "create_voice_prompt": ".voice_management",
    "auto_transcribe_audio": ".voice_management",
    "get_prompt_table_data": ".voice_management",
    "preview_voice": ".voice_management",
    "rename_voice": ".voice_management",
    "delete_voice": ".voice_management",
    "set_voice_default": ".voice_management",
    # Model management
    "get_model_table_data": ".model_management",
    "toggle_model": ".model_management",
    "toggle_asr": ".model_management",
    "update_startup_defaults": ".model_management",
    "get_model_status_html": ".model_management",
    "get_audio_loader_setting": ".model_management",
    "set_audio_loader_setting": ".model_management",
}

# Cache for imported modules
_module_cache = {}


# Submodules that can be lazily imported
_SUBMODULES = {
    "_facade",
    "shared",
    "generation",
    "voice_management",
    "model_management",
}


def __getattr__(name):
    """Lazy import symbols when accessed."""
    # Handle submodule access (e.g., qwen3_tts.interface.ui.voice_management)
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", package=__name__)

    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        if module_name not in _module_cache:
            _module_cache[module_name] = importlib.import_module(
                module_name, package=__name__
            )
        return getattr(_module_cache[module_name], name)

    # Special case: TTSClient re-exported from server.client for backward compatibility
    if name == "TTSClient":
        from qwen3_tts.server.client import TTSClient as _TTSClient

        return _TTSClient

    # Config constants re-exported for backward compatibility
    if name in (
        "VALID_MODEL_SIZES",
        "VALID_MLX_QUANTIZATIONS",
        "get_backend",
        "get_model_size",
        "get_mlx_quantization",
    ):
        from qwen3_tts.core import config

        return getattr(config, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
