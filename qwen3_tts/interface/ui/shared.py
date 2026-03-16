#!/usr/bin/env python3
"""Shared utilities and helpers for the Gradio UI.

This module contains:
- Constants like SPEAKER_CHOICES
- Status and history helpers
- Model settings utilities
- AI description enhancement
"""

import logging
import os

import gradio as gr

from qwen3_tts.core.config import (
    CUSTOM_VOICE_SPEAKERS,
    VALID_MODEL_SIZES,
    VALID_MLX_QUANTIZATIONS,
    get_backend,
    get_model_size,
    get_mlx_quantization,
    get_server_url,
    is_server_running,
    auth_headers,
    load_config,
)

logger = logging.getLogger("tts.ui")

# Constants
MAX_HISTORY_SIZE = 10

# Derive speaker choices from canonical source
SPEAKER_CHOICES = [
    f"{key} ({info['lang']}) - {info['desc']}"
    for key, info in CUSTOM_VOICE_SPEAKERS.items()
]


def enhance_description_with_ai(description):
    """Enhance a brief voice description using an LLM API."""
    if not description or not description.strip():
        raise gr.Error("Please enter a description to enhance")

    config = load_config()
    enhancer_config = config.get("prompt_enhancer", {})

    if not enhancer_config.get("enabled", False):
        raise gr.Error("AI enhancement is not enabled. Set prompt_enhancer.enabled=true in config.json")

    api_key_env = enhancer_config.get("api_key_env", "ANTHROPIC_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise gr.Error(f"API key not found. Set the {api_key_env} environment variable")

    provider = enhancer_config.get("provider", "anthropic")
    model = enhancer_config.get("model", "claude-haiku-4-5-20251001")

    system_prompt = (
        "You are a TTS voice description specialist. Expand the user's brief voice description "
        "into a detailed, TTS-optimized description. Include gender, age range, tone, texture, "
        "pace, and accent details. Keep it under 100 words. Output ONLY the description, "
        "no preamble or explanation."
    )

    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=200,
                messages=[
                    {"role": "user", "content": f"Expand this voice description: {description}"},
                ],
                system=system_prompt,
            )
            return response.content[0].text.strip()
        else:
            raise gr.Error(f"Unsupported provider: {provider}")
    except Exception as e:
        logger.error(f"AI enhancement failed: {e}")
        raise gr.Error(f"Enhancement failed: {e}")


def is_enhancer_available():
    """Check if AI description enhancement is available."""
    config = load_config()
    enhancer_config = config.get("prompt_enhancer", {})
    if not enhancer_config.get("enabled", False):
        return False
    api_key_env = enhancer_config.get("api_key_env", "ANTHROPIC_API_KEY")
    return bool(os.environ.get(api_key_env))


def get_current_model_settings():
    """Get current model size, quantization, and backend from server."""
    backend = get_backend()
    model_size = get_model_size()
    mlx_quant = get_mlx_quantization()

    if is_server_running(load_config()):
        try:
            import requests
            url = get_server_url(load_config())
            resp = requests.get(f"{url}/models", timeout=5, headers=auth_headers())
            if resp.status_code == 200:
                data = resp.json()
                # Get settings from server if available
                if "settings" in data:
                    settings = data["settings"]
                    return (
                        settings.get("model_size", model_size),
                        settings.get("mlx_quantization", mlx_quant),
                        settings.get("backend", backend),
                    )
        except Exception as e:
            logger.warning(f"Could not fetch model settings from server: {e}")

    return model_size, mlx_quant, backend


def apply_model_settings(model_size, mlx_quantization):
    """Apply model settings to server."""
    if not is_server_running(load_config()):
        return "Server not running", format_status_display()

    try:
        import requests
        url = get_server_url(load_config())
        payload = {
            "model_size": model_size,
        }
        backend = get_backend()
        if backend == "mlx" and mlx_quantization:
            payload["mlx_quantization"] = mlx_quantization

        resp = requests.post(
            f"{url}/update-model-config",
            json=payload,
            timeout=10,
            headers=auth_headers(),
        )
        if resp.status_code == 200:
            return "Settings applied (takes effect on next generation)", format_status_display()
        else:
            error = resp.json().get("error", "Unknown error")
            return f"Failed: {error}", format_status_display()
    except Exception as e:
        return f"Error: {e}", format_status_display()


def update_text_info(text):
    """Update text info display with character count, word count, and chunk estimate."""
    if not text or not text.strip():
        return ""
    chars = len(text)
    words = len(text.split())
    # Estimate chunks (500 chars default)
    chunks = max(1, (chars + 499) // 500)
    if chunks > 1:
        return f"{chars} chars | ~{chunks} chunks"
    return f"{chars} chars"


def get_server_status():
    """Get current server status, memory usage, loaded models, and backend info.

    Returns:
        tuple: (status_str, memory_str, models_str, backend_str)
    """
    from qwen3_tts.server.client import TTSClient
    client = TTSClient()

    if not client.is_server_running():
        return "Disconnected", "N/A", "N/A", "N/A"

    try:
        stats = client.get_stats()
        # Use explicit None check so 0.0 is a valid value, not skipped as falsy
        memory_val = None
        for _key in ('mlx_memory_active_mb', 'mps_memory_allocated_mb', 'cuda_memory_allocated_mb'):
            _v = stats.get(_key)
            if _v is not None:
                memory_val = _v
                break
        memory = f"{memory_val:.1f}MB" if isinstance(memory_val, (int, float)) else "N/A"

        loaded_models = []
        if stats.get("clone_model_loaded"):
            loaded_models.append("Clone")
        if stats.get("design_model_loaded"):
            loaded_models.append("Design")
        if stats.get("custom_model_loaded"):
            loaded_models.append("Custom")
        models_str = ", ".join(loaded_models) if loaded_models else "None"

        backend = stats.get("backend", "torch")
        model_size = stats.get("model_size", "1.7B")
        if backend == "mlx":
            quant = stats.get("mlx_quantization", "8bit")
            backend_str = f"MLX ({quant}, {model_size})"
        else:
            dtype = stats.get("dtype", "float32")
            backend_str = f"PyTorch ({dtype}, {model_size})"

        return "Connected", memory, models_str, backend_str
    except Exception as e:
        return f"Error: {str(e)}", "N/A", "N/A", "N/A"


def format_status_display():
    """Format server status as HTML for display."""
    status, memory, models, backend = get_server_status()

    if status == "Connected":
        status_html = '<span style="color: green; font-weight: bold;">Connected</span>'
    elif status == "Disconnected":
        status_html = '<span style="color: red; font-weight: bold;">Disconnected</span>'
    else:
        status_html = f'<span style="color: orange;">{status}</span>'

    return f"""
    <div style="padding: 10px; background: var(--block-background-fill, #f5f5f5); border-radius: 5px; margin-bottom: 15px; border: 1px solid var(--block-border-color, #e0e0e0);">
        <strong>Status:</strong> {status_html} |
        <strong>Backend:</strong> {backend} |
        <strong>Memory:</strong> {memory} |
        <strong>Models:</strong> {models}
    </div>
    """


def get_voice_prompts():
    """Get list of available voice prompts."""
    from qwen3_tts.core.config import VOICE_PROMPTS_DIR

    try:
        files = os.listdir(VOICE_PROMPTS_DIR)
    except OSError:
        return []

    # Include .pt files and MLX prompt pairs (.wav + .txt)
    pt_prompts = {f for f in files if f.endswith('.pt')}
    txt_bases = {f[:-4] for f in files if f.endswith('.txt')}
    mlx_prompts = {f for f in files if f.endswith('.wav') and f[:-4] in txt_bases}

    return sorted(pt_prompts | mlx_prompts)


def get_presets():
    """Get list of available generation presets."""
    config = load_config()
    return ["(none)"] + list(config.get("presets", {}).keys())


def add_to_history(history_list, mode, text, output_path, duration_chunks):
    """Add a generation to history.

    Args:
        history_list: Existing history (not mutated).
        mode: Generation mode string (e.g. "clone"). Stored capitalized.
        text: Generated text. Truncated to 40 chars + "..." if longer.
        output_path: Path to the output audio file.
        duration_chunks: Number of audio chunks (int).

    Returns:
        New list with the entry prepended, capped at MAX_HISTORY_SIZE.
    """
    import time
    entry = {
        "timestamp": time.time(),
        "mode": mode.capitalize() if mode else mode,
        "text": text[:40] + "..." if len(text) > 40 else text,
        "path": output_path,
        "chunks": duration_chunks if isinstance(duration_chunks, int) else 0,
    }
    new_list = [entry] + list(history_list)
    return new_list[:MAX_HISTORY_SIZE]


def get_history_data(history_list):
    """Convert history list to list-of-lists format.

    Returns:
        List of [time, mode, text, chunks] rows.
    """
    import datetime

    if not history_list:
        return []

    rows = []
    for entry in history_list:
        ts = entry.get("timestamp", 0)
        time_str = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""
        rows.append([
            time_str,
            entry.get("mode", "?"),
            entry.get("text", ""),
            entry.get("chunks", 0),
        ])

    return rows
