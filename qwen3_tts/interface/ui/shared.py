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
    """Get server status including model loading state and MLX memory stats."""
    config = load_config()
    if not is_server_running(config):
        return {
            "running": False,
            "models": {},
            "memory": None,
            "backend": get_backend(),
        }

    try:
        import requests
        url = get_server_url(config)
        resp = requests.get(f"{url}/models", timeout=5, headers=auth_headers())
        if resp.status_code == 200:
            data = resp.json()
            # Include MLX-specific memory fields if available
            status = {
                "running": True,
                "models": data.get("models", {}),
                "memory": data.get("memory_mb"),
                "backend": data.get("backend", get_backend()),
            }
            # Add MLX memory stats if present
            for key in ('mlx_memory_active_mb', 'mps_memory_allocated_mb', 'cuda_memory_allocated_mb'):
                if key in data:
                    status[key] = data[key]
            return status
    except Exception as e:
        logger.warning(f"Could not fetch server status: {e}")

    return {
        "running": True,
        "models": {},
        "memory": None,
        "backend": get_backend(),
    }


def format_status_display():
    """Format server status as HTML for display."""
    status = get_server_status()

    if not status["running"]:
        return """
        <div style="padding: 10px; background: #ffebee; border-radius: 8px;">
            <b>Server:</b> <span style="color: red;">Not running</span><br>
            <small>Start with: <code>tts server start</code></small>
        </div>
        """

    models = status.get("models", {})
    memory = status.get("memory")
    backend = status.get("backend", "unknown")

    model_indicators = []
    for model_type in ["clone", "design", "custom"]:
        model_info = models.get(model_type, {})
        loaded = model_info.get("loaded", False)
        color = "green" if loaded else "gray"
        status_text = "loaded" if loaded else "not loaded"
        model_indicators.append(
            f"<span style='color: {color};'>{model_type}: {status_text}</span>"
        )

    memory_str = f"{memory:.0f}MB" if memory else "unknown"

    return f"""
    <div style="padding: 10px; background: #e8f5e9; border-radius: 8px;">
        <b>Server:</b> <span style="color: green;">Running</span> |
        <b>Backend:</b> {backend} |
        <b>Memory:</b> {memory_str}<br>
        <b>Models:</b> {" | ".join(model_indicators)}
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
    return list(config.get("presets", {}).keys())


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
