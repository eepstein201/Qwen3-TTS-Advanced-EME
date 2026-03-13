#!/usr/bin/env python3
"""Model management for the Gradio UI.

This module contains:
- Model load/unload functions
- Model status display
- ASR load/unload
- Startup config updates
- Audio loader settings
"""

import logging

import gradio as gr

from qwen3_tts.core.config import (
    get_server_url,
    is_server_running,
    auth_headers,
    load_config,
    save_config,
    get_backend,
)
from qwen3_tts.interface.ui.shared import format_status_display

logger = logging.getLogger("tts.ui")


def get_model_table_data():
    """Get model status as table data for display.

    Returns:
        List of [model_type, status, memory_mb, startup_load] rows
    """
    config = load_config()

    if not is_server_running(config):
        return [["clone", "server not running", "-", "—"],
                ["design", "server not running", "-", "—"],
                ["custom", "server not running", "-", "—"]]

    try:
        import requests
        url = get_server_url(config)
        resp = requests.get(f"{url}/models", timeout=5, headers=auth_headers())

        if resp.status_code != 200:
            return [["clone", "error", "-", "—"],
                    ["design", "error", "-", "—"],
                    ["custom", "error", "-", "—"]]

        data = resp.json()
        models = data.get("models", {})
        startup_config = config.get("models", {})

        rows = []
        for model_type in ["clone", "design", "custom"]:
            info = models.get(model_type, {})
            loaded = info.get("loaded", False)
            memory = info.get("memory_mb", 0)
            load_at_startup = startup_config.get(model_type, {}).get("load_at_startup", False)

            status = "✅ Loaded" if loaded else "Not loaded"
            memory_str = f"{memory:.0f}MB" if memory else "—"
            startup_str = "Yes" if load_at_startup else "No"

            rows.append([model_type, status, memory_str, startup_str])

        return rows

    except Exception as e:
        logger.error(f"Failed to get model table data: {e}")
        return [["clone", f"error: {e}", "-", "—"],
                ["design", f"error: {e}", "-", "—"],
                ["custom", f"error: {e}", "-", "—"]]


def toggle_model(model_type, action):
    """Load or unload a model.

    Args:
        model_type: 'clone', 'design', or 'custom'
        action: 'load' or 'unload'

    Returns:
        Tuple of (status_message, model_table, status_html)
    """
    config = load_config()

    if not is_server_running(config):
        return "Server not running", get_model_table_data(), format_status_display()

    try:
        import requests
        url = get_server_url(config)

        if action == "load":
            endpoint = f"{url}/load-model"
        else:
            endpoint = f"{url}/unload-model"

        resp = requests.post(
            endpoint,
            json={"model_type": model_type},
            timeout=120,
            headers=auth_headers(),
        )

        if resp.status_code == 200:
            result = resp.json()
            status = result.get("status", "done")
            return f"Model {model_type}: {status}", get_model_table_data(), format_status_display()
        else:
            error = resp.json().get("error", "Unknown error")
            return f"Failed: {error}", get_model_table_data(), format_status_display()

    except Exception as e:
        logger.error(f"Model toggle failed: {e}")
        return f"Error: {e}", get_model_table_data(), format_status_display()


def toggle_asr(action):
    """Load or unload the ASR model.

    Args:
        action: 'load' or 'unload'

    Returns:
        Tuple of (status_message, status_html)
    """
    config = load_config()

    if not is_server_running(config):
        return "Server not running", format_status_display()

    try:
        import requests
        url = get_server_url(config)

        if action == "load":
            endpoint = f"{url}/load-asr"
        else:
            endpoint = f"{url}/unload-asr"

        resp = requests.post(
            endpoint,
            timeout=60,
            headers=auth_headers(),
        )

        if resp.status_code == 200:
            result = resp.json()
            status = result.get("status", "done")
            return f"ASR: {status}", format_status_display()
        else:
            error = resp.json().get("error", "Unknown error")
            return f"Failed: {error}", format_status_display()

    except Exception as e:
        logger.error(f"ASR toggle failed: {e}")
        return f"Error: {e}", format_status_display()


def update_startup_defaults(clone_startup, design_startup, custom_startup):
    """Update which models load at server startup.

    Args:
        clone_startup: Whether clone model should load at startup
        design_startup: Whether design model should load at startup
        custom_startup: Whether custom model should load at startup

    Returns:
        Tuple of (status_message, model_table)
    """
    config = load_config()

    if "models" not in config:
        config["models"] = {}

    config["models"]["clone"] = {"load_at_startup": clone_startup}
    config["models"]["design"] = {"load_at_startup": design_startup}
    config["models"]["custom"] = {"load_at_startup": custom_startup}

    save_config(config)

    return "Startup config updated (restart server to apply)", get_model_table_data()


def get_model_status_html(model_type):
    """Get HTML status indicator for a specific model.

    Args:
        model_type: 'clone', 'design', or 'custom'

    Returns:
        HTML string with status indicator
    """
    config = load_config()

    if not is_server_running(config):
        return f'<span style="color: gray;">Server not running</span>'

    try:
        import requests
        url = get_server_url(config)
        resp = requests.get(f"{url}/models", timeout=5, headers=auth_headers())

        if resp.status_code != 200:
            return f'<span style="color: red;">Error</span>'

        data = resp.json()
        models = data.get("models", {})
        info = models.get(model_type, {})
        loaded = info.get("loaded", False)
        memory = info.get("memory_mb", 0)

        if loaded:
            return f'<span style="color: green;">✓ Loaded ({memory:.0f}MB)</span>'
        else:
            return f'<span style="color: gray;">Not loaded</span>'

    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        return f'<span style="color: red;">Error</span>'


def get_audio_loader_setting():
    """Get current audio loader setting.

    Returns:
        Current audio loader value
    """
    config = load_config()
    return config.get("advanced", {}).get("audio_loader", "torchaudio")


def set_audio_loader_setting(loader):
    """Set audio loader setting.

    Args:
        loader: 'torchaudio' or 'librosa'

    Returns:
        Status message
    """
    config = load_config()

    if "advanced" not in config:
        config["advanced"] = {}

    config["advanced"]["audio_loader"] = loader
    save_config(config)

    return f"Audio loader set to: {loader} (restart server to apply)"
