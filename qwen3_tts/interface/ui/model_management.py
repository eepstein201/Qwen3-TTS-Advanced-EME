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


from qwen3_tts.core.config import (
    get_server_url,
    is_server_running,
    auth_headers,
    load_config,
    save_config,
)
from qwen3_tts.interface.ui.components import (
    ProgressIndicator,
    poll_model_load_progress,
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
                ["custom", "server not running", "-", "—"],
                ["asr", "server not running", "-", "—"]]

    try:
        import requests
        url = get_server_url(config)
        resp = requests.get(f"{url}/models", timeout=5, headers=auth_headers())

        if resp.status_code != 200:
            return [["clone", "error", "-", "—"],
                    ["design", "error", "-", "—"],
                    ["custom", "error", "-", "—"],
                    ["asr", "error", "-", "—"]]

        data = resp.json()
        models = data.get("models", {})
        startup_config = config.get("models", {})

        rows = []
        for model_type in ["clone", "design", "custom"]:
            info = models.get(model_type, {})
            loaded = info.get("loaded", False)
            memory = info.get("memory_mb", 0)
            load_at_startup = startup_config.get(model_type, {}).get("load_at_startup", False)

            status = "Loaded" if loaded else "Not loaded"
            memory_str = f"{memory:.0f}MB" if memory else "—"
            startup_str = "Yes" if load_at_startup else "No"

            rows.append([model_type, status, memory_str, startup_str])

        # Add ASR row from the 'asr' key in the response
        asr_info = data.get("asr", {})
        asr_loaded = asr_info.get("loaded", False)
        asr_model = asr_info.get("model_name", "")
        asr_status = f"Loaded ({asr_model})" if asr_loaded else "Not loaded"
        rows.append(["asr", asr_status, "—", "—"])

        return rows

    except Exception as e:
        logger.error("Failed to get model table data: %s", e)
        return [["clone", f"error: {e}", "-", "—"],
                ["design", f"error: {e}", "-", "—"],
                ["custom", f"error: {e}", "-", "—"],
                ["asr", f"error: {e}", "-", "—"]]


def toggle_model(model_type, action):
    """Load or unload a model — yields a ProgressIndicator before the
    blocking HTTP call so the UI never goes silent during long loads.

    Args:
        model_type: 'clone', 'design', or 'custom'
        action: 'load' or 'unload'

    Yields:
        Tuples of (status_message, model_table, status_html). First yield is
        the in-flight progress indicator; final yield is the result. UI uses
        poll_model_load_progress under the hood (via /models loading:bool).
    """
    config = load_config()

    if not is_server_running(config):
        yield "Server not running", get_model_table_data(), format_status_display()
        return

    # Surface immediate progress so the UI doesn't appear frozen during the
    # synchronous /load-model call (which blocks until the model is ready).
    if action == "load":
        # Seed an indeterminate-mode progress; the next /models poll will
        # produce a real percent via poll_model_load_progress when wired into
        # _facade.py's polling loop.
        progress = poll_model_load_progress(model_type)
        eta_s = progress.get("eta_s")
        progress_html = ProgressIndicator(
            mode="indeterminate",
            message=f"{action.capitalize()}ing {model_type}…"
            + (f" (~{int(eta_s)}s expected)" if eta_s else ""),
        ).render()
        yield progress_html, get_model_table_data(), format_status_display()

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
            yield (
                f"Model {model_type}: {status}",
                get_model_table_data(),
                format_status_display(),
            )
        else:
            error = resp.json().get("error", "Unknown error")
            yield (
                f"Failed: {error}",
                get_model_table_data(),
                format_status_display(),
            )

    except Exception as e:
        logger.error("Model toggle failed: %s", e)
        yield f"Error: {e}", get_model_table_data(), format_status_display()


def toggle_asr(action):
    """Load or unload the ASR model — yields indeterminate ProgressIndicator
    before the blocking call (ASR has no per-percent signal).

    Args:
        action: 'load' or 'unload'

    Yields:
        Tuples of (status_message, status_html). First yield is the
        ProgressIndicator HTML; final yield is the result.
    """
    config = load_config()

    if not is_server_running(config):
        yield "Server not running", format_status_display()
        return

    if action == "load":
        progress_html = ProgressIndicator(
            mode="indeterminate",
            message="Loading ASR model…",
        ).render()
        yield progress_html, format_status_display()

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
            yield f"ASR: {status}", format_status_display()
        else:
            error = resp.json().get("error", "Unknown error")
            yield f"Failed: {error}", format_status_display()

    except Exception as e:
        logger.error("ASR toggle failed: %s", e)
        yield f"Error: {e}", format_status_display()


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
    existing_models = config.get("models", {})
    new_models = {
        **existing_models,
        "clone": {"load_at_startup": clone_startup},
        "design": {"load_at_startup": design_startup},
        "custom": {"load_at_startup": custom_startup},
    }
    save_config({**config, "models": new_models})

    return "Startup config updated (restart server to apply)", get_model_table_data()


def get_model_status_html(model_type):
    """Get HTML status indicator for a specific model.

    Reflects live /models state (not stale config). When the server reports
    `loading: True` the UI shows a "Loading" badge so it doesn't claim a model
    is "Loaded" mid-download (Phase 0 bug #4).

    Args:
        model_type: 'clone', 'design', or 'custom'

    Returns:
        HTML string with accessible status badge (SVG + text + aria-label).
    """
    from qwen3_tts.interface.ui.components import status_badge

    config = load_config()

    if not is_server_running(config):
        return status_badge("Server not running", severity="warning")

    try:
        import requests
        url = get_server_url(config)
        resp = requests.get(f"{url}/models", timeout=5, headers=auth_headers())

        if resp.status_code != 200:
            return status_badge("Error", severity="error")

        data = resp.json()
        info = data.get("models", {}).get(model_type, {}) or {}
        loaded = info.get("loaded", False)
        loading = info.get("loading", False)
        memory = info.get("memory_mb", 0)

        if loaded:
            return status_badge(f"Loaded ({memory:.0f}MB)", severity="success")
        if loading:
            return status_badge(f"Loading {model_type}...", severity="loading")
        return status_badge("Not loaded", severity="info")

    except Exception as e:
        logger.error("Failed to get model status: %s", e)
        return status_badge("Error", severity="error")


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
    adv = config.get("advanced", {})
    save_config({**config, "advanced": {**adv, "audio_loader": loader}})

    return f"Audio loader set to: {loader} (restart server to apply)"
