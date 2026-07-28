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
    is_server_running,
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
        return [
            ["clone", "server not running", "-", "—"],
            ["design", "server not running", "-", "—"],
            ["custom", "server not running", "-", "—"],
            ["asr", "server not running", "-", "—"],
        ]

    try:
        from qwen3_tts.core.http_client import server_request

        resp = server_request("GET", "/models", timeout=5)

        if resp.status_code != 200:
            return [
                ["clone", "error", "-", "—"],
                ["design", "error", "-", "—"],
                ["custom", "error", "-", "—"],
                ["asr", "error", "-", "—"],
            ]

        data = resp.json()
        models = data.get("models", {})
        startup_config = config.get("models", {})

        rows = []
        for model_type in ["clone", "design", "custom"]:
            info = models.get(model_type, {})
            loaded = info.get("loaded", False)
            memory = info.get("memory_mb", 0)
            load_at_startup = startup_config.get(model_type, {}).get(
                "load_at_startup", False
            )

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
        return [
            ["clone", f"error: {e}", "-", "—"],
            ["design", f"error: {e}", "-", "—"],
            ["custom", f"error: {e}", "-", "—"],
            ["asr", f"error: {e}", "-", "—"],
        ]


def toggle_model(model_type, action):
    """Load or unload a model.

    Phase 1b: ProgressIndicator is constructed at start so Gradio toasts /
    inline progress can wire into it; the function itself stays a normal
    return-tuple handler because the existing wiring in _facade.py uses a
    `lambda mt: toggle_model(mt, action)` — Gradio does not iterate generators
    returned from lambdas.

    Args:
        model_type: 'clone', 'design', or 'custom'
        action: 'load' or 'unload'

    Returns:
        Tuple of (status_message, model_table, status_html)
    """
    config = load_config()

    if not is_server_running(config):
        return "Server not running", get_model_table_data(), format_status_display()

    if action == "load":
        # Build a ProgressIndicator instance so log entries / future inline
        # progress wiring can read message + ETA. Also probes the server's
        # `loading: bool` field via poll_model_load_progress (Phase 1b
        # contract). Not yielded today because the blocking sync call leaves
        # no opportunity to update outputs mid-flight under the current
        # lambda-wrapped wiring.
        progress = poll_model_load_progress(model_type)
        eta_s = progress.get("eta_s")
        ProgressIndicator(
            mode="indeterminate",
            message=f"Loading {model_type}…"
            + (f" (~{int(eta_s)}s expected)" if eta_s else ""),
        )

    try:
        from qwen3_tts.core.http_client import server_request

        if action == "load":
            path = "/load-model"
        else:
            path = "/unload-model"

        resp = server_request(
            "POST",
            path,
            json={"model_type": model_type},
            timeout=120,
        )

        if resp.status_code == 200:
            result = resp.json()
            status = result.get("status", "done")
            return (
                f"Model {model_type}: {status}",
                get_model_table_data(),
                format_status_display(),
            )
        else:
            error = resp.json().get("error", "Unknown error")
            return f"Failed: {error}", get_model_table_data(), format_status_display()

    except Exception as e:
        logger.error("Model toggle failed: %s", e)
        return f"Error: {e}", get_model_table_data(), format_status_display()


def toggle_asr(action):
    """Load or unload the ASR model.

    Phase 1b: ProgressIndicator constructed at start for log/toast surfacing.
    Returns a tuple — generator-yield wiring incompatible with current
    `lambda: toggle_asr(action)` wrapper in _facade.py.

    Args:
        action: 'load' or 'unload'

    Returns:
        Tuple of (status_message, status_html)
    """
    config = load_config()

    if not is_server_running(config):
        return "Server not running", format_status_display()

    if action == "load":
        ProgressIndicator(mode="indeterminate", message="Loading ASR model…")

    try:
        from qwen3_tts.core.http_client import server_request

        if action == "load":
            path = "/load-asr"
        else:
            path = "/unload-asr"

        resp = server_request("POST", path, timeout=60)

        if resp.status_code == 200:
            result = resp.json()
            status = result.get("status", "done")
            return f"ASR: {status}", format_status_display()
        else:
            error = resp.json().get("error", "Unknown error")
            return f"Failed: {error}", format_status_display()

    except Exception as e:
        logger.error("ASR toggle failed: %s", e)
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
    existing_models = config.get("models", {})
    new_models = {
        **existing_models,
        "clone": {"load_at_startup": clone_startup},
        "design": {"load_at_startup": design_startup},
        "custom": {"load_at_startup": custom_startup},
    }
    save_config({**config, "models": new_models})

    return "Startup config updated (restart server to apply)", get_model_table_data()


#: Model types that carry a status badge in the UI, in display order.
MODEL_INDICATOR_TYPES = ("clone", "design", "custom")


def _badge_from_models_payload(data, model_type):
    """Render the status badge for *model_type* from a ``/models`` payload."""
    from qwen3_tts.interface.ui.components import status_badge

    info = data.get("models", {}).get(model_type, {}) or {}
    loaded = info.get("loaded", False)
    loading = info.get("loading", False)
    memory = info.get("memory_mb", 0)

    if loaded:
        return status_badge(f"Loaded ({memory:.0f}MB)", severity="success")
    if loading:
        return status_badge(f"Loading {model_type}...", severity="loading")
    return status_badge("Not loaded", severity="info")


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
        from qwen3_tts.core.http_client import server_request

        resp = server_request("GET", "/models", timeout=5)

        if resp.status_code != 200:
            return status_badge("Error", severity="error")

        return _badge_from_models_payload(resp.json(), model_type)

    except Exception as e:
        logger.error("Failed to get model status: %s", e)
        return status_badge("Error", severity="error")


def get_all_model_status_html():
    """Status badges for every mode indicator, from a single ``/models`` call.

    Equivalent to calling :func:`get_model_status_html` once per model type but
    issues one HTTP request instead of three — it is polled by the UI's refresh
    timer, so the request count matters.

    Returns:
        tuple: one HTML badge per entry in :data:`MODEL_INDICATOR_TYPES`.
    """
    from qwen3_tts.interface.ui.components import status_badge

    config = load_config()

    if not is_server_running(config):
        badge = status_badge("Server not running", severity="warning")
        return tuple(badge for _ in MODEL_INDICATOR_TYPES)

    try:
        from qwen3_tts.core.http_client import server_request

        resp = server_request("GET", "/models", timeout=5)

        if resp.status_code != 200:
            badge = status_badge("Error", severity="error")
            return tuple(badge for _ in MODEL_INDICATOR_TYPES)

        data = resp.json()
        return tuple(
            _badge_from_models_payload(data, model_type)
            for model_type in MODEL_INDICATOR_TYPES
        )

    except Exception as e:
        logger.error("Failed to get model status: %s", e)
        badge = status_badge("Error", severity="error")
        return tuple(badge for _ in MODEL_INDICATOR_TYPES)


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
