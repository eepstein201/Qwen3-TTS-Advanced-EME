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
import shutil
import threading
import time

import gradio as gr

from qwen3_tts.core.config import (
    CUSTOM_VOICE_SPEAKERS,
    get_backend,
    get_generation_presets,
    get_mlx_quantization,
    get_model_size,
    is_server_running,
    load_config,
)

logger = logging.getLogger("tts.ui")

# Constants
MAX_HISTORY_SIZE = 10

# Glyph shown in the "Remove" column of the Recent Generations table. Clicking
# that cell routes through on_history_select (column-aware) to delete the row.
HISTORY_REMOVE_GLYPH = "✕"
# Glyph for the "Download" column — copies the row's file into Manual Downloads
# (wired in Task 4; the column exists so get_history_data's row shape is built
# once). Clicking it currently falls through to the default replay branch.
HISTORY_DOWNLOAD_GLYPH = "⭳"
# Cell text shown while a row's action is armed (waiting for the confirming
# second click within DELETE_CONFIRM_TIMEOUT_S). Distinct from the resting
# glyph so the user sees the armed state without an extra status read.
HISTORY_REMOVE_ARMED_LABEL = "Confirm?"
HISTORY_DOWNLOAD_ARMED_LABEL = "Overwrite?"

# Thread-safe lock for history state updates. Shared by add_to_history
# (generation.py) and the clear/remove handlers (_facade.py) so concurrent
# generations and UI clicks can't race on history_state.
history_lock = threading.Lock()

# Derive speaker choices from canonical source
SPEAKER_CHOICES = [
    f"{key} ({info['lang']}) - {info['desc']}"
    for key, info in CUSTOM_VOICE_SPEAKERS.items()
]


def enhance_description_with_ai(description):
    """Enhance a brief voice description using an LLM API.

    Phase 1b: surfaces a `gr.Info` toast and constructs an inline
    ProgressIndicator while the LLM call is in flight, so the user knows
    something is happening during the round-trip (~2-5s).
    """
    from qwen3_tts.interface.ui.components import ProgressIndicator

    if not description or not description.strip():
        raise gr.Error("Please enter a description to enhance")

    # Visible toast + structured progress object (the indicator HTML is also
    # available for any inline gr.HTML that wires into this handler).
    progress = ProgressIndicator(mode="indeterminate", message="Enhancing description…")
    try:
        gr.Info(progress.message)
    except Exception:  # nosec B110  # gr.Info raises in non-event contexts (e.g. tests); cosmetic UI toast, safe to swallow
        pass

    config = load_config()
    enhancer_config = config.get("prompt_enhancer", {})

    if not enhancer_config.get("enabled", False):
        raise gr.Error(
            "AI enhancement is not enabled. Set prompt_enhancer.enabled=true in config.json"
        )

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
                    {
                        "role": "user",
                        "content": f"Expand this voice description: {description}",
                    },
                ],
                system=system_prompt,
            )
            return response.content[0].text.strip()
        else:
            raise gr.Error(f"Unsupported provider: {provider}")
    except Exception as e:
        logger.error("AI enhancement failed: %s", e)
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
            from qwen3_tts.core.http_client import server_request

            resp = server_request("GET", "/models", timeout=5)
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
            logger.warning("Could not fetch model settings from server: %s", e)

    return model_size, mlx_quant, backend


def apply_model_settings(model_size, mlx_quantization):
    """Apply model settings to server, reloading previously loaded models."""
    if not is_server_running(load_config()):
        return "Server not running", format_status_display()

    try:
        from qwen3_tts.core.http_client import server_request

        # Step 1: Remember which models are loaded
        models_loaded = []
        try:
            resp = server_request("GET", "/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for model_type in ("clone", "design", "custom"):
                    if data.get("models", {}).get(model_type, {}).get("loaded"):
                        models_loaded.append(model_type)
        except Exception as e:
            logger.warning(
                "Could not determine loaded-model state (%s); skipping reload", e
            )

        # Step 2: Apply config change (unloads all models)
        payload = {"model_size": model_size}
        backend = get_backend()
        if backend == "mlx" and mlx_quantization:
            payload["mlx_quantization"] = mlx_quantization

        resp = server_request("POST", "/update-model-config", json=payload, timeout=10)
        if resp.status_code != 200:
            error = resp.json().get("error", "Unknown error")
            return f"Failed: {error}", format_status_display()

        # Step 3: Reload previously loaded models with new settings
        if models_loaded:
            reloaded = []
            for model_type in models_loaded:
                try:
                    r = server_request(
                        "POST",
                        "/load-model",
                        json={"model_type": model_type},
                        timeout=120,
                    )
                    if r.status_code == 200:
                        reloaded.append(model_type)
                        # Wait for model to be actually loaded (check /health endpoint)
                        for _ in range(60):  # Wait up to 30 seconds
                            time.sleep(0.5)
                            health_resp = server_request("GET", "/health", timeout=5)
                            if health_resp.status_code == 200:
                                health_data = health_resp.json()
                                model_key = f"{model_type}_model_loaded"
                                if health_data.get(model_key):
                                    break
                except Exception as e:
                    logger.warning(
                        "Failed to reload %s model after settings change: %s",
                        model_type,
                        e,
                    )
            if reloaded:
                return (
                    f"Settings applied. Reloaded: {', '.join(reloaded)}",
                    format_status_display(),
                )
            return (
                "Settings applied. Models unloaded (reload failed)",
                format_status_display(),
            )

        return "Settings applied (models were not loaded)", format_status_display()
    except Exception as e:
        return f"Error: {e}", format_status_display()


def update_text_info(text):
    """Update text info display with character count, chunk estimate, and limit warning."""
    if not text or not text.strip():
        return ""
    chars = len(text)
    config = load_config()
    max_chunk_chars = config.get("generation", {}).get("max_chunk_chars", 500)
    if max_chunk_chars and max_chunk_chars > 0:
        chunks = max(1, (chars + max_chunk_chars - 1) // max_chunk_chars)
    else:
        chunks = 1
    info = f"{chars} chars"
    if chunks > 1:
        info += f" | ~{chunks} chunks"
    max_text_length = config.get("security", {}).get("max_text_length", 50000)
    if chars > max_text_length:
        info += f" ⚠️ exceeds {max_text_length} char limit — generation will be rejected"
    return info


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
        for _key in (
            "mlx_memory_active_mb",
            "mps_memory_allocated_mb",
            "cuda_memory_allocated_mb",
        ):
            _v = stats.get(_key)
            if _v is not None:
                memory_val = _v
                break
        memory = (
            f"{memory_val:.1f}MB" if isinstance(memory_val, (int, float)) else "N/A"
        )

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
        import html as html_mod

        status_html = f'<span style="color: orange;">{html_mod.escape(status)}</span>'

    import html as html_mod

    return f"""
    <div role="status" aria-live="polite" style="padding: 10px; background: var(--block-background-fill, #f5f5f5); border-radius: 5px; margin-bottom: 15px; border: 1px solid var(--block-border-color, #e0e0e0);">
        <strong>Status:</strong> {status_html} |
        <strong>Backend:</strong> {html_mod.escape(str(backend))} |
        <strong>Memory:</strong> {html_mod.escape(str(memory))} |
        <strong>Models:</strong> {html_mod.escape(str(models))}
    </div>
    """


def get_voice_prompts():
    """Get list of available voice prompts, filtered by current backend."""
    from qwen3_tts.core.config import VOICE_PROMPTS_DIR

    backend = get_backend()
    try:
        files = os.listdir(VOICE_PROMPTS_DIR)
    except OSError:
        return []
    if backend == "mlx":
        txt_bases = {f[:-4] for f in files if f.endswith(".txt")}
        return sorted(f for f in files if f.endswith(".wav") and f[:-4] in txt_bases)
    else:
        return sorted(f for f in files if f.endswith(".pt"))


def get_presets():
    """Get list of available generation presets (defaults + user-defined)."""
    config = load_config()
    return ["(none)"] + list(get_generation_presets(config).keys())


def get_voice_metadata(name: str) -> dict:
    """Get metadata for a voice prompt.

    Args:
        name: Voice prompt name

    Returns:
        Dict with keys: name, formats, size_bytes, created, is_default, size_mb, duration
    """
    from qwen3_tts.server.client import TTSClient

    try:
        client = TTSClient()
        if not client.is_server_running():
            return {"name": name, "error": "Server not running"}

        details = client.get_prompt_details(name=name)

        # Add computed fields
        size_mb = details.get("size_bytes", 0) / (1024 * 1024)
        result = {
            **details,
            "size_mb": round(size_mb, 2),
        }

        # Try to get duration from .wav file
        if ".wav" in details.get("formats", []):
            try:
                import soundfile as sf

                from qwen3_tts.core.config import (
                    VOICE_PROMPTS_DIR,
                    safe_path_join,
                )

                # safe_path_join rejects traversal in the (user-supplied) name;
                # a ValueError is caught below and surfaces as duration "N/A".
                wav_path = safe_path_join(str(VOICE_PROMPTS_DIR), f"{name}.wav")
                if os.path.exists(wav_path):
                    info = sf.info(wav_path)
                    duration = info.duration
                    result["duration"] = f"{duration:.1f}s"
            except Exception as e:
                logger.warning("Could not read WAV duration for %s: %s", name, e)
                result["duration"] = "N/A"
        else:
            result["duration"] = "N/A"

        return result

    except Exception as e:
        logger.error("Failed to get voice metadata for '%s': %s", name, e)
        return {"name": name, "error": str(e)}


def add_to_history(history_list, mode, text, output_path, duration_chunks, seed=None):
    """Add a generation to history.

    Args:
        history_list: Existing history (not mutated).
        mode: Generation mode string (e.g. "clone"). Stored capitalized.
        text: Generated text. Truncated to 40 chars + "..." if longer.
        output_path: Path to the output audio file.
        duration_chunks: Number of audio chunks (int).
        seed: Optional integer seed used for generation.

    Returns:
        New list with the entry prepended, capped at MAX_HISTORY_SIZE.
    """
    import time

    entry = {
        "timestamp": time.time(),
        "mode": mode.capitalize() if mode else mode,
        "text": text[:40] + "..." if len(text) > 40 else text,
        # Full transcript retained for copy-to-clipboard (text above is the
        # truncated display form). Not persisted by add_to_history itself; the
        # .json sidecar already stores the full text and repopulates this on load.
        "full_text": text,
        "path": output_path,
        "chunks": duration_chunks if isinstance(duration_chunks, int) else 0,
        "seed": seed,
    }
    new_list = [entry] + list(history_list)
    return new_list[:MAX_HISTORY_SIZE]


def remove_history_row(history_list, row_index):
    """Return a new history list with the row at row_index removed.

    Immutable: the input list is never mutated. Out-of-range or negative
    indices return an unchanged copy. Non-list input returns [].
    """
    if not isinstance(history_list, list):
        return []
    if isinstance(row_index, int) and 0 <= row_index < len(history_list):
        return history_list[:row_index] + history_list[row_index + 1 :]
    return list(history_list)


def remove_history_row_by_path(history_list, path):
    """Return a new history list with the entry whose "path" matches removed.

    Keyed by path rather than row index: a generation completing between a
    two-step confirm's two clicks prepends a row and shifts every index, so an
    index-keyed confirm could delete the wrong entry. Immutable — the input
    list is never mutated; a fresh list is always returned.
    """
    if not isinstance(history_list, list):
        return []
    return [e for e in history_list if e.get("path") != path]


def clear_history(history_list=None):
    """Return a fresh empty history list (list-only clear; disk files untouched)."""
    return []


def get_history_data(history_list, armed_delete_path=None, armed_download_path=None):
    """Convert history list to list-of-lists format.

    Args:
        armed_delete_path: when set, the matching row's "Remove" cell renders
            HISTORY_REMOVE_ARMED_LABEL ("Confirm?") instead of the glyph, so the
            user sees the armed two-step state inline.
        armed_download_path: same idea for the "Download" cell (Task 4).

    Returns:
        List of [time, mode, text, seed, chunks, remove, download] rows.
    """
    import datetime

    if not history_list:
        return []

    rows = []
    for entry in history_list:
        ts = entry.get("timestamp", 0)
        time_str = (
            datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""
        )
        seed_val = entry.get("seed")
        seed_str = str(seed_val) if seed_val is not None else "-"
        is_armed_delete = (
            armed_delete_path is not None and entry.get("path") == armed_delete_path
        )
        is_armed_download = (
            armed_download_path is not None
            and entry.get("path") == armed_download_path
        )
        rows.append(
            [
                time_str,
                entry.get("mode", "?"),
                entry.get("text", ""),
                seed_str,
                entry.get("chunks", 0),
                HISTORY_REMOVE_ARMED_LABEL if is_armed_delete else HISTORY_REMOVE_GLYPH,
                (
                    HISTORY_DOWNLOAD_ARMED_LABEL
                    if is_armed_download
                    else HISTORY_DOWNLOAD_GLYPH
                ),
            ]
        )

    return rows


def _resolve_output_dir(config: dict) -> str:
    """Resolve and validate output_directory from config.

    Falls back to ~/Downloads if path resolves outside home directory.
    """
    raw = config.get("output_directory", "~/Downloads")
    resolved = os.path.realpath(os.path.expanduser(raw))
    home = os.path.realpath(os.path.expanduser("~"))
    if not (resolved == home or resolved.startswith(home + os.sep)):
        logger.warning(
            "output_directory %r resolves outside home; falling back to ~/Downloads",
            raw,
        )
        resolved = os.path.realpath(os.path.expanduser("~/Downloads"))
    return resolved


# Fixed subfolder names beneath history_output_directory. Deliberately not
# configurable: only their shared parent is, so one setting moves both.
AUTOMATED_OUTPUT_SUBDIR = "Automated Output"
MANUAL_DOWNLOADS_SUBDIR = "Manual Downloads"
DEFAULT_HISTORY_OUTPUT_DIR = "~/Downloads/Qwen3-TTS Output"


def resolve_history_output_dir(config: dict) -> str:
    """Resolve the parent folder for web-UI generation output.

    Falls back to the default when the configured path escapes the home
    directory (traversal or an absolute path elsewhere). Does not create
    anything — callers that need the directory to exist call
    ``ensure_history_dirs``.
    """
    raw = config.get("history_output_directory", DEFAULT_HISTORY_OUTPUT_DIR)
    resolved = os.path.realpath(os.path.expanduser(raw))
    home = os.path.realpath(os.path.expanduser("~"))
    if not (resolved == home or resolved.startswith(home + os.sep)):
        logger.warning(
            "history_output_directory %r resolves outside home; using default",
            raw,
        )
        return os.path.realpath(os.path.expanduser(DEFAULT_HISTORY_OUTPUT_DIR))
    return resolved


def resolve_automated_output_dir(config: dict) -> str:
    """Resolve the subfolder every web-UI generation is saved into."""
    return os.path.join(resolve_history_output_dir(config), AUTOMATED_OUTPUT_SUBDIR)


def resolve_manual_downloads_dir(config: dict) -> str:
    """Resolve the subfolder the per-row Download action copies into."""
    return os.path.join(resolve_history_output_dir(config), MANUAL_DOWNLOADS_SUBDIR)


def ensure_history_dirs(config: dict) -> tuple:
    """Create both history subfolders if absent. Idempotent.

    Returns (automated_output_dir, manual_downloads_dir).
    """
    automated = resolve_automated_output_dir(config)
    manual = resolve_manual_downloads_dir(config)
    os.makedirs(automated, exist_ok=True)
    os.makedirs(manual, exist_ok=True)
    return automated, manual


def delete_generation_files(path: str, config: dict) -> bool:
    """Delete a generation's .wav and .json sidecar from Automated Output.

    Returns True when the path was inside Automated Output (whether or not the
    files still existed), False when it was refused for being outside. Refusing
    rather than raising keeps a stale history row from breaking the UI: a row
    whose file is already gone, or whose path somehow escaped the folder, must
    not raise and leave the panel wedged.

    The containment check is the only thing standing between a user click and an
    ``os.remove``, so it is strict: the resolved path must live strictly beneath
    the resolved Automated Output root (``root + os.sep``), never the root itself.
    """
    automated = resolve_automated_output_dir(config)
    resolved = os.path.realpath(os.path.expanduser(path))
    root = os.path.realpath(automated)
    if not resolved.startswith(root + os.sep):
        logger.warning("Refusing to delete %r: outside Automated Output", path)
        return False
    for target in (resolved, resolved.replace(".wav", ".json")):
        try:
            os.remove(target)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not delete %r: %s", target, exc)
    return True


def copy_to_manual_downloads(path: str, config: dict, overwrite: bool = False) -> str:
    """Copy a generation into Manual Downloads under its original filename.

    Returns:
        "copied"  — the file was written (new, or overwrite=True)
        "exists"  — a file of that name is already there and overwrite is False
        "refused" — the source is outside Automated Output

    Same strict containment as delete_generation_files: the source must live
    strictly beneath the resolved Automated Output root, so a crafted history
    row can't exfiltrate an arbitrary file into the user's Downloads.
    """
    automated = os.path.realpath(resolve_automated_output_dir(config))
    resolved = os.path.realpath(os.path.expanduser(path))
    if not resolved.startswith(automated + os.sep):
        logger.warning("Refusing to copy %r: outside Automated Output", path)
        return "refused"

    manual = resolve_manual_downloads_dir(config)
    os.makedirs(manual, exist_ok=True)
    dest = os.path.join(manual, os.path.basename(resolved))
    if os.path.exists(dest) and not overwrite:
        return "exists"
    shutil.copy2(resolved, dest)
    return "copied"


def save_generation_metadata(wav_path: str, metadata: dict) -> None:
    """Save generation metadata as JSON sidecar alongside a .wav file.

    Args:
        wav_path: Path to the .wav file.
        metadata: Dict of generation params (not mutated).
    """
    import json as json_mod

    from qwen3_tts.core.config import safe_path_join

    # Security: validate wav_path against traversal before deriving json_path
    expanded = os.path.expanduser(wav_path)
    if os.path.isabs(expanded):
        # Absolute paths: reject if they contain traversal sequences
        if ".." in expanded:
            raise ValueError(f"Path traversal detected in wav_path: {wav_path}")
        safe_wav_path = expanded
    else:
        # Relative paths: validate to prevent traversal
        safe_wav_path = safe_path_join(os.getcwd(), expanded)

    # Verify safe_wav_path is under home directory (user data should be in home)
    home = os.path.realpath(os.path.expanduser("~"))
    resolved_wav = os.path.realpath(safe_wav_path)
    if not (resolved_wav == home or resolved_wav.startswith(home + os.sep)):
        raise ValueError(f"wav_path must be under home directory: {wav_path}")

    json_path = safe_wav_path.replace(".wav", ".json")
    with open(json_path, "w") as f:
        json_mod.dump(metadata, f, indent=2)


def load_history_from_disk(output_dir: str) -> list:
    """Load generation history from JSON sidecar files in output directory.

    Scan for voice_ui_*.json files, pair with .wav files, return
    newest-first list capped at MAX_HISTORY_SIZE.
    """
    import glob
    import json as json_mod

    from qwen3_tts.core.config import safe_path_join

    # Security: validate output_dir against traversal before globbing
    expanded = os.path.expanduser(output_dir)
    if os.path.isabs(expanded):
        # Absolute paths: reject if they contain traversal sequences
        if ".." in expanded:
            raise ValueError(f"Path traversal detected in output_dir: {output_dir}")
        safe_output_dir = expanded
    else:
        # Relative paths: validate to prevent traversal
        safe_output_dir = safe_path_join(os.getcwd(), expanded)

    # Verify safe_output_dir is under home directory
    home = os.path.realpath(os.path.expanduser("~"))
    resolved_dir = os.path.realpath(safe_output_dir)
    if not (resolved_dir == home or resolved_dir.startswith(home + os.sep)):
        raise ValueError(f"output_dir must be under home directory: {output_dir}")

    json_files = glob.glob(os.path.join(safe_output_dir, "voice_ui_*.json"))
    entries: list = []
    for jf in json_files:
        wav_path = jf.replace(".json", ".wav")
        if not os.path.exists(wav_path):
            continue
        try:
            with open(jf) as f:
                data = json_mod.load(f)
            text = data.get("text", "")
            entries.append(
                {
                    "timestamp": data.get("timestamp", 0),
                    "mode": (data.get("mode") or "?").capitalize(),
                    "text": text[:40] + "..." if len(text) > 40 else text,
                    # Sidecar stores the full transcript under "text"; retain it
                    # verbatim for copy-to-clipboard (falls back to display text).
                    "full_text": text,
                    "path": wav_path,
                    "chunks": data.get("chunks", 0),
                    "seed": data.get("seed"),
                    "temperature": data.get("temperature"),
                    "top_k": data.get("top_k"),
                    "top_p": data.get("top_p"),
                    "repetition_penalty": data.get("repetition_penalty"),
                    "prompt_file": data.get("prompt_file"),
                    "voice_description": data.get("voice_description"),
                    "speaker": data.get("speaker"),
                }
            )
        except (ValueError, OSError, KeyError):
            continue
    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return entries[:MAX_HISTORY_SIZE]


def load_history_from_disk_for_config(config: dict) -> list:
    """Load history from the configured Automated Output subfolder.

    Thin wrapper over load_history_from_disk that resolves the directory from
    config, so callers don't each re-derive the path. Returns [] when the
    folder does not exist yet (fresh install, before the first generation).
    """
    automated = resolve_automated_output_dir(config)
    if not os.path.isdir(automated):
        return []
    return load_history_from_disk(automated)


def refresh_history_from_disk(
    history_list, config, armed_delete_path=None, armed_download_path=None
):
    """Re-derive ``(history_state, history_df)`` from disk, not ``history_list``.

    ``demo.load()``'s preload and a generation chain's refresh are independent
    Gradio events with no guaranteed delivery order, so a stale in-memory list
    could previously win and render an unrelated row. Both paths now re-derive
    from the same source of truth (the Automated Output sidecars), making the
    outcome order-independent. ``history_list`` is accepted for API symmetry
    with :func:`get_history_data` but intentionally ignored — disk wins.

    Returns both halves because every caller needs both: the generation chains
    write them to ``[history_state, history_df]`` in one step. Returning only
    the rows is what left ``_facade._refresh_history`` open-coding the same
    two lines (repo-audit-2026-07-31 P1-2).

    Safe only because Remove hard-deletes the file: a soft delete would let a
    removed row reappear on the next re-read.
    """
    entries = load_history_from_disk_for_config(config)
    return entries, get_history_data(
        entries,
        armed_delete_path=armed_delete_path,
        armed_download_path=armed_download_path,
    )


def get_gradio_launch_kwargs(config: dict) -> dict:
    """Shared Gradio launch() kwargs -- single source of truth for all UI entry points."""
    import tempfile

    from qwen3_tts.core.config import IN_COLAB

    output_dir = _resolve_output_dir(config)
    downloads = os.path.realpath(os.path.expanduser("~/Downloads"))
    allowed = list({output_dir, downloads, tempfile.gettempdir()})

    return {
        "server_name": "0.0.0.0" if IN_COLAB else "127.0.0.1",  # nosec B104
        "allowed_paths": allowed,
        "theme": gr.themes.Soft(),
        "css": ".gr-hidden { display: none !important; height: 0 !important; overflow: hidden !important; }",
    }
