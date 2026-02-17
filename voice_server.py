#!/usr/bin/env python3
"""Persistent TTS server that keeps models loaded in memory for fast generation."""

import json as _json
import logging
import logging.handlers
import os
import secrets
import shutil
import signal
import sys
import tempfile
import threading
import time
from collections import deque

from flask import Flask, request, jsonify, send_file
import soundfile as sf

logger = logging.getLogger("tts")

from voice_config import (
    CONFIG_PATH,
    VOICE_PROMPTS_DIR,
    PID_FILE,
    TOKEN_FILE,
    MODEL_INFO,
    MLX_MODEL_INFO,
    CUSTOM_VOICE_SPEAKERS,
    load_config,
    save_config,
    get_backend,
    get_default_clone_prompt,
    set_default_clone_prompt,
    get_torch_dtype_name,
    get_mlx_quantization,
    get_model_size,
)
from voice_engine import (
    load_model,
    load_voice_prompt,
    run_inference,
    run_inference_streaming,
    voice_prompt_cache_info,
    clear_voice_prompt_cache,
)

# Pre-computed valid speaker names (keys + display names)
_VALID_SPEAKER_NAMES = frozenset(CUSTOM_VOICE_SPEAKERS.keys()) | frozenset(
    v["name"] for v in CUSTOM_VOICE_SPEAKERS.values()
)

# Auth token for this server session
auth_token = None

app = Flask(__name__)
app.json.sort_keys = False


def generate_auth_token():
    """Generate a new auth token and write it to TOKEN_FILE."""
    global auth_token
    auth_token = secrets.token_hex(32)
    with open(TOKEN_FILE, "w") as f:
        f.write(auth_token)
    os.chmod(TOKEN_FILE, 0o600)
    return auth_token


# Endpoints that don't require authentication
PUBLIC_ENDPOINTS = {"health", "generation_status", "static"}


@app.before_request
def check_auth():
    """Verify Bearer token on all endpoints except public ones."""
    if auth_token is None:
        return jsonify({"error": "Server not ready", "recovery": "retry"}), 503

    endpoint = request.endpoint
    if endpoint in PUBLIC_ENDPOINTS:
        return

    auth_header = request.headers.get("Authorization", "")
    if auth_header == f"Bearer {auth_token}":
        return

    return jsonify({
        "error": "Authentication required",
        "detail": "Include 'Authorization: Bearer <token>' header. Token is in ~/.voice_server_token",
        "recovery": "restart",
    }), 401


# Global model holders
clone_model = None
design_model = None
custom_model = None
model_load_times = {}  # model_type -> seconds

# Thread safety for generation
generation_lock = threading.Lock()
request_queue = set()
max_concurrent = 1  # TTS generation is memory-intensive, serialize by default

# Auto-shutdown timer
shutdown_timer = None
last_activity = time.time()

# Store config globally for auto-shutdown settings
server_config = {}

# Generation state for progress tracking
generation_state = {
    "active": False,
    "start_time": 0.0,
    "text_length": 0,
    "mode": "",
    "batch_index": 0,
    "batch_total": 0,
    "chunk_index": 0,
    "chunk_total": 0,
    "generation_id": None,
    "cancelled": False,
}


def reset_activity_timer():
    """Reset the auto-shutdown timer on activity."""
    global shutdown_timer, last_activity
    last_activity = time.time()

    auto_shutdown_minutes = server_config.get("auto_shutdown_minutes", 0)
    if auto_shutdown_minutes <= 0:
        return  # Auto-shutdown disabled

    # Cancel existing timer
    if shutdown_timer is not None:
        shutdown_timer.cancel()

    # Start new timer
    shutdown_timer = threading.Timer(
        auto_shutdown_minutes * 60,
        auto_shutdown
    )
    shutdown_timer.daemon = True
    shutdown_timer.start()


def auto_shutdown():
    """Auto-shutdown due to inactivity."""
    logger.info("Auto-shutdown: No activity for %d minutes.", server_config.get("auto_shutdown_minutes", 0))
    cleanup_pid()


def _get_model(model_type):
    """Return the loaded model for a given type, or None."""
    if model_type == "clone":
        return clone_model
    elif model_type == "design":
        return design_model
    elif model_type == "custom":
        return custom_model
    return None


def load_single_model(model_type):
    """Load a single model by type using voice_engine."""
    global clone_model, design_model, custom_model

    valid_types = ("clone", "design", "custom")
    if model_type not in valid_types:
        return False

    # Get model info for logging (use current size)
    from voice_config import get_model_info
    info = get_model_info(model_type)
    model_name = info.get("name", info.get("name_template", model_type))

    logger.info("Loading %s...", model_name)
    t0 = time.time()
    model = load_model(model_type)

    if model_type == "clone":
        clone_model = model
    elif model_type == "design":
        design_model = model
    elif model_type == "custom":
        custom_model = model

    elapsed = round(time.time() - t0, 1)
    model_load_times[model_type] = elapsed
    logger.info("Loaded %s model successfully in %.1fs.", model_type, elapsed)
    return True


def load_models():
    """Load TTS models based on config."""
    models_config = server_config.get("models", {})

    # Default: load clone model if no config
    if not models_config:
        models_config = {"clone": {"load_at_startup": True}}

    models_to_load = []
    for model_type, settings in models_config.items():
        if settings.get("load_at_startup", False):
            models_to_load.append(model_type)

    if not models_to_load:
        logger.warning("No models configured to load at startup.")
        return

    logger.info("Loading %d model(s): %s", len(models_to_load), ", ".join(models_to_load))

    for model_type in models_to_load:
        load_single_model(model_type)

    logger.info("Model loading complete.")


@app.route("/health", methods=["GET"])
def health():
    reset_activity_timer()
    backend = get_backend()
    data = {
        "status": "ok",
        "backend": backend,
        "model_size": get_model_size(),
        "clone_model_loaded": clone_model is not None,
        "design_model_loaded": design_model is not None,
        "custom_model_loaded": custom_model is not None,
        "model_load_times": dict(model_load_times),
    }
    if backend == "mlx":
        data["mlx_quantization"] = get_mlx_quantization()
    else:
        data["dtype"] = get_torch_dtype_name()
    return jsonify(data)


@app.route("/generation-status", methods=["GET"])
def generation_status():
    """Return current generation state for progress display. No auth required."""
    state = dict(generation_state)
    if state["active"]:
        state["elapsed_sec"] = round(time.time() - state["start_time"], 1)
        # Estimate ETA from history median chars/sec
        state["eta_sec"] = _estimate_eta(state["text_length"], state["elapsed_sec"])
    return jsonify(state)


@app.route("/cancel-generation", methods=["POST"])
def cancel_generation():
    """Cancel the current streaming generation. Requires auth."""
    if not generation_state["active"]:
        return jsonify({"status": "no_active_generation"})

    generation_state["cancelled"] = True
    logger.info("Generation cancellation requested")
    return jsonify({"status": "cancellation_requested", "generation_id": generation_state.get("generation_id")})


# ---------------------------------------------------------------------------
# ETA cache — avoids reading .jsonl on every 1s poll
# ---------------------------------------------------------------------------

_eta_cache = {"median_rate": None, "last_updated": 0}
_ETA_CACHE_TTL = 30  # seconds


def _estimate_eta(text_length, elapsed_sec):
    """Estimate remaining seconds from history data.

    Uses a cached median chars/sec rate (refreshed every 30s) to avoid
    reading the history file on every 1-second progress poll.
    """
    from voice_config import HISTORY_FILE

    now = time.time()

    # Refresh cache if stale
    if now - _eta_cache["last_updated"] > _ETA_CACHE_TTL:
        try:
            if not os.path.exists(HISTORY_FILE):
                _eta_cache["median_rate"] = None
            else:
                with open(HISTORY_FILE, "r") as f:
                    lines = deque(f, maxlen=20)
                rates = []
                for line in lines:
                    entry = _json.loads(line)
                    dur = entry.get("duration_sec")
                    tl = entry.get("text_length")
                    if dur and tl and dur > 0:
                        rates.append(tl / dur)
                if rates:
                    rates.sort()
                    _eta_cache["median_rate"] = rates[len(rates) // 2]
                else:
                    _eta_cache["median_rate"] = None
        except Exception:
            _eta_cache["median_rate"] = None
        _eta_cache["last_updated"] = now

    median_rate = _eta_cache["median_rate"]
    if median_rate is None:
        return None

    estimated_total = text_length / median_rate
    remaining = max(0, estimated_total - elapsed_sec)
    return round(remaining, 1)


# ---------------------------------------------------------------------------
# Generation result cache — caches recent results by input hash
# ---------------------------------------------------------------------------

import hashlib

_gen_cache = {}  # key -> {"file": path, "sample_rate": int, "timestamp": float}
_gen_cache_lock = threading.Lock()
_GEN_CACHE_MAX = 5


def _gen_cache_key(text, mode, gen_params, prompt_file=None, voice_description=None,
                   speaker=None, instruct=None):
    """Generate a hash key for generation cache lookup."""
    key_parts = [text, mode, str(sorted(gen_params.items()))]
    if prompt_file:
        key_parts.append(prompt_file)
    if voice_description:
        key_parts.append(voice_description)
    if speaker:
        key_parts.append(speaker)
    if instruct:
        key_parts.append(instruct)
    raw = "|".join(key_parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _gen_cache_get(key):
    """Get a cached generation result. Returns file path or None."""
    with _gen_cache_lock:
        entry = _gen_cache.get(key)
        if entry and os.path.exists(entry["file"]):
            return entry
        # Clean up stale entry
        if entry:
            _gen_cache.pop(key, None)
        return None


def _gen_cache_put(key, file_path, sample_rate):
    """Store a generation result in cache, evicting oldest if full."""
    with _gen_cache_lock:
        if len(_gen_cache) >= _GEN_CACHE_MAX:
            # Evict oldest by timestamp
            oldest_key = min(_gen_cache, key=lambda k: _gen_cache[k]["timestamp"])
            old_entry = _gen_cache.pop(oldest_key)
            # Clean up old file
            try:
                if os.path.exists(old_entry["file"]):
                    os.remove(old_entry["file"])
            except OSError:
                pass
        _gen_cache[key] = {
            "file": file_path,
            "sample_rate": sample_rate,
            "timestamp": time.time(),
        }


def _gen_cache_invalidate():
    """Invalidate all cached generation results."""
    with _gen_cache_lock:
        for entry in _gen_cache.values():
            try:
                if os.path.exists(entry["file"]):
                    os.remove(entry["file"])
            except OSError:
                pass
        _gen_cache.clear()


@app.route("/stats", methods=["GET"])
def stats():
    """Return server statistics including memory usage."""
    reset_activity_timer()

    idle_seconds = int(time.time() - last_activity)
    auto_shutdown_minutes = server_config.get("auto_shutdown_minutes", 0)

    cache_info = voice_prompt_cache_info()

    backend = get_backend()
    stats_data = {
        "status": "ok",
        "backend": backend,
        "clone_model_loaded": clone_model is not None,
        "design_model_loaded": design_model is not None,
        "custom_model_loaded": custom_model is not None,
        "voice_prompts_cached": cache_info.currsize,
        "voice_prompts_cache_hits": cache_info.hits,
        "idle_seconds": idle_seconds,
        "auto_shutdown_minutes": auto_shutdown_minutes if auto_shutdown_minutes > 0 else "disabled",
        "generation_queue_size": len(request_queue),
    }
    if backend == "mlx":
        stats_data["mlx_quantization"] = get_mlx_quantization()
    else:
        stats_data["dtype"] = get_torch_dtype_name()

    # GPU memory stats (lazy torch import — only when torch backend is active)
    try:
        import torch
        if torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory()
                stats_data["mps_memory_allocated_mb"] = round(allocated / (1024 * 1024), 2)
            except Exception:
                stats_data["mps_memory_allocated_mb"] = "unavailable"

        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                stats_data["cuda_memory_allocated_mb"] = round(allocated / (1024 * 1024), 2)
                stats_data["cuda_memory_reserved_mb"] = round(reserved / (1024 * 1024), 2)
            except Exception:
                pass
    except ImportError:
        pass

    # MLX memory stats (when mlx backend is active)
    if backend == "mlx":
        try:
            import mlx.core as mx
            active_mem = mx.metal.get_active_memory()
            peak_mem = mx.metal.get_peak_memory()
            stats_data["mlx_memory_active_mb"] = round(active_mem / (1024 * 1024), 2)
            stats_data["mlx_memory_peak_mb"] = round(peak_mem / (1024 * 1024), 2)
        except Exception:
            pass

    return jsonify(stats_data)


@app.route("/models", methods=["GET"])
def list_models():
    """Return information about available models and their load status."""
    reset_activity_timer()

    backend = get_backend()
    model_size = get_model_size()

    # Get size-specific model info
    size_model_info = MODEL_INFO.get(model_size, MODEL_INFO["1.7B"])
    size_mlx_info = MLX_MODEL_INFO.get(model_size, MLX_MODEL_INFO["1.7B"])

    models_data = {}
    for model_type, info in size_model_info.items():
        loaded = _get_model(model_type) is not None
        # Check startup config
        models_cfg = server_config.get("models", {})
        load_at_startup = models_cfg.get(model_type, {}).get("load_at_startup", False)

        entry = {
            "loaded": loaded,
            "description": info["description"],
            "memory_mb": info["memory_mb"],
            "repo_id": info["name"],
            "load_at_startup": load_at_startup,
            "load_time_sec": model_load_times.get(model_type),
        }
        # Include MLX repo ID when using MLX backend
        if backend == "mlx":
            mlx_info = size_mlx_info.get(model_type)
            if mlx_info:
                from voice_config import get_mlx_model_name
                entry["repo_id"] = get_mlx_model_name(model_type)
                entry["memory_mb"] = mlx_info["memory_mb"]
        models_data[model_type] = entry

    return jsonify({"models": models_data, "backend": backend, "model_size": model_size})


@app.route("/load-model", methods=["POST"])
def load_model_endpoint():
    """Load a model on demand."""
    reset_activity_timer()

    data = request.json
    model_type = data.get("model_type")

    if not model_type:
        return jsonify({"error": "model_type required", "recovery": "config"}), 400

    valid_types = ("clone", "design", "custom")
    if model_type not in valid_types:
        return jsonify({
            "error": f"Unknown model type: {model_type}. Valid: {', '.join(valid_types)}",
            "recovery": "config",
        }), 400

    # Check if already loaded
    if _get_model(model_type) is not None:
        return jsonify({"status": "already_loaded", "model": model_type})

    # Load the model (this may take a while)
    try:
        with generation_lock:
            success = load_single_model(model_type)
    except ImportError as e:
        return jsonify({
            "error": f"Backend not available for model loading: {model_type}",
            "detail": str(e),
            "recovery": "config",
        }), 500
    except Exception as e:
        logger.error("Failed to load model %s: %s", model_type, e, exc_info=True)
        return jsonify({
            "error": f"Failed to load model: {model_type}",
            "detail": str(e),
            "recovery": "restart",
        }), 500

    if success:
        return jsonify({"status": "loaded", "model": model_type})
    else:
        return jsonify({
            "error": f"Failed to load model: {model_type}",
            "recovery": "restart",
        }), 500


@app.route("/update-model-config", methods=["POST"])
def update_model_config():
    """Update model size and/or quantization settings.

    Accepts JSON body with optional keys:
      - model_size: "1.7B" or "0.6B"
      - mlx_quantization: "4bit", "8bit", or "bf16"

    After updating config.json, unloads current models so the new
    settings take effect on next generation.
    """
    reset_activity_timer()
    data = request.json or {}

    new_size = data.get("model_size")
    new_quant = data.get("mlx_quantization")

    if not new_size and not new_quant:
        return jsonify({
            "error": "At least one of model_size or mlx_quantization required",
            "recovery": "config",
        }), 400

    # Validate values
    valid_sizes = ("1.7B", "0.6B")
    valid_quants = ("4bit", "8bit", "bf16")

    if new_size and new_size not in valid_sizes:
        return jsonify({
            "error": f"Invalid model_size: {new_size}. Valid: {', '.join(valid_sizes)}",
            "recovery": "config",
        }), 400

    if new_quant and new_quant not in valid_quants:
        return jsonify({
            "error": f"Invalid mlx_quantization: {new_quant}. Valid: {', '.join(valid_quants)}",
            "recovery": "config",
        }), 400

    # Update config.json
    from voice_config import load_config, save_config
    config = load_config()
    if "advanced" not in config:
        config["advanced"] = {}

    changes = []
    if new_size:
        config["advanced"]["model_size"] = new_size
        changes.append(f"model_size={new_size}")
    if new_quant:
        config["advanced"]["mlx_quantization"] = new_quant
        changes.append(f"mlx_quantization={new_quant}")

    save_config(config)

    # Unload all models so new settings take effect
    global clone_model, design_model, custom_model
    with generation_lock:
        clone_model = None
        design_model = None
        custom_model = None

    # Invalidate generation cache — results from old model are stale
    _gen_cache_invalidate()

    # Sync audio loader cache if config changed
    new_loader = config.get("advanced", {}).get("audio_loader")
    if new_loader:
        try:
            from voice_engine import set_audio_loader
            set_audio_loader(new_loader)
        except (ValueError, ImportError):
            pass

    logger.info("Model config updated: %s. Models unloaded. Generation cache cleared.", ", ".join(changes))

    return jsonify({
        "status": "config_updated",
        "changes": changes,
        "models_unloaded": True,
        "note": "New model will be loaded on next generation",
    })


@app.route("/unload-model", methods=["POST"])
def unload_model():
    """Unload a single model to free memory."""
    reset_activity_timer()
    data = request.json or {}
    model_type = data.get("model_type")

    if not model_type:
        return jsonify({"error": "model_type required", "recovery": "config"}), 400

    valid_types = ("clone", "design", "custom")
    if model_type not in valid_types:
        return jsonify({
            "error": f"Unknown model type: {model_type}. Valid: {', '.join(valid_types)}",
            "recovery": "config",
        }), 400

    # Check if generation is active for this mode
    if generation_state["active"] and generation_state["mode"] == model_type:
        return jsonify({
            "error": f"Cannot unload {model_type} model while generation is active",
            "recovery": "retry",
        }), 409

    global clone_model, design_model, custom_model
    with generation_lock:
        if model_type == "clone":
            if clone_model is None:
                return jsonify({"status": "already_unloaded", "model": model_type})
            clone_model = None
        elif model_type == "design":
            if design_model is None:
                return jsonify({"status": "already_unloaded", "model": model_type})
            design_model = None
        elif model_type == "custom":
            if custom_model is None:
                return jsonify({"status": "already_unloaded", "model": model_type})
            custom_model = None

    from voice_engine import unload_model_cleanup
    unload_model_cleanup()
    _gen_cache_invalidate()
    model_load_times.pop(model_type, None)

    logger.info("Unloaded %s model.", model_type)
    return jsonify({"status": "unloaded", "model": model_type})


@app.route("/update-startup-config", methods=["POST"])
def update_startup_config():
    """Update which models load at startup in config.json."""
    reset_activity_timer()
    data = request.json or {}

    valid_types = ("clone", "design", "custom")
    changes = []
    config = load_config()
    if "models" not in config:
        config["models"] = {}

    for model_type in valid_types:
        if model_type in data:
            val = bool(data[model_type])
            if model_type not in config["models"]:
                config["models"][model_type] = {}
            config["models"][model_type]["load_at_startup"] = val
            changes.append(f"{model_type}={'on' if val else 'off'}")

    if not changes:
        return jsonify({"error": "No valid model types provided", "recovery": "config"}), 400

    save_config(config)
    logger.info("Startup config updated: %s", ", ".join(changes))
    return jsonify({"status": "updated", "changes": changes})


@app.route("/prompts", methods=["GET"])
def list_prompts():
    reset_activity_timer()
    backend = get_backend()
    all_files = set(os.listdir(VOICE_PROMPTS_DIR))
    if backend == "mlx":
        # MLX uses .wav+.txt pairs; list voice names that have both files
        wav_files = {f[:-4] for f in all_files if f.endswith('.wav')}
        txt_files = {f[:-4] for f in all_files if f.endswith('.txt')}
        names = sorted(wav_files & txt_files)
        prompts = [f"{n}.wav" for n in names]
    else:
        # Torch uses .pt files, but also include voices with .wav+.txt
        # or .wav-only (server auto-creates .pt on first use)
        pt_names = {f[:-3] for f in all_files if f.endswith('.pt')}
        wav_names = {f[:-4] for f in all_files if f.endswith('.wav')}
        all_names = sorted(pt_names | wav_names)
        prompts = [f"{n}.pt" for n in all_names]
    return jsonify({"prompts": prompts})


# ---------------------------------------------------------------------------
# Voice prompt management endpoints
# ---------------------------------------------------------------------------

def _validate_prompt_name(name):
    """Validate a prompt name for path traversal. Returns error response or None."""
    if not name:
        return jsonify({"error": "Missing prompt name", "recovery": "config"}), 400
    if ".." in name or "/" in name:
        return jsonify({"error": "Invalid prompt name: path traversal not allowed", "recovery": "config"}), 400
    return None


@app.route("/delete-prompt", methods=["POST"])
def delete_prompt():
    """Delete a voice prompt and all its format files (.pt, .wav, .txt)."""
    reset_activity_timer()
    data = request.get_json(silent=True) or {}
    name = data.get("name", "")

    err = _validate_prompt_name(name)
    if err:
        return err

    # Strip extension to get base name
    base = name
    for ext in (".pt", ".wav", ".txt"):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break

    # Find and delete all matching files
    files_removed = []
    for ext in (".pt", ".wav", ".txt"):
        path = os.path.join(VOICE_PROMPTS_DIR, f"{base}{ext}")
        if os.path.exists(path):
            os.remove(path)
            files_removed.append(f"{base}{ext}")

    if not files_removed:
        return jsonify({"error": f"Voice prompt '{base}' not found", "recovery": "config"}), 404

    # If deleted prompt was the default, clear it
    try:
        config = load_config()
        current_default = config.get("default_clone_prompt", "")
        default_base = current_default
        for ext in (".pt", ".wav", ".txt"):
            if default_base.endswith(ext):
                default_base = default_base[:-len(ext)]
                break
        if default_base == base:
            config["default_clone_prompt"] = ""
            save_config(config)
    except Exception:
        pass

    # Clear voice prompt cache
    clear_voice_prompt_cache()

    logger.info("Deleted voice prompt '%s': %s", base, files_removed)
    return jsonify({"status": "deleted", "name": base, "files_removed": files_removed})


@app.route("/rename-prompt", methods=["POST"])
def rename_prompt():
    """Rename a voice prompt (all format files) with rollback on partial failure."""
    reset_activity_timer()
    data = request.get_json(silent=True) or {}
    old_name = data.get("old_name", "")
    new_name = data.get("new_name", "")

    for name_val in (old_name, new_name):
        err = _validate_prompt_name(name_val)
        if err:
            return err

    # Strip extensions to get base names
    old_base = old_name
    new_base = new_name
    for ext in (".pt", ".wav", ".txt"):
        if old_base.endswith(ext):
            old_base = old_base[:-len(ext)]
            break
    for ext in (".pt", ".wav", ".txt"):
        if new_base.endswith(ext):
            new_base = new_base[:-len(ext)]
            break

    if old_base == new_base:
        return jsonify({"error": "Old and new names are the same", "recovery": "config"}), 400

    # Collision check — ensure no files with new name exist
    for ext in (".pt", ".wav", ".txt"):
        if os.path.exists(os.path.join(VOICE_PROMPTS_DIR, f"{new_base}{ext}")):
            return jsonify({
                "error": f"Voice prompt '{new_base}' already exists",
                "recovery": "config",
            }), 409

    # Check that at least one old file exists
    old_exists = any(
        os.path.exists(os.path.join(VOICE_PROMPTS_DIR, f"{old_base}{ext}"))
        for ext in (".pt", ".wav", ".txt")
    )
    if not old_exists:
        return jsonify({"error": f"Voice prompt '{old_base}' not found", "recovery": "config"}), 404

    # Rename with rollback on partial failure
    renamed = []
    try:
        for ext in (".pt", ".wav", ".txt"):
            old_path = os.path.join(VOICE_PROMPTS_DIR, f"{old_base}{ext}")
            new_path = os.path.join(VOICE_PROMPTS_DIR, f"{new_base}{ext}")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                renamed.append((new_path, old_path))
    except OSError as e:
        # Rollback successful renames
        for current, rollback_to in renamed:
            try:
                os.rename(current, rollback_to)
            except OSError:
                pass
        return jsonify({"error": f"Rename failed: {e}", "recovery": "retry"}), 500

    # Update default if the renamed prompt was the default
    try:
        config = load_config()
        current_default = config.get("default_clone_prompt", "")
        default_base = current_default
        for ext in (".pt", ".wav", ".txt"):
            if default_base.endswith(ext):
                default_base = default_base[:-len(ext)]
                break
        if default_base == old_base:
            # Preserve the extension format of the original default
            if current_default.endswith(".pt"):
                config["default_clone_prompt"] = f"{new_base}.pt"
            else:
                config["default_clone_prompt"] = new_base
            save_config(config)
    except Exception:
        pass

    # Clear voice prompt cache
    clear_voice_prompt_cache()

    files_renamed = [os.path.basename(new) for new, _ in renamed]
    logger.info("Renamed voice prompt '%s' -> '%s': %s", old_base, new_base, files_renamed)
    return jsonify({"status": "renamed", "old_name": old_base, "new_name": new_base, "files_renamed": files_renamed})


@app.route("/preview-prompt", methods=["GET"])
def preview_prompt():
    """Return the .wav file for a voice prompt as audio/wav."""
    reset_activity_timer()
    name = request.args.get("name", "")

    err = _validate_prompt_name(name)
    if err:
        return err

    # Strip extension to get base name
    base = name
    for ext in (".pt", ".wav", ".txt"):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break

    wav_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.wav")
    if not os.path.exists(wav_path):
        return jsonify({"error": f"No .wav file found for prompt '{base}'", "recovery": "config"}), 404

    return send_file(wav_path, mimetype="audio/wav")


@app.route("/prompt-details", methods=["GET"])
def prompt_details():
    """Return metadata for voice prompts.

    If ?name=X is provided, returns details for that prompt.
    Otherwise returns details for all prompts.
    """
    reset_activity_timer()
    name = request.args.get("name")

    # Get current default
    current_default = get_default_clone_prompt() or ""
    default_base = current_default
    for ext in (".pt", ".wav", ".txt"):
        if default_base.endswith(ext):
            default_base = default_base[:-len(ext)]
            break

    def _prompt_info(base):
        """Build metadata dict for a single prompt."""
        formats = []
        total_size = 0
        created = None
        for ext in (".pt", ".wav", ".txt"):
            path = os.path.join(VOICE_PROMPTS_DIR, f"{base}{ext}")
            if os.path.exists(path):
                formats.append(ext)
                total_size += os.path.getsize(path)
                mtime = os.path.getmtime(path)
                if created is None or mtime < created:
                    created = mtime
        return {
            "name": base,
            "formats": formats,
            "size_bytes": total_size,
            "created": created,
            "is_default": (base == default_base),
        }

    if name:
        err = _validate_prompt_name(name)
        if err:
            return err
        base = name
        for ext in (".pt", ".wav", ".txt"):
            if base.endswith(ext):
                base = base[:-len(ext)]
                break
        info = _prompt_info(base)
        if not info["formats"]:
            return jsonify({"error": f"Voice prompt '{base}' not found", "recovery": "config"}), 404
        return jsonify(info)

    # All prompts
    try:
        all_files = os.listdir(VOICE_PROMPTS_DIR)
    except OSError:
        return jsonify({"prompts": []})

    # Collect unique base names
    bases = set()
    for f in all_files:
        for ext in (".pt", ".wav", ".txt"):
            if f.endswith(ext):
                bases.add(f[:-len(ext)])
                break

    prompts = [_prompt_info(b) for b in sorted(bases)]
    return jsonify({"prompts": prompts})


def _validate_generation_request(data, is_batch=True):
    """Validate and normalize generation request data.
    Returns (normalized_data, error_response) — error_response is None on success.
    """
    security = server_config.get("security", {})
    max_text_length = security.get("max_text_length", 10000)
    max_batch_size = security.get("max_batch_size", 20)

    if is_batch:
        texts = data.get("texts", [])
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return None, (jsonify({"error": "No texts provided", "recovery": "config"}), 400)
        if len(texts) > max_batch_size:
            return None, (jsonify({"error": f"Batch size {len(texts)} exceeds limit of {max_batch_size}", "recovery": "config"}), 400)
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                return None, (jsonify({"error": f"Text at index {i} is empty or invalid", "recovery": "config"}), 400)
            if len(t) > max_text_length:
                return None, (jsonify({"error": f"Text at index {i} exceeds {max_text_length} character limit ({len(t)} chars)", "recovery": "config"}), 400)
    else:
        text = data.get("text", "")
        if not text:
            return None, (jsonify({"error": "No text provided", "recovery": "config"}), 400)
        if len(text) > max_text_length:
            return None, (jsonify({"error": f"Text exceeds {max_text_length} character limit ({len(text)} chars)", "recovery": "config"}), 400)

    mode = data.get("mode", "clone")
    if mode not in ("clone", "design", "custom"):
        return None, (jsonify({"error": f"Invalid mode: {mode}. Must be clone, design, or custom", "recovery": "config"}), 400)

    prompt_file = data.get("prompt_file")
    if prompt_file and (".." in prompt_file or "/" in prompt_file):
        return None, (jsonify({"error": "Invalid prompt_file: path traversal not allowed", "recovery": "config"}), 400)

    speaker = data.get("speaker")
    if mode == "custom" and speaker:
        speaker_key = speaker.lower() if isinstance(speaker, str) else ""
        if speaker_key not in CUSTOM_VOICE_SPEAKERS and speaker not in _VALID_SPEAKER_NAMES:
            return None, (jsonify({"error": f"Unknown speaker: {speaker}. Valid: {', '.join(CUSTOM_VOICE_SPEAKERS.keys())}", "recovery": "config"}), 400)

    return data, None


def _create_temp_audio_copy(source_path):
    """Create a secure temp copy of an audio file. Returns temp path."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    os.chmod(temp_file.name, 0o600)
    try:
        shutil.copy2(source_path, temp_file.name)
        return temp_file.name
    except Exception:
        os.unlink(temp_file.name)
        raise


def _prepare_mode_params(mode, data):
    """Load voice prompt for clone, validate speaker for custom.
    Returns (voice_prompt, error_response_or_None).
    """
    voice_prompt = None
    if mode == "clone":
        prompt_file = data.get("prompt_file")
        if not prompt_file:
            return None, (jsonify({"error": "prompt_file required for clone mode", "recovery": "config"}), 400)
        voice_prompt = load_voice_prompt(prompt_file)
        if voice_prompt is None:
            return None, (jsonify({"error": f"Voice prompt not found: {prompt_file}", "detail": "Check available prompts with 'changeVoice --list-prompts'", "recovery": "config"}), 404)
    elif mode == "custom":
        speaker = data.get("speaker")
        if not speaker:
            return None, (jsonify({"error": "speaker required for custom mode", "recovery": "config"}), 400)
    return voice_prompt, None


@app.route("/generate", methods=["POST"])
def generate():
    reset_activity_timer()

    data = request.json
    data, err = _validate_generation_request(data, is_batch=True)
    if err:
        return err
    texts = data.get("texts", [])
    if isinstance(texts, str):
        texts = [texts]

    mode = data.get("mode", "clone")
    prompt_file = data.get("prompt_file")
    voice_description = data.get("voice_description", "")
    language = data.get("language", "English")
    speaker = data.get("speaker")
    instruct = data.get("instruct", "")
    x_vector_only_mode = data.get("x_vector_only_mode", False)

    # Check if required model is loaded
    model = _get_model(mode)
    if model is None:
        return jsonify({
            "error": "model_not_loaded",
            "detail": f"The '{mode}' model is not loaded. {MODEL_INFO.get(get_model_size(), {}).get(mode, {}).get('description', '')}",
            "recovery": "restart",
            "model_type": mode,
        }), 503

    # Generation parameters
    gen_params = {
        "temperature": data.get("temperature", 0.7),
        "top_k": data.get("top_k", 50),
        "top_p": data.get("top_p", 0.95),
        "repetition_penalty": data.get("repetition_penalty", 1.05),
        "max_new_tokens": data.get("max_new_tokens", 2048),
    }

    seed = data.get("seed")
    if seed is not None:
        gen_params["seed"] = seed

    # Text chunking — None means use config default, 0 means disable
    max_chunk_chars = data.get("max_chunk_chars")

    # Track this request in queue
    request_id = id(request)
    request_queue.add(request_id)

    try:
        # --- Double-checked locking: first cache check BEFORE acquiring lock ---
        # For single-text requests, a cache hit avoids blocking on the lock entirely.
        pre_lock_cache_keys = {}
        pre_lock_results = {}
        for i, text in enumerate(texts):
            cache_key = _gen_cache_key(
                text, mode, gen_params,
                prompt_file=prompt_file,
                voice_description=voice_description,
                speaker=speaker, instruct=instruct,
            )
            pre_lock_cache_keys[i] = cache_key
            cached = _gen_cache_get(cache_key)
            if cached:
                temp_path = _create_temp_audio_copy(cached["file"])
                pre_lock_results[i] = {"index": i, "file": temp_path, "sample_rate": cached["sample_rate"]}
                logger.info("Generation cache hit (pre-lock) for text %d/%d", i + 1, len(texts))

        # If ALL texts hit cache, skip the lock entirely
        if len(pre_lock_results) == len(texts):
            results = [pre_lock_results[i] for i in range(len(texts))]
            # Skip to response (no lock needed)
            request_queue.discard(request_id)
            return jsonify({"results": results})

        # Acquire lock for thread-safe generation
        with generation_lock:
            results = []

            for i, text in enumerate(texts):
                # Use pre-lock cache hit if available
                if i in pre_lock_results:
                    results.append(pre_lock_results[i])
                    continue

                # --- Double-checked locking: second cache check AFTER acquiring lock ---
                # Another thread may have generated this while we waited for the lock.
                cache_key = pre_lock_cache_keys[i]
                cached = _gen_cache_get(cache_key)
                if cached:
                    temp_path = _create_temp_audio_copy(cached["file"])
                    results.append({"index": i, "file": temp_path, "sample_rate": cached["sample_rate"]})
                    logger.info("Generation cache hit (post-lock) for text %d/%d", i + 1, len(texts))
                    continue

                # Update generation state for progress tracking
                generation_state.update({
                    "active": True,
                    "start_time": time.time(),
                    "text_length": len(text),
                    "mode": mode,
                    "batch_index": i,
                    "batch_total": len(texts),
                })

                # Resolve voice prompt for clone mode
                voice_prompt, mode_err = _prepare_mode_params(mode, data)
                if mode_err:
                    return mode_err

                def _chunk_progress(chunk_idx, chunk_total):
                    generation_state.update({
                        "chunk_index": chunk_idx,
                        "chunk_total": chunk_total,
                    })

                wav, sr = run_inference(
                    model=model,
                    text=text,
                    mode=mode,
                    gen_params=gen_params,
                    language=language,
                    voice_prompt=voice_prompt,
                    voice_description=voice_description,
                    speaker=speaker,
                    instruct=instruct,
                    max_chunk_chars=max_chunk_chars,
                    progress_callback=_chunk_progress,
                    x_vector_only_mode=x_vector_only_mode,
                )

                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                os.chmod(temp_file.name, 0o600)
                try:
                    sf.write(temp_file.name, wav, sr)
                except Exception:
                    os.unlink(temp_file.name)
                    raise
                results.append({"index": i, "file": temp_file.name, "sample_rate": sr})

                # Store in generation cache (copy the file so original can be moved)
                cache_path = _create_temp_audio_copy(temp_file.name)
                _gen_cache_put(cache_key, cache_path, sr)

            return jsonify({"results": results})
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        return jsonify({
            "error": "Audio generation failed",
            "detail": str(e),
            "recovery": "retry",
        }), 500
    finally:
        # Clear generation state
        generation_state.update({"active": False, "start_time": 0.0, "text_length": 0,
                                 "mode": "", "batch_index": 0, "batch_total": 0,
                                 "chunk_index": 0, "chunk_total": 0})
        # Remove from queue
        if request_id in request_queue:
            request_queue.discard(request_id)


@app.route("/generate-stream", methods=["POST"])
def generate_stream():
    """Stream audio generation — returns chunked audio as it's produced.

    Request body same as /generate, but for single text only.
    Returns multipart audio chunks as they're generated.
    """
    from flask import Response

    reset_activity_timer()

    data = request.json
    data, err = _validate_generation_request(data, is_batch=False)
    if err:
        return err

    text = data.get("text", "")
    mode = data.get("mode", "clone")
    voice_description = data.get("voice_description", "")
    language = data.get("language", "English")
    speaker = data.get("speaker")
    instruct = data.get("instruct", "")
    x_vector_only_mode = data.get("x_vector_only_mode", False)

    model = _get_model(mode)
    if model is None:
        return jsonify({
            "error": "model_not_loaded",
            "detail": f"The '{mode}' model is not loaded",
            "recovery": "restart",
        }), 503

    gen_params = {
        "temperature": data.get("temperature", 0.7),
        "top_k": data.get("top_k", 50),
        "top_p": data.get("top_p", 0.95),
        "repetition_penalty": data.get("repetition_penalty", 1.05),
        "max_new_tokens": data.get("max_new_tokens", 2048),
    }
    seed = data.get("seed")
    if seed is not None:
        gen_params["seed"] = seed

    voice_prompt, mode_err = _prepare_mode_params(mode, data)
    if mode_err:
        return mode_err

    def generate_chunks():
        """Generator that yields audio chunks with length-prefixed format.

        Each chunk: [4-byte sample_rate][4-byte audio_length][audio_bytes]
        Checks for cancellation between chunks.
        """
        import struct
        import uuid

        gen_id = str(uuid.uuid4())[:8]
        generation_state.update({
            "active": True,
            "start_time": time.time(),
            "text_length": len(text),
            "mode": mode,
            "generation_id": gen_id,
            "cancelled": False,
        })

        try:
            chunk_idx = 0
            for wav_chunk, sr in run_inference_streaming(
                model=model,
                text=text,
                mode=mode,
                gen_params=gen_params,
                language=language,
                voice_prompt=voice_prompt,
                voice_description=voice_description,
                speaker=speaker,
                instruct=instruct,
                x_vector_only_mode=x_vector_only_mode,
            ):
                # Check for cancellation
                if generation_state["cancelled"]:
                    logger.info("Generation cancelled after %d chunks", chunk_idx)
                    break

                chunk_idx += 1
                generation_state["chunk_index"] = chunk_idx

                # Length-prefixed format: [sample_rate:4][length:4][audio:length]
                audio_bytes = wav_chunk.astype("<f4").tobytes()
                header = struct.pack("<II", sr, len(audio_bytes))
                yield header + audio_bytes

        finally:
            # Only reset if this is still our generation (not a new one)
            if generation_state.get("generation_id") == gen_id:
                generation_state.update({
                    "active": False, "start_time": 0.0, "text_length": 0,
                    "mode": "", "chunk_index": 0, "chunk_total": 0,
                    "generation_id": None, "cancelled": False,
                })

    return Response(
        generate_chunks(),
        mimetype="application/octet-stream",
        headers={"X-Content-Type": "audio/raw-float32"}
    )


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Graceful shutdown — works even when Gradio prevents sys.exit()."""
    global shutdown_timer
    if shutdown_timer is not None:
        shutdown_timer.cancel()
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)

    func = request.environ.get("werkzeug.server.shutdown")
    if func:
        func()
        return jsonify({"status": "shutting down"})

    # Fallback: schedule os._exit in a thread (Gradio blocks sys.exit/SIGTERM)
    def _force_exit():
        time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=_force_exit, daemon=True).start()
    return jsonify({"status": "shutting down"})


def write_pid():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def cleanup_pid(signum=None, frame=None):
    global shutdown_timer
    if shutdown_timer is not None:
        shutdown_timer.cancel()
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)
    sys.exit(0)


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(description="TTS Server")
    _parser.add_argument("--public", action="store_true",
                         help="Bind to 0.0.0.0 (accessible from network)")
    _args = _parser.parse_args()

    # Configure logging: RotatingFileHandler + stderr
    from voice_config import LOG_FILE
    _log_fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")
    _file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=1)
    _file_handler.setFormatter(_log_fmt)
    _stderr_handler = logging.StreamHandler(sys.stderr)
    _stderr_handler.setFormatter(_log_fmt)
    logging.getLogger("tts").setLevel(logging.DEBUG)
    logging.getLogger("tts").addHandler(_file_handler)
    logging.getLogger("tts").addHandler(_stderr_handler)

    config = load_config()
    server_config = config.get("server", {})
    # Store models config in server_config for access by load_models()
    server_config["models"] = config.get("models", {})
    server_config["security"] = config.get("security", {})
    host = server_config.get("host", "127.0.0.1")
    port = server_config.get("port", 5123)

    if _args.public:
        host = "0.0.0.0"
        logger.warning("Binding to 0.0.0.0 — server is accessible from the network.")

    from voice_config import IN_COLAB
    if IN_COLAB:
        host = "0.0.0.0"
        logger.info("Colab detected — binding to 0.0.0.0 for tunnel access.")

    # Handle shutdown signals
    signal.signal(signal.SIGTERM, cleanup_pid)
    signal.signal(signal.SIGINT, cleanup_pid)

    # Generate auth token
    generate_auth_token()
    print(f"Auth token written to {TOKEN_FILE}")

    # Load models before starting server
    load_models()

    # Migrate orphan MLX prompts to .pt format (torch backend only)
    if get_backend() == "torch":
        try:
            from voice_engine import migrate_orphan_mlx_prompts
            migrate_orphan_mlx_prompts()
        except Exception as e:
            logger.warning("MLX prompt migration failed: %s", e)

    # Write PID file
    write_pid()

    # Start auto-shutdown timer if configured
    auto_shutdown_minutes = server_config.get("auto_shutdown_minutes", 0)
    if auto_shutdown_minutes > 0:
        print(f"Auto-shutdown enabled: {auto_shutdown_minutes} minutes of inactivity")
        reset_activity_timer()

    print(f"\nTTS Server running on http://{host}:{port}")
    print("Use stopTTSServer to shut down.\n")

    # Enable threading for concurrent request handling
    # Generation is serialized via lock, but multiple requests can queue
    app.run(host=host, port=port, threaded=True)
