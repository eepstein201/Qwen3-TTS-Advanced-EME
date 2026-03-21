"""Model and stats endpoint handlers.

Extracted from app.py to keep each module under 800 lines.
"""

import logging
import os
import time

from fastapi import HTTPException

from qwen3_tts.core.config import (
    MODEL_INFO,
    MLX_MODEL_INFO,
    get_backend,
    get_model_size,
    get_mlx_quantization,
    get_torch_dtype_name,
    get_mlx_model_name,
    save_config,
)
from qwen3_tts.server.app_lifespan import _get_queue_size
from qwen3_tts.server.validation import _error_response

logger = logging.getLogger("tts")


def handle_stats(state, server_config):
    """Build and return server statistics dict.

    Includes model status, cache info, memory stats, and queue size.
    Torch and MLX imports are lazy (inside this function only).
    """
    idle_seconds = int(time.time() - state.last_activity)
    auto_shutdown_minutes = server_config.get("auto_shutdown_minutes", 0)

    from qwen3_tts.core.engine import voice_prompt_cache_info
    cache_info = voice_prompt_cache_info()

    backend = get_backend()
    stats_data = {
        "status": "ok",
        "backend": backend,
        "clone_model_loaded": state.models.get("clone") is not None,
        "design_model_loaded": state.models.get("design") is not None,
        "custom_model_loaded": state.models.get("custom") is not None,
        "voice_prompts_cached": cache_info.currsize,
        "voice_prompts_cache_hits": cache_info.hits,
        "idle_seconds": idle_seconds,
        "auto_shutdown_minutes": auto_shutdown_minutes if auto_shutdown_minutes > 0 else "disabled",
        "generation_queue_size": _get_queue_size(state),
    }
    if backend == "mlx":
        stats_data["mlx_quantization"] = get_mlx_quantization()
    else:
        stats_data["dtype"] = get_torch_dtype_name()

    # GPU memory stats (lazy torch import)
    try:
        import torch
        if torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory()
                stats_data["mps_memory_allocated_mb"] = round(allocated / (1024 * 1024), 2)
            except (AttributeError, RuntimeError):
                stats_data["mps_memory_allocated_mb"] = "unavailable"

        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                stats_data["cuda_memory_allocated_mb"] = round(allocated / (1024 * 1024), 2)
                stats_data["cuda_memory_reserved_mb"] = round(reserved / (1024 * 1024), 2)
            except (AttributeError, RuntimeError):
                pass
    except ImportError:
        pass

    # MLX memory stats
    if backend == "mlx":
        try:
            import mlx.core as mx
            try:
                active_mem = mx.get_active_memory()
                peak_mem = mx.get_peak_memory()
            except AttributeError:
                active_mem = mx.metal.get_active_memory()
                peak_mem = mx.metal.get_peak_memory()
            stats_data["mlx_memory_active_mb"] = round(active_mem / (1024 * 1024), 2)
            stats_data["mlx_memory_peak_mb"] = round(peak_mem / (1024 * 1024), 2)
        except (ImportError, AttributeError, RuntimeError):
            pass

    return stats_data


def handle_list_models(state, server_config):
    """Build and return model status listing.

    Returns dict with models info, backend, and model_size.
    """
    backend = get_backend()
    model_size = get_model_size()

    size_model_info = MODEL_INFO.get(model_size, MODEL_INFO["1.7B"])
    size_mlx_info = MLX_MODEL_INFO.get(model_size, MLX_MODEL_INFO["1.7B"])

    models_data = {}
    for model_type, info in size_model_info.items():
        loaded = state.models.get(model_type) is not None
        models_cfg = server_config.get("models", {})
        load_at_startup = models_cfg.get(model_type, {}).get("load_at_startup", False)

        entry = {
            "loaded": loaded,
            "description": info["description"],
            "memory_mb": info["memory_mb"],
            "repo_id": info["name"],
            "load_at_startup": load_at_startup,
            "load_time_sec": state.model_load_times.get(model_type),
        }
        if backend == "mlx":
            mlx_info = size_mlx_info.get(model_type)
            if mlx_info:
                entry["repo_id"] = get_mlx_model_name(model_type)
                entry["memory_mb"] = mlx_info["memory_mb"]
        models_data[model_type] = entry

    return {"models": models_data, "backend": backend, "model_size": model_size}


def handle_load_model(state, req):
    """Load a model on demand.

    Raises HTTPException on invalid model_type or load failure.
    Returns status dict.
    """
    model_type = req.model_type

    valid_types = ("clone", "design", "custom")
    if model_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: {model_type}. Valid: {', '.join(valid_types)}",
        )

    # Check if already loaded
    if state.models.get(model_type) is not None:
        return {"status": "already_loaded", "model": model_type}

    # Load the model
    try:
        from qwen3_tts.core.engine import load_model
        from qwen3_tts.core.config import get_model_info

        info = get_model_info(model_type)
        model_name = info.get("name", info.get("name_template", model_type))
        logger.info("Loading %s...", model_name)
        t0 = time.time()
        model = load_model(model_type)
        state.models[model_type] = model
        state.model_load_times[model_type] = round(time.time() - t0, 1)
        logger.info("Loaded %s model successfully in %.1fs.", model_type, state.model_load_times[model_type])
        # Clear any previous load error for this model
        state.model_load_errors[model_type] = None
    except ImportError as e:
        logger.error("Backend not available for model loading %s: %s", model_type, e, exc_info=True)
        state.model_load_errors[model_type] = str(e)
        _error_response(500, "import_error", str(e), "config")
    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Failed to load model %s: %s", model_type, e, exc_info=True)
        state.model_load_errors[model_type] = str(e)
        _error_response(500, "load_failed", str(e), "restart")
    except Exception as e:
        logger.error("Unexpected error loading model %s: %s", model_type, e, exc_info=True)
        state.model_load_errors[model_type] = str(e)
        _error_response(500, "unknown_error", str(e), "bug")

    return {"status": "loaded", "model": model_type}


def handle_unload_model(state, req):
    """Unload a model to free memory.

    Cleans up generation cache and runs backend-specific cleanup.
    Raises HTTPException if model_type invalid or generation active.
    """
    model_type = req.model_type

    valid_types = ("clone", "design", "custom")
    if model_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: {model_type}. Valid: {', '.join(valid_types)}",
        )

    # Check if generation is active for this mode
    if state.generation_state["active"] and state.generation_state["mode"] == model_type:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot unload {model_type} model while generation is active",
        )

    if state.models.get(model_type) is None:
        return {"status": "already_unloaded", "model": model_type}

    state.models[model_type] = None

    from qwen3_tts.core.engine import unload_model_cleanup
    unload_model_cleanup()

    # Invalidate generation cache
    with state.gen_cache_lock:
        for entry in state.gen_cache.values():
            try:
                main_file = entry.get("main_file") or entry.get("file")
                if main_file and os.path.exists(main_file):
                    os.remove(main_file)
            except OSError:
                pass
        state.gen_cache.clear()

    state.model_load_times.pop(model_type, None)
    logger.info("Unloaded %s model.", model_type)

    return {"status": "unloaded", "model": model_type}


async def handle_update_model_config(state, req, config_fn):
    """Update model size and/or quantization settings.

    Uses immutable config updates (Phase 10a).
    Unloads all models and clears generation cache so new settings take effect.

    Args:
        state: app.state
        req: UpdateModelConfigRequest
        config_fn: callable returning current config dict
    """
    new_size = req.model_size
    new_quant = req.mlx_quantization

    if not new_size and not new_quant:
        raise HTTPException(
            status_code=400,
            detail="At least one of model_size or mlx_quantization required",
        )

    valid_sizes = ("1.7B", "0.6B")
    valid_quants = ("4bit", "8bit", "bf16")

    if new_size and new_size not in valid_sizes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_size: {new_size}. Valid: {', '.join(valid_sizes)}",
        )

    if new_quant and new_quant not in valid_quants:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mlx_quantization: {new_quant}. Valid: {', '.join(valid_quants)}",
        )

    # Immutable config update (Phase 10a)
    config = config_fn()
    adv = dict(config.get("advanced", {}))
    changes = []
    if new_size:
        adv["model_size"] = new_size
        changes.append(f"model_size={new_size}")
    if new_quant:
        adv["mlx_quantization"] = new_quant
        changes.append(f"mlx_quantization={new_quant}")
    save_config({**config, "advanced": adv})

    # Unload all models so new settings take effect
    async with state.generation_lock:
        for name in ("clone", "design", "custom"):
            state.models[name] = None

    # Invalidate generation cache
    with state.gen_cache_lock:
        for entry in state.gen_cache.values():
            try:
                main_file = entry.get("main_file") or entry.get("file")
                if main_file and os.path.exists(main_file):
                    os.remove(main_file)
            except OSError:
                pass
        state.gen_cache.clear()

    # Sync audio loader cache if config changed
    new_loader = adv.get("audio_loader")
    if new_loader:
        try:
            from qwen3_tts.core.engine import set_audio_loader
            set_audio_loader(new_loader)
        except (ValueError, ImportError):
            pass

    logger.info("Model config updated: %s. Models unloaded. Generation cache cleared.", ", ".join(changes))

    return {
        "status": "config_updated",
        "changes": changes,
        "models_unloaded": True,
        "note": "New model will be loaded on next generation",
    }


def handle_update_startup_config(state, req, config_fn):
    """Update which models load at startup in config.json.

    Uses immutable config updates (Phase 10b).

    Args:
        state: app.state
        req: UpdateStartupConfigRequest
        config_fn: callable returning current config dict
    """
    valid_types = ("clone", "design", "custom")
    changes = []

    # Immutable config update (Phase 10b)
    config = config_fn()
    models = dict(config.get("models", {}))

    for model_type in valid_types:
        val = getattr(req, model_type, None)
        if val is not None:
            val_bool = bool(val)
            models[model_type] = {**models.get(model_type, {}), "load_at_startup": val_bool}
            changes.append(f"{model_type}={'on' if val_bool else 'off'}")

    if not changes:
        raise HTTPException(status_code=400, detail="No valid model types provided")

    save_config({**config, "models": models})

    # Update server config cache
    state.server_config["models"] = models

    logger.info("Startup config updated: %s", ", ".join(changes))
    return {"status": "updated", "changes": changes}
