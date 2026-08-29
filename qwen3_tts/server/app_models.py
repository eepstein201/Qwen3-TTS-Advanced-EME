"""Model and stats endpoint handlers.

Extracted from app.py to keep each module under 800 lines.
"""

import asyncio
import logging
import os
import time

from fastapi import HTTPException

from qwen3_tts.core.config import (
    MLX_MODEL_INFO,
    MODEL_INFO,
    VALID_MLX_QUANTIZATIONS,
    get_backend,
    get_mlx_model_name,
    get_mlx_quantization,
    get_model_size,
    get_torch_dtype_name,
    sanitize_log,
    save_config,
)
from qwen3_tts.server.app_lifespan import _get_queue_size, _sanitize_error
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

    from qwen3_tts.server.app_lifespan import detect_degraded_generation

    backend = get_backend()
    stats_data = {
        "status": "ok",
        "backend": backend,
        # Full detail is safe here — /stats requires auth, so exposing the
        # in-flight request's size (via sec_per_char) is not the information
        # leak it would be on the public /health, which gets the bool only.
        "generation_health": detect_degraded_generation(state),
        "model_size": get_model_size(),
        "clone_model_loaded": state.models.get("clone") is not None,
        "design_model_loaded": state.models.get("design") is not None,
        "custom_model_loaded": state.models.get("custom") is not None,
        "voice_prompts_cached": cache_info.currsize,
        "voice_prompts_cache_hits": cache_info.hits,
        "idle_seconds": idle_seconds,
        "auto_shutdown_minutes": auto_shutdown_minutes
        if auto_shutdown_minutes > 0
        else "disabled",
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
                stats_data["mps_memory_allocated_mb"] = round(
                    allocated / (1024 * 1024), 2
                )
            except (AttributeError, RuntimeError):
                stats_data["mps_memory_allocated_mb"] = "unavailable"

        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                stats_data["cuda_memory_allocated_mb"] = round(
                    allocated / (1024 * 1024), 2
                )
                stats_data["cuda_memory_reserved_mb"] = round(
                    reserved / (1024 * 1024), 2
                )
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
    loading_map = getattr(state, "models_loading", None) or {}
    for model_type, info in size_model_info.items():
        loaded = state.models.get(model_type) is not None
        models_cfg = server_config.get("models", {})
        load_at_startup = models_cfg.get(model_type, {}).get("load_at_startup", False)
        # `loading` is mutually exclusive with `loaded`: a model that is fully
        # loaded cannot also be in flight. UI polls this to drive Phase 1b
        # progress indicators (poll_model_loading_state in components.py).
        loading = bool(loading_map.get(model_type, False)) and not loaded

        entry = {
            "loaded": loaded,
            "loading": loading,
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

    # Add ASR model info (lazy import — no heavy deps at module scope)
    from qwen3_tts.core.engine import get_asr_model_info

    asr_info = get_asr_model_info()

    return {
        "models": models_data,
        "asr": asr_info,
        "backend": backend,
        "model_size": model_size,
    }


def _recover_from_failed_load(state, model_type: str) -> None:
    """Reclaim backend memory and reset state after a failed model swap (PRF-5).

    A partially-constructed model leaves allocations behind that slow down
    every later generation (upstream mlx-audio #827 reports ~2.4x on Base
    cloning, and the server has a known-red "dies under repeated
    load/unload"). Recording the error is not enough — this mirrors the
    cleanup the unload path runs and drops the state that would otherwise let
    /models describe the model as healthy.

    Never raises: the caller still has to surface the original load failure.
    """
    try:
        state.models[model_type] = None
        state.model_load_times.pop(model_type, None)

        from qwen3_tts.core.engine import unload_model_cleanup

        unload_model_cleanup()
    except Exception as e:
        logger.warning(
            "Recovery after failed %s load did not complete: %s",
            sanitize_log(model_type),
            sanitize_log(e),
        )


async def handle_load_model(state, req):
    """Load a model on demand.

    The load itself runs WITHOUT inference_lock — minutes of download and
    weight construction must not starve /generate. The design warm-up
    inference afterwards runs UNDER inference_lock, acquired as a leaf
    (nothing else held), matching the global lock order where
    inference_lock is outermost (/generate at app_generation.py, /ws).
    Warm-up-vs-generation was the issue #192 trigger pair — the one this
    closes. /transcribe and /create-voice-prompt were the same class and
    are both serialized now (leaf acquisitions) — with them, every MLX
    inference reachable through the API serializes on inference_lock.

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

    # Mark loading=True so /models polling reflects in-flight state.
    loading_map = getattr(state, "models_loading", None)
    if loading_map is not None:
        loading_map[model_type] = True

    # Load the model
    try:
        from qwen3_tts.core.config import get_model_info
        from qwen3_tts.core.engine import load_model
        from qwen3_tts.core.engine.model_loader import (
            _warmup_disabled,
            _warmup_model,
        )

        info = get_model_info(model_type)
        model_name = info.get("name", info.get("name_template", model_type))
        logger.info("Loading %s...", sanitize_log(model_name))
        t0 = time.time()
        # Load without the warm-up — the server serializes it below so the
        # warm-up never runs concurrently with a generation (#192).
        model = await asyncio.to_thread(load_model, model_type, warmup=False)
        # Only design weights warm up (see _warmup_model's own guard — keep
        # in sync), and the knob is checked BEFORE the lock so ablation
        # runs don't queue behind generations for a no-op; clone/custom
        # skip the lock round-trip entirely.
        if model_type == "design" and not _warmup_disabled():
            async with state.inference_lock:
                await asyncio.to_thread(
                    _warmup_model, model, model_type, get_backend()
                )
        state.models[model_type] = model
        state.model_load_times[model_type] = round(time.time() - t0, 1)
        logger.info(
            "Loaded %s model successfully in %.1fs.",
            sanitize_log(model_type),
            state.model_load_times[model_type],
        )
        # Clear any previous load error for this model
        state.model_load_errors[model_type] = None
    except ImportError as e:
        logger.error(
            "Backend not available for model loading %s: %s",
            sanitize_log(model_type),
            sanitize_log(e),
            exc_info=True,
        )
        state.model_load_errors[model_type] = _sanitize_error(str(e))
        _recover_from_failed_load(state, model_type)
        _error_response(500, "import_error", _sanitize_error(str(e)), "config")
        return  # explicit guard — _error_response raises, but this ensures no fall-through
    except (RuntimeError, OSError, ValueError) as e:
        logger.error(
            "Failed to load model %s: %s",
            sanitize_log(model_type),
            sanitize_log(e),
            exc_info=True,
        )
        state.model_load_errors[model_type] = _sanitize_error(str(e))
        _recover_from_failed_load(state, model_type)
        _error_response(500, "load_failed", _sanitize_error(str(e)), "restart")
        return
    except Exception as e:
        logger.error(
            "Unexpected error loading model %s: %s",
            sanitize_log(model_type),
            sanitize_log(e),
            exc_info=True,
        )
        state.model_load_errors[model_type] = _sanitize_error(str(e))
        _recover_from_failed_load(state, model_type)
        _error_response(500, "unknown_error", _sanitize_error(str(e)), "bug")
        return
    finally:
        if loading_map is not None:
            loading_map[model_type] = False

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
    if (
        state.generation_state["active"]
        and state.generation_state["mode"] == model_type
    ):
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
    logger.info("Unloaded %s model.", sanitize_log(model_type))

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
    valid_quants = VALID_MLX_QUANTIZATIONS

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
    await asyncio.to_thread(save_config, {**config, "advanced": adv})

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
        except (ValueError, ImportError) as e:
            logger.warning(
                "Failed to sync audio loader cache to %r: %s", new_loader, e
            )

    logger.info(
        "Model config updated: %s. Models unloaded. Generation cache cleared.",
        sanitize_log(", ".join(changes)),
    )

    return {
        "status": "config_updated",
        "changes": changes,
        "models_unloaded": True,
        "note": "All models unloaded. Reload required before generation.",
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
            models[model_type] = {
                **models.get(model_type, {}),
                "load_at_startup": val_bool,
            }
            changes.append(f"{model_type}={'on' if val_bool else 'off'}")

    if not changes:
        raise HTTPException(status_code=400, detail="No valid model types provided")

    save_config({**config, "models": models})

    # Update server config cache
    state.server_config = {**state.server_config, "models": models}

    logger.info("Startup config updated: %s", ", ".join(changes))
    return {"status": "updated", "changes": changes}


# ---------------------------------------------------------------------------
# ASR endpoint handlers
# ---------------------------------------------------------------------------


def handle_load_asr(state):
    """Load the ASR model for transcription.

    Returns status dict. Raises HTTPException on failure.
    """
    from qwen3_tts.core.engine import is_asr_loaded, load_asr_model

    if is_asr_loaded():
        return {"status": "already_loaded"}

    try:
        t0 = time.time()
        load_asr_model()
        elapsed = round(time.time() - t0, 1)
        logger.info("ASR model loaded in %.1fs", elapsed)
        return {"status": "loaded", "load_time_sec": elapsed}
    except ImportError as e:
        logger.error("ASR backend not available: %s", sanitize_log(e), exc_info=True)
        _error_response(500, "import_error", _sanitize_error(str(e)), "config")
        return
    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Failed to load ASR model: %s", sanitize_log(e), exc_info=True)
        _error_response(500, "load_failed", _sanitize_error(str(e)), "restart")
        return
    except Exception as e:
        logger.error("Unexpected error loading ASR: %s", sanitize_log(e), exc_info=True)
        _error_response(500, "unknown_error", _sanitize_error(str(e)), "bug")
        return


def handle_unload_asr(state):
    """Unload the ASR model to free memory.

    Returns status dict.
    """
    from qwen3_tts.core.engine import unload_asr_model

    unload_asr_model()
    logger.info("ASR model unloaded.")
    return {"status": "unloaded"}


def _decode_audio(audio_b64):
    """Decode the request's base64 payload (blocking CPU for large clips)."""
    import base64

    return base64.b64decode(audio_b64)


def _stage_tempfile(audio_bytes):
    """Write decoded audio to a 0600 tempfile for ASR (blocking file IO)."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
        try:
            tmp.write(audio_bytes)
            os.chmod(path, 0o600)
        except OSError:
            # The handler's finally only sees the path once staging
            # returns, so a write/chmod failure must clean up here —
            # unlike the pre-#192 shape, which leaked the write-failure
            # case. Unlink-then-close is fine on the POSIX targets.
            try:
                os.remove(path)
            except OSError:
                pass
            raise
    return path


def _remove_tempfile(path):
    """Best-effort tempfile cleanup (blocking file IO)."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


async def handle_transcribe(state, req):
    """Transcribe audio to text using the ASR model.

    Decodes base64 audio, writes to tempfile, transcribes, cleans up.

    The mlx-whisper generate is real MLX inference and runs under
    ``inference_lock`` (#192: unsynchronized concurrent MLX inference is
    upstream-unsafe — ml-explore/mlx#3078, Blaizzy/mlx-audio#638/#733). It is
    a leaf acquisition: the handler holds nothing else when it takes the
    lock, so the global inference_lock-outermost order is preserved. The lazy
    ASR model load stays OUTSIDE the lock — minutes of download + weight
    construction must not starve /generate (same split as /load-model's
    load/warm-up, PR #211). /create-voice-prompt was the last of the
    class and is serialized the same way — with it, all MLX inference
    reachable through the API serializes on inference_lock.

    Args:
        state: app.state
        req: TranscribeRequest with audio_base64 and language

    Returns:
        Dict with transcript text.
    """
    from qwen3_tts.core.engine import (
        is_asr_loaded,
        load_asr_model,
        transcribe_audio,
    )

    # Decode audio. Off the event loop: a near-100 MB body makes b64decode
    # + the tempfile write sub-second-but-blocking work (async-offload
    # policy, cf. tests/test_server_async_offload.py).
    try:
        audio_bytes = await asyncio.to_thread(_decode_audio, req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    # Write to tempfile for ASR processing
    tmp_path = None
    try:
        tmp_path = await asyncio.to_thread(_stage_tempfile, audio_bytes)

        # First /transcribe after startup pays the ASR model load here
        # (preload_asr_model never preloads on the MLX backend). Unlocked —
        # and it must complete before the generate queues for the lock, or
        # the load would run inside the generation-serialized section.
        if not is_asr_loaded():
            await asyncio.to_thread(load_asr_model)

        async with state.inference_lock:
            # Re-check under the lock (#214 item 2). The ensure above is
            # deliberately unlocked, which leaves a check-then-use window: a
            # /unload-asr landing in it means transcribe_audio — which lazily
            # loads on first call — would rebuild the model INSIDE
            # inference_lock, blocking every /generate for minutes. Bail
            # instead and let the caller retry; re-loading here would trade a
            # cheap 503 for the starvation this lock exists to prevent.
            if not is_asr_loaded():
                logger.warning(
                    "ASR was unloaded while /transcribe waited for "
                    "inference_lock; asking the caller to retry"
                )
                _error_response(
                    503, "asr_unloaded", "ASR model was unloaded concurrently", "retry"
                )
                return  # explicit guard — _error_response raises, but it is
                # typed -> None, not NoReturn, so nothing structurally stops a
                # fall-through into transcribe_audio with ASR unloaded.
            transcript = await asyncio.to_thread(
                transcribe_audio, tmp_path, req.language
            )
        return {"transcript": transcript}
    except HTTPException:
        # _error_response raises HTTPException; the catch-all below would
        # otherwise rewrap the 503 above as recovery="bug" and strip its
        # retry classification.
        raise
    except ImportError as e:
        logger.error("ASR not available: %s", sanitize_log(e))
        _error_response(500, "import_error", _sanitize_error(str(e)), "config")
    except (RuntimeError, OSError) as e:
        logger.error("Transcription failed: %s", sanitize_log(e))
        _error_response(500, "transcription_failed", _sanitize_error(str(e)), "retry")
    except Exception as e:
        logger.error("Unexpected transcription error: %s", sanitize_log(e))
        _error_response(500, "unknown_error", _sanitize_error(str(e)), "bug")
    finally:
        await asyncio.to_thread(_remove_tempfile, tmp_path)
