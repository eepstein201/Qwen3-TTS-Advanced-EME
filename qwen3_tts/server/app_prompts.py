"""Voice prompt endpoint handlers.

Extracted from app.py to keep each module under 800 lines.
"""

import json
import logging
import os

from fastapi import HTTPException
from fastapi.responses import FileResponse

from qwen3_tts.core.config import (
    VOICE_PROMPTS_DIR,
    get_default_clone_prompt,
    safe_path_join,
    sanitize_log,
    save_config,
)
from qwen3_tts.server.app_lifespan import _sanitize_error
from qwen3_tts.server.validation import (
    _error_response,
    _strip_extension,
    _validate_prompt_name,
)

logger = logging.getLogger("tts")


def handle_list_prompts(state, backend, query_params):
    """List voice prompts with optional pagination.

    Args:
        state: app.state (unused currently, reserved for future cache)
        backend: current backend string ("mlx" or "torch")
        query_params: request.query_params for offset/limit

    Returns:
        Dict with prompts list, total count, offset, and limit.
    """
    try:
        all_files = set(os.listdir(VOICE_PROMPTS_DIR))
    except OSError:
        return {"prompts": [], "total": 0}

    if backend == "mlx":
        # MLX uses .wav+.txt pairs
        wav_files = {f[:-4] for f in all_files if f.endswith(".wav")}
        txt_files = {f[:-4] for f in all_files if f.endswith(".txt")}
        names = sorted(wav_files & txt_files)
        prompts = [f"{n}.wav" for n in names]
    else:
        # Torch uses .pt files only (not .wav files)
        pt_names = {f[:-3] for f in all_files if f.endswith(".pt")}
        prompts = sorted(f"{n}.pt" for n in pt_names)

    total = len(prompts)

    # Pagination (R-24) -- default limit=0 means return all (backward compat)
    try:
        offset = max(0, int(query_params.get("offset", 0)))
    except (ValueError, TypeError):
        offset = 0
    try:
        limit = max(0, int(query_params.get("limit", 0)))
    except (ValueError, TypeError):
        limit = 0

    if offset > 0 or limit > 0:
        if limit > 0:
            prompts = prompts[offset : offset + limit]
        else:
            prompts = prompts[offset:]

    return {"prompts": prompts, "total": total, "offset": offset, "limit": limit}


def handle_delete_prompt(state, req, config_fn):
    """Delete a voice prompt and all its format files.

    Uses immutable config update (Phase 10c) when clearing default prompt.

    Args:
        state: app.state (unused currently, reserved for future cache)
        req: DeletePromptRequest
        config_fn: callable returning current config dict
    """
    name = req.name
    err = _validate_prompt_name(name)
    if err:
        raise HTTPException(status_code=err[1], detail=err[0]["error"])

    base = _strip_extension(name)

    # Find and delete all matching files with per-file error handling (R-41)
    files_removed = []
    files_failed = []
    for ext in (".pt", ".wav", ".txt"):
        path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}{ext}")
        if os.path.exists(path):
            try:
                os.remove(path)
                files_removed.append(f"{base}{ext}")
            except OSError as e:
                logger.warning("Failed to delete %s: %s", path, e)
                files_failed.append(f"{base}{ext}")

    if not files_removed and not files_failed:
        raise HTTPException(status_code=404, detail=f"Voice prompt '{base}' not found")

    # If deleted prompt was the default, clear it (immutable — Phase 10c)
    try:
        config = config_fn()
        current_default = config.get("default_clone_prompt", "")
        default_base = _strip_extension(current_default)
        if default_base == base:
            save_config({**config, "default_clone_prompt": ""})
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.warning(
            "Deleted prompt '%s' but failed to clear it as default_clone_prompt: %s",
            base,
            e,
        )

    # Clear voice prompt cache
    from qwen3_tts.core.engine import clear_voice_prompt_cache

    clear_voice_prompt_cache()

    logger.info(
        "Deleted voice prompt '%s': %s", sanitize_log(base), sanitize_log(files_removed)
    )
    result = {"status": "deleted", "name": base, "files_removed": files_removed}
    if files_failed:
        result["files_failed"] = files_failed
    return result


def handle_rename_prompt(state, req, config_fn):
    """Rename a voice prompt (all format files) with rollback on partial failure.

    Uses immutable config update (Phase 10d) when updating default prompt.

    Args:
        state: app.state (unused currently, reserved for future cache)
        req: RenamePromptRequest
        config_fn: callable returning current config dict
    """
    for name_val in (req.old_name, req.new_name):
        err = _validate_prompt_name(name_val)
        if err:
            raise HTTPException(status_code=err[1], detail=err[0]["error"])

    old_base = _strip_extension(req.old_name)
    new_base = _strip_extension(req.new_name)

    if old_base == new_base:
        raise HTTPException(status_code=400, detail="Old and new names are the same")

    # Collision check
    for ext in (".pt", ".wav", ".txt"):
        if os.path.exists(safe_path_join(VOICE_PROMPTS_DIR, f"{new_base}{ext}")):
            raise HTTPException(
                status_code=409, detail=f"Voice prompt '{new_base}' already exists"
            )

    # Check that at least one old file exists
    old_exists = any(
        os.path.exists(safe_path_join(VOICE_PROMPTS_DIR, f"{old_base}{ext}"))
        for ext in (".pt", ".wav", ".txt")
    )
    if not old_exists:
        raise HTTPException(
            status_code=404, detail=f"Voice prompt '{old_base}' not found"
        )

    # Rename with rollback on partial failure
    renamed = []
    try:
        for ext in (".pt", ".wav", ".txt"):
            old_path = safe_path_join(VOICE_PROMPTS_DIR, f"{old_base}{ext}")
            new_path = safe_path_join(VOICE_PROMPTS_DIR, f"{new_base}{ext}")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                renamed.append((new_path, old_path))
    except OSError as e:
        # Rollback
        for current, rollback_to in renamed:
            try:
                os.rename(current, rollback_to)
            except OSError:
                pass
        logger.error(
            "Rename failed %s -> %s: %s",
            sanitize_log(req.old_name),
            sanitize_log(req.new_name),
            sanitize_log(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Rename failed. Check server logs for details."
        )

    # Update default if the renamed prompt was the default (immutable — Phase 10d)
    try:
        config = config_fn()
        current_default = config.get("default_clone_prompt", "")
        default_base = _strip_extension(current_default)
        if default_base == old_base:
            if current_default.endswith(".pt"):
                save_config({**config, "default_clone_prompt": f"{new_base}.pt"})
            else:
                save_config({**config, "default_clone_prompt": new_base})
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.warning(
            "Renamed prompt to '%s' but failed to update default_clone_prompt: %s",
            new_base,
            e,
        )

    # Clear voice prompt cache
    from qwen3_tts.core.engine import clear_voice_prompt_cache

    clear_voice_prompt_cache()

    files_renamed = [os.path.basename(new) for new, _ in renamed]
    logger.info(
        "Renamed voice prompt '%s' -> '%s': %s",
        sanitize_log(old_base),
        sanitize_log(new_base),
        sanitize_log(files_renamed),
    )
    return {
        "status": "renamed",
        "old_name": old_base,
        "new_name": new_base,
        "files_renamed": files_renamed,
    }


def handle_preview_prompt(name_param):
    """Return the .wav file for a voice prompt as audio/wav.

    Includes symlink resolution to prevent path traversal (R-20).

    Args:
        name_param: prompt name from query params

    Returns:
        FileResponse with the .wav file.
    """
    err = _validate_prompt_name(name_param)
    if err:
        raise HTTPException(status_code=err[1], detail=err[0]["error"])

    base = _strip_extension(name_param)
    wav_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.wav")

    # Symlink resolution -- prevent path traversal via symlinks (R-20)
    real_path = os.path.realpath(wav_path)
    real_prompts_dir = os.path.realpath(str(VOICE_PROMPTS_DIR))
    if not real_path.startswith(real_prompts_dir + os.sep):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path outside voice prompts directory",
        )

    if not os.path.exists(real_path):
        raise HTTPException(
            status_code=404, detail=f"No .wav file found for prompt '{base}'"
        )

    return FileResponse(real_path, media_type="audio/wav")


def handle_prompt_details(name_param):
    """Return metadata for voice prompts.

    If name_param is provided, returns info for that single prompt.
    If name_param is None, returns info for all prompts.

    Args:
        name_param: prompt name from query params, or None for all

    Returns:
        Single prompt info dict, or {"prompts": [...]} for all.
    """
    # Get current default
    current_default = get_default_clone_prompt() or ""
    default_base = _strip_extension(current_default)

    def _prompt_info(base):
        """Build metadata dict for a single prompt."""
        formats = []
        total_size = 0
        created = None
        for ext in (".pt", ".wav", ".txt"):
            path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}{ext}")
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

    if name_param:
        err = _validate_prompt_name(name_param)
        if err:
            raise HTTPException(status_code=err[1], detail=err[0]["error"])
        base = _strip_extension(name_param)
        info = _prompt_info(base)
        if not info["formats"]:
            raise HTTPException(
                status_code=404, detail=f"Voice prompt '{base}' not found"
            )
        return info

    # All prompts
    try:
        all_files = os.listdir(VOICE_PROMPTS_DIR)
    except OSError:
        return {"prompts": []}

    # Collect unique base names
    bases = set()
    for f in all_files:
        for ext in (".pt", ".wav", ".txt"):
            if f.endswith(ext):
                bases.add(f[: -len(ext)])
                break

    prompts = [_prompt_info(b) for b in sorted(bases)]
    return {"prompts": prompts}


def handle_create_voice_prompt(state, req):
    """Create a voice clone prompt from uploaded audio.

    Decodes base64 audio, loads it for cloning, creates the voice prompt
    tensor, and saves it to VOICE_PROMPTS_DIR.

    Args:
        state: app.state (provides loaded models)
        req: CreateVoicePromptRequest with audio_base64, name, transcript, no_transcript

    Returns:
        Dict with status and prompt name.
    """
    import base64
    import tempfile

    # Validate prompt name
    err = _validate_prompt_name(req.name)
    if err:
        raise HTTPException(status_code=err[1], detail=err[0]["error"])

    base = _strip_extension(req.name)

    # Verify clone model is loaded
    if state.models.get("clone") is None:
        raise HTTPException(
            status_code=503,
            detail="Clone model must be loaded to create voice prompts",
        )

    # Decode audio
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    # Write to tempfile, load audio, create prompt
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        os.chmod(tmp_path, 0o600)

        from qwen3_tts.core.engine import create_voice_prompt, load_audio_for_cloning

        ref_audio, ref_sr = load_audio_for_cloning(tmp_path)
        transcript = "" if req.no_transcript else (req.transcript or "")
        voice_prompt = create_voice_prompt(
            state.models["clone"],
            ref_audio,
            ref_sr,
            transcript,
            x_vector_only_mode=req.no_transcript,
        )

        # Save the .pt file
        import torch

        pt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.pt")
        torch.save(voice_prompt, pt_path)

        # Clear voice prompt cache so new prompt is visible
        from qwen3_tts.core.engine import clear_voice_prompt_cache

        clear_voice_prompt_cache()

        logger.info("Created voice prompt '%s'", sanitize_log(base))
        return {"status": "created", "name": base}

    except HTTPException:
        raise
    except ImportError as e:
        logger.error("Backend not available for voice creation: %s", sanitize_log(e))
        _error_response(500, "import_error", _sanitize_error(str(e)), "config")
        return
    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Voice prompt creation failed: %s", sanitize_log(e))
        _error_response(500, "creation_failed", _sanitize_error(str(e)), "retry")
        return
    except Exception as e:
        logger.error("Unexpected error creating voice prompt: %s", sanitize_log(e))
        _error_response(500, "unknown_error", _sanitize_error(str(e)), "bug")
        return
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
