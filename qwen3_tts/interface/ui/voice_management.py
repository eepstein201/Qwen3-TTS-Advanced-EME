#!/usr/bin/env python3
"""Voice prompt management for the Gradio UI.

This module contains:
- Voice prompt creation
- Voice prompt listing/renaming/deletion
- Voice prompt preview
"""

import logging
import os
import tempfile

import gradio as gr

from qwen3_tts.core.config import (
    VOICE_PROMPTS_DIR,
    get_backend,
    get_default_clone_prompt,
    is_server_running,
    load_config,
    safe_path_join,
    set_default_clone_prompt,
    validate_voice_name,
)
from qwen3_tts.interface.ui.shared import (
    get_voice_prompts,
)
from qwen3_tts.interface.voice_helpers import (
    strip_extension,
    validate_prompt_name,
)

logger = logging.getLogger("tts.ui")


def create_voice_prompt(
    audio_path, transcript, voice_name, no_transcript=False
):
    """Create a voice prompt from audio file and transcript.

    Args:
        audio_path: Path to the audio file
        transcript: Transcript text (or None if no_transcript=True)
        voice_name: Name for the voice prompt
        no_transcript: If True, use x_vector_only mode without transcript

    Returns:
        Tuple of (status_message, prompt_list, default_prompt)
    """
    if not audio_path:
        raise gr.Error("Please upload an audio file")

    if not voice_name or not voice_name.strip():
        raise gr.Error("Please enter a voice name")

    voice_name = voice_name.strip()

    # Validate name
    validation_error = validate_prompt_name(voice_name)
    if validation_error:
        raise gr.Error(validation_error[0]["error"])

    config = load_config()
    backend = config.get("advanced", {}).get("backend", "mlx")

    # Build output path
    base_name = strip_extension(voice_name)
    try:
        validate_voice_name(base_name)
    except ValueError as exc:
        raise gr.Error(str(exc))
    if backend == "mlx":
        # MLX needs .wav + .txt pair
        wav_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base_name}.wav")
        txt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base_name}.txt")
    else:
        # Torch uses .pt files
        pt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base_name}.pt")

    # Check if prompt already exists
    if backend == "mlx":
        if os.path.exists(wav_path) or os.path.exists(txt_path):
            raise gr.Error(f"Voice prompt '{base_name}' already exists")
    else:
        if os.path.exists(pt_path):
            raise gr.Error(f"Voice prompt '{base_name}' already exists")

    try:
        if backend == "mlx":
            # For MLX, copy the audio and transcript
            import shutil

            shutil.copy(audio_path, wav_path)

            if no_transcript:
                # Create empty transcript marker for x_vector_only mode
                with open(txt_path, "w") as f:
                    f.write("")
            else:
                if not transcript or not transcript.strip():
                    # Clean up wav file
                    os.remove(wav_path)
                    raise gr.Error(
                        "Please provide a transcript or enable 'no transcript' mode"
                    )
                with open(txt_path, "w") as f:
                    f.write(transcript.strip())

            status = f"Created MLX voice prompt: {base_name}"
        else:
            # For torch, use server to create .pt file
            if not is_server_running(config):
                raise gr.Error("Server must be running to create torch voice prompts")

            # Upload audio and transcript to server
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            import base64

            payload = {
                "audio_base64": base64.b64encode(audio_bytes).decode(),
                "transcript": transcript.strip() if transcript else "",
                "name": base_name,
                "no_transcript": no_transcript,
            }

            from qwen3_tts.core.http_client import server_request

            resp = server_request(
                "POST",
                "/create-voice-prompt",
                json=payload,
                timeout=60,
            )

            if resp.status_code != 200:
                error = resp.json().get("error", "Unknown error")
                raise gr.Error(f"Failed to create prompt: {error}")

            status = f"Created torch voice prompt: {base_name}"

        # Refresh prompt list
        prompts = get_voice_prompts()
        default = get_default_clone_prompt(config)

        return (
            status,
            gr.update(choices=prompts),
            gr.update(choices=prompts, value=default),
        )

    except gr.Error:
        raise
    except Exception as e:
        logger.error("Failed to create voice prompt: %s", e)
        raise gr.Error(f"Failed to create prompt: {e}")


def auto_transcribe_audio(audio_path):
    """Auto-transcribe audio using server ASR.

    Phase 1b: surfaces a `gr.Info` toast + ProgressIndicator while the
    transcribe round-trip is in flight (typically 1-3s).

    Args:
        audio_path: Path to the audio file

    Returns:
        Transcript text
    """
    from qwen3_tts.interface.ui.components import ProgressIndicator

    if not audio_path:
        raise gr.Error("Please upload an audio file first")

    config = load_config()
    if not is_server_running(config):
        raise gr.Error("Server must be running for auto-transcription")

    progress = ProgressIndicator(mode="indeterminate", message="Transcribing audio…")
    try:
        gr.Info(progress.message)
    except Exception:
        pass

    try:
        import base64

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        payload = {
            "audio_base64": base64.b64encode(audio_bytes).decode(),
        }

        from qwen3_tts.core.http_client import server_request

        resp = server_request(
            "POST",
            "/transcribe",
            json=payload,
            timeout=60,
        )

        if resp.status_code != 200:
            error = resp.json().get("error", "Unknown error")
            raise gr.Error(f"Transcription failed: {error}")

        return resp.json().get("transcript", "")

    except gr.Error:
        raise
    except Exception as e:
        logger.error("Auto-transcription failed: %s", e)
        raise gr.Error(f"Transcription failed: {e}")


def get_prompt_table_data():
    """Get voice prompts as table data for display.

    Returns:
        List of [name, format, default] rows
    """
    prompts = get_voice_prompts()
    config = load_config()
    default_prompt = config.get("default_clone_prompt", "")

    rows = []
    for prompt in prompts:
        base = strip_extension(prompt)
        is_default = "✓" if prompt == default_prompt or base == default_prompt else ""

        # Determine format
        if prompt.endswith(".pt"):
            fmt = "torch (.pt)"
        else:
            fmt = "mlx (.wav+.txt)"

        rows.append([base, fmt, is_default])

    return rows


def preview_voice(name):
    """Preview a voice prompt by generating a sample.

    Args:
        name: Voice prompt name

    Returns:
        Path to preview audio file
    """
    if not name:
        raise gr.Error("Please select a voice prompt")

    config = load_config()
    if not is_server_running(config):
        raise gr.Error("Server must be running for preview")

    tmp_path = None
    try:
        from qwen3_tts.core.http_client import server_request

        resp = server_request(
            "GET",
            "/preview-prompt",
            params={"name": name},
            timeout=60,
        )

        if resp.status_code != 200:
            error = resp.json().get("error", "Unknown error")
            raise gr.Error(f"Preview failed: {error}")

        # Save to temp file; track path for cleanup on failure
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = f.name
        try:
            f.write(resp.content)
        finally:
            f.close()
        return tmp_path

    except gr.Error:
        raise
    except Exception as e:
        logger.error("Voice preview failed: %s", e)
        if tmp_path is not None:
            try:
                import os

                os.unlink(tmp_path)
            except OSError:
                pass
        return None


def rename_voice(old_name, new_name):
    """Rename a voice prompt.

    Args:
        old_name: Current name
        new_name: New name

    Returns:
        Tuple of (status_message, prompt_table, dropdown_update)
    """
    if not old_name:
        raise gr.Error("Please select a voice prompt to rename")
    if not new_name or not new_name.strip():
        raise gr.Error("Please enter a new name")

    new_name = new_name.strip()

    # Validate new name
    validation_error = validate_prompt_name(new_name)
    if validation_error:
        raise gr.Error(validation_error[0]["error"])

    config = load_config()

    # Use server to rename (handles all format complexity)
    if not is_server_running(config):
        raise gr.Error("Server must be running for rename")

    try:
        from qwen3_tts.core.http_client import server_request

        resp = server_request(
            "POST",
            "/rename-prompt",
            json={"old_name": old_name, "new_name": new_name},
            timeout=10,
        )

        if resp.status_code != 200:
            error = resp.json().get("error", "Unknown error")
            raise gr.Error(f"Rename failed: {error}")

        prompts = get_voice_prompts()
        return (
            f"Renamed '{old_name}' to '{new_name}'",
            get_prompt_table_data(),
            gr.update(choices=prompts),
        )

    except gr.Error:
        raise
    except Exception as e:
        logger.error("Rename failed: %s", e)
        raise gr.Error(f"Rename failed: {e}")


def delete_voice(name):
    """Delete a voice prompt.

    Args:
        name: Voice prompt name

    Returns:
        Tuple of (status_message, prompt_table, dropdown_update)
    """
    if not name:
        raise gr.Error("Please select a voice prompt to delete")

    config = load_config()

    # Use server to delete (handles all formats)
    if not is_server_running(config):
        raise gr.Error("Server must be running for delete")

    try:
        from qwen3_tts.core.http_client import server_request

        resp = server_request(
            "POST",
            "/delete-prompt",
            json={"name": name},
            timeout=10,
        )

        if resp.status_code != 200:
            error = resp.json().get("error", "Unknown error")
            raise gr.Error(f"Delete failed: {error}")

        prompts = get_voice_prompts()
        return f"Deleted '{name}'", get_prompt_table_data(), gr.update(choices=prompts)

    except gr.Error:
        raise
    except Exception as e:
        logger.error("Delete failed: %s", e)
        raise gr.Error(f"Delete failed: {e}")


def set_voice_default(name):
    """Set a voice prompt as the default.

    Args:
        name: Voice prompt name

    Returns:
        Tuple of (status_message, prompt_table)
    """
    if not name:
        raise gr.Error("Please select a voice prompt")

    backend = get_backend()
    base = strip_extension(name)
    name = f"{base}.wav" if backend == "mlx" else f"{base}.pt"

    set_default_clone_prompt(name)

    return f"Set '{name}' as default", get_prompt_table_data()
