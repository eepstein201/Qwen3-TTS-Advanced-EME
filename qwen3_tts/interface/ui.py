#!/usr/bin/env python3
"""
Qwen3-TTS Web Interface - Gradio-based UI for TTS generation.

Launch with:
    tts ui

Opens a web browser at http://localhost:7860
"""

import logging
import os
import sys
import threading
import time
import uuid
import gradio as gr

logger = logging.getLogger("tts.ui")

# Add the user files directory to path for imports


from qwen3_tts.server.client import TTSClient
import tempfile

from qwen3_tts.core.config import (
    CUSTOM_VOICE_SPEAKERS,
    VOICE_PROMPTS_DIR,
    VOICE_DESCRIPTION_ATTRIBUTES,
    VALID_MODEL_SIZES,
    VALID_MLX_QUANTIZATIONS,
    get_default_clone_prompt,
    set_default_clone_prompt,
    get_server_url,
    get_backend,
    get_model_size,
    get_mlx_quantization,
    is_server_running,
    auth_headers,
    load_config,
    get_prosody_presets,
)
from qwen3_tts.interface.voice_helpers import (
    get_prosody_choices,
    apply_prosody_preset,
    compose_voice_description,
    strip_extension,
    validate_prompt_name,
)

# Derive speaker choices from canonical source
SPEAKER_CHOICES = [
    f"{key} ({info['lang']}) - {info['desc']}"
    for key, info in CUSTOM_VOICE_SPEAKERS.items()
]


def get_prosody_choices():
    """Return list of prosody preset choices for dropdown, with (none) first."""
    presets = get_prosody_presets()
    return ["(none)"] + [f"{name} - {text}" for name, text in sorted(presets.items())]


def apply_prosody_preset(choice, existing_text=None):
    """When a prosody preset is selected, append to existing text or fill in.

    If the target field already has content, appends with ". " separator.
    If empty, fills with the prosody text directly.
    """
    if not choice or choice == "(none)":
        return existing_text or ""
    name = choice.split(" - ")[0].strip()
    presets = get_prosody_presets()
    prosody_text = presets.get(name, "")
    if not prosody_text:
        return existing_text or ""
    if existing_text and existing_text.strip():
        return f"{existing_text.strip()}. {prosody_text}"
    return prosody_text


# =============================================================================
# Voice Description Builder
# =============================================================================

def compose_voice_description(gender, age, tone, texture, pace, accent):
    """Compose a voice description from dropdown selections."""
    parts = []
    if age and age != "(none)":
        # Extract the age range text
        age_text = age.lower().split(" (")[0] if " (" in age else age.lower()
        if gender and gender != "(none)":
            parts.append(f"A {age_text} {gender.lower()}")
        else:
            parts.append(f"A {age_text} speaker")
    elif gender and gender != "(none)":
        parts.append(f"A {gender.lower()} speaker")

    qualifiers = []
    if tone and tone != "(none)":
        qualifiers.append(tone.lower())
    if texture and texture != "(none)":
        qualifiers.append(texture.lower())
    if qualifiers:
        parts.append(f"with a {', '.join(qualifiers)} voice")

    if pace and pace != "(none)":
        parts.append(f"who speaks at a {pace.lower()} pace")

    if accent and accent != "(none)" and accent != "None/Default":
        parts.append(f"with a {accent} accent")

    if not parts:
        return ""
    desc = " ".join(parts)
    if not desc.endswith("."):
        desc += "."
    return desc


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
                system=system_prompt,
                messages=[{"role": "user", "content": description}],
            )
            return response.content[0].text.strip()
        else:
            raise gr.Error(f"Unsupported provider: {provider}")
    except ImportError:
        raise gr.Error("anthropic package not installed. Run: pip install anthropic")
    except Exception as e:
        raise gr.Error(f"Enhancement failed: {e}")


def is_enhancer_available():
    """Check if the AI enhancer is configured and available."""
    try:
        config = load_config()
        enhancer = config.get("prompt_enhancer", {})
        if not enhancer.get("enabled", False):
            return False
        api_key_env = enhancer.get("api_key_env", "ANTHROPIC_API_KEY")
        return bool(os.environ.get(api_key_env))
    except Exception:
        return False


# =============================================================================
# Model Settings Functions
# =============================================================================

def get_current_model_settings():
    """Get current model size and quantization from server or config."""
    client = TTSClient()
    try:
        if client.is_server_running():
            stats = client.get_stats()
            size = stats.get("model_size", "1.7B")
            quant = stats.get("mlx_quantization", "8bit")
            backend = stats.get("backend", "mlx")
        else:
            # Fall back to config
            size = get_model_size()
            quant = get_mlx_quantization()
            backend = get_backend()
        return size, quant, backend
    except Exception:
        return "1.7B", "8bit", "mlx"


def apply_model_settings(model_size, mlx_quantization):
    """Apply new model settings via server endpoint."""
    client = TTSClient()

    if not client.is_server_running():
        return "Error: Server not running. Start with 'tts server start'.", format_status_display()

    try:
        result = client.update_model_config(
            model_size=model_size,
            mlx_quantization=mlx_quantization
        )

        if result.get("status") == "config_updated":
            changes = result.get("changes", [])
            # changes is a list of strings like ["model_size=1.7B", "mlx_quantization=8bit"]
            change_summary = ", ".join(changes) if changes else "no changes"

            if result.get("models_unloaded"):
                msg = f"Settings applied ({change_summary}). Models unloaded - new model loads on next generation."
            else:
                msg = f"Settings applied ({change_summary})."
            return msg, format_status_display()
        else:
            return f"Unexpected response: {result}", format_status_display()
    except Exception as e:
        return f"Error: {str(e)}", format_status_display()


# =============================================================================
# Text Info Helper
# =============================================================================

def update_text_info(text):
    """Show character count and estimated chunks."""
    if not text:
        return ""
    chars = len(text)
    # Estimate chunks (500 chars default)
    chunks = max(1, (chars + 499) // 500)
    if chunks > 1:
        return f"{chars} chars | ~{chunks} chunks"
    return f"{chars} chars"


# =============================================================================
# Voice Creation Functions (via subprocess)
# =============================================================================

def create_voice_prompt(audio_path, transcript, voice_name, no_transcript=False, auto_transcribed=False):
    """Create voice prompt by calling create_custom_voice directly."""
    if not audio_path:
        raise gr.Error("Please upload an audio file")
    if not no_transcript and (not transcript or not transcript.strip()):
        raise gr.Error("Please enter or auto-transcribe a transcript")
    if not voice_name or not voice_name.strip():
        raise gr.Error("Please enter a name for the voice")

    # Sanitize voice name
    voice_name = voice_name.strip().replace(" ", "_").replace("/", "_").replace("\\", "_").replace("..", "")

    try:
        from qwen3_tts.tools.create_voice import create_and_save_voice_prompt
        from qwen3_tts.core.config import get_backend, IN_COLAB
        backend = get_backend()
        # On Colab, always use mlx_only mode to avoid loading a second model
        # copy into VRAM. The .wav/.txt files are saved, and the server will
        # create the .pt on-demand via /create-prompt if needed.
        # On local Mac, respect the backend setting.
        mlx_only = (backend == "mlx") or IN_COLAB
        effective_transcript = "" if no_transcript else transcript
        create_and_save_voice_prompt(
            audio_path, effective_transcript, voice_name,
            test_generation=False, mlx_only=mlx_only,
        )
        prompts = get_voice_prompts()
        return f"Created voice: {voice_name}", gr.update(choices=prompts), gr.update(choices=prompts)
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")


def auto_transcribe_audio(audio_path):
    """Auto-transcribe using MLX ASR."""
    if not audio_path:
        raise gr.Error("Please upload an audio file first")

    try:
        from qwen3_tts.core.engine import transcribe_audio, is_asr_available
        if not is_asr_available():
            raise gr.Error("Auto-transcribe requires MLX backend")
        transcript = transcribe_audio(audio_path)
        # Free ASR model to reclaim VRAM for TTS generation
        from qwen3_tts.core.engine import unload_asr_model
        unload_asr_model()
        return transcript
    except ImportError:
        raise gr.Error("ASR not available - enter transcript manually")
    except Exception as e:
        raise gr.Error(f"Transcription failed: {str(e)}")


# =============================================================================
# Server Status Functions
# =============================================================================

def get_server_status():
    """Get current server status, memory usage, loaded models, and backend info."""
    client = TTSClient()

    if not client.is_server_running():
        return "Disconnected", "N/A", "N/A", "N/A"

    try:
        stats = client.get_stats()
        # Check for MLX memory first, then MPS, then CUDA
        # Use explicit None check so 0.0 is a valid value, not skipped
        memory_val = None
        for _key in ('mlx_memory_active_mb', 'mps_memory_allocated_mb', 'cuda_memory_allocated_mb'):
            _v = stats.get(_key)
            if _v is not None:
                memory_val = _v
                break
        if isinstance(memory_val, (int, float)):
            memory = f"{memory_val:.1f}MB"
        else:
            memory = 'N/A'

        # Check for loaded models - server returns clone_model_loaded, etc.
        loaded_models = []
        if stats.get("clone_model_loaded"):
            loaded_models.append("Clone")
        if stats.get("design_model_loaded"):
            loaded_models.append("Design")
        if stats.get("custom_model_loaded"):
            loaded_models.append("Custom")

        models_str = ", ".join(loaded_models) if loaded_models else "None"

        # Backend info
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
    """Format server status for display."""
    status, memory, models, backend = get_server_status()

    if status == "Connected":
        status_html = f'<span style="color: green; font-weight: bold;">Connected</span>'
    elif status == "Disconnected":
        status_html = f'<span style="color: red; font-weight: bold;">Disconnected</span>'
    else:
        status_html = f'<span style="color: orange;">{status}</span>'

    return f"""
    <div style="padding: 10px; background: var(--block-background-fill, #f5f5f5); border-radius: 5px; margin-bottom: 15px; border: 1px solid var(--block-border-color, #e0e0e0);">
        <strong>Status:</strong> {status_html} |
        <strong>Backend:</strong> {backend} |
        <strong>Memory:</strong> {memory} |
        <strong>Models:</strong> {models}
    </div>
    """


def get_voice_prompts():
    """Get list of available voice prompts."""
    client = TTSClient()
    try:
        prompts = client.list_prompts()
        return prompts if prompts else ["No prompts available"]
    except Exception:
        return ["No prompts available"]


def get_presets():
    """Get list of available presets."""
    client = TTSClient()
    try:
        presets = list(client.list_presets().keys())
        return ["(none)"] + presets if presets else ["(none)"]
    except Exception:
        return ["(none)"]


def _ensure_model_loaded(client, mode, progress):
    """Check if the required model is loaded; if not, load it on demand.

    Args:
        client: TTSClient instance
        mode: Model type (clone/design/custom)
        progress: gr.Progress() callback for status updates

    Raises:
        gr.Error: If model loading fails
    """
    try:
        health = client.get_health()
        key = f"{mode}_model_loaded"
        if health.get(key):
            progress(1.0, desc=f"✓ {mode.capitalize()} model ready")
            return  # already loaded
    except Exception:
        progress(0.1, desc="Connecting to server...")
        # Can't reach server; generate will report the error
        return

    # Check for previous load error (NO try-except - let gr.Error bubble up)
    errors = health.get("model_load_errors", {})
    if mode in errors and errors[mode]:
        progress(0.0, desc=f"✗ {mode.capitalize()} model failed to load")
        raise gr.Error(f"{mode.capitalize()} model error: {errors[mode]}")

    progress(0.2, desc=f"Loading {mode} model...")
    try:
        client.load_model(mode)
        progress(1.0, desc=f"✓ {mode.capitalize()} model loaded")
    except Exception as e:
        progress(0.0, desc=f"✗ Failed to load {mode} model")
        raise gr.Error(f"Failed to load {mode} model: {e}")


def _poll_progress(server_url, progress_fn, stop_event):
    """Poll /generation-status and update Gradio progress bar."""
    import requests as _requests
    while not stop_event.is_set():
        try:
            resp = _requests.get(f"{server_url}/generation-status", timeout=2)
            if resp.status_code == 200:
                state = resp.json()
                if state.get("active"):
                    elapsed = state.get("elapsed_sec", 0)
                    eta = state.get("eta_sec")
                    if eta is not None and (elapsed + eta) > 0:
                        pct = min(0.95, elapsed / (elapsed + eta))
                    else:
                        pct = min(0.95, elapsed / max(elapsed + 10, 30))
                    chunk_total = state.get("chunk_total", 0)
                    if chunk_total > 1:
                        chunk_idx = state.get("chunk_index", 0) + 1
                        desc = f"Generating chunk {chunk_idx}/{chunk_total}... {elapsed:.0f}s"
                    else:
                        desc = f"Generating... {elapsed:.0f}s"
                    progress_fn(pct, desc=desc)
        except Exception:  # nosec B110
            pass
        stop_event.wait(1.0)


# =============================================================================
# Generation History
# =============================================================================

# Per-session history (via gr.State); MAX_HISTORY_SIZE is module-level constant
MAX_HISTORY_SIZE = 10


def add_to_history(history_list, mode, text, output_path, duration_chunks):
    """Add a generation to history. Returns a new list capped at MAX_HISTORY_SIZE (does not mutate input)."""
    import datetime
    entry = {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "mode": mode.capitalize(),
        "text": text[:40] + "..." if len(text) > 40 else text,
        "chunks": duration_chunks,
        "path": output_path,
    }
    new_history = history_list.copy()
    new_history.insert(0, entry)
    if len(new_history) > MAX_HISTORY_SIZE:
        new_history.pop()
    return new_history


def get_history_data(history_list):
    """Return history as a list of lists for Dataframe display."""
    return [[h["time"], h["mode"], h["text"], f"{h['chunks']} chunks"] for h in history_list]


def get_history_audio(evt: gr.SelectData, history_list):
    """Return the audio file path for the selected history row."""
    if evt.index[0] < len(history_list):
        return history_list[evt.index[0]]["path"]
    return None


# =============================================================================
# Generation Functions
# =============================================================================

def _check_generation_cancelled():
    """Check if the current generation was cancelled on the server."""
    try:
        import requests
        from qwen3_tts.core.config import load_config
        config = load_config()
        server_url = get_server_url(config)
        resp = requests.get(f"{server_url}/generation-status", timeout=2)
        if resp.status_code == 200:
            state = resp.json()
            return state.get("cancelled", False)
    except Exception:  # nosec B110
        pass
    return False


def cancel_streaming_generation():
    """Cancel the current streaming generation and clear audio output."""
    client = TTSClient()
    try:
        result = client.cancel_generation()
        status = result.get("status", "unknown")
        if status == "cancellation_requested":
            # Return None for audio to clear the player
            return None, "Generation cancelled", format_status_display()
        elif status == "no_active_generation":
            return None, "No active generation to cancel", format_status_display()
        return None, f"Cancel status: {status}", format_status_display()
    except Exception as e:
        return None, f"Cancel failed: {str(e)}", format_status_display()


def _create_audio_reset_js():
    """JavaScript to reset Audio component state before each generation.

    Returns:
        str: JavaScript function code for Gradio's js parameter.

    The JavaScript finds all audio elements, stops playback, clears the src
    attribute (unloading buffered data), and removes any Gradio-internal state.

    IMPORTANT: This function MUST receive input values as arguments and
    return them unchanged. Gradio passes the component values as positional
    arguments to the js function, and uses the return value as the actual
    inputs to the Python handler. Returning undefined would block generation.
    """
    return """
    (...inputs) => {
        try {
            // Find all audio elements in the document
            const audioEls = document.querySelectorAll('audio');

            audioEls.forEach((audio, idx) => {
                console.log('[Audio Reset] Resetting audio element', idx, ': paused=', audio.paused, ', currentTime=', audio.currentTime, ', currentSrc=', audio.currentSrc);

                // Stop any ongoing playback
                if (!audio.paused) {
                    audio.pause();
                }

                // Reset playback position
                audio.currentTime = 0;

                // Remove source to unload buffered data
                if (audio.currentSrc) {
                    audio.removeAttribute('src');
                    audio.load();
                }

                // Clear Gradio's internal audio state
                // Gradio stores state on the parent container
                const container = audio.closest('[data-testid]') || audio.parentElement;
                if (container) {
                    // Clear any Gradio-internal tracking
                    delete container._gradio_audio_streaming;
                    delete container._gradio_audio_chunks;
                    delete container._gradioAudioState;
                }
            });

            console.log('[Audio Reset] Reset complete:', audioEls.length, 'audio elements reset');
        } catch (e) {
            console.error('[Audio Reset] Error:', e.message, e.stack);
        }
        // CRITICAL: Return inputs unchanged for Gradio to pass to Python handler
        return inputs;
    }
    """


def _save_streaming_audio(all_chunks, sample_rate):
    """Save accumulated streaming chunks to a temp file and return path."""
    import numpy as np
    import soundfile as sf

    if not all_chunks:
        return None

    # Filter out empty chunks to avoid issues with np.concatenate
    non_empty_chunks = [c for c in all_chunks if c is not None and len(c) > 0]
    if not non_empty_chunks:
        return None

    combined = np.concatenate(non_empty_chunks)
    # Double-check the result has samples
    if len(combined) == 0:
        return None

    # Use UUID to avoid filename collisions when multiple generations happen quickly
    unique_id = uuid.uuid4().hex[:8]
    output_path = os.path.expanduser(f"~/Downloads/voice_ui_{unique_id}.wav")
    sf.write(output_path, combined, sample_rate)
    return output_path


# =============================================================================
# Unified generation helpers (reduces code duplication)
# =============================================================================

def _validate_inputs(mode, text, description=None):
    """Validate inputs for generation. Returns error message or None."""
    if not text or not text.strip():
        return "Error: Please enter some text."
    if mode == "design" and (not description or not description.strip()):
        return "Error: Please enter a voice description."
    return None


def _get_mode_kwargs(mode, prompt=None, description=None, speaker_choice=None, instruct=None):
    """Build mode-specific kwargs for client.generate() or client.generate_streaming()."""
    if mode == "clone":
        return {"prompt": prompt}
    elif mode == "design":
        return {"description": description}
    elif mode == "custom":
        speaker = speaker_choice.split(" ")[0] if speaker_choice else "ryan"
        kwargs = {"speaker": speaker}
        if instruct and instruct.strip():
            kwargs["instruct"] = instruct
        return kwargs
    return {}


def _generate_streaming_impl(mode, text, preset, temperature, top_k, top_p, rep_penalty, seed,
                              prompt=None, description=None, speaker_choice=None, instruct=None,
                              x_vector_only_mode=False, history_list=None, progress=gr.Progress()):
    """Unified streaming generation for all modes."""
    if history_list is None:
        history_list = []
    # Validate inputs
    error = _validate_inputs(mode, text, description)
    if error:
        yield None, error, gr.update(), history_list, gr.update()
        return

    client = TTSClient()
    if not client.is_server_running():
        yield None, "Error: TTS server is not running.", gr.update(), history_list, gr.update()
        return

    try:
        _ensure_model_loaded(client, mode, progress)

        seed_val = int(seed) if seed and str(seed).strip() else None
        preset_val = preset if preset and preset != "(none)" else None
        mode_kwargs = _get_mode_kwargs(mode, prompt, description, speaker_choice, instruct)

        all_chunks = []
        sample_rate = None
        chunk_count = 0

        for wav_chunk, sr in client.generate_streaming(
            text=text,
            mode=mode,
            preset=preset_val,
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            seed=seed_val,
            repetition_penalty=rep_penalty,
            x_vector_only_mode=x_vector_only_mode,
            **mode_kwargs,
        ):
            all_chunks.append(wav_chunk)
            sample_rate = sr
            chunk_count += 1
            yield (sr, wav_chunk), f"Streaming... {chunk_count} chunks", gr.update(), history_list, gr.update()

        # Check if cancelled before yielding final state
        if _check_generation_cancelled():
            yield None, f"Cancelled after {chunk_count} chunks", format_status_display(), history_list, gr.update()
            return

        output_path = _save_streaming_audio(all_chunks, sample_rate)
        if output_path:
            history_list = add_to_history(history_list, mode, text, output_path, chunk_count)
            # Explicitly clear audio with gr.update() to signal streaming session end
            # This helps Gradio recognize the session is complete and prepare for next generation
            yield gr.update(value=None), f"Complete: {chunk_count} chunks — saved to {os.path.basename(output_path)}", format_status_display(), history_list, get_history_data(history_list)
        else:
            yield gr.update(value=None), "Error: No audio was generated", format_status_display(), history_list, gr.update()

    except Exception as e:
        # Streaming failed - attempt fallback to non-streaming
        import logging
        logging.warning(f"Streaming failed for {mode} mode, attempting fallback: {e}")
        yield None, "Streaming stalled - trying file mode...", format_status_display(), history_list, gr.update()

        try:
            # Fallback to non-streaming generation
            seed_val = int(seed) if seed and str(seed).strip() else None
            preset_val = preset if preset and preset != "(none)" else None

            # Build kwargs for non-streaming generate (different signature than streaming)
            fallback_kwargs = {
                "trim_silence": False,
                "normalize": False,
                "speed": 1.0,
                "pitch": 1.0,
            }
            if mode == "clone":
                fallback_kwargs["prompt"] = prompt
                fallback_kwargs["no_transcript"] = x_vector_only_mode
            elif mode == "design":
                fallback_kwargs["description"] = description
            elif mode == "custom":
                fallback_kwargs["speaker_choice"] = speaker_choice
                fallback_kwargs["instruct"] = instruct

            output_path = client.generate(
                text=text,
                mode=mode,
                preset=preset_val,
                temperature=temperature,
                top_k=int(top_k),
                top_p=top_p,
                seed=seed_val,
                repetition_penalty=rep_penalty,
                **fallback_kwargs,
            )
            if output_path:
                history_list = add_to_history(history_list, mode, text, output_path, 0)
                yield None, f"Complete (file mode): {os.path.basename(output_path)}", format_status_display(), history_list, get_history_data(history_list)
            else:
                yield None, "Error: Fallback generated no audio", format_status_display(), history_list, gr.update()
        except Exception as fallback_error:
            yield None, f"Error: Streaming and fallback both failed: {fallback_error}", format_status_display(), history_list, gr.update()


def _generate_non_streaming_impl(mode, text, preset, temperature, top_k, top_p, rep_penalty, seed,
                                  trim_silence, normalize, speed, pitch, progress=gr.Progress(),
                                  prompt=None, description=None, speaker_choice=None, instruct=None,
                                  x_vector_only_mode=False, history_list=None):
    """Unified non-streaming generation for all modes."""
    if history_list is None:
        history_list = []
    # Validate inputs
    error = _validate_inputs(mode, text, description)
    if error:
        return None, error, gr.update(), history_list, gr.update()

    client = TTSClient()
    if not client.is_server_running():
        return None, "Error: TTS server is not running. Start it with 'tts server start'.", gr.update(), history_list, gr.update()

    try:
        _ensure_model_loaded(client, mode, progress)

        seed_val = int(seed) if seed and str(seed).strip() else None
        preset_val = preset if preset and preset != "(none)" else None
        output_path = os.path.expanduser(f"~/Downloads/voice_ui_{uuid.uuid4().hex[:8]}.wav")
        mode_kwargs = _get_mode_kwargs(mode, prompt, description, speaker_choice, instruct)

        progress(0, desc="Starting generation...")
        stop_event = threading.Event()
        poll_thread = threading.Thread(
            target=_poll_progress,
            args=(client.server_url, progress, stop_event),
            daemon=True,
        )
        poll_thread.start()

        try:
            result = client.generate(
                text=text,
                output=output_path,
                mode=mode,
                preset=preset_val,
                temperature=temperature,
                top_k=int(top_k),
                top_p=top_p,
                repetition_penalty=rep_penalty,
                seed=seed_val,
                trim_silence=trim_silence,
                normalize=normalize,
                speed=speed if speed != 1.0 else None,
                pitch=pitch if pitch != 0 else None,
                x_vector_only_mode=x_vector_only_mode,
                **mode_kwargs,
            )
        finally:
            stop_event.set()
            poll_thread.join(timeout=2)

        progress(1.0, desc="Complete")
        history_list = add_to_history(history_list, mode, text, result, 1)
        return result, f"Generated: {os.path.basename(result)}", format_status_display(), history_list, get_history_data(history_list)
    except Exception as e:
        error_msg = str(e)
        if "restart" in error_msg.lower() or "not running" in error_msg.lower():
            gr.Warning("Server issue — try restarting with 'tts server start'")
        return None, f"Error: {error_msg}", format_status_display(), history_list, gr.update()


# =============================================================================
# Mode-specific wrappers (thin wrappers for Gradio click handlers)
# =============================================================================

def generate_clone_streaming(text, prompt, preset, temperature, top_k, top_p, rep_penalty, seed,
                             trim_silence, normalize, speed, pitch, no_transcript=False, history_list=None):
    """Generate audio with streaming for clone mode."""
    yield from _generate_streaming_impl(
        "clone", text, preset, temperature, top_k, top_p, rep_penalty, seed,
        prompt=prompt, x_vector_only_mode=no_transcript, history_list=history_list)


def generate_design_streaming(text, description, preset, temperature, top_k, top_p, rep_penalty, seed,
                              trim_silence, normalize, speed, pitch, history_list=None):
    """Generate audio with streaming for design mode."""
    yield from _generate_streaming_impl(
        "design", text, preset, temperature, top_k, top_p, rep_penalty, seed,
        description=description, history_list=history_list)


def generate_custom_streaming(text, speaker_choice, instruct, preset, temperature, top_k, top_p, rep_penalty, seed,
                              trim_silence, normalize, speed, pitch, history_list=None):
    """Generate audio with streaming for custom mode."""
    yield from _generate_streaming_impl(
        "custom", text, preset, temperature, top_k, top_p, rep_penalty, seed,
        speaker_choice=speaker_choice, instruct=instruct, history_list=history_list)


def generate_clone(text, prompt, preset, temperature, top_k, top_p, rep_penalty, seed,
                   trim_silence, normalize, speed, pitch, no_transcript=False, history_list=None, progress=gr.Progress()):
    """Generate audio using clone mode."""
    return _generate_non_streaming_impl(
        "clone", text, preset, temperature, top_k, top_p, rep_penalty, seed,
        trim_silence, normalize, speed, pitch, progress,
        prompt=prompt, x_vector_only_mode=no_transcript, history_list=history_list)


def generate_design(text, description, preset, temperature, top_k, top_p, rep_penalty, seed,
                    trim_silence, normalize, speed, pitch, history_list=None, progress=gr.Progress()):
    """Generate audio using design mode."""
    return _generate_non_streaming_impl(
        "design", text, preset, temperature, top_k, top_p, rep_penalty, seed,
        trim_silence, normalize, speed, pitch, progress,
        description=description, history_list=history_list)


def generate_custom(text, speaker_choice, instruct, preset, temperature, top_k, top_p, rep_penalty, seed,
                    trim_silence, normalize, speed, pitch, history_list=None, progress=gr.Progress()):
    """Generate audio using custom mode with premium speakers."""
    return _generate_non_streaming_impl(
        "custom", text, preset, temperature, top_k, top_p, rep_penalty, seed,
        trim_silence, normalize, speed, pitch, progress,
        speaker_choice=speaker_choice, instruct=instruct, history_list=history_list)


# =============================================================================
# Build UI
# =============================================================================

def stop_server():
    """Stop the TTS server via /shutdown endpoint."""
    import requests as _requests
    client = TTSClient()
    if not client.is_server_running():
        return format_status_display()

    try:
        _requests.post(f"{client.server_url}/shutdown", timeout=5, headers=auth_headers())
    except Exception:  # nosec B110
        pass  # Server shuts down immediately, may not respond

    # Wait briefly and re-check
    time.sleep(1)
    return format_status_display()


# =============================================================================
# Voice Management helpers
# =============================================================================

def get_prompt_table_data():
    """Fetch prompt details from server and format as table rows."""
    client = TTSClient()
    try:
        details = client.get_prompt_details()
        prompts = details.get("prompts", [])
        rows = []
        for p in prompts:
            fmts = ", ".join(p.get("formats", []))
            size_mb = f"{p.get('size_bytes', 0) / (1024 * 1024):.1f}"
            default = "Yes" if p.get("is_default") else ""
            rows.append([p["name"], fmts, size_mb, default])
        return rows
    except Exception as e:
        logger.warning("Failed to get prompt details: %s", e)
        return []


def preview_voice(name):
    """Preview a voice prompt. Returns path to temp .wav file for gr.Audio."""
    if not name:
        return None
    client = TTSClient()
    tmp = None
    tmp_path = None
    try:
        audio_bytes = client.preview_prompt(name)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.write(audio_bytes)
        tmp.close()
        return tmp_path
    except Exception as e:
        # Clean up temp file if it was created but not returned
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        logger.warning("Failed to preview prompt '%s': %s", name, e)
        return None


def rename_voice(old_name, new_name):
    """Rename a voice prompt. Returns (status_msg, table_data, dropdown_update)."""
    if not old_name or not new_name:
        return "Please provide both old and new names.", get_prompt_table_data(), gr.update()
    client = TTSClient()
    try:
        result = client.rename_prompt(old_name, new_name)
        new_prompts = get_voice_prompts()
        return (
            f"Renamed '{old_name}' to '{new_name}'",
            get_prompt_table_data(),
            gr.update(choices=new_prompts, value=new_prompts[0] if new_prompts else None),
        )
    except Exception as e:
        return f"Rename failed: {e}", get_prompt_table_data(), gr.update()


def delete_voice(name):
    """Delete a voice prompt. Returns (status_msg, table_data, dropdown_update)."""
    if not name:
        return "No voice selected.", get_prompt_table_data(), gr.update()
    client = TTSClient()
    try:
        result = client.delete_prompt(name)
        new_prompts = get_voice_prompts()
        return (
            f"Deleted '{name}' ({', '.join(result.get('files_removed', []))})",
            get_prompt_table_data(),
            gr.update(choices=new_prompts, value=new_prompts[0] if new_prompts else None),
        )
    except Exception as e:
        return f"Delete failed: {e}", get_prompt_table_data(), gr.update()


def set_voice_default(name):
    """Set a voice as the default clone prompt. Returns (status_msg, table_data)."""
    if not name:
        return "No voice selected.", get_prompt_table_data()
    try:
        set_default_clone_prompt(name)
        return f"Set '{name}' as default voice.", get_prompt_table_data()
    except Exception as e:
        return f"Failed to set default: {e}", get_prompt_table_data()


# =============================================================================
# Model Management helpers
# =============================================================================

def get_model_table_data():
    """Fetch model info from server and format as table rows."""
    client = TTSClient()
    try:
        if not client.is_server_running():
            return []
        models_resp = client.get_models()
        models = models_resp.get("models", {})
        rows = []
        for model_type in ("clone", "design", "custom"):
            info = models.get(model_type, {})
            status = "Loaded" if info.get("loaded") else "Unloaded"
            memory = f"{info.get('memory_mb', '?')} MB"
            load_time = info.get("load_time_sec")
            load_time_str = f"{load_time:.1f}s" if load_time else "-"
            startup = "Yes" if info.get("load_at_startup") else "No"
            label = "Clone / Create" if model_type == "clone" else model_type.capitalize()
            rows.append([label, status, memory, load_time_str, startup])
        # ASR row
        from qwen3_tts.core.engine import is_asr_loaded, get_asr_model_info
        asr_info = get_asr_model_info()
        asr_status = "Loaded" if asr_info.get("loaded") else "Unloaded"
        asr_model = asr_info.get("model_name", "-") or "-"
        rows.append(["ASR (Whisper)", asr_status, asr_model, "-", "-"])
        return rows
    except Exception as e:
        logger.warning("Failed to get model table data: %s", e)
        return []


def toggle_model(model_type, action):
    """Load or unload a TTS model. Returns (status_msg, table_data, status_html)."""
    client = TTSClient()
    if not client.is_server_running():
        return "Server not running", get_model_table_data(), format_status_display()
    try:
        if action == "load":
            result = client.load_model(model_type)
            msg = f"{model_type.capitalize()} model: {result.get('status', 'done')}"
        else:
            result = client.unload_model(model_type)
            msg = f"{model_type.capitalize()} model: {result.get('status', 'done')}"
        return msg, get_model_table_data(), format_status_display()
    except Exception as e:
        return f"Error: {e}", get_model_table_data(), format_status_display()


def toggle_asr(action):
    """Load or unload ASR model. Returns (status_msg, table_data)."""
    try:
        if action == "load":
            from qwen3_tts.core.engine import is_asr_loaded, load_asr_model
            if is_asr_loaded():
                return "ASR already loaded", get_model_table_data()
            load_asr_model()
            return "ASR model loaded", get_model_table_data()
        else:
            from qwen3_tts.core.engine import unload_asr_model
            unload_asr_model()
            return "ASR model unloaded", get_model_table_data()
    except Exception as e:
        return f"Error: {e}", get_model_table_data()


def update_startup_defaults(clone_startup, design_startup, custom_startup):
    """Update which models load at server startup. Returns status message."""
    client = TTSClient()
    if not client.is_server_running():
        return "Server not running"
    try:
        result = client.update_startup_config(
            clone=clone_startup, design=design_startup, custom=custom_startup
        )
        changes = result.get("changes", [])
        return f"Startup config updated: {', '.join(changes)}"
    except Exception as e:
        return f"Error: {e}"


def get_model_status_html(model_type):
    """Compact colored indicator for a model's load status.

    Shows:
    - Green if model is loaded
    - Red with error message if load failed
    - Yellow if not loaded (will load on demand)
    - Gray if server offline
    """
    client = TTSClient()
    try:
        if not client.is_server_running():
            return '<span style="color: gray;">⚪ Server offline</span>'

        health = client.get_health()
        loaded = health.get(f"{model_type}_model_loaded", False)
        errors = health.get("model_load_errors", {})
        error_msg = errors.get(model_type)

        if error_msg:
            # Truncate very long error messages
            display_error = error_msg if len(error_msg) <= 60 else error_msg[:57] + "..."
            return f'<span style="color: red;">🔴 {model_type.capitalize()} error: {display_error}</span>'
        elif loaded:
            return f'<span style="color: green; font-weight: bold;">🟢 {model_type.capitalize()} loaded</span>'
        else:
            return f'<span style="color: #b0b000;">🟡 {model_type.capitalize()} not loaded — will load on demand</span>'
    except Exception:
        return '<span style="color: gray;">⚪ Unknown</span>'


def get_audio_loader_setting():
    """Get current audio loader setting from engine cache."""
    try:
        from qwen3_tts.core.engine import get_audio_loader
        return get_audio_loader()
    except Exception:
        return "torchaudio"


def set_audio_loader_setting(loader):
    """Set audio loader preference. Returns status message."""
    try:
        from qwen3_tts.core.engine import set_audio_loader
        set_audio_loader(loader)
        # Also persist to config
        config = load_config()
        if "advanced" not in config:
            config["advanced"] = {}
        config["advanced"]["audio_loader"] = loader
        from qwen3_tts.core.config import save_config
        save_config(config)
        return f"Audio loader set to: {loader}"
    except Exception as e:
        return f"Error: {e}"


def _build_common_controls(audio_processing_label="Audio Processing"):
    """Build the common right-column controls shared by all generation tabs.

    Returns dict with keys: streaming, temp, top_k, top_p, rep, seed,
    trim, norm, speed, pitch.
    """
    streaming = gr.Checkbox(
        label="Enable Streaming",
        value=True,
        info="Hear audio as it generates (MLX: native, torch: chunked)"
    )

    with gr.Accordion("Advanced Settings", open=False):
        temp = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
        top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-P")
        rep = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="Repetition Penalty")
        seed = gr.Textbox(label="Seed (empty for random)", value="")

    with gr.Accordion(audio_processing_label, open=False):
        if "Style" in audio_processing_label:
            gr.Markdown("Use speed/pitch to modify the cloned voice's delivery style.", elem_classes=["info-text"])
        trim = gr.Checkbox(label="Trim Silence", value=False)
        norm = gr.Checkbox(label="Normalize", value=False)
        speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
        pitch = gr.Slider(-12, 12, value=0, step=1, label="Pitch (semitones)")

    return {
        "streaming": streaming, "temp": temp, "top_k": top_k, "top_p": top_p,
        "rep": rep, "seed": seed, "trim": trim, "norm": norm,
        "speed": speed, "pitch": pitch,
    }


def _build_generate_buttons_and_output():
    """Build the Generate/Stop buttons and output components.

    Returns dict with keys: btn, cancel_btn, output, status.
    """
    with gr.Row():
        btn = gr.Button("Generate", variant="primary")
        cancel_btn = gr.Button("Stop", variant="stop")
    output = gr.Audio(label="Output", streaming=True, autoplay=True)
    status = gr.Textbox(label="Status", interactive=False)
    return {"btn": btn, "cancel_btn": cancel_btn, "output": output, "status": status}


def _wire_generation_tab(mode, btn, cancel_btn, output, status, model_indicator,
                         text, text_info, inputs_list, status_html, history_df,
                         handler, api_name=None, history_state=None):
    """Wire up the common event handlers for a generation tab.

    Args:
        mode: The generation mode ('clone', 'design', 'custom').
        btn: The generate button component.
        cancel_btn: The cancel button component.
        output: The audio output component.
        status: The status textbox component.
        model_indicator: The model status HTML component.
        text: The input textbox component.
        text_info: The text info textbox component.
        inputs_list: List of input components for the handler.
        status_html: The status HTML component.
        history_df: The history dataframe component.
        handler: The generation handler function.
        api_name: Optional API name for the endpoint.
        history_state: Optional history state component.
    """
    # Include JavaScript reset to clear audio state between generations
    click_kwargs = {
        "fn": handler,
        "inputs": inputs_list,
        "outputs": [output, status, status_html, history_state, history_df],
        "js": _create_audio_reset_js(),  # Reset audio state before each generation
    }
    if api_name:
        click_kwargs["api_name"] = api_name

    btn.click(**click_kwargs).then(
        fn=lambda: get_model_status_html(mode),
        outputs=model_indicator
    )

    # Cancel only updates output/status/status_html; history_state and history_df
    # are intentionally not updated on cancel (no history entry for cancelled generations).
    cancel_btn.click(
        fn=cancel_streaming_generation,
        outputs=[output, status, status_html]
    )

    text.change(fn=update_text_info, inputs=text, outputs=text_info)


def build_ui():
    """Build the Gradio interface."""

    with gr.Blocks(title="Qwen3-TTS Web Interface") as demo:
        gr.Markdown("# Qwen3-TTS Web Interface")

        # Status bar
        status_html = gr.HTML(value=format_status_display())
        with gr.Row():
            refresh_btn = gr.Button("Refresh Status", size="sm")
            stop_btn = gr.Button("Stop Server", size="sm", variant="stop")
        refresh_btn.click(fn=format_status_display, outputs=status_html)
        if hasattr(gr, 'Timer'):
            gr.Timer(value=5).tick(fn=format_status_display, outputs=status_html)
        stop_btn.click(
            fn=stop_server,
            outputs=status_html,
        )

        # Model Settings (MLX-first architecture)
        current_size, current_quant, current_backend = get_current_model_settings()
        with gr.Accordion("Model Settings", open=False):
            gr.Markdown("Change model size or quantization. Settings apply on next generation.")
            with gr.Row():
                model_size_dropdown = gr.Dropdown(
                    label="Model Size",
                    choices=list(VALID_MODEL_SIZES),
                    value=current_size,
                    info="1.7B: higher quality | 0.6B: ~40% faster, lower memory"
                )
                mlx_quant_dropdown = gr.Dropdown(
                    label="MLX Quantization",
                    choices=list(VALID_MLX_QUANTIZATIONS),
                    value=current_quant,
                    info="4bit: smallest | 8bit: balanced | bf16: highest quality",
                    visible=(current_backend == "mlx")
                )
            with gr.Row():
                apply_settings_btn = gr.Button("Apply Settings", variant="secondary", size="sm")
                settings_status = gr.Textbox(
                    label="", show_label=False, interactive=False,
                    max_lines=1, container=False, scale=3
                )

            apply_settings_btn.click(
                fn=apply_model_settings,
                inputs=[model_size_dropdown, mlx_quant_dropdown],
                outputs=[settings_status, status_html]
            )

        # Per-session history state
        history_state = gr.State([])

        # History Panel (defined before tabs so history_df can be referenced in click handlers)
        with gr.Accordion("Recent Generations", open=False):
            history_df = gr.Dataframe(
                headers=["Time", "Mode", "Text Preview", "Chunks"],
                value=[],
                interactive=False,
                wrap=True,
            )
            history_audio = gr.Audio(label="Selected Generation", visible=False)
            history_df.select(
                fn=get_history_audio,
                inputs=[history_state],
                outputs=history_audio
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=history_audio
            )
            refresh_history_btn = gr.Button("Refresh History", size="sm")
            refresh_history_btn.click(
                fn=get_history_data,
                inputs=[history_state],
                outputs=history_df
            )

        # Tabs for different modes
        with gr.Tabs():
            # ---- Clone Mode Tab ----
            with gr.Tab("Clone Mode") as clone_tab:
                gr.Markdown(
                    "Use a voice prompt file to clone a specific voice. "
                    "Clone mode reproduces the voice from your reference audio. "
                    "For voice design from descriptions, use Design mode. "
                    "To create a reusable designed voice, generate in Design mode then save as a voice prompt."
                )
                clone_model_indicator = gr.HTML(value=get_model_status_html("clone"))

                with gr.Row():
                    with gr.Column(scale=2):
                        clone_text = gr.Textbox(label="Text Input", placeholder="Enter text to synthesize...", lines=3)
                        clone_text_info = gr.Textbox(label="", show_label=False, interactive=False, max_lines=1, container=False)
                        _default_prompt = get_default_clone_prompt()
                        _prompts = get_voice_prompts()
                        clone_prompt = gr.Dropdown(
                            label="Voice Prompt", choices=_prompts,
                            value=_default_prompt if _default_prompt in _prompts else (_prompts[0] if _prompts else None)
                        )
                        clone_preset = gr.Dropdown(label="Preset", choices=get_presets(), value="(none)")

                    with gr.Column(scale=1):
                        clone_ctrls = _build_common_controls(audio_processing_label="Audio Processing (Style Adjustment)")
                        clone_no_transcript = gr.Checkbox(
                            label="Speaker embedding only", value=False,
                            info="Clone using x-vector only (no transcript needed, lower fidelity)"
                        )

                clone_btns = _build_generate_buttons_and_output()

                def clone_handler(text, prompt, preset, temp, top_k, top_p, rep, seed,
                                  trim, norm, speed, pitch, streaming, no_transcript, history_list):
                    if streaming:
                        yield from generate_clone_streaming(
                            text, prompt, preset, temp, top_k, top_p, rep, seed,
                            trim, norm, speed, pitch, no_transcript=no_transcript, history_list=history_list)
                    else:
                        yield generate_clone(
                            text, prompt, preset, temp, top_k, top_p, rep, seed,
                            trim, norm, speed, pitch, no_transcript=no_transcript, history_list=history_list)

                _wire_generation_tab(
                    "clone", clone_btns["btn"], clone_btns["cancel_btn"],
                    clone_btns["output"], clone_btns["status"], clone_model_indicator,
                    clone_text, clone_text_info,
                    inputs_list=[clone_text, clone_prompt, clone_preset,
                                 clone_ctrls["temp"], clone_ctrls["top_k"], clone_ctrls["top_p"],
                                 clone_ctrls["rep"], clone_ctrls["seed"],
                                 clone_ctrls["trim"], clone_ctrls["norm"],
                                 clone_ctrls["speed"], clone_ctrls["pitch"],
                                 clone_ctrls["streaming"], clone_no_transcript,
                                 history_state],
                    status_html=status_html, history_df=history_df,
                    handler=clone_handler, api_name="generate_clone",
                    history_state=history_state,
                )

            # ---- Design Mode Tab ----
            with gr.Tab("Design Mode") as design_tab:
                gr.Markdown("Generate a voice from a text description.")
                design_model_indicator = gr.HTML(value=get_model_status_html("design"))

                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(label="Text Input", placeholder="Enter text to synthesize...", lines=3)
                        design_text_info = gr.Textbox(label="", show_label=False, interactive=False, max_lines=1, container=False)
                        design_desc = gr.Textbox(
                            label="Voice Description",
                            placeholder="Describe the voice (e.g., 'A warm, friendly female voice with clear articulation')",
                            lines=2
                        )
                        with gr.Row():
                            design_prosody = gr.Dropdown(
                                label="Style Preset", choices=get_prosody_choices(), value="(none)",
                                info="Appends style to description", scale=2,
                            )
                            _enhancer_visible = is_enhancer_available()
                            design_enhance_btn = gr.Button(
                                "Enhance with AI", size="sm", variant="secondary",
                                visible=_enhancer_visible, scale=1,
                            )

                        with gr.Accordion("Description Builder", open=False):
                            gr.Markdown("Build a voice description from attributes:")
                            _none_opt = ["(none)"]
                            with gr.Row():
                                db_gender = gr.Dropdown(label="Gender", choices=_none_opt + VOICE_DESCRIPTION_ATTRIBUTES["gender"], value="(none)")
                                db_age = gr.Dropdown(label="Age", choices=_none_opt + VOICE_DESCRIPTION_ATTRIBUTES["age"], value="(none)")
                            with gr.Row():
                                db_tone = gr.Dropdown(label="Tone", choices=_none_opt + VOICE_DESCRIPTION_ATTRIBUTES["tone"], value="(none)")
                                db_texture = gr.Dropdown(label="Texture", choices=_none_opt + VOICE_DESCRIPTION_ATTRIBUTES["texture"], value="(none)")
                            with gr.Row():
                                db_pace = gr.Dropdown(label="Pace", choices=_none_opt + VOICE_DESCRIPTION_ATTRIBUTES["pace"], value="(none)")
                                db_accent = gr.Dropdown(label="Accent", choices=_none_opt + VOICE_DESCRIPTION_ATTRIBUTES["accent"], value="(none)")
                            db_compose_btn = gr.Button("Compose Description", size="sm", variant="secondary")

                        design_preset = gr.Dropdown(label="Preset", choices=get_presets(), value="(none)")

                    with gr.Column(scale=1):
                        design_ctrls = _build_common_controls()

                design_btns = _build_generate_buttons_and_output()

                # Save as Voice Prompt (Design-then-Clone pipeline)
                with gr.Accordion("Save as Voice Prompt", open=False):
                    gr.Markdown("Save the generated audio as a reusable voice clone prompt.")
                    design_save_name = gr.Textbox(label="Voice Name", placeholder="e.g., designed_voice", max_lines=1)
                    design_save_btn = gr.Button("Save as Voice Prompt", size="sm", variant="secondary")
                    design_save_status = gr.Textbox(label="", show_label=False, interactive=False, max_lines=1, container=False)

                def design_handler(text, desc, preset, temp, top_k, top_p, rep, seed,
                                   trim, norm, speed, pitch, streaming, history_list):
                    if streaming:
                        yield from generate_design_streaming(
                            text, desc, preset, temp, top_k, top_p, rep, seed, trim, norm, speed, pitch,
                            history_list=history_list)
                    else:
                        yield generate_design(
                            text, desc, preset, temp, top_k, top_p, rep, seed, trim, norm, speed, pitch,
                            history_list=history_list)

                _wire_generation_tab(
                    "design", design_btns["btn"], design_btns["cancel_btn"],
                    design_btns["output"], design_btns["status"], design_model_indicator,
                    design_text, design_text_info,
                    inputs_list=[design_text, design_desc, design_preset,
                                 design_ctrls["temp"], design_ctrls["top_k"], design_ctrls["top_p"],
                                 design_ctrls["rep"], design_ctrls["seed"],
                                 design_ctrls["trim"], design_ctrls["norm"],
                                 design_ctrls["speed"], design_ctrls["pitch"],
                                 design_ctrls["streaming"],
                                 history_state],
                    status_html=status_html, history_df=history_df, handler=design_handler,
                    history_state=history_state,
                )

                design_prosody.change(fn=apply_prosody_preset, inputs=[design_prosody, design_desc], outputs=design_desc)

                # Wire up Description Builder
                db_compose_btn.click(
                    fn=compose_voice_description,
                    inputs=[db_gender, db_age, db_tone, db_texture, db_pace, db_accent],
                    outputs=design_desc,
                )

                # Wire up Enhance button
                design_enhance_btn.click(fn=enhance_description_with_ai, inputs=[design_desc], outputs=design_desc)

                # Wire up Save as Voice Prompt
                def save_design_as_prompt(voice_name, history_list):
                    """Save the most recent Design mode output as a voice prompt."""
                    if not voice_name or not voice_name.strip():
                        return "Please enter a voice name.", gr.update()
                    voice_name = voice_name.strip().replace(" ", "_").replace("/", "_").replace("\\", "_").replace("..", "")
                    for entry in history_list:
                        if entry.get("mode") == "Design" and entry.get("path"):
                            audio_path = entry["path"]
                            if os.path.exists(audio_path):
                                try:
                                    from qwen3_tts.tools.create_voice import create_and_save_voice_prompt
                                    from qwen3_tts.core.config import IN_COLAB
                                    backend = get_backend()
                                    mlx_only = (backend == "mlx") or IN_COLAB
                                    create_and_save_voice_prompt(
                                        audio_path, "", voice_name,
                                        test_generation=False, mlx_only=mlx_only,
                                    )
                                    prompts = get_voice_prompts()
                                    return f"Saved voice prompt: {voice_name}", gr.update(choices=prompts)
                                except Exception as e:
                                    return f"Error: {e}", gr.update()
                    return "No recent Design mode output found. Generate audio first.", gr.update()

                design_save_btn.click(
                    fn=save_design_as_prompt, inputs=[design_save_name, history_state],
                    outputs=[design_save_status, clone_prompt],
                )

            # ---- Custom Mode Tab ----
            with gr.Tab("Custom Mode") as custom_tab:
                gr.Markdown("Use premium pre-trained speakers.")
                custom_model_indicator = gr.HTML(value=get_model_status_html("custom"))

                with gr.Row():
                    with gr.Column(scale=2):
                        custom_text = gr.Textbox(label="Text Input", placeholder="Enter text to synthesize...", lines=3)
                        custom_text_info = gr.Textbox(label="", show_label=False, interactive=False, max_lines=1, container=False)
                        custom_speaker = gr.Dropdown(label="Speaker", choices=SPEAKER_CHOICES, value=SPEAKER_CHOICES[0])
                        custom_prosody = gr.Dropdown(
                            label="Style Preset", choices=get_prosody_choices(), value="(none)",
                            info="Select a preset to fill the instruction field, or type your own below"
                        )
                        custom_instruct = gr.Textbox(
                            label="Style Instruction (optional)",
                            placeholder="e.g., 'Speak with enthusiasm' or 'Read slowly and clearly'", lines=1
                        )
                        custom_preset = gr.Dropdown(label="Preset", choices=get_presets(), value="(none)")

                    with gr.Column(scale=1):
                        custom_ctrls = _build_common_controls()

                custom_btns = _build_generate_buttons_and_output()

                def custom_handler(text, speaker, instruct, preset, temp, top_k, top_p, rep, seed,
                                   trim, norm, speed, pitch, streaming, history_list):
                    if streaming:
                        yield from generate_custom_streaming(
                            text, speaker, instruct, preset, temp, top_k, top_p, rep, seed,
                            trim, norm, speed, pitch, history_list=history_list)
                    else:
                        yield generate_custom(
                            text, speaker, instruct, preset, temp, top_k, top_p, rep, seed,
                            trim, norm, speed, pitch, history_list=history_list)

                _wire_generation_tab(
                    "custom", custom_btns["btn"], custom_btns["cancel_btn"],
                    custom_btns["output"], custom_btns["status"], custom_model_indicator,
                    custom_text, custom_text_info,
                    inputs_list=[custom_text, custom_speaker, custom_instruct, custom_preset,
                                 custom_ctrls["temp"], custom_ctrls["top_k"], custom_ctrls["top_p"],
                                 custom_ctrls["rep"], custom_ctrls["seed"],
                                 custom_ctrls["trim"], custom_ctrls["norm"],
                                 custom_ctrls["speed"], custom_ctrls["pitch"],
                                 custom_ctrls["streaming"],
                                 history_state],
                    status_html=status_html, history_df=history_df, handler=custom_handler,
                    history_state=history_state,
                )

                custom_prosody.change(fn=apply_prosody_preset, inputs=[custom_prosody, custom_instruct], outputs=custom_instruct)

            # Tab selection handlers - update model status when switching tabs
            clone_tab.select(
                fn=lambda: get_model_status_html("clone"),
                outputs=clone_model_indicator
            )
            design_tab.select(
                fn=lambda: get_model_status_html("design"),
                outputs=design_model_indicator
            )
            custom_tab.select(
                fn=lambda: get_model_status_html("custom"),
                outputs=custom_model_indicator
            )

            # Create Voice Tab
            with gr.Tab("Create Voice"):
                gr.Markdown("Create a new voice clone from reference audio.")

                with gr.Row():
                    with gr.Column(scale=2):
                        create_audio = gr.Audio(
                            label="Reference Audio",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        create_transcript = gr.Textbox(
                            label="Transcript",
                            placeholder="Enter the exact words spoken in the audio...",
                            lines=3
                        )
                        with gr.Row():
                            auto_transcribe_btn = gr.Button("Auto-Transcribe", size="sm")

                    with gr.Column(scale=1):
                        create_name = gr.Textbox(
                            label="Voice Name",
                            placeholder="e.g., my_voice",
                            info="Will create my_voice.pt + .wav + .txt"
                        )
                        create_no_transcript = gr.Checkbox(
                            label="Speaker embedding only",
                            value=False,
                            info="Create without transcript (x-vector only, lower fidelity)"
                        )
                        create_btn = gr.Button("Create Voice Prompt", variant="primary")
                        create_status = gr.Textbox(label="Status", interactive=False)
                        voice_list = gr.Dropdown(
                            label="Available Voices",
                            choices=get_voice_prompts(),
                            interactive=False
                        )

                auto_transcribe_btn.click(
                    fn=auto_transcribe_audio,
                    inputs=[create_audio],
                    outputs=[create_transcript]
                )

                create_btn.click(
                    fn=create_voice_prompt,
                    inputs=[create_audio, create_transcript, create_name, create_no_transcript],
                    outputs=[create_status, voice_list, clone_prompt]
                )

            # Manage Voices Tab
            with gr.Tab("Manage Voices"):
                gr.Markdown("View, preview, rename, and delete voice prompts.")

                with gr.Row():
                    with gr.Column(scale=2):
                        manage_table = gr.Dataframe(
                            headers=["Name", "Formats", "Size (MB)", "Default"],
                            value=get_prompt_table_data(),
                            interactive=False,
                            wrap=True,
                        )
                        manage_refresh_btn = gr.Button("Refresh List", size="sm")

                    with gr.Column(scale=1):
                        manage_preview_audio = gr.Audio(label="Preview", visible=True)
                        manage_selected = gr.Textbox(
                            label="Selected Voice", interactive=False,
                            max_lines=1, container=True
                        )
                        manage_new_name = gr.Textbox(
                            label="New Name (for rename)",
                            placeholder="Enter new name...",
                            max_lines=1
                        )
                        with gr.Row():
                            manage_preview_btn = gr.Button("Preview", size="sm", interactive=False)
                            manage_default_btn = gr.Button("Set Default", size="sm", interactive=False)
                        with gr.Row():
                            manage_rename_btn = gr.Button("Rename", size="sm", variant="secondary", interactive=False)
                            manage_delete_btn = gr.Button("Delete", size="sm", variant="stop", interactive=False)
                        manage_status = gr.Textbox(
                            label="", show_label=False, interactive=False,
                            max_lines=2, container=False
                        )

                # Row selection -> populate selected name
                def on_table_select(evt: gr.SelectData, table_data):
                    active = gr.update(interactive=True)
                    inactive = gr.update(interactive=False)
                    try:
                        row_idx = evt.index[0]
                        if hasattr(table_data, "iloc"):
                            if row_idx < len(table_data):
                                return str(table_data.iloc[row_idx, 0]), active, active, active, active
                        elif table_data and row_idx < len(table_data):
                            return table_data[row_idx][0], active, active, active, active
                    except (IndexError, TypeError, KeyError):
                        pass
                    return "", inactive, inactive, inactive, inactive

                manage_table.select(
                    fn=on_table_select,
                    inputs=[manage_table],
                    outputs=[manage_selected, manage_preview_btn, manage_default_btn, manage_rename_btn, manage_delete_btn]
                )

                manage_refresh_btn.click(
                    fn=get_prompt_table_data,
                    outputs=[manage_table]
                )

                manage_preview_btn.click(
                    fn=preview_voice,
                    inputs=[manage_selected],
                    outputs=[manage_preview_audio]
                )

                manage_default_btn.click(
                    fn=set_voice_default,
                    inputs=[manage_selected],
                    outputs=[manage_status, manage_table]
                )

                manage_rename_btn.click(
                    fn=rename_voice,
                    inputs=[manage_selected, manage_new_name],
                    outputs=[manage_status, manage_table, clone_prompt]
                )

                manage_delete_btn.click(
                    fn=delete_voice,
                    inputs=[manage_selected],
                    outputs=[manage_status, manage_table, clone_prompt],
                )

            # Manage Models Tab
            with gr.Tab("Manage Models"):
                gr.Markdown("Load/unload models, configure startup defaults, and audio loader.")

                with gr.Row():
                    with gr.Column(scale=2):
                        model_table = gr.Dataframe(
                            headers=["Model", "Status", "Memory", "Load Time", "Startup"],
                            value=get_model_table_data(),
                            interactive=False,
                            wrap=True,
                        )
                        model_refresh_btn = gr.Button("Refresh", size="sm")

                    with gr.Column(scale=1):
                        gr.Markdown("### Load / Unload")
                        with gr.Row():
                            model_type_select = gr.Dropdown(
                                label="Model",
                                choices=["clone", "design", "custom"],
                                value="clone",
                            )
                        with gr.Row():
                            model_load_btn = gr.Button("Load", size="sm", variant="primary")
                            model_unload_btn = gr.Button("Unload", size="sm", variant="stop")
                        model_manage_status = gr.Textbox(
                            label="", show_label=False, interactive=False,
                            max_lines=2, container=False
                        )

                        gr.Markdown("### ASR (Whisper)")
                        with gr.Row():
                            asr_load_btn = gr.Button("Load ASR", size="sm")
                            asr_unload_btn = gr.Button("Unload ASR", size="sm", variant="stop")

                        gr.Markdown("### Startup Defaults")
                        startup_clone = gr.Checkbox(label="Clone/Create at startup", value=True)
                        startup_design = gr.Checkbox(label="Design at startup", value=False)
                        startup_custom = gr.Checkbox(label="Custom at startup", value=False)
                        startup_save_btn = gr.Button("Save Startup Config", size="sm")
                        startup_status = gr.Textbox(
                            label="", show_label=False, interactive=False,
                            max_lines=1, container=False
                        )

                        gr.Markdown("### Audio Loader")
                        audio_loader_select = gr.Dropdown(
                            label="Audio Loader",
                            choices=["torchaudio", "librosa"],
                            value=get_audio_loader_setting(),
                            info="torchaudio: faster C++ | librosa: broader format support"
                        )
                        audio_loader_save_btn = gr.Button("Save Audio Loader", size="sm")
                        audio_loader_status = gr.Textbox(
                            label="", show_label=False, interactive=False,
                            max_lines=1, container=False
                        )

                # Wire up Manage Models handlers
                model_refresh_btn.click(
                    fn=get_model_table_data,
                    outputs=model_table
                )

                model_load_btn.click(
                    fn=lambda mt: toggle_model(mt, "load"),
                    inputs=[model_type_select],
                    outputs=[model_manage_status, model_table, status_html]
                ).then(
                    fn=lambda mt: (
                        get_model_status_html("clone") if mt == "clone" else gr.update(),
                        get_model_status_html("design") if mt == "design" else gr.update(),
                        get_model_status_html("custom") if mt == "custom" else gr.update(),
                    ),
                    inputs=[model_type_select],
                    outputs=[clone_model_indicator, design_model_indicator, custom_model_indicator]
                )

                model_unload_btn.click(
                    fn=lambda mt: toggle_model(mt, "unload"),
                    inputs=[model_type_select],
                    outputs=[model_manage_status, model_table, status_html]
                ).then(
                    fn=lambda mt: (
                        get_model_status_html("clone") if mt == "clone" else gr.update(),
                        get_model_status_html("design") if mt == "design" else gr.update(),
                        get_model_status_html("custom") if mt == "custom" else gr.update(),
                    ),
                    inputs=[model_type_select],
                    outputs=[clone_model_indicator, design_model_indicator, custom_model_indicator]
                )

                asr_load_btn.click(
                    fn=lambda: toggle_asr("load"),
                    outputs=[model_manage_status, model_table]
                )

                asr_unload_btn.click(
                    fn=lambda: toggle_asr("unload"),
                    outputs=[model_manage_status, model_table]
                )

                startup_save_btn.click(
                    fn=update_startup_defaults,
                    inputs=[startup_clone, startup_design, startup_custom],
                    outputs=startup_status
                )

                audio_loader_save_btn.click(
                    fn=set_audio_loader_setting,
                    inputs=[audio_loader_select],
                    outputs=audio_loader_status
                )

        # Update all model status indicators on UI load
        demo.load(
            fn=lambda: (
                get_model_status_html("clone"),
                get_model_status_html("design"),
                get_model_status_html("custom")
            ),
            outputs=[clone_model_indicator, design_model_indicator, custom_model_indicator]
        )

        # Footer
        gr.Markdown("""
        ---
        **Tips:**
        - Start the TTS server first: `tts server start`
        - Models auto-load on first use — no need to pre-load all three
        - Use **Model Settings** above to switch between model sizes (0.6B/1.7B) and quantizations (4bit/8bit/bf16)
        - Clone mode uses a voice prompt (.pt for PyTorch, .wav+.txt for MLX)
        - Design mode creates voices from text descriptions
        - Custom mode uses premium pre-trained speakers
        - Run `tts config` to optimize settings for your hardware
        """)

    # Preload ASR model in background (non-blocking)
    try:
        from qwen3_tts.core.engine import is_asr_available, preload_asr_model
        if is_asr_available():
            preload_asr_model()
            logger.info("ASR preload started in background")
    except Exception as e:
        logger.warning("ASR preload setup failed: %s", e)

    return demo


# =============================================================================
# Main
# =============================================================================

def _find_available_port(preferred, max_tries=10):
    """Return *preferred* port if free, otherwise the next available port.

    Scans preferred .. preferred+max_tries-1.  Returns None if all are taken.
    """
    import socket
    from qwen3_tts.core.config import IN_COLAB
    bind_addr = "0.0.0.0" if IN_COLAB else "127.0.0.1"  # nosec B104
    for offset in range(max_tries):
        port = preferred + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((bind_addr, port))
                return port
        except OSError:
            continue
    return None


def main():
    """Main entry point."""
    import argparse
    from qwen3_tts.core.config import load_config

    config = load_config()
    default_port = config.get("ui", {}).get("port", 7860)

    parser = argparse.ArgumentParser(description="Qwen3-TTS Web Interface")
    parser.add_argument("--port", type=int, default=default_port,
                        help=f"Port to run on (default: {default_port})")
    parser.add_argument("--share", action="store_true", help="Create public URL")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    args = parser.parse_args()

    # Find an available port (fallback to next in range if busy)
    port = _find_available_port(args.port)
    if port is None:
        print(f"Error: No available port in range {args.port}-{args.port + 9}.")
        sys.exit(1)
    if port != args.port:
        print(f"Port {args.port} is in use, using {port} instead.")

    # Check server status
    client = TTSClient()
    if not client.is_server_running():
        print("\n" + "=" * 60)
        print("WARNING: TTS Server is not running!")
        print("=" * 60)
        print("\nStart the server first for best experience:")
        print("  tts server start")
        print("\nThe UI will still load, but generation will fail until")
        print("the server is running.")
        print("=" * 60 + "\n")

    from qwen3_tts.core.config import IN_COLAB
    demo = build_ui()
    server_name = "0.0.0.0" if IN_COLAB else "127.0.0.1"  # nosec B104
    share = args.share or IN_COLAB
    inbrowser = not args.no_browser and not IN_COLAB
    # Allow Gradio to serve audio files from output and temp directories
    allowed = [os.path.expanduser("~/Downloads"), tempfile.gettempdir()]
    demo.launch(
        server_name=server_name,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        theme=gr.themes.Soft(),
        allowed_paths=allowed,
    )


if __name__ == "__main__":
    main()
