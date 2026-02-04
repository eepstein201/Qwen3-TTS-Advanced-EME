#!/usr/bin/env python3
"""
Qwen3-TTS Web Interface - Gradio-based UI for TTS generation.

Launch with:
    ttsUI
    # or
    python ~/Qwen3-TTS_UserFiles/tts_ui.py

Opens a web browser at http://localhost:7860
"""

import logging
import os
import sys
import threading
import time
import gradio as gr

logger = logging.getLogger("tts.ui")

# Add the user files directory to path for imports
sys.path.insert(0, os.path.expanduser("~/Qwen3-TTS_UserFiles"))

from tts_client import TTSClient
from tts_config import (
    CUSTOM_VOICE_SPEAKERS,
    VOICE_PROMPTS_DIR,
    get_default_clone_prompt,
    get_server_url,
    is_server_running,
    auth_headers,
)

# Derive speaker choices from canonical source
SPEAKER_CHOICES = [
    f"{key} ({info['lang']}) - {info['desc']}"
    for key, info in CUSTOM_VOICE_SPEAKERS.items()
]


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

def create_voice_prompt(audio_path, transcript, voice_name, auto_transcribed=False):
    """Create voice prompt via subprocess (handles env switching)."""
    import subprocess

    if not audio_path:
        raise gr.Error("Please upload an audio file")
    if not transcript or not transcript.strip():
        raise gr.Error("Please enter or auto-transcribe a transcript")
    if not voice_name or not voice_name.strip():
        raise gr.Error("Please enter a name for the voice")

    # Sanitize voice name
    voice_name = voice_name.strip().replace(" ", "_").replace("/", "_")

    cmd = [
        os.path.expanduser("~/bin/createVoice"),
        audio_path,
        "-n", voice_name,
        "-t", transcript,
        "--no-test",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise gr.Error(f"Failed: {result.stderr or result.stdout}")
        return f"✓ Created voice: {voice_name}.pt", get_voice_prompts()
    except subprocess.TimeoutExpired:
        raise gr.Error("Voice creation timed out (>120s)")
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")


def auto_transcribe_audio(audio_path):
    """Auto-transcribe using MLX ASR."""
    if not audio_path:
        raise gr.Error("Please upload an audio file first")

    try:
        from tts_engine import transcribe_audio, is_asr_available
        if not is_asr_available():
            raise gr.Error("Auto-transcribe requires MLX backend")
        transcript = transcribe_audio(audio_path)
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
        memory_val = stats.get('mps_memory_allocated_mb', 'N/A')
        if isinstance(memory_val, (int, float)):
            memory = f"{memory_val:.0f}MB"
        else:
            memory = str(memory_val)

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
    <div style="padding: 10px; background: #f5f5f5; border-radius: 5px; margin-bottom: 15px;">
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
    """Check if the required model is loaded; if not, load it on demand."""
    try:
        health = client.get_health()
        key = f"{mode}_model_loaded"
        if health.get(key):
            return  # already loaded
    except Exception:
        return  # can't reach server; generate will report the error

    progress(0, desc=f"Loading {mode} model (first use)...")
    try:
        client.load_model(mode)
    except Exception as e:
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
        except Exception:
            pass
        stop_event.wait(1.0)


# =============================================================================
# Generation History
# =============================================================================

# Session-level history (shared across tabs)
generation_history = []
MAX_HISTORY_SIZE = 10


def add_to_history(mode, text, output_path, duration_chunks):
    """Add a generation to history."""
    import datetime
    entry = {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "mode": mode.capitalize(),
        "text": text[:40] + "..." if len(text) > 40 else text,
        "chunks": duration_chunks,
        "path": output_path,
    }
    generation_history.insert(0, entry)
    if len(generation_history) > MAX_HISTORY_SIZE:
        generation_history.pop()


def get_history_data():
    """Return history as a list of lists for Dataframe display."""
    return [[h["time"], h["mode"], h["text"], f"{h['chunks']} chunks"] for h in generation_history]


def get_history_audio(evt: gr.SelectData):
    """Return the audio file path for the selected history row."""
    if evt.index[0] < len(generation_history):
        return generation_history[evt.index[0]]["path"]
    return None


# =============================================================================
# Generation Functions
# =============================================================================

def cancel_streaming_generation():
    """Cancel the current streaming generation."""
    client = TTSClient()
    try:
        result = client.cancel_generation()
        status = result.get("status", "unknown")
        if status == "cancellation_requested":
            return "Cancellation requested...", format_status_display()
        elif status == "no_active_generation":
            return "No active generation to cancel", format_status_display()
        return f"Cancel status: {status}", format_status_display()
    except Exception as e:
        return f"Cancel failed: {str(e)}", format_status_display()


def _save_streaming_audio(all_chunks, sample_rate):
    """Save accumulated streaming chunks to a temp file and return path."""
    import numpy as np
    import soundfile as sf

    if not all_chunks:
        return None

    combined = np.concatenate(all_chunks)
    timestamp = int(time.time())
    output_path = os.path.expanduser(f"~/Downloads/tts_ui_{timestamp}.wav")
    sf.write(output_path, combined, sample_rate)
    return output_path


def generate_clone_streaming(text, prompt, preset, temperature, top_k, top_p, rep_penalty, seed,
                             trim_silence, normalize, speed, pitch):
    """Generate audio with streaming - yields NEW chunk only for real-time playback."""
    import numpy as np

    if not text or not text.strip():
        yield None, "Error: Please enter some text.", gr.update()
        return

    client = TTSClient()
    if not client.is_server_running():
        yield None, "Error: TTS server is not running.", gr.update()
        return

    try:
        _ensure_model_loaded(client, "clone", lambda p, desc="": None)

        seed_val = int(seed) if seed and str(seed).strip() else None
        preset_val = preset if preset and preset != "(none)" else None

        all_chunks = []
        sample_rate = None
        chunk_count = 0

        for wav_chunk, sr in client.generate_streaming(
            text=text,
            mode="clone",
            prompt=prompt,
            preset=preset_val,
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            seed=seed_val,
            repetition_penalty=rep_penalty,
        ):
            all_chunks.append(wav_chunk)
            sample_rate = sr
            chunk_count += 1
            # Yield ONLY the new chunk for streaming playback
            yield (sr, wav_chunk), f"Streaming... {chunk_count} chunks", gr.update()

        # Final: save complete audio and return file path
        output_path = _save_streaming_audio(all_chunks, sample_rate)
        add_to_history("clone", text, output_path, chunk_count)
        yield output_path, f"Complete: {chunk_count} chunks", format_status_display()

    except Exception as e:
        yield None, f"Error: {str(e)}", format_status_display()


def generate_design_streaming(text, description, preset, temperature, top_k, top_p, rep_penalty, seed,
                              trim_silence, normalize, speed, pitch):
    """Generate audio with streaming for design mode."""
    import numpy as np

    if not text or not text.strip():
        yield None, "Error: Please enter some text.", gr.update()
        return

    if not description or not description.strip():
        yield None, "Error: Please enter a voice description.", gr.update()
        return

    client = TTSClient()
    if not client.is_server_running():
        yield None, "Error: TTS server is not running.", gr.update()
        return

    try:
        _ensure_model_loaded(client, "design", lambda p, desc="": None)

        seed_val = int(seed) if seed and str(seed).strip() else None
        preset_val = preset if preset and preset != "(none)" else None

        all_chunks = []
        sample_rate = None
        chunk_count = 0

        for wav_chunk, sr in client.generate_streaming(
            text=text,
            mode="design",
            description=description,
            preset=preset_val,
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            seed=seed_val,
            repetition_penalty=rep_penalty,
        ):
            all_chunks.append(wav_chunk)
            sample_rate = sr
            chunk_count += 1
            yield (sr, wav_chunk), f"Streaming... {chunk_count} chunks", gr.update()

        output_path = _save_streaming_audio(all_chunks, sample_rate)
        add_to_history("design", text, output_path, chunk_count)
        yield output_path, f"Complete: {chunk_count} chunks", format_status_display()

    except Exception as e:
        yield None, f"Error: {str(e)}", format_status_display()


def generate_custom_streaming(text, speaker_choice, instruct, preset, temperature, top_k, top_p, rep_penalty, seed,
                              trim_silence, normalize, speed, pitch):
    """Generate audio with streaming for custom mode."""
    import numpy as np

    if not text or not text.strip():
        yield None, "Error: Please enter some text.", gr.update()
        return

    client = TTSClient()
    if not client.is_server_running():
        yield None, "Error: TTS server is not running.", gr.update()
        return

    try:
        _ensure_model_loaded(client, "custom", lambda p, desc="": None)

        speaker = speaker_choice.split(" ")[0] if speaker_choice else "ryan"
        seed_val = int(seed) if seed and str(seed).strip() else None
        preset_val = preset if preset and preset != "(none)" else None

        all_chunks = []
        sample_rate = None
        chunk_count = 0

        for wav_chunk, sr in client.generate_streaming(
            text=text,
            mode="custom",
            speaker=speaker,
            instruct=instruct if instruct and instruct.strip() else None,
            preset=preset_val,
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            seed=seed_val,
            repetition_penalty=rep_penalty,
        ):
            all_chunks.append(wav_chunk)
            sample_rate = sr
            chunk_count += 1
            yield (sr, wav_chunk), f"Streaming... {chunk_count} chunks", gr.update()

        output_path = _save_streaming_audio(all_chunks, sample_rate)
        add_to_history("custom", text, output_path, chunk_count)
        yield output_path, f"Complete: {chunk_count} chunks", format_status_display()

    except Exception as e:
        yield None, f"Error: {str(e)}", format_status_display()


def generate_clone(text, prompt, preset, temperature, top_k, top_p, rep_penalty, seed,
                   trim_silence, normalize, speed, pitch, progress=gr.Progress()):
    """Generate audio using clone mode."""
    if not text or not text.strip():
        return None, "Error: Please enter some text to generate.", gr.update()

    client = TTSClient()

    if not client.is_server_running():
        return None, "Error: TTS server is not running. Start it with 'startTTSServer'.", gr.update()

    try:
        _ensure_model_loaded(client, "clone", progress)

        seed_val = int(seed) if seed and str(seed).strip() else None
        preset_val = preset if preset and preset != "(none)" else None
        timestamp = int(time.time())
        output_path = os.path.expanduser(f"~/Downloads/tts_ui_{timestamp}.wav")

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
                text=text, output=output_path, mode="clone", prompt=prompt,
                preset=preset_val, temperature=temperature, top_k=int(top_k),
                top_p=top_p, repetition_penalty=rep_penalty, seed=seed_val,
                trim_silence=trim_silence, normalize=normalize,
                speed=speed if speed != 1.0 else None,
                pitch=pitch if pitch != 0 else None,
            )
        finally:
            stop_event.set()
            poll_thread.join(timeout=2)

        progress(1.0, desc="Complete")
        return result, f"Generated: {os.path.basename(result)}", format_status_display()
    except Exception as e:
        error_msg = str(e)
        if "restart" in error_msg.lower() or "not running" in error_msg.lower():
            gr.Warning("Server issue — try restarting with 'startTTSServer'")
        return None, f"Error: {error_msg}", format_status_display()


def generate_design(text, description, preset, temperature, top_k, top_p, rep_penalty, seed,
                    trim_silence, normalize, speed, pitch, progress=gr.Progress()):
    """Generate audio using design mode."""
    if not text or not text.strip():
        return None, "Error: Please enter some text to generate.", gr.update()

    if not description or not description.strip():
        return None, "Error: Please enter a voice description.", gr.update()

    client = TTSClient()

    if not client.is_server_running():
        return None, "Error: TTS server is not running. Start it with 'startTTSServer'.", gr.update()

    try:
        _ensure_model_loaded(client, "design", progress)

        seed_val = int(seed) if seed and str(seed).strip() else None
        preset_val = preset if preset and preset != "(none)" else None
        timestamp = int(time.time())
        output_path = os.path.expanduser(f"~/Downloads/tts_ui_{timestamp}.wav")

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
                text=text, output=output_path, mode="design", description=description,
                preset=preset_val, temperature=temperature, top_k=int(top_k),
                top_p=top_p, repetition_penalty=rep_penalty, seed=seed_val,
                trim_silence=trim_silence, normalize=normalize,
                speed=speed if speed != 1.0 else None,
                pitch=pitch if pitch != 0 else None,
            )
        finally:
            stop_event.set()
            poll_thread.join(timeout=2)

        progress(1.0, desc="Complete")
        return result, f"Generated: {os.path.basename(result)}", format_status_display()
    except Exception as e:
        error_msg = str(e)
        if "restart" in error_msg.lower() or "not running" in error_msg.lower():
            gr.Warning("Server issue — try restarting with 'startTTSServer'")
        return None, f"Error: {error_msg}", format_status_display()


def generate_custom(text, speaker_choice, instruct, preset, temperature, top_k, top_p, rep_penalty, seed,
                    trim_silence, normalize, speed, pitch, progress=gr.Progress()):
    """Generate audio using custom mode with premium speakers."""
    if not text or not text.strip():
        return None, "Error: Please enter some text to generate.", gr.update()

    client = TTSClient()

    if not client.is_server_running():
        return None, "Error: TTS server is not running. Start it with 'startTTSServer'.", gr.update()

    try:
        _ensure_model_loaded(client, "custom", progress)

        speaker = speaker_choice.split(" ")[0] if speaker_choice else "ryan"
        seed_val = int(seed) if seed and str(seed).strip() else None
        preset_val = preset if preset and preset != "(none)" else None
        timestamp = int(time.time())
        output_path = os.path.expanduser(f"~/Downloads/tts_ui_{timestamp}.wav")

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
                text=text, output=output_path, mode="custom", speaker=speaker,
                instruct=instruct if instruct and instruct.strip() else None,
                preset=preset_val, temperature=temperature, top_k=int(top_k),
                top_p=top_p, repetition_penalty=rep_penalty, seed=seed_val,
                trim_silence=trim_silence, normalize=normalize,
                speed=speed if speed != 1.0 else None,
                pitch=pitch if pitch != 0 else None,
            )
        finally:
            stop_event.set()
            poll_thread.join(timeout=2)

        progress(1.0, desc="Complete")
        return result, f"Generated: {os.path.basename(result)}", format_status_display()
    except Exception as e:
        error_msg = str(e)
        if "restart" in error_msg.lower() or "not running" in error_msg.lower():
            gr.Warning("Server issue — try restarting with 'startTTSServer'")
        return None, f"Error: {error_msg}", format_status_display()


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
    except Exception:
        pass  # Server shuts down immediately, may not respond

    # Wait briefly and re-check
    time.sleep(1)
    return format_status_display()


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
        stop_btn.click(
            fn=stop_server,
            outputs=status_html,
            js="(x) => { if (!confirm('Stop the TTS server? Generation will be unavailable until you restart.')) throw new Error('Cancelled'); return x; }",
        )

        # Tabs for different modes
        with gr.Tabs():
            # Clone Mode Tab
            with gr.Tab("Clone Mode"):
                gr.Markdown("Use a voice prompt file to clone a specific voice.")

                with gr.Row():
                    with gr.Column(scale=2):
                        clone_text = gr.Textbox(
                            label="Text Input",
                            placeholder="Enter text to synthesize...",
                            lines=3
                        )
                        clone_text_info = gr.Textbox(
                            label="", show_label=False, interactive=False,
                            max_lines=1, container=False
                        )
                        _default_prompt = get_default_clone_prompt()
                        _prompts = get_voice_prompts()
                        clone_prompt = gr.Dropdown(
                            label="Voice Prompt",
                            choices=_prompts,
                            value=_default_prompt if _default_prompt in _prompts else (_prompts[0] if _prompts else None)
                        )
                        clone_preset = gr.Dropdown(
                            label="Preset",
                            choices=get_presets(),
                            value="(none)"
                        )

                    with gr.Column(scale=1):
                        clone_streaming = gr.Checkbox(
                            label="Enable Streaming",
                            value=True,
                            info="Hear audio as it generates (MLX: native, torch: chunked)"
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            clone_temp = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                            clone_top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
                            clone_top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-P")
                            clone_rep = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="Repetition Penalty")
                            clone_seed = gr.Textbox(label="Seed (empty for random)", value="")

                        with gr.Accordion("Audio Processing", open=False):
                            clone_trim = gr.Checkbox(label="Trim Silence", value=False)
                            clone_norm = gr.Checkbox(label="Normalize", value=False)
                            clone_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                            clone_pitch = gr.Slider(-12, 12, value=0, step=1, label="Pitch (semitones)")

                with gr.Row():
                    clone_btn = gr.Button("Generate", variant="primary")
                    clone_cancel_btn = gr.Button("Stop", variant="stop")
                clone_output = gr.Audio(label="Output", streaming=True, autoplay=True)
                clone_status = gr.Textbox(label="Status", interactive=False)

                # Dynamic handler based on streaming checkbox
                def clone_handler(text, prompt, preset, temp, top_k, top_p, rep, seed,
                                  trim, norm, speed, pitch, streaming):
                    if streaming:
                        yield from generate_clone_streaming(
                            text, prompt, preset, temp, top_k, top_p, rep, seed,
                            trim, norm, speed, pitch
                        )
                    else:
                        # Non-streaming: use original function
                        result = generate_clone(
                            text, prompt, preset, temp, top_k, top_p, rep, seed,
                            trim, norm, speed, pitch
                        )
                        yield result

                clone_btn.click(
                    fn=clone_handler,
                    inputs=[clone_text, clone_prompt, clone_preset, clone_temp, clone_top_k,
                            clone_top_p, clone_rep, clone_seed, clone_trim, clone_norm,
                            clone_speed, clone_pitch, clone_streaming],
                    outputs=[clone_output, clone_status, status_html]
                )

                clone_cancel_btn.click(
                    fn=cancel_streaming_generation,
                    outputs=[clone_status, status_html]
                )

                clone_text.change(fn=update_text_info, inputs=clone_text, outputs=clone_text_info)

            # Design Mode Tab
            with gr.Tab("Design Mode"):
                gr.Markdown("Generate a voice from a text description.")

                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(
                            label="Text Input",
                            placeholder="Enter text to synthesize...",
                            lines=3
                        )
                        design_text_info = gr.Textbox(
                            label="", show_label=False, interactive=False,
                            max_lines=1, container=False
                        )
                        design_desc = gr.Textbox(
                            label="Voice Description",
                            placeholder="Describe the voice (e.g., 'A warm, friendly female voice with clear articulation')",
                            lines=2
                        )
                        design_preset = gr.Dropdown(
                            label="Preset",
                            choices=get_presets(),
                            value="(none)"
                        )

                    with gr.Column(scale=1):
                        design_streaming = gr.Checkbox(
                            label="Enable Streaming",
                            value=True,
                            info="Hear audio as it generates (MLX: native, torch: chunked)"
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            design_temp = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                            design_top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
                            design_top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-P")
                            design_rep = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="Repetition Penalty")
                            design_seed = gr.Textbox(label="Seed (empty for random)", value="")

                        with gr.Accordion("Audio Processing", open=False):
                            design_trim = gr.Checkbox(label="Trim Silence", value=False)
                            design_norm = gr.Checkbox(label="Normalize", value=False)
                            design_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                            design_pitch = gr.Slider(-12, 12, value=0, step=1, label="Pitch (semitones)")

                with gr.Row():
                    design_btn = gr.Button("Generate", variant="primary")
                    design_cancel_btn = gr.Button("Stop", variant="stop")
                design_output = gr.Audio(label="Output", streaming=True, autoplay=True)
                design_status = gr.Textbox(label="Status", interactive=False)

                def design_handler(text, desc, preset, temp, top_k, top_p, rep, seed,
                                   trim, norm, speed, pitch, streaming):
                    if streaming:
                        yield from generate_design_streaming(
                            text, desc, preset, temp, top_k, top_p, rep, seed,
                            trim, norm, speed, pitch
                        )
                    else:
                        result = generate_design(
                            text, desc, preset, temp, top_k, top_p, rep, seed,
                            trim, norm, speed, pitch
                        )
                        yield result

                design_btn.click(
                    fn=design_handler,
                    inputs=[design_text, design_desc, design_preset, design_temp, design_top_k,
                            design_top_p, design_rep, design_seed, design_trim, design_norm,
                            design_speed, design_pitch, design_streaming],
                    outputs=[design_output, design_status, status_html]
                )

                design_cancel_btn.click(
                    fn=cancel_streaming_generation,
                    outputs=[design_status, status_html]
                )

                design_text.change(fn=update_text_info, inputs=design_text, outputs=design_text_info)

            # Custom Mode Tab
            with gr.Tab("Custom Mode"):
                gr.Markdown("Use premium pre-trained speakers.")

                with gr.Row():
                    with gr.Column(scale=2):
                        custom_text = gr.Textbox(
                            label="Text Input",
                            placeholder="Enter text to synthesize...",
                            lines=3
                        )
                        custom_text_info = gr.Textbox(
                            label="", show_label=False, interactive=False,
                            max_lines=1, container=False
                        )
                        custom_speaker = gr.Dropdown(
                            label="Speaker",
                            choices=SPEAKER_CHOICES,
                            value=SPEAKER_CHOICES[0]
                        )
                        custom_instruct = gr.Textbox(
                            label="Style Instruction (optional)",
                            placeholder="e.g., 'Speak with enthusiasm' or 'Read slowly and clearly'",
                            lines=1
                        )
                        custom_preset = gr.Dropdown(
                            label="Preset",
                            choices=get_presets(),
                            value="(none)"
                        )

                    with gr.Column(scale=1):
                        custom_streaming = gr.Checkbox(
                            label="Enable Streaming",
                            value=True,
                            info="Hear audio as it generates (MLX: native, torch: chunked)"
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            custom_temp = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                            custom_top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
                            custom_top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-P")
                            custom_rep = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="Repetition Penalty")
                            custom_seed = gr.Textbox(label="Seed (empty for random)", value="")

                        with gr.Accordion("Audio Processing", open=False):
                            custom_trim = gr.Checkbox(label="Trim Silence", value=False)
                            custom_norm = gr.Checkbox(label="Normalize", value=False)
                            custom_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                            custom_pitch = gr.Slider(-12, 12, value=0, step=1, label="Pitch (semitones)")

                with gr.Row():
                    custom_btn = gr.Button("Generate", variant="primary")
                    custom_cancel_btn = gr.Button("Stop", variant="stop")
                custom_output = gr.Audio(label="Output", streaming=True, autoplay=True)
                custom_status = gr.Textbox(label="Status", interactive=False)

                def custom_handler(text, speaker, instruct, preset, temp, top_k, top_p, rep, seed,
                                   trim, norm, speed, pitch, streaming):
                    if streaming:
                        yield from generate_custom_streaming(
                            text, speaker, instruct, preset, temp, top_k, top_p, rep, seed,
                            trim, norm, speed, pitch
                        )
                    else:
                        result = generate_custom(
                            text, speaker, instruct, preset, temp, top_k, top_p, rep, seed,
                            trim, norm, speed, pitch
                        )
                        yield result

                custom_btn.click(
                    fn=custom_handler,
                    inputs=[custom_text, custom_speaker, custom_instruct, custom_preset,
                            custom_temp, custom_top_k, custom_top_p, custom_rep, custom_seed,
                            custom_trim, custom_norm, custom_speed, custom_pitch, custom_streaming],
                    outputs=[custom_output, custom_status, status_html]
                )

                custom_cancel_btn.click(
                    fn=cancel_streaming_generation,
                    outputs=[custom_status, status_html]
                )

                custom_text.change(fn=update_text_info, inputs=custom_text, outputs=custom_text_info)

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
                            auto_transcribe_btn = gr.Button("Auto-Transcribe (MLX)", size="sm")

                    with gr.Column(scale=1):
                        create_name = gr.Textbox(
                            label="Voice Name",
                            placeholder="e.g., my_voice",
                            info="Will create my_voice.pt + .wav + .txt"
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
                    inputs=[create_audio, create_transcript, create_name],
                    outputs=[create_status, voice_list]
                )

        # History Panel
        with gr.Accordion("Recent Generations", open=False):
            history_df = gr.Dataframe(
                headers=["Time", "Mode", "Text Preview", "Chunks"],
                value=get_history_data,
                interactive=False,
                wrap=True,
            )
            history_audio = gr.Audio(label="Selected Generation", visible=False)
            history_df.select(
                fn=get_history_audio,
                outputs=history_audio
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=history_audio
            )
            refresh_history_btn = gr.Button("Refresh History", size="sm")
            refresh_history_btn.click(
                fn=get_history_data,
                outputs=history_df
            )

        # Footer
        gr.Markdown("""
        ---
        **Tips:**
        - Start the TTS server first: `startTTSServer`
        - Models auto-load on first use — no need to pre-load all three
        - Clone mode uses a voice prompt (.pt for PyTorch, .wav+.txt for MLX)
        - Design mode creates voices from text descriptions
        - Custom mode uses premium pre-trained speakers
        - Switch backends in `config.json` → `advanced.backend` ("torch" or "mlx")
        """)

    return demo


# =============================================================================
# Main
# =============================================================================

def _find_available_port(preferred, max_tries=10):
    """Return *preferred* port if free, otherwise the next available port.

    Scans preferred .. preferred+max_tries-1.  Returns None if all are taken.
    """
    import socket
    for offset in range(max_tries):
        port = preferred + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return None


def main():
    """Main entry point."""
    import argparse
    from tts_config import load_config

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
        print("  startTTSServer")
        print("\nThe UI will still load, but generation will fail until")
        print("the server is running.")
        print("=" * 60 + "\n")

    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=args.share,
        inbrowser=not args.no_browser,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
