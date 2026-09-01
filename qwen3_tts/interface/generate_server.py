#!/usr/bin/env python3
"""Server interaction functions for Qwen3-TTS generation.

Handles server-side generation (HTTP), streaming, local generation,
server lifecycle, and UI launch.
"""

import logging
import os
import subprocess  # nosec B404
import sys
import time

logger = logging.getLogger("tts.cli")


class TTSGenericError(RuntimeError):
    """Raised for TTS server-interaction failures that have no more specific type."""


# get_server_url / load_config are part of this module's namespace contract: they
# are patched by tests (see tests/test_generate_server.py). Keep them imported.
from qwen3_tts.core.config import (  # noqa: E402, F401
    CONFIG_PATH,
    VOICE_PROMPTS_DIR,
    get_backend,
    get_server_url,
    is_server_running,
    load_config,
    safe_path_join,
)

# Wire-format constants and the single shared frame parser. core.stream_protocol
# imports no FastAPI, torch or mlx, so the CLI can share the server's definition
# of the format instead of re-declaring it — these used to be local copies that
# drifted (a 200 MB cap here vs 100 MB in TTSClient). The two constants are
# re-exported because tests/test_stream_error_frame.py asserts this module's
# names stay in lockstep with the server's.
from qwen3_tts.core.stream_protocol import (  # noqa: E402, F401
    MAX_STREAM_CHUNK_BYTES,
    STREAM_ERROR_SENTINEL_SR,
    StreamProtocolError,
    iter_stream_chunks,
)
from qwen3_tts.interface.generate_helpers import (  # noqa: E402
    _build_generation_payload,
    _decode_base64_result,
    _save_base64_result,
    log_generation,
    play_audio,
    process_audio_args,
    voice_prompt_exists,
)

# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------


def ensure_server_running(config):
    """Ensure the TTS server is running, starting it if necessary."""
    if is_server_running(config):
        return True

    print("TTS Server is not running.")
    print(
        "Starting server (this may take 30-180 seconds on first run to download models)..."
    )

    import shutil

    tts_cmd = shutil.which("tts")
    if tts_cmd:
        result = subprocess.run([tts_cmd, "server", "start"], capture_output=False)  # nosec B603
        return result.returncode == 0

    # Fallback: start server directly as a module
    from qwen3_tts.core.config import LOG_FILE

    with open(LOG_FILE, "w") as log:
        subprocess.Popen(  # nosec B603
            [sys.executable, "-m", "qwen3_tts.server.app"],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )

    print("Waiting for server to be ready...")
    for i in range(300):
        if is_server_running(config):
            print("TTS Server is ready!")
            return True
        time.sleep(1)
        if (i + 1) % 10 == 0:
            print(f"  Still loading models... ({i + 1} seconds)")

    print("Error: Server failed to start. Check log:", LOG_FILE)
    return False


def build_ui_and_launch(config):
    """Build and launch the Gradio UI in-process."""
    from qwen3_tts.core.config import IN_COLAB
    from qwen3_tts.interface.ui import _find_available_port, build_ui
    from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

    # `tts ui --port N` arrives as TTS_UI_PORT (ui_command sets it), the same
    # way --share and --no-browser arrive below. It was previously never read
    # back, so the flag was accepted and silently dropped: the UI started fine
    # on the config/default port and the user's browser hit a dead address.
    # An explicit flag outranks the configured default; an unparseable value
    # falls back rather than taking down startup.
    preferred = config.get("ui", {}).get("port", 7860)
    requested = os.environ.get("TTS_UI_PORT")
    if requested:
        try:
            preferred = int(requested)
        except ValueError:
            print(f"Ignoring invalid TTS_UI_PORT={requested!r}; using {preferred}.")

    port = _find_available_port(preferred)
    if port is None:
        print(f"No available port found near {preferred}.")
        return
    share = bool(os.environ.get("TTS_UI_SHARE")) or IN_COLAB
    inbrowser = not bool(os.environ.get("TTS_UI_NO_BROWSER")) and not IN_COLAB
    demo = build_ui()
    demo.launch(
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        **get_gradio_launch_kwargs(config),
    )


def launch_gradio_ui(config):
    """Launch the Gradio web interface, starting server if needed."""
    if not ensure_server_running(config):
        print("\nCannot launch web interface without the server.")
        print("Please check the server logs and try again.")
        return

    print("\nLaunching Gradio web interface...")
    build_ui_and_launch(config)


def load_model_on_server(config, model_type):
    """Request the server to load a model on demand."""
    from qwen3_tts.core.http_client import (
        LOAD_MODEL_TIMEOUT_SEC,
        server_request,
    )

    print(f"Loading {model_type} model on server (this may take 30-60 seconds)...")
    resp = server_request(
        "POST",
        "/load-model",
        json={"model_type": model_type},
        timeout=LOAD_MODEL_TIMEOUT_SEC,
    )
    if resp.status_code == 200:
        result = resp.json()
        if result.get("status") == "loaded":
            print(f"Model '{model_type}' loaded successfully!")
            return True
        elif result.get("status") == "already_loaded":
            print(f"Model '{model_type}' was already loaded.")
            return True
    # _error_response nests the structured body under "detail"; read it
    # through the shared unwrap so the classified error/retry hint survives
    # (identical defect class fixed in voice_management).
    from qwen3_tts.server.client._base import _error_payload

    payload = _error_payload(resp)
    error = payload.get("detail") or payload.get("error") or "Unknown error"
    print(f"Failed to load model: {error}")
    return False


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------


def generate_via_server(
    texts,
    mode,
    config,
    gen_params,
    prompt_file=None,
    voice_description=None,
    speaker=None,
    instruct=None,
    auto_load_model=True,
    max_chunk_chars=None,
    x_vector_only_mode=False,
):
    """Generate audio via the TTS server."""
    import requests  # lazy (used for exception types only)

    from qwen3_tts.core.http_client import server_request
    from qwen3_tts.interface.generate_interactive import _ProgressPoller
    from qwen3_tts.server.client.generator import _generation_timeout

    payload = _build_generation_payload(
        mode,
        config,
        gen_params,
        prompt_file=prompt_file,
        voice_description=voice_description,
        speaker=speaker,
        instruct=instruct,
        x_vector_only_mode=x_vector_only_mode,
        max_chunk_chars=max_chunk_chars,
    )
    payload["texts"] = texts
    gen_timeout = _generation_timeout(sum(len(t) for t in texts))

    # Start progress polling
    progress = _ProgressPoller(batch_total=len(texts))
    progress.start()

    try:
        resp = server_request("POST", "/generate", json=payload, timeout=gen_timeout)
    finally:
        progress.stop()

    # Handle model not loaded
    if resp.status_code == 503:
        try:
            error_data = resp.json()
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            raise TTSGenericError(
                "Server returned HTTP 503 (non-JSON response)"
            ) from e

        if error_data.get("error") == "model_not_loaded":
            model_type = error_data.get("model_type")
            description = error_data.get("description")
            print(f"\nThe '{model_type}' model is not loaded.")
            print(f"  Purpose: {description}")
            print()

            if auto_load_model:
                choice = (
                    input(
                        f"Would you like to load the '{model_type}' model now? [Y/n]: "
                    )
                    .strip()
                    .lower()
                )
                if choice != "n":
                    if load_model_on_server(config, model_type):
                        progress = _ProgressPoller(batch_total=len(texts))
                        progress.start()
                        try:
                            resp = server_request(
                                "POST", "/generate", json=payload, timeout=gen_timeout
                            )
                        finally:
                            progress.stop()
                    else:
                        raise TTSGenericError(f"Failed to load {model_type} model")
                else:
                    raise TTSGenericError(
                        f"Model '{model_type}' not loaded. Enable in config.json or load with server."
                    )

    if resp.status_code != 200:
        try:
            error_data = resp.json()
            error_msg = error_data.get("error", "Unknown error")
            detail = error_data.get("detail", "")
            recovery = error_data.get("recovery", "")
        except (ValueError, requests.exceptions.JSONDecodeError):
            error_msg = f"Server returned HTTP {resp.status_code} (non-JSON response)"
            detail = ""
            recovery = ""

        msg = f"Server error: {error_msg}"
        if detail:
            msg += f" [{detail}]"
        if recovery == "restart":
            msg += "\n  Suggestion: Try restarting the server with 'tts server start'."
        elif recovery == "config":
            msg += f"\n  Suggestion: Check your configuration in {CONFIG_PATH}."
        elif recovery == "retry":
            msg += "\n  Suggestion: Try again; the issue may be transient."
        raise TTSGenericError(msg)

    response_json = resp.json()
    results = response_json.get("results")
    if results is None:
        raise TTSGenericError(
            "Server response missing expected 'results' key; "
            f"top-level keys present: {sorted(response_json.keys())}"
        )

    # A 200 with a SHORT results list is the cancellation shape: the server
    # stops the batch mid-loop and reports it with a top-level `cancelled`
    # flag. Returning the short list let callers under-deliver silently —
    # cli/srt.py and cli/dialogue.py index results[0] (bare IndexError), and
    # interface/generate.py iterates it, writing fewer .wav files than the
    # user asked for with exit code 0 and no warning. Fail loudly instead.
    # Mirror handle_generate's own normalization: a bare string is one text.
    expected = 1 if isinstance(texts, str) else len(texts)
    if len(results) != expected:
        reason = (
            "generation was cancelled"
            if response_json.get("cancelled")
            else "the server returned an incomplete batch"
        )
        if not results:
            raise TTSGenericError(
                f"Server returned no audio for {expected} requested text(s): "
                f"{reason}."
            )
        raise TTSGenericError(
            f"Server returned {len(results)} result(s) for {expected} "
            f"requested text(s): {reason}."
        )
    return results


def generate_streaming(
    text,
    mode,
    config,
    gen_params,
    output_path,
    prompt_file=None,
    voice_description=None,
    speaker=None,
    instruct=None,
    x_vector_only_mode=False,
):
    """Generate and stream audio playback in real-time (MLX backend).

    Streams from server and plays audio chunks as they arrive.
    Also saves the complete audio to output_path.
    """
    import tempfile

    import numpy as np
    import requests  # lazy
    import soundfile as sf

    payload = _build_generation_payload(
        mode,
        config,
        gen_params,
        prompt_file=prompt_file,
        voice_description=voice_description,
        speaker=speaker,
        instruct=instruct,
        x_vector_only_mode=x_vector_only_mode,
    )
    payload["text"] = text

    print("Streaming generation...")

    try:
        from qwen3_tts.core.http_client import server_request

        resp = server_request(
            "POST",
            "/generate-stream",
            json=payload,
            timeout=600,  # stream=True: inter-chunk gap timeout, not total time — no scaling needed
            stream=True,
        )

        if resp.status_code != 200:
            error_data = resp.json()
            raise TTSGenericError(
                f"Server error: {error_data.get('error', 'Unknown')}"
            )

        # Collect all chunks for saving
        all_chunks = []
        sample_rate = None
        chunk_count = 0

        # Wire-format parsing lives in core/stream_protocol.py — ONE parser
        # shared with TTSClient.generate_streaming(). This used to be a private
        # copy, and the two drifted (different size caps; only this one checked
        # the terminal error sentinel).
        try:
            for chunk, sr in iter_stream_chunks(resp.iter_content(chunk_size=4096)):
                if sample_rate is None:
                    sample_rate = sr

                all_chunks.append(chunk)
                chunk_count += 1

                # Play chunk using platform-aware player
                temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(temp.name, chunk, sr)
                try:
                    play_audio(temp.name)
                except Exception:
                    logger.debug("Streaming audio playback failed: skipping")
                finally:
                    os.unlink(temp.name)
        except StreamProtocolError as e:
            # Terminal error frame: the server cannot change the status code
            # once streaming has begun, so a mid-stream failure arrives in
            # band. Never treat the partial audio already received as a
            # complete generation.
            raise TTSGenericError(f"Server error during streaming: {e}") from e

        print(f"Streaming complete: {chunk_count} chunks received")

        # Save combined audio
        if all_chunks and sample_rate:
            combined = np.concatenate(all_chunks)
            sf.write(output_path, combined, sample_rate)
            print(f"Saved: {output_path}")
            return output_path

        return None

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Streaming request failed: {e}") from e


# ---------------------------------------------------------------------------
# Local generation (lazy import of qwen3_tts.core.engine)
# ---------------------------------------------------------------------------


def generate_local(
    text,
    mode,
    gen_params,
    language="auto",
    prompt_file=None,
    voice_description=None,
    speaker=None,
    instruct=None,
    max_chunk_chars=None,
):
    """Generate speech locally using qwen3_tts.core.engine (imports torch on first call)."""
    from qwen3_tts.core.engine import load_model, load_voice_prompt, run_inference

    print(f"Loading {mode} model locally...")
    model = load_model(mode)

    voice_prompt = None
    if mode == "clone":
        if not prompt_file:
            print("Error: Voice prompt required for clone mode")
            sys.exit(1)
        if not voice_prompt_exists(prompt_file):
            backend = get_backend()
            if backend == "mlx":
                base = prompt_file[:-3] if prompt_file.endswith(".pt") else prompt_file
                print(f"Error: MLX voice prompt not found for '{base}'.")
                print(f"  Need: voice_prompts/{base}.wav + voice_prompts/{base}.txt")
                print(
                    f"  Create with: tts voice create <audio> -t <transcript> -n {base} --mlx-only"
                )
            else:
                print(
                    f"Error: Voice prompt not found: {safe_path_join(VOICE_PROMPTS_DIR, prompt_file)}"
                )
            sys.exit(1)
        print(f"Loading voice prompt: {prompt_file}")
        voice_prompt = load_voice_prompt(prompt_file)
    elif mode == "custom":
        if not speaker:
            speaker = "Ryan"
        print(f"Using speaker: {speaker}")
        if instruct:
            print(f"With instruction: {instruct}")
    elif mode == "design":
        print(f"Using voice description: {voice_description}")

    print("Generating audio...")
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
    )
    return wav, sr


# ---------------------------------------------------------------------------
# Single generation execution
# ---------------------------------------------------------------------------


def _voice_param_for_log(mode, prompt_file, voice_description, speaker_name, instruct):
    """Build the voice_param string for log_generation()."""
    if mode == "clone":
        return prompt_file
    elif mode == "design":
        return voice_description
    return f"{speaker_name}" + (f" ({instruct})" if instruct else "")


def _run_single_generation(
    text,
    args,
    config,
    gen_params,
    use_server,
    max_chunk_chars,
    output_path,
    mode,
    language,
    prompt_file,
    voice_description,
    speaker_name,
    *,
    instruct="",
):
    """Execute a single text generation (streaming, server, or local) and save output."""
    import soundfile as sf  # lazy

    gen_start = time.time()

    if getattr(args, "stream", False) and use_server:
        print("Using TTS server (streaming mode)...")
        if mode == "clone":
            generate_streaming(
                text,
                mode,
                config,
                gen_params,
                output_path,
                prompt_file=prompt_file,
                x_vector_only_mode=getattr(args, "no_transcript", False),
            )
        elif mode == "design":
            generate_streaming(
                text,
                mode,
                config,
                gen_params,
                output_path,
                voice_description=voice_description,
            )
        else:
            generate_streaming(
                text,
                mode,
                config,
                gen_params,
                output_path,
                speaker=speaker_name,
                instruct=instruct,
            )
        gen_duration = time.time() - gen_start
        print(f"Streaming complete ({gen_duration:.1f}s)")
        voice_param = _voice_param_for_log(
            mode, prompt_file, voice_description, speaker_name, instruct
        )
        log_generation(
            text, mode, voice_param, output_path, gen_params, duration_sec=gen_duration
        )
        return use_server

    if use_server:
        print("Using TTS server...")
        if mode == "clone":
            results = generate_via_server(
                [text],
                mode,
                config,
                gen_params,
                prompt_file=prompt_file,
                max_chunk_chars=max_chunk_chars,
                x_vector_only_mode=getattr(args, "no_transcript", False),
            )
        elif mode == "design":
            results = generate_via_server(
                [text],
                mode,
                config,
                gen_params,
                voice_description=voice_description,
                max_chunk_chars=max_chunk_chars,
            )
        else:
            results = generate_via_server(
                [text],
                mode,
                config,
                gen_params,
                speaker=speaker_name,
                instruct=instruct,
                max_chunk_chars=max_chunk_chars,
            )
        needs_processing = (
            args.trim_silence or args.normalize or args.speed or args.pitch
        )
        if needs_processing:
            wav, sr = _decode_base64_result(results[0])
            wav = process_audio_args(wav, sr, args)
            sf.write(output_path, wav, sr)
        else:
            _save_base64_result(results[0], output_path)
    else:
        if mode == "custom":
            wav, sr = generate_local(
                text,
                mode,
                gen_params,
                language,
                speaker=speaker_name,
                instruct=instruct,
                max_chunk_chars=max_chunk_chars,
            )
        elif mode == "design":
            wav, sr = generate_local(
                text,
                mode,
                gen_params,
                language,
                voice_description=voice_description,
                max_chunk_chars=max_chunk_chars,
            )
        else:
            wav, sr = generate_local(
                text,
                mode,
                gen_params,
                language,
                prompt_file=prompt_file,
                max_chunk_chars=max_chunk_chars,
            )
        wav = process_audio_args(wav, sr, args)
        sf.write(output_path, wav, sr)

    gen_duration = time.time() - gen_start
    print(f"Saved to: {output_path} ({gen_duration:.1f}s)")
    voice_param = _voice_param_for_log(
        mode, prompt_file, voice_description, speaker_name, instruct
    )
    log_generation(
        text, mode, voice_param, output_path, gen_params, duration_sec=gen_duration
    )

    if args.play:
        play_audio(output_path)
    elif not args.no_open:
        from qwen3_tts.interface.generate_helpers import open_file

        open_file(output_path)

    return use_server
