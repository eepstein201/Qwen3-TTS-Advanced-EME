#!/usr/bin/env python3
"""Generation tab logic for the Gradio UI.

This module contains:
- Streaming configuration preparation
- Generation validation
- Audio saving utilities
- Generation tab wiring
"""

import base64
import logging
import os

import gradio as gr

from qwen3_tts.core.config import (
    get_server_url,
    is_server_running,
    auth_headers,
    load_config,
    get_prosody_presets,
)
from qwen3_tts.interface.ui.shared import (
    add_to_history,
    format_status_display,
)
from qwen3_tts.interface.voice_helpers import (  # noqa: F401 (re-exported via ui/__init__.py)
    get_prosody_choices,
    apply_prosody_preset,
)

logger = logging.getLogger("tts.ui")


def cancel_streaming_generation():
    """Cancel any ongoing streaming generation."""
    config = load_config()
    if not is_server_running(config):
        return "Server not running", format_status_display()

    try:
        import requests
        url = get_server_url(config)
        resp = requests.post(
            f"{url}/cancel-generation",
            timeout=5,
            headers=auth_headers(),
        )
        if resp.status_code == 200:
            return "Generation cancelled", format_status_display()
        else:
            return f"Cancel failed: {resp.json().get('error', 'Unknown')}", format_status_display()
    except Exception as e:
        return f"Error: {e}", format_status_display()


def _prepare_streaming_config(mode, text, preset, temperature, top_k, top_p,
                              repetition_penalty, seed, prompt_file=None,
                              description=None, speaker=None, instruct=None,
                              prosody_preset=None, no_transcript=False):
    """Prepare streaming configuration for the given mode.

    Returns (config_dict_or_None, status_text) tuple.
    Returns None as config on validation/server errors.
    """
    config = load_config()

    if not text or not text.strip():
        return None, "Error: Please enter text to generate"

    if mode == "design" and not description:
        return None, "Error: Please enter a voice description for design mode"

    # Check server is running before further validation
    if not is_server_running(config):
        return None, "Error: TTS server is not running."

    # Validate mode-specific requirements
    if mode == "clone" and not prompt_file:
        return None, "Error: Please select a voice prompt for clone mode"
    if mode == "custom" and not speaker:
        return None, "Error: Please select a speaker for custom mode"

    # Apply prosody preset to instruct if specified
    if prosody_preset and mode in ("custom", "design"):
        prosody_presets = get_prosody_presets(config)
        if prosody_preset in prosody_presets:
            instruct = prosody_presets[prosody_preset]

    # Build generation params
    gen_params = {
        "temperature": temperature,
        "top_k": int(top_k),
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    if seed and seed.strip():
        try:
            gen_params["seed"] = int(seed)
        except ValueError:
            pass

    # Apply preset if specified
    if preset and preset != "(none)":
        presets = config.get("presets", {})
        if preset in presets:
            gen_params.update(presets[preset])

    # Build request payload for the server
    payload = {
        "mode": mode,
        "text": text,
        "language": config.get("language", "English"),
        **gen_params,
    }

    if mode == "clone":
        payload["prompt_file"] = prompt_file
        if no_transcript:
            payload["x_vector_only_mode"] = True
    elif mode == "design":
        payload["voice_description"] = description
    elif mode == "custom":
        # Dropdown sends full display string like "ryan (English) - ..."; extract key
        if speaker and " (" in speaker:
            speaker = speaker.split(" (")[0]
        payload["speaker"] = speaker
        payload["instruct"] = instruct or ""

    # In Colab, the browser can't reach 127.0.0.1 — return sentinel so JS
    # skips fetch() and the Python fallback generates audio server-side.
    from qwen3_tts.core.config import IN_COLAB
    if IN_COLAB:
        return {"colab_fallback": True, "payload": payload}, "Generating (Colab mode)..."

    # Build streaming config: nested structure matching JS expectations
    # JS reads config.server_url, config.auth_token, config.payload
    server_url = get_server_url(config)
    with open(os.path.expanduser("~/.voice_server_token")) as f:
        auth_token = f.read().strip()
    return {"server_url": server_url, "auth_token": auth_token, "payload": payload}, "Connecting..."


def _save_completed_audio(base64_wav, mode, text, history_list, stream_config=None):
    """Save completed base64 audio to file and update history.

    Returns (status_text, status_html, history_list, history_df_data).
    """
    import gradio as gr
    from qwen3_tts.interface.ui.shared import get_history_data

    if history_list is None:
        history_list = []

    if not base64_wav or base64_wav == '':
        return "Cancelled", format_status_display(), history_list, gr.update()

    if base64_wav == 'TIMEOUT':
        return "Error: Timed out waiting for audio", format_status_display(), history_list, gr.update()

    if base64_wav.startswith('ERROR:'):
        error_msg = base64_wav[6:]
        return f"Error: {error_msg}", format_status_display(), history_list, gr.update()

    try:
        # Decode and save audio
        audio_bytes = base64.b64decode(base64_wav)

        # Generate output path
        import uuid
        output_path = os.path.expanduser(f"~/Downloads/voice_ui_{uuid.uuid4().hex[:8]}.wav")

        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        # Add to history
        history_list = add_to_history(history_list, mode, text, output_path, 0)

        return (f"Generated: {os.path.basename(output_path)}",
                format_status_display(), history_list, get_history_data(history_list))

    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        return f"Error saving audio: {e}", format_status_display(), history_list, gr.update()


def _generate_colab_fallback(base64_wav, mode, text, history_list, stream_config):
    """Handle generation for Colab environment where JS streaming doesn't work.

    If JS streaming succeeded (base64_wav is non-empty), delegates to _save_completed_audio.
    In Colab mode (stream_config has colab_fallback=True and base64_wav is empty),
    generates audio server-side via TTSClient and returns the file path.

    Returns:
        tuple: (audio_path_or_none, status_text, status_html, history_list, history_df_data)
    """
    import uuid
    from qwen3_tts.interface.ui.shared import get_history_data, add_to_history
    import gradio as gr

    if history_list is None:
        history_list = []

    # JS streaming returned an error
    if base64_wav and base64_wav.startswith('ERROR:'):
        error_msg = base64_wav[6:]
        return None, f"Error: {error_msg}", format_status_display(), history_list, gr.update()

    # JS timed out waiting for audio
    if base64_wav == 'TIMEOUT':
        return None, "Error: Timed out waiting for audio — try again", format_status_display(), history_list, gr.update()

    # JS streaming succeeded — save as usual
    if base64_wav and not base64_wav.startswith('ERROR:'):
        status, html, hist, df = _save_completed_audio(base64_wav, mode, text, history_list)
        # Return the saved file path so the JS .then() step can load it into WaveSurfer
        saved_path = hist[-1].get("path") if hist else None
        return saved_path, status, html, hist, df

    # Check if this is a Colab fallback request
    is_colab = (isinstance(stream_config, dict) and stream_config.get("colab_fallback"))
    if not is_colab:
        # stream_config is None when validation failed — preserve the error
        # status already set by Step 1 instead of overwriting with "Cancelled"
        if stream_config is None:
            return None, gr.update(), format_status_display(), history_list, gr.update()
        # Non-Colab, empty result with valid config means user cancelled
        return None, "Cancelled", format_status_display(), history_list, gr.update()

    # Colab fallback: generate via Python TTSClient
    try:
        from qwen3_tts.server.client import TTSClient
        payload = stream_config.get("payload", {})
        unique_id = uuid.uuid4().hex[:8]
        output_path = os.path.expanduser(f"~/Downloads/voice_ui_{unique_id}.wav")
        client = TTSClient()
        client.generate(
            text=payload.get("text", ""),
            output=output_path,
            mode=payload.get("mode", "clone"),
            prompt=payload.get("prompt_file"),
            description=payload.get("voice_description"),
            speaker=payload.get("speaker"),
            instruct=payload.get("instruct"),
            temperature=payload.get("temperature", 0.7),
            top_k=payload.get("top_k", 50),
            top_p=payload.get("top_p", 0.95),
            repetition_penalty=payload.get("repetition_penalty", 1.05),
            seed=payload.get("seed"),
            preset=payload.get("preset"),
            x_vector_only_mode=payload.get("x_vector_only_mode", False),
        )

        history_list = add_to_history(history_list, mode, text, output_path, 0)
        return (
            output_path,
            f"Generated: {os.path.basename(output_path)}",
            format_status_display(),
            history_list,
            get_history_data(history_list),
        )
    except Exception as e:
        logger.error(f"Colab fallback generation failed: {e}")
        return None, f"Error: {e}", format_status_display(), history_list, get_history_data(history_list)


def _validate_inputs(mode, text, description=None):
    """Validate generation inputs for the given mode.

    Returns (is_valid, error_message).
    """
    if not text or not text.strip():
        return False, "Please enter text to generate"

    if mode == "design" and (not description or not description.strip()):
        return False, "Please enter a voice description for design mode"

    return True, None


def _build_common_controls():
    """Build the common right-column controls shared by all generation tabs.

    Returns dict with keys: temp, top_k, top_p, rep, seed.
    Audio processing (trim, normalize, speed, pitch) is not supported
    in streaming mode and has been removed from the WaveSurfer UI.
    """
    with gr.Accordion("Advanced Settings", open=False):
        temp = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
        top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-P")
        rep = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="Repetition Penalty")
        seed = gr.Textbox(label="Seed (empty for random)", value="")

    return {
        "temp": temp, "top_k": top_k, "top_p": top_p,
        "rep": rep, "seed": seed,
    }


def _build_generate_buttons_and_output(tab_id):
    """Build the Generate/Stop buttons and WaveSurfer output components.

    Args:
        tab_id: Unique identifier for this tab (e.g., 'clone', 'design', 'custom').

    Returns dict with keys: btn, cancel_btn, audio_url_converter, stream_config,
                            result_data, mode_hidden, text_hidden, status.
    """
    from qwen3_tts.interface.wavesurfer_js import (
        get_player_html,
    )

    with gr.Row():
        btn = gr.Button("Generate", variant="primary")
        cancel_btn = gr.Button("Stop", variant="stop")
    gr.HTML(
        value=get_player_html(tab_id),
        label="Audio Player",
    )
    # Hidden gr.Audio for file URL conversion (Gradio serves local files as HTTP URLs)
    # NOTE: Use elem_classes=["gr-hidden"] instead of visible=False because Gradio 6
    # removes visible=False components from the DOM entirely, breaking JS↔Python
    # data flow in .then() chains.
    audio_url_converter = gr.Audio(elem_classes=["gr-hidden"])
    status = gr.Textbox(label="Status", interactive=False)
    # Hidden components for JS<->Python data flow
    stream_config = gr.JSON(elem_classes=["gr-hidden"])
    result_data = gr.Textbox(elem_classes=["gr-hidden"], elem_id=f"{tab_id}-result-data")
    mode_hidden = gr.Textbox(value=tab_id, elem_classes=["gr-hidden"])
    text_hidden = gr.Textbox(elem_classes=["gr-hidden"])
    return {
        "btn": btn, "cancel_btn": cancel_btn,
        "audio_url_converter": audio_url_converter,
        "stream_config": stream_config,
        "result_data": result_data, "mode_hidden": mode_hidden,
        "text_hidden": text_hidden, "status": status,
    }


def _wire_generation_tab(mode, btn, cancel_btn, status, stream_config, result_data,
                         mode_hidden, text_hidden, model_indicator,
                         text, text_info, inputs_list, status_html, history_df,
                         config_handler, api_name=None, history_state=None,
                         audio_url_converter=None):
    """Wire up the generation flow: Python validates → JS streams → Python saves/fallback.

    Args:
        mode: The generation mode ('clone', 'design', 'custom').
        btn: The generate button component.
        cancel_btn: The cancel button component.
        status: The status textbox component.
        stream_config: Hidden gr.JSON for Python→JS config.
        result_data: Hidden gr.Textbox for JS→Python base64 result.
        mode_hidden: Hidden gr.Textbox with mode name.
        text_hidden: Hidden gr.Textbox that captures text for save step.
        model_indicator: The model status HTML component.
        text: The input textbox component.
        text_info: The text info textbox component.
        inputs_list: List of input components for the config_handler.
        status_html: The status HTML component.
        history_df: The history dataframe component.
        config_handler: Function returning (config_dict, status_text).
        api_name: Optional API name for the endpoint.
        history_state: Optional history state component.
        audio_url_converter: Hidden gr.Audio for Colab fallback file URL conversion.
    """
    from qwen3_tts.interface.ui.model_management import get_model_status_html
    from qwen3_tts.interface.wavesurfer_js import (
        get_streaming_trigger_js,
        get_cancel_js,
        get_load_into_player_js,
    )
    from qwen3_tts.interface.ui.shared import update_text_info

    # Step 1: Python validates inputs, returns streaming config JSON
    click_kwargs = {
        "fn": config_handler,
        "inputs": inputs_list,
        "outputs": [stream_config, status],
    }
    if api_name:
        click_kwargs["api_name"] = api_name

    # Gradio 6 breaks .then() chains after JS-only steps (fn=None, js=...).
    # Workaround: provide a passthrough fn alongside js so the chain continues.
    btn.click(**click_kwargs).then(
        # Also capture the text input for the save step
        fn=lambda t: t,
        inputs=[text],
        outputs=[text_hidden],
    ).then(
        # Step 2: JS reads config and starts streaming via fetch()
        # In Colab, config has no server_url so JS returns '' immediately
        # NOTE: fn=passthrough required for Gradio 6 .then() chain continuity
        fn=lambda x: x,
        js=get_streaming_trigger_js(mode),
        inputs=[stream_config],
        outputs=[result_data],
    ).then(
        # Step 3: Handle result — JS streaming success OR Colab Python fallback
        fn=_generate_colab_fallback,
        inputs=[result_data, mode_hidden, text_hidden, history_state, stream_config],
        outputs=[audio_url_converter, status, status_html, history_state, history_df],
    ).then(
        # Step 4: Load saved file into tab's WaveSurfer player via hidden gr.Audio URL
        # NOTE: fn=passthrough required for Gradio 6 .then() chain continuity
        fn=lambda x: x,
        js=get_load_into_player_js(mode),
        inputs=[audio_url_converter],
        outputs=[audio_url_converter],
    ).then(
        fn=lambda: get_model_status_html(mode),
        outputs=model_indicator,
    )

    # Cancel button — Python cancels server-side, then JS aborts fetch + stops player
    cancel_btn.click(
        fn=cancel_streaming_generation,
        outputs=[status, status_html],
    ).then(
        # NOTE: fn=passthrough required for Gradio 6 .then() chain continuity
        fn=lambda x: x,
        js=get_cancel_js(mode),
        inputs=[stream_config],
        outputs=[status],
    )

    text.change(fn=update_text_info, inputs=text, outputs=text_info)
