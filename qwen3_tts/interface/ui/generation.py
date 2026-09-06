#!/usr/bin/env python3
"""Generation tab logic for the Gradio UI.

This module contains:
- Streaming configuration preparation
- Generation validation
- Audio saving utilities
- Generation tab wiring
"""

import logging
import os
import shutil
import tempfile
import time
import uuid

import gradio as gr

from qwen3_tts.core.config import (
    ModelNotLoadedError,
    get_generation_defaults,
    get_generation_presets,
    get_prosody_presets,
    is_server_running,
    load_config,
)
from qwen3_tts.interface.ui import shared
from qwen3_tts.interface.ui.shared import (
    add_to_history,
    format_status_display,
    history_lock,
    save_generation_metadata,
)
from qwen3_tts.interface.voice_helpers import (  # noqa: E402, F401 (intentional lazy import, re-exported via ui/__init__.py)
    apply_prosody_preset,
    get_prosody_choices,
)

logger = logging.getLogger("tts.ui")

# Unified stop/cancel vocabulary (single source of truth — A11Y-1).
STATUS_STOP_LABEL = "Stop"
STATUS_STOP_CONFIRM_ARM = "Stop — click again to confirm"
STATUS_STOP_CONFIRM_HINT = "Click again within 5s to confirm stop."
STATUS_GENERATION_STOPPING = "Stopping generation…"
STATUS_GENERATION_STOPPED = "Generation stopped"
STATUS_STOP_CANCELED = "Stop canceled"


def _announce_status(msg: str | None) -> str:
    """Visually-hidden screen-reader live region for an action status message.

    gr.Textbox (the visible Status field) has no aria-live in Gradio 6, so action
    outcomes would go unannounced. Returns an sr-only role=status aria-live=polite
    block. Inline styles (no external CSS) + kept in DOM (visible=False is removed
    from the DOM in Gradio 6, which would defeat the live region).
    """
    import html as html_mod

    text = html_mod.escape(str(msg)) if msg else ""
    return (
        '<div role="status" aria-live="polite" '
        'style="position:absolute;width:1px;height:1px;padding:0;'
        'margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;'
        'border:0">'
        f"{text}</div>"
    )


def generate_guard_check(gen_guard_state: dict | None) -> tuple:
    """Pre-generate guard: detects in-flight generation and applies two-step confirm.

    Returns (new_state, status_text, is_blocked).
    - is_blocked=False: proceed — generation is allowed to start.
    - is_blocked=True:  do NOT proceed — generation was blocked (state was armed).

    Wire before the config handler step in _wire_generation_tab.
    """
    from qwen3_tts.interface.ui.components import confirm_step

    if not isinstance(gen_guard_state, dict):
        gen_guard_state = {}

    if not gen_guard_state.get("generating", False):
        new_state = {**gen_guard_state, "generating": True, "armed": False, "ts": 0.0}
        return new_state, "", False

    new_state, _, confirmed = confirm_step(
        gen_guard_state,
        arm_label=STATUS_STOP_CONFIRM_ARM,
        original_label="Generate",
    )
    if not confirmed:
        return (
            {**new_state, "generating": True},
            "Generation in progress. Click again to stop and restart.",
            True,
        )

    cancel_streaming_generation()
    return {**new_state, "generating": False}, STATUS_GENERATION_STOPPING, False


def cancel_streaming_generation():
    """Cancel any ongoing streaming generation."""
    config = load_config()
    if not is_server_running(config):
        return "Server not running", format_status_display()

    try:
        from qwen3_tts.core.http_client import server_request

        resp = server_request(
            "POST",
            "/cancel-generation",
            timeout=5,
        )
        if resp.status_code == 200:
            return STATUS_GENERATION_STOPPED, format_status_display()
        else:
            return (
                f"Stop failed: {resp.json().get('error', 'Unknown')}",
                format_status_display(),
            )
    except Exception as e:
        return f"Error: {e}", format_status_display()


def _prepare_cancel_confirmation():
    """Fetch generation progress for cancel confirmation.

    Returns:
        Tuple of (status_message, should_confirm, progress_pct, chunks, eta)
    """
    config = load_config()
    if not is_server_running(config):
        return "Server not running", False, 0, 0, "N/A"

    try:
        from qwen3_tts.core.http_client import server_request

        resp = server_request("GET", "/generation-status", timeout=2)
        if resp.status_code != 200:
            # Generation not active - can proceed without confirmation
            return "Ready to stop", True, 0, 0, "N/A"

        status = resp.json()
        if not status.get("active", False):
            return "No active generation", True, 0, 0, "N/A"

        # Calculate progress percentage
        chunk_index = status.get("chunk_index", 0)
        chunk_total = status.get("chunk_total", 1)
        progress_pct = int(chunk_index / chunk_total * 100) if chunk_total > 0 else 0

        chunks = chunk_index
        eta = status.get("eta_sec", "N/A")

        # Fast path: <10% progress, no confirmation needed
        if progress_pct < 10:
            return "Stopping...", True, progress_pct, chunks, eta

        # Full confirmation required
        return (
            f"Stop generation?\nProgress: {progress_pct}%\nChunks: {chunks}\nETA: {eta}s",
            False,  # requires confirmation
            progress_pct,
            chunks,
            eta,
        )

    except Exception as e:
        logger.error("Failed to fetch generation status: %s", e)
        return f"Error: {e}", True, 0, 0, "N/A"


def _prepare_streaming_config(
    mode,
    text,
    preset,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    seed,
    prompt_file=None,
    description=None,
    speaker=None,
    instruct=None,
    prosody_preset=None,
    no_transcript=False,
    seed_lock_chunks=False,
):
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

    # Apply preset if specified (defaults merged with user-defined)
    if preset and preset != "(none)":
        presets = get_generation_presets(config)
        if preset in presets:
            gen_params.update(presets[preset])

    # Build request payload for the server
    payload = {
        "mode": mode,
        "text": text,
        "language": config.get("language", "auto"),
        "seed_lock_chunks": seed_lock_chunks,
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

    # Generate server-side via Python TTSClient — auth token never reaches the browser
    return {"server_side": True, "payload": payload}, "Generating..."


def _generate_server_side(mode, text, history_list, stream_config):
    """Generate audio server-side via Python TTSClient (no token in browser).

    Auth token stays in the Python process — never sent to the browser JS layer.

    Returns:
        tuple: (audio_path_or_none, status_text, status_html, history_list)
    """
    import gradio as gr

    # Ensure history_list is always a valid list (defensive against Gradio state issues)
    if history_list is None:
        history_list = []
    elif not isinstance(history_list, list):
        logger.warning(
            f"history_list is not a list: {type(history_list)}, resetting to []"
        )
        history_list = []
    else:
        # Make a copy to avoid shared state issues in concurrent requests
        history_list = list(history_list)

    # stream_config is None when validation failed — preserve the error
    if stream_config is None:
        return None, gr.update(), format_status_display(), history_list

    if not isinstance(stream_config, dict) or not stream_config.get("server_side"):
        return None, STATUS_GENERATION_STOPPED, format_status_display(), history_list

    try:
        from qwen3_tts.server.client import TTSClient

        payload = stream_config.get("payload", {})

        # The engine logs this to `tts.engine`, which for the web UI means
        # .voice_server.log — a browser user never sees it and just watches a
        # generation run on and on. Surface it where they are, before the wait
        # starts. Advisory only: the generation still proceeds.
        if payload.get("mode", "clone") == "clone" and payload.get("prompt_file"):
            rate_warning = shared.low_rate_prompt_warning(payload["prompt_file"])
            if rate_warning:
                gr.Warning(rate_warning)

        filename = f"voice_ui_{uuid.uuid4().hex[:8]}.wav"
        # Save to temp dir (Gradio always allows tempdir paths)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        client = TTSClient()
        # Generation sampling defaults come from config.json (single source of
        # truth) so the fallbacks here can't drift from the slider defaults.
        _defaults = get_generation_defaults()
        client.generate(
            text=payload.get("text", ""),
            output=temp_path,
            mode=payload.get("mode", "clone"),
            prompt=payload.get("prompt_file"),
            description=payload.get("voice_description"),
            speaker=payload.get("speaker"),
            instruct=payload.get("instruct"),
            temperature=payload.get("temperature", _defaults["temperature"]),
            top_k=payload.get("top_k", _defaults["top_k"]),
            top_p=payload.get("top_p", _defaults["top_p"]),
            repetition_penalty=payload.get(
                "repetition_penalty", _defaults["repetition_penalty"]
            ),
            seed=payload.get("seed"),
            preset=payload.get("preset"),
            x_vector_only_mode=payload.get("x_vector_only_mode", False),
            seed_lock_chunks=payload.get("seed_lock_chunks", False),
        )
        chunks = getattr(client, "last_chunk_count", 0)
        # Prefer the actual seed the server used (set even for random seeds);
        # fall back to whatever the user supplied in the payload.
        used_seed = getattr(client, "last_seed", None)
        if used_seed is None:
            used_seed = payload.get("seed")

        # Copy to the web-UI Automated Output folder for persistent access.
        config = load_config()
        from qwen3_tts.interface.ui.shared import ensure_history_dirs

        # Web-UI generations go to Automated Output so Remove can safely delete
        # them; resolve_* already enforces home containment and rejects traversal.
        output_dir, _ = ensure_history_dirs(config)
        persistent_path = os.path.join(output_dir, filename)
        shutil.copy2(temp_path, persistent_path)

        # Save metadata JSON sidecar for persistent history + seed reuse
        generation_metadata = {
            "timestamp": time.time(),
            "mode": payload.get("mode", mode),
            "text": payload.get("text", ""),
            "seed": used_seed,
            "chunks": chunks,
            "temperature": payload.get("temperature"),
            "top_k": payload.get("top_k"),
            "top_p": payload.get("top_p"),
            "repetition_penalty": payload.get("repetition_penalty"),
            "prompt_file": payload.get("prompt_file"),
            "voice_description": payload.get("voice_description"),
            "speaker": payload.get("speaker"),
            "output_file": os.path.basename(persistent_path),
        }
        save_generation_metadata(persistent_path, generation_metadata)

        # History tracks persistent path; Gradio gets temp path
        # Use lock to prevent concurrent modification issues
        with history_lock:
            history_list = add_to_history(
                history_list,
                mode,
                text,
                persistent_path,
                chunks,
                seed=used_seed,
            )
            # Make a copy for return to avoid external mutation
            history_list_copy = list(history_list)
        return (
            temp_path,
            # First line stays the bare filename: e2e tests parse the wav
            # basename from line 1 of this status ("Generated: <name>.wav").
            f"Generated: {os.path.basename(persistent_path)}\n"
            f"Saved to: {output_dir}",
            format_status_display(),
            history_list_copy,
        )
    except ModelNotLoadedError as e:
        # Surface a clear, actionable message instead of the bare "model is not
        # loaded" string the server returns — the latter previously read as a
        # silent failure because it gave the user no recovery path. Names the
        # Manage Models tab and the load_at_startup config key directly.
        model_type = e.model_type or mode
        logger.warning("Generation blocked: '%s' model not loaded", model_type)
        safe_history_list = (
            list(history_list) if isinstance(history_list, list) else []
        )
        status_msg = (
            f"The '{model_type}' model is not loaded. "
            "Load it in the 'Manage Models' tab, or enable "
            f"'models.{model_type}.load_at_startup' in config to auto-load it."
        )
        return None, status_msg, format_status_display(), safe_history_list
    except Exception as e:
        logger.error("Server-side generation failed: %s", e)
        # Ensure we always return a valid list even on error
        safe_history_list = list(history_list) if isinstance(history_list, list) else []
        return None, f"Error: {e}", format_status_display(), safe_history_list


def _build_common_controls():
    """Build the common right-column controls shared by all generation tabs.

    Returns dict with keys: temp, top_k, top_p, rep, seed, seed_lock.
    Per-generation audio-processing controls were removed when generation
    moved server-side; post-processing (echo trim, loudness) is applied by
    the engine.
    """
    # Slider defaults are driven by config.json (single source of truth) so the
    # UI tracks the user's configured generation defaults, not hardcoded values.
    _defaults = get_generation_defaults()
    with gr.Accordion("Advanced Settings", open=False):
        temp = gr.Slider(0.1, 1.5, value=_defaults["temperature"], step=0.05, label="Temperature")
        top_k = gr.Slider(1, 100, value=_defaults["top_k"], step=1, label="Top-K")
        top_p = gr.Slider(0.1, 1.0, value=_defaults["top_p"], step=0.01, label="Top-P")
        rep = gr.Slider(1.0, 2.0, value=_defaults["repetition_penalty"], step=0.01, label="Repetition Penalty")
        seed = gr.Textbox(
            label="Seed (empty = random)",
            value="",
            info="The seed actually used is saved to Recent Generations — click a row to reuse it",
        )
        seed_lock = gr.Checkbox(
            label="Lock voice across chunks",
            value=False,
            info="Re-seed before each chunk for consistent voice timbre in long texts",
        )

    return {
        "temp": temp,
        "top_k": top_k,
        "top_p": top_p,
        "rep": rep,
        "seed": seed,
        "seed_lock": seed_lock,
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
    # Visually-hidden aria-live region mirroring `status` (Textbox has no
    # aria-live in Gradio 6). In-DOM + sr-only inline styles — NOT visible=False.
    status_announcer = gr.HTML(value=_announce_status(""))
    # Hidden components for JS<->Python data flow
    stream_config = gr.JSON(elem_classes=["gr-hidden"])
    result_data = gr.Textbox(
        elem_classes=["gr-hidden"], elem_id=f"{tab_id}-result-data"
    )
    mode_hidden = gr.Textbox(value=tab_id, elem_classes=["gr-hidden"])
    text_hidden = gr.Textbox(elem_classes=["gr-hidden"])
    return {
        "btn": btn,
        "cancel_btn": cancel_btn,
        "audio_url_converter": audio_url_converter,
        "stream_config": stream_config,
        "result_data": result_data,
        "mode_hidden": mode_hidden,
        "text_hidden": text_hidden,
        "status": status,
        "status_announcer": status_announcer,
    }


def _wire_generation_tab(
    mode,
    btn,
    cancel_btn,
    status,
    stream_config,
    result_data,
    mode_hidden,
    text_hidden,
    model_indicator,
    text,
    text_info,
    inputs_list,
    status_html,
    config_handler,
    api_name=None,
    history_state=None,
    audio_url_converter=None,
    gen_guard_state=None,
    status_announcer=None,
):
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
        config_handler: Function returning (config_dict, status_text).
        api_name: Optional API name for the endpoint.
        history_state: Optional history state component.
        audio_url_converter: Hidden gr.Audio for server-side file URL conversion.
        status_announcer: Optional sr-only gr.HTML mirroring `status` for SR users.

    Returns:
        The final event chain object so callers can append further .then() steps
        (e.g. to update a history dataframe rendered below the tabs).
    """
    from qwen3_tts.interface.ui.model_management import get_model_status_html
    from qwen3_tts.interface.ui.shared import update_text_info
    from qwen3_tts.interface.wavesurfer_js import (
        get_cancel_js,
        get_load_into_player_js,
    )

    # Step 1: Python validates inputs, returns generation config JSON.
    # If gen_guard_state is wired, a guard check precedes config_handler and
    # sets stream_config=None when generation is already in flight (blocks
    # _generate_server_side from firing).
    if gen_guard_state is not None:

        def _guarded_config(guard_state, *config_inputs):
            new_guard_state, guard_status, blocked = generate_guard_check(guard_state)
            if blocked:
                return new_guard_state, None, guard_status
            cfg, cfg_status = config_handler(*config_inputs)
            return new_guard_state, cfg, cfg_status

        click_kwargs = {
            "fn": _guarded_config,
            "inputs": [gen_guard_state, *inputs_list],
            "outputs": [gen_guard_state, stream_config, status],
        }
    else:
        click_kwargs = {
            "fn": config_handler,
            "inputs": inputs_list,
            "outputs": [stream_config, status],
        }
    if api_name:
        click_kwargs["api_name"] = api_name

    # Generation flow: Python validates → Python generates server-side → JS loads audio
    # Auth token stays in the Python process and never reaches the browser.
    chain = (
        btn.click(**click_kwargs)
        .then(
            # Capture the text input for the generation step
            fn=lambda t: t,
            inputs=[text],
            outputs=[text_hidden],
        )
        .then(
            # Step 2: Generate audio server-side via TTSClient (no JS streaming)
            fn=_generate_server_side,
            inputs=[mode_hidden, text_hidden, history_state, stream_config],
            outputs=[audio_url_converter, status, status_html, history_state],
        )
        .then(
            # Step 3: Load saved file into tab's WaveSurfer player via hidden gr.Audio URL
            # NOTE: fn=passthrough required for Gradio 6 .then() chain continuity
            fn=lambda x: x,
            js=get_load_into_player_js(mode),
            inputs=[audio_url_converter],
            outputs=[audio_url_converter],
        )
        .then(
            fn=lambda: get_model_status_html(mode),
            outputs=model_indicator,
        )
    )

    # Reset generating flag after the chain completes (including errors).
    if gen_guard_state is not None:
        chain = chain.then(
            fn=lambda s: (
                {**s, "generating": False}
                if isinstance(s, dict)
                else {"generating": False}
            ),
            inputs=[gen_guard_state],
            outputs=[gen_guard_state],
        )

    # Mirror the visible status into the sr-only aria-live announcer so screen
    # readers announce generation outcomes (Textbox has no aria-live in Gradio 6).
    if status_announcer is not None:
        chain = chain.then(
            fn=_announce_status,
            inputs=[status],
            outputs=[status_announcer],
        )

    # Cancel button — with confirmation for >10% progress
    from qwen3_tts.interface.ui.components import ConfirmButton

    cancel_confirm_btn = ConfirmButton(
        arm_label=STATUS_STOP_CONFIRM_ARM,
        original_label=STATUS_STOP_LABEL,
        timeout_s=5.0,
        status_message=STATUS_STOP_CONFIRM_HINT,
    )

    cancel_confirm_state = gr.State({"armed": False, "ts": 0.0})

    def on_cancel_click(state):
        # First click: show progress and request confirmation if needed
        if not state.get("armed", False):
            banner_msg, should_proceed, progress_pct, chunks, eta = (
                _prepare_cancel_confirmation()
            )
            if should_proceed:
                # Fast path: <10% or no active generation, proceed immediately
                status_msg, status_html_msg = cancel_streaming_generation()
                return state, gr.update(), status_msg, status_html_msg

            # ConfirmButton.click returns a 4-tuple; only the state dict goes
            # into cancel_confirm_state. Binding the whole tuple to new_state
            # stored a tuple in the gr.State, and the next click called .get()
            # on it. Every branch must also return one value per wired output
            # (cancel_confirm_state, cancel_btn, status, status_html) or Gradio
            # raises ValueError from validate_outputs.
            new_state, _btn_update, _status_update, _confirmed = (
                cancel_confirm_btn.click(state)
            )
            return (
                new_state,
                gr.update(value=STATUS_STOP_CONFIRM_ARM),
                banner_msg,
                gr.update(),
            )

        # Second click: user confirmed
        new_state, btn_update, status_update, confirmed = cancel_confirm_btn.click(
            state
        )
        if not confirmed:
            return (
                new_state,
                gr.update(value=STATUS_STOP_LABEL),
                STATUS_STOP_CANCELED,
                gr.update(),
            )

        status_msg, status_html_msg = cancel_streaming_generation()
        return new_state, gr.update(value=STATUS_STOP_LABEL), status_msg, status_html_msg

    cancel_chain = cancel_btn.click(
        fn=on_cancel_click,
        inputs=[cancel_confirm_state],
        outputs=[cancel_confirm_state, cancel_btn, status, status_html],
    ).then(
        # NOTE: fn=passthrough required for Gradio 6 .then() chain continuity
        fn=lambda x: x,
        js=get_cancel_js(mode),
        inputs=[stream_config],
        outputs=[status],
    )

    # Mirror the cancel outcome into the sr-only aria-live announcer.
    if status_announcer is not None:
        cancel_chain = cancel_chain.then(
            fn=_announce_status,
            inputs=[status],
            outputs=[status_announcer],
        )

    text.change(fn=update_text_info, inputs=text, outputs=text_info)

    return chain
