#!/usr/bin/env python3
"""Facade module for Qwen3-TTS Gradio UI.

This module contains the main entry points:
- build_ui(): Build the Gradio interface
- main(): CLI entry point
- stop_server(): Stop the TTS server
- _find_available_port(): Find an available port
"""

import logging
import os
import re
import shutil
import sys
import tempfile
import time

import gradio as gr

from qwen3_tts.core.config import (
    VOICE_DESCRIPTION_ATTRIBUTES,
    VALID_MODEL_SIZES,
    VALID_MLX_QUANTIZATIONS,
    IN_COLAB,
    get_default_clone_prompt,
    get_backend,
    load_config,
)
from qwen3_tts.interface.voice_helpers import (
    get_prosody_choices,
    apply_prosody_preset,
    compose_voice_description,
)
from qwen3_tts.interface.wavesurfer_js import (
    get_wavesurfer_loader_js,
    get_streaming_player_js,
    get_player_html,
    get_script_reexecutor_fn,
    get_load_into_player_js,
)
from qwen3_tts.server.client import TTSClient

# Import from sibling modules
from qwen3_tts.interface.ui.shared import (
    SPEAKER_CHOICES,
    enhance_description_with_ai,
    is_enhancer_available,
    get_current_model_settings,
    apply_model_settings,
    format_status_display,
    get_voice_prompts,
    get_presets,
)
from qwen3_tts.interface.ui.generation import (
    _prepare_streaming_config,
    _build_common_controls,
    _build_generate_buttons_and_output,
    _wire_generation_tab,
)
from qwen3_tts.interface.ui.voice_management import (
    create_voice_prompt,
    auto_transcribe_audio,
    get_prompt_table_data,
    preview_voice,
    rename_voice,
    delete_voice,
    set_voice_default,
)
from qwen3_tts.interface.ui.model_management import (
    get_model_table_data,
    toggle_model,
    toggle_asr,
    update_startup_defaults,
    get_model_status_html,
    get_audio_loader_setting,
    set_audio_loader_setting,
)

logger = logging.getLogger("tts.ui")


def stop_server():
    """Stop the TTS server from the UI."""
    client = TTSClient()

    try:
        result = client.shutdown()
        logger.info("Server shutdown initiated: %s", result)
    except Exception as e:
        logger.warning("Failed to send shutdown request: %s", e)

    # Poll for up to 5 seconds to verify shutdown
    for _ in range(10):
        time.sleep(0.5)
        if not client.is_server_running():
            return format_status_display()

    return format_status_display()


def _find_available_port(preferred, max_tries=10):
    """Return *preferred* port if free, otherwise the next available port.

    Scans preferred .. preferred+max_tries-1.  Returns None if all are taken.
    """
    import socket
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


_SAFE_NAME_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')


def _sanitize_voice_name(raw: str) -> tuple:
    """Validate and sanitize a voice name using allowlist.

    Returns (sanitized_name, error_or_None).
    """
    name = raw.strip().replace(" ", "_")
    if not _SAFE_NAME_RE.match(name):
        return "", "Voice name may only contain letters, numbers, underscores, and hyphens (1-64 chars)"
    return name, None


def on_history_select(evt: gr.SelectData, history_list):
    """Handle click on a history row — return file path for WaveSurfer playback.

    Defense-in-depth: validates path against safe roots and copies to tempdir
    so Gradio can always serve it (tempdir is always in allowed_paths).
    """
    if not (isinstance(history_list, list) and history_list):
        return None
    if not (hasattr(evt, 'index') and isinstance(evt.index, (list, tuple))
            and len(evt.index) >= 1 and 0 <= evt.index[0] < len(history_list)):
        return None
    path = history_list[evt.index[0]].get("path", "")
    if not path:
        return None
    resolved = os.path.realpath(path)
    # Containment check: only serve files from known-safe directories
    from qwen3_tts.interface.ui.shared import _resolve_output_dir
    config = load_config()
    output_dir = _resolve_output_dir(config)
    safe_roots = {
        os.path.realpath(tempfile.gettempdir()),
        os.path.realpath(os.path.expanduser("~/Downloads")),
        output_dir,
    }
    if not any(resolved == r or resolved.startswith(r + os.sep) for r in safe_roots):
        return None
    if not os.path.exists(resolved):
        return None
    # Copy to temp for Gradio compatibility (tempdir always in allowed_paths)
    temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(resolved))
    if not os.path.exists(temp_path):
        shutil.copy2(resolved, temp_path)
    return temp_path


def _build_clone_tab(status_html, history_df, history_state):
    """Build Clone Mode tab components and wiring.

    Returns (clone_prompt, clone_model_indicator) for cross-tab references.
    """
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
            clone_ctrls = _build_common_controls()
            clone_no_transcript = gr.Checkbox(
                label="Speaker embedding only", value=False,
                info="Clone using x-vector only (no transcript needed, lower fidelity)"
            )

    clone_btns = _build_generate_buttons_and_output("clone")

    def clone_config_handler(text, prompt, preset, temp, top_k, top_p, rep, seed,
                             no_transcript):
        return _prepare_streaming_config(
            "clone", text, preset, temp, top_k, top_p, rep, seed,
            prompt_file=prompt, no_transcript=no_transcript)

    _wire_generation_tab(
        "clone", clone_btns["btn"], clone_btns["cancel_btn"],
        clone_btns["status"], clone_btns["stream_config"],
        clone_btns["result_data"], clone_btns["mode_hidden"],
        clone_btns["text_hidden"], clone_model_indicator,
        clone_text, clone_text_info,
        inputs_list=[clone_text, clone_prompt, clone_preset,
                     clone_ctrls["temp"], clone_ctrls["top_k"], clone_ctrls["top_p"],
                     clone_ctrls["rep"], clone_ctrls["seed"],
                     clone_no_transcript],
        status_html=status_html, history_df=history_df,
        config_handler=clone_config_handler, api_name="generate_clone",
        history_state=history_state,
        audio_url_converter=clone_btns["audio_url_converter"],
    )
    return clone_prompt, clone_model_indicator


def _build_design_tab(status_html, history_df, history_state, clone_prompt):
    """Build Design Mode tab components and wiring.

    Returns design_model_indicator for cross-tab references.
    """
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

    design_btns = _build_generate_buttons_and_output("design")

    # Save as Voice Prompt (Design-then-Clone pipeline)
    with gr.Accordion("Save as Voice Prompt", open=False):
        gr.Markdown("Save the generated audio as a reusable voice clone prompt.")
        design_save_name = gr.Textbox(label="Voice Name", placeholder="e.g., designed_voice", max_lines=1)
        design_save_btn = gr.Button("Save as Voice Prompt", size="sm", variant="secondary")
        design_save_status = gr.Textbox(label="", show_label=False, interactive=False, max_lines=1, container=False)

    def design_config_handler(text, desc, preset, temp, top_k, top_p, rep, seed):
        return _prepare_streaming_config(
            "design", text, preset, temp, top_k, top_p, rep, seed,
            description=desc)

    _wire_generation_tab(
        "design", design_btns["btn"], design_btns["cancel_btn"],
        design_btns["status"], design_btns["stream_config"],
        design_btns["result_data"], design_btns["mode_hidden"],
        design_btns["text_hidden"], design_model_indicator,
        design_text, design_text_info,
        inputs_list=[design_text, design_desc, design_preset,
                     design_ctrls["temp"], design_ctrls["top_k"], design_ctrls["top_p"],
                     design_ctrls["rep"], design_ctrls["seed"]],
        status_html=status_html, history_df=history_df,
        config_handler=design_config_handler,
        history_state=history_state,
        audio_url_converter=design_btns["audio_url_converter"],
    )

    design_prosody.change(fn=apply_prosody_preset, inputs=[design_prosody, design_desc], outputs=design_desc)
    db_compose_btn.click(
        fn=compose_voice_description,
        inputs=[db_gender, db_age, db_tone, db_texture, db_pace, db_accent],
        outputs=design_desc,
    )
    design_enhance_btn.click(fn=enhance_description_with_ai, inputs=[design_desc], outputs=design_desc)

    def save_design_as_prompt(voice_name, history_list):
        """Save the most recent Design mode output as a voice prompt."""
        if not voice_name or not voice_name.strip():
            return "Please enter a voice name.", gr.update()
        voice_name, err = _sanitize_voice_name(voice_name)
        if err:
            return err, gr.update()
        for entry in history_list:
            if entry.get("mode") == "Design" and entry.get("path"):
                audio_path = entry["path"]
                if os.path.exists(audio_path):
                    try:
                        from qwen3_tts.tools.create_voice import create_and_save_voice_prompt
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
    return design_model_indicator


def _build_custom_tab(status_html, history_df, history_state):
    """Build Custom Mode tab components and wiring.

    Returns custom_model_indicator for cross-tab references.
    """
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

    custom_btns = _build_generate_buttons_and_output("custom")

    def custom_config_handler(text, speaker, instruct, preset, temp, top_k, top_p, rep, seed):
        return _prepare_streaming_config(
            "custom", text, preset, temp, top_k, top_p, rep, seed,
            speaker=speaker, instruct=instruct)

    _wire_generation_tab(
        "custom", custom_btns["btn"], custom_btns["cancel_btn"],
        custom_btns["status"], custom_btns["stream_config"],
        custom_btns["result_data"], custom_btns["mode_hidden"],
        custom_btns["text_hidden"], custom_model_indicator,
        custom_text, custom_text_info,
        inputs_list=[custom_text, custom_speaker, custom_instruct, custom_preset,
                     custom_ctrls["temp"], custom_ctrls["top_k"], custom_ctrls["top_p"],
                     custom_ctrls["rep"], custom_ctrls["seed"]],
        status_html=status_html, history_df=history_df,
        config_handler=custom_config_handler,
        history_state=history_state,
        audio_url_converter=custom_btns["audio_url_converter"],
    )
    custom_prosody.change(fn=apply_prosody_preset, inputs=[custom_prosody, custom_instruct], outputs=custom_instruct)
    return custom_model_indicator


def _build_create_voice_tab(clone_prompt):
    """Build Create Voice tab components and wiring."""
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


def _build_manage_voices_tab(clone_prompt):
    """Build Manage Voices tab components and wiring."""
    gr.Markdown("View, preview, rename, and delete voice prompts.")

    with gr.Row():
        with gr.Column(scale=2):
            manage_table = gr.Dataframe(
                headers=["Name", "Format", "Default"],
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
    manage_refresh_btn.click(fn=get_prompt_table_data, outputs=[manage_table])
    manage_preview_btn.click(fn=preview_voice, inputs=[manage_selected], outputs=[manage_preview_audio])
    manage_default_btn.click(fn=set_voice_default, inputs=[manage_selected], outputs=[manage_status, manage_table])
    manage_rename_btn.click(
        fn=rename_voice, inputs=[manage_selected, manage_new_name],
        outputs=[manage_status, manage_table, clone_prompt]
    )
    manage_delete_btn.click(
        fn=delete_voice, inputs=[manage_selected],
        outputs=[manage_status, manage_table, clone_prompt],
    )


def _build_manage_models_tab(status_html, clone_model_indicator,
                             design_model_indicator, custom_model_indicator):
    """Build Manage Models tab components and wiring."""
    gr.Markdown("Load/unload models, configure startup defaults, and audio loader.")

    with gr.Row():
        with gr.Column(scale=2):
            model_table = gr.Dataframe(
                headers=["Model", "Status", "Memory", "Startup"],
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

    model_refresh_btn.click(fn=get_model_table_data, outputs=model_table)

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

    asr_load_btn.click(fn=lambda: toggle_asr("load"), outputs=[model_manage_status, status_html])
    asr_unload_btn.click(fn=lambda: toggle_asr("unload"), outputs=[model_manage_status, status_html])
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


def build_ui():
    """Build the Gradio interface."""

    with gr.Blocks(title="Qwen3-TTS Web Interface") as demo:
        gr.Markdown("# Qwen3-TTS Web Interface")

        # Inject WaveSurfer.js and StreamingPlayer class
        gr.HTML(value=get_wavesurfer_loader_js() +
                "<script type='module'>" + get_streaming_player_js() + "</script>")

        # Status bar
        status_html = gr.HTML(value=format_status_display())
        with gr.Row():
            refresh_btn = gr.Button("Refresh Status", size="sm")
            stop_btn = gr.Button("Stop Server", size="sm", variant="stop")
        refresh_btn.click(fn=format_status_display, outputs=status_html)
        if hasattr(gr, 'Timer'):
            gr.Timer(value=5).tick(fn=format_status_display, outputs=status_html)
        stop_btn.click(fn=stop_server, outputs=status_html)

        # Model Settings (MLX-first architecture)
        current_size, current_quant, current_backend = get_current_model_settings()
        with gr.Accordion("Model Settings", open=False):
            gr.Markdown("Change model size or quantization. Settings apply on next generation.")
            with gr.Row():
                model_size_dropdown = gr.Dropdown(
                    label="Model Size", choices=list(VALID_MODEL_SIZES), value=current_size,
                    info="1.7B: higher quality | 0.6B: ~40% faster, lower memory"
                )
                mlx_quant_dropdown = gr.Dropdown(
                    label="MLX Quantization", choices=list(VALID_MLX_QUANTIZATIONS),
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

        # Per-session history state (shared across tabs)
        history_state = gr.State([])

        # History (defined before tabs for wiring, renders here)
        with gr.Accordion("Recent Generations", open=True):
            history_df = gr.Dataframe(
                headers=["Time", "Mode", "Text Preview", "Seed", "Chunks"],
                value=[], interactive=False, wrap=True,
            )
            gr.HTML(value=get_player_html("history"))
            history_audio_url = gr.Audio(elem_classes=["gr-hidden"])

        history_df.select(
            fn=on_history_select, inputs=[history_state], outputs=[history_audio_url],
        ).then(
            fn=lambda x: x, js=get_load_into_player_js("history"),
            inputs=[history_audio_url], outputs=[history_audio_url],
        )

        # Tabs for different modes
        with gr.Tabs():
            with gr.Tab("Clone Mode") as clone_tab:
                clone_prompt, clone_model_indicator = _build_clone_tab(
                    status_html, history_df, history_state)
            with gr.Tab("Design Mode") as design_tab:
                design_model_indicator = _build_design_tab(
                    status_html, history_df, history_state, clone_prompt)
            with gr.Tab("Custom Mode") as custom_tab:
                custom_model_indicator = _build_custom_tab(
                    status_html, history_df, history_state)

            clone_tab.select(fn=lambda: get_model_status_html("clone"), outputs=clone_model_indicator)
            design_tab.select(fn=lambda: get_model_status_html("design"), outputs=design_model_indicator)
            custom_tab.select(fn=lambda: get_model_status_html("custom"), outputs=custom_model_indicator)

            with gr.Tab("Create Voice"):
                _build_create_voice_tab(clone_prompt)
            with gr.Tab("Manage Voices"):
                _build_manage_voices_tab(clone_prompt)
            with gr.Tab("Manage Models"):
                _build_manage_models_tab(
                    status_html, clone_model_indicator,
                    design_model_indicator, custom_model_indicator)

        demo.load(
            fn=lambda: (
                get_model_status_html("clone"),
                get_model_status_html("design"),
                get_model_status_html("custom")
            ),
            js=get_script_reexecutor_fn(),
            outputs=[clone_model_indicator, design_model_indicator, custom_model_indicator]
        )

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


def main():
    """Main entry point."""
    import argparse

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

    from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

    demo = build_ui()
    share = args.share or IN_COLAB
    inbrowser = not args.no_browser and not IN_COLAB
    demo.launch(
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        **get_gradio_launch_kwargs(config),
    )


if __name__ == "__main__":
    main()
