#!/usr/bin/env python3
"""Clone / Design / Custom tab builders for the Gradio UI.

Each ``_build_*_tab`` function builds a mode tab's components and wires its
generation chain via ``generation._wire_generation_tab``. Collaborators are
imported module-style so tests can patch them at their definition site.
"""

import os
import re

import gradio as gr

from qwen3_tts.core import config as core_config
from qwen3_tts.interface import voice_helpers
from qwen3_tts.interface.ui import generation, model_management, shared

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def _sanitize_voice_name(raw: str) -> tuple:
    """Validate and sanitize a voice name using allowlist.

    Returns (sanitized_name, error_or_None).
    """
    name = raw.strip().replace(" ", "_")
    if not _SAFE_NAME_RE.match(name):
        return (
            "",
            "Voice name may only contain letters, numbers, underscores, and hyphens (1-64 chars)",
        )
    return name, None


def _build_clone_tab(status_html, history_state):
    """Build Clone Mode tab components and wiring.

    Returns (clone_prompt, clone_model_indicator, clone_chain, clone_seed) for cross-tab references.
    clone_chain is the final event chain object; callers may append .then() steps to it
    (e.g. to update a history dataframe rendered outside this tab).
    """
    gr.Markdown(
        "Use a voice prompt file to clone a specific voice. "
        "Clone mode reproduces the voice from your reference audio. "
        "For voice design from descriptions, use Design mode. "
        "To create a reusable designed voice, generate in Design mode then save as a voice prompt."
    )
    clone_model_indicator = gr.HTML(value=model_management.get_model_status_html("clone"))

    with gr.Row():
        with gr.Column(scale=2):
            clone_text = gr.Textbox(
                label="Text Input", placeholder="Enter text to synthesize...", lines=3
            )
            clone_text_info = gr.Textbox(
                label="",
                show_label=False,
                interactive=False,
                max_lines=1,
                container=False,
            )
            _default_prompt = core_config.get_default_clone_prompt()
            _prompts = shared.get_voice_prompts()
            clone_prompt = gr.Dropdown(
                label="Voice Prompt",
                choices=_prompts,
                value=_default_prompt
                if _default_prompt in _prompts
                else (_prompts[0] if _prompts else None),
            )
            clone_preset = gr.Dropdown(
                label="Preset", choices=shared.get_presets(), value="(none)"
            )

        with gr.Column(scale=1):
            clone_ctrls = generation._build_common_controls()
            clone_no_transcript = gr.Checkbox(
                label="Speaker embedding only",
                value=False,
                info="Clone using x-vector only (no transcript needed, lower fidelity)",
            )

    clone_btns = generation._build_generate_buttons_and_output("clone")
    gen_guard_state = gr.State({"generating": False, "armed": False, "ts": 0.0})

    def clone_config_handler(
        text, prompt, preset, temp, top_k, top_p, rep, seed, no_transcript, seed_lock
    ):
        return generation._prepare_streaming_config(
            "clone",
            text,
            preset,
            temp,
            top_k,
            top_p,
            rep,
            seed,
            prompt_file=prompt,
            no_transcript=no_transcript,
            seed_lock_chunks=seed_lock,
        )

    clone_chain = generation._wire_generation_tab(
        "clone",
        clone_btns["btn"],
        clone_btns["cancel_btn"],
        clone_btns["status"],
        clone_btns["stream_config"],
        clone_btns["result_data"],
        clone_btns["mode_hidden"],
        clone_btns["text_hidden"],
        clone_model_indicator,
        clone_text,
        clone_text_info,
        inputs_list=[
            clone_text,
            clone_prompt,
            clone_preset,
            clone_ctrls["temp"],
            clone_ctrls["top_k"],
            clone_ctrls["top_p"],
            clone_ctrls["rep"],
            clone_ctrls["seed"],
            clone_no_transcript,
            clone_ctrls["seed_lock"],
        ],
        status_html=status_html,
        status_announcer=clone_btns["status_announcer"],
        config_handler=clone_config_handler,
        api_name="generate_clone",
        history_state=history_state,
        audio_url_converter=clone_btns["audio_url_converter"],
        gen_guard_state=gen_guard_state,
    )
    return clone_prompt, clone_model_indicator, clone_chain, clone_ctrls["seed"]


def _build_design_tab(status_html, history_state, clone_prompt):
    """Build Design Mode tab components and wiring.

    Returns (design_model_indicator, design_chain, design_seed) for cross-tab references.
    design_chain is the final event chain object; callers may append .then() steps to it
    (e.g. to update a history dataframe rendered outside this tab).
    """
    gr.Markdown("Generate a voice from a text description.")
    design_model_indicator = gr.HTML(value=model_management.get_model_status_html("design"))

    with gr.Row():
        with gr.Column(scale=2):
            design_text = gr.Textbox(
                label="Text Input", placeholder="Enter text to synthesize...", lines=3
            )
            design_text_info = gr.Textbox(
                label="",
                show_label=False,
                interactive=False,
                max_lines=1,
                container=False,
            )
            design_desc = gr.Textbox(
                label="Voice Description",
                placeholder="Describe the voice (e.g., 'A warm, friendly female voice with clear articulation')",
                lines=2,
            )
            with gr.Row():
                design_prosody = gr.Dropdown(
                    label="Style Preset",
                    choices=voice_helpers.get_prosody_choices(),
                    value="(none)",
                    info="Appends style to description",
                    scale=2,
                )
                _enhancer_visible = shared.is_enhancer_available()
                design_enhance_btn = gr.Button(
                    "Enhance with AI",
                    size="sm",
                    variant="secondary",
                    visible=_enhancer_visible,
                    scale=1,
                )

            with gr.Accordion("Description Builder", open=False):
                gr.Markdown("Build a voice description from attributes:")
                _none_opt = ["(none)"]
                with gr.Row():
                    db_gender = gr.Dropdown(
                        label="Gender",
                        choices=_none_opt + core_config.VOICE_DESCRIPTION_ATTRIBUTES["gender"],
                        value="(none)",
                    )
                    db_age = gr.Dropdown(
                        label="Age",
                        choices=_none_opt + core_config.VOICE_DESCRIPTION_ATTRIBUTES["age"],
                        value="(none)",
                    )
                with gr.Row():
                    db_tone = gr.Dropdown(
                        label="Tone",
                        choices=_none_opt + core_config.VOICE_DESCRIPTION_ATTRIBUTES["tone"],
                        value="(none)",
                    )
                    db_texture = gr.Dropdown(
                        label="Texture",
                        choices=_none_opt + core_config.VOICE_DESCRIPTION_ATTRIBUTES["texture"],
                        value="(none)",
                    )
                with gr.Row():
                    db_pace = gr.Dropdown(
                        label="Pace",
                        choices=_none_opt + core_config.VOICE_DESCRIPTION_ATTRIBUTES["pace"],
                        value="(none)",
                    )
                    db_accent = gr.Dropdown(
                        label="Accent",
                        choices=_none_opt + core_config.VOICE_DESCRIPTION_ATTRIBUTES["accent"],
                        value="(none)",
                    )
                db_compose_btn = gr.Button(
                    "Compose Description", size="sm", variant="secondary"
                )

            design_preset = gr.Dropdown(
                label="Preset", choices=shared.get_presets(), value="(none)"
            )

        with gr.Column(scale=1):
            design_ctrls = generation._build_common_controls()

    design_btns = generation._build_generate_buttons_and_output("design")
    gen_guard_state = gr.State({"generating": False, "armed": False, "ts": 0.0})

    # Save as Voice Prompt (Design-then-Clone pipeline)
    with gr.Accordion("Save as Voice Prompt", open=False):
        gr.Markdown("Save the generated audio as a reusable voice clone prompt.")
        design_save_name = gr.Textbox(
            label="Voice Name", placeholder="e.g., designed_voice", max_lines=1
        )
        design_save_btn = gr.Button(
            "Save as Voice Prompt", size="sm", variant="secondary"
        )
        design_save_status = gr.Textbox(
            label="", show_label=False, interactive=False, max_lines=1, container=False
        )

    def design_config_handler(
        text, desc, preset, temp, top_k, top_p, rep, seed, seed_lock
    ):
        return generation._prepare_streaming_config(
            "design",
            text,
            preset,
            temp,
            top_k,
            top_p,
            rep,
            seed,
            description=desc,
            seed_lock_chunks=seed_lock,
        )

    design_chain = generation._wire_generation_tab(
        "design",
        design_btns["btn"],
        design_btns["cancel_btn"],
        design_btns["status"],
        design_btns["stream_config"],
        design_btns["result_data"],
        design_btns["mode_hidden"],
        design_btns["text_hidden"],
        design_model_indicator,
        design_text,
        design_text_info,
        inputs_list=[
            design_text,
            design_desc,
            design_preset,
            design_ctrls["temp"],
            design_ctrls["top_k"],
            design_ctrls["top_p"],
            design_ctrls["rep"],
            design_ctrls["seed"],
            design_ctrls["seed_lock"],
        ],
        status_html=status_html,
        status_announcer=design_btns["status_announcer"],
        config_handler=design_config_handler,
        history_state=history_state,
        audio_url_converter=design_btns["audio_url_converter"],
        gen_guard_state=gen_guard_state,
    )

    design_prosody.change(
        fn=voice_helpers.apply_prosody_preset,
        inputs=[design_prosody, design_desc],
        outputs=design_desc,
    )
    db_compose_btn.click(
        fn=voice_helpers.compose_voice_description,
        inputs=[db_gender, db_age, db_tone, db_texture, db_pace, db_accent],
        outputs=design_desc,
    )
    design_enhance_btn.click(
        fn=shared.enhance_description_with_ai, inputs=[design_desc], outputs=design_desc
    )

    def save_design_as_prompt(voice_name, history_list):
        """Save the most recent Design mode output as a voice prompt."""
        if not voice_name or not voice_name.strip():
            return "Please enter a voice name.", gr.update()
        voice_name, err = _sanitize_voice_name(voice_name)
        if err:
            return err, gr.update()
        try:
            core_config.validate_voice_name(voice_name)
        except ValueError as exc:
            return str(exc), gr.update()
        for entry in history_list:
            if entry.get("mode") == "Design" and entry.get("path"):
                audio_path = entry["path"]
                # Security: validate audio_path from history before using
                audio_expanded = os.path.expanduser(audio_path)
                if os.path.isabs(audio_expanded):
                    if ".." in audio_expanded:
                        return (
                            f"Path traversal detected in history path: {audio_path}",
                            gr.update(),
                        )
                    safe_audio_path = audio_expanded
                else:
                    safe_audio_path = core_config.safe_path_join(os.getcwd(), audio_expanded)

                # Verify path is under home directory
                home = os.path.realpath(os.path.expanduser("~"))
                resolved = os.path.realpath(safe_audio_path)
                if not (resolved == home or resolved.startswith(home + os.sep)):
                    return (
                        f"History path must be under home directory: {audio_path}",
                        gr.update(),
                    )

                if os.path.exists(safe_audio_path):
                    try:
                        from qwen3_tts.tools.create_voice import (
                            create_and_save_voice_prompt,
                        )

                        backend = core_config.get_backend()
                        mlx_only = (backend == "mlx") or core_config.IN_COLAB
                        create_and_save_voice_prompt(
                            safe_audio_path,
                            "",
                            voice_name,
                            test_generation=False,
                            mlx_only=mlx_only,
                        )
                        prompts = shared.get_voice_prompts()
                        return f"Saved voice prompt: {voice_name}", gr.update(
                            choices=prompts
                        )
                    except Exception as e:
                        return f"Error: {e}", gr.update()
        return "No recent Design mode output found. Generate audio first.", gr.update()

    design_save_btn.click(
        fn=save_design_as_prompt,
        inputs=[design_save_name, history_state],
        outputs=[design_save_status, clone_prompt],
    )
    return design_model_indicator, design_chain, design_ctrls["seed"]


def _build_custom_tab(status_html, history_state):
    """Build Custom Mode tab components and wiring.

    Returns (custom_model_indicator, custom_chain, custom_seed) for cross-tab references.
    custom_chain is the final event chain object; callers may append .then() steps to it
    (e.g. to update a history dataframe rendered outside this tab).
    """
    gr.Markdown("Use premium pre-trained speakers.")
    custom_model_indicator = gr.HTML(value=model_management.get_model_status_html("custom"))

    with gr.Row():
        with gr.Column(scale=2):
            custom_text = gr.Textbox(
                label="Text Input", placeholder="Enter text to synthesize...", lines=3
            )
            custom_text_info = gr.Textbox(
                label="",
                show_label=False,
                interactive=False,
                max_lines=1,
                container=False,
            )
            custom_speaker = gr.Dropdown(
                label="Speaker", choices=shared.SPEAKER_CHOICES, value=shared.SPEAKER_CHOICES[0]
            )
            custom_prosody = gr.Dropdown(
                label="Style Preset",
                choices=voice_helpers.get_prosody_choices(),
                value="(none)",
                info="Select a preset to fill the instruction field, or type your own below",
            )
            custom_instruct = gr.Textbox(
                label="Style Instruction (optional)",
                placeholder="e.g., 'Speak with enthusiasm' or 'Read slowly and clearly'",
                lines=1,
            )
            custom_preset = gr.Dropdown(
                label="Preset", choices=shared.get_presets(), value="(none)"
            )

        with gr.Column(scale=1):
            custom_ctrls = generation._build_common_controls()

    custom_btns = generation._build_generate_buttons_and_output("custom")
    gen_guard_state = gr.State({"generating": False, "armed": False, "ts": 0.0})

    def custom_config_handler(
        text, speaker, instruct, preset, temp, top_k, top_p, rep, seed, seed_lock
    ):
        return generation._prepare_streaming_config(
            "custom",
            text,
            preset,
            temp,
            top_k,
            top_p,
            rep,
            seed,
            speaker=speaker,
            instruct=instruct,
            seed_lock_chunks=seed_lock,
        )

    custom_chain = generation._wire_generation_tab(
        "custom",
        custom_btns["btn"],
        custom_btns["cancel_btn"],
        custom_btns["status"],
        custom_btns["stream_config"],
        custom_btns["result_data"],
        custom_btns["mode_hidden"],
        custom_btns["text_hidden"],
        custom_model_indicator,
        custom_text,
        custom_text_info,
        inputs_list=[
            custom_text,
            custom_speaker,
            custom_instruct,
            custom_preset,
            custom_ctrls["temp"],
            custom_ctrls["top_k"],
            custom_ctrls["top_p"],
            custom_ctrls["rep"],
            custom_ctrls["seed"],
            custom_ctrls["seed_lock"],
        ],
        status_html=status_html,
        status_announcer=custom_btns["status_announcer"],
        config_handler=custom_config_handler,
        history_state=history_state,
        audio_url_converter=custom_btns["audio_url_converter"],
        gen_guard_state=gen_guard_state,
    )
    custom_prosody.change(
        fn=voice_helpers.apply_prosody_preset,
        inputs=[custom_prosody, custom_instruct],
        outputs=custom_instruct,
    )
    return custom_model_indicator, custom_chain, custom_ctrls["seed"]
