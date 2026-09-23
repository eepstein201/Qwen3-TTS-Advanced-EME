#!/usr/bin/env python3
"""Clone / Design / Custom tab builders for the Gradio UI.

Each ``_build_*_tab`` function builds a mode tab's components and wires its
generation chain via ``generation._wire_generation_tab``. Collaborators are
imported module-style so tests can patch them at their definition site.
"""

import os
import re
import time

import gradio as gr

from qwen3_tts.core import config as core_config
from qwen3_tts.interface import voice_helpers
from qwen3_tts.interface.ui import generation, model_management, shared

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")

# Sentinel shown as the "nothing selected" entry in every preset/prosody
# dropdown. Handlers compare against it, so it must stay a single literal.
NONE_CHOICE = "(none)"

# --- User-defined prosody presets (Custom-tab save/delete) ------------------
#
# Target-keyed two-step confirm (history_panel precedent): the handler owns
# the name comparison; confirm applies only when armed_name matches the
# current stripped target AND within the window — anything else re-arms.
# Handlers never call load_config/save_config directly and never re-implement
# validation: branch decisions go through core_config, all writes through the
# CRUD functions, whose (ok, message) tuples are displayed verbatim.

_PROSODY_CONFIRM_WINDOW_SEC = 5.0
_PROSODY_SAVE_BTN_BASE = "Save as preset"
_PROSODY_SAVE_BTN_ARM = "Confirm Overwrite? (click again)"
_PROSODY_DELETE_BTN_BASE = "Delete preset"
_PROSODY_DELETE_BTN_ARM = "Confirm Delete? (click again)"
_PROSODY_DISARMED_STATE = {"armed": False, "ts": 0.0, "armed_name": None}


def _prosody_arm_is_fresh(state, name, now):
    return (
        bool(state.get("armed"))
        and state.get("armed_name") == name
        and (now - float(state.get("ts") or 0.0)) <= _PROSODY_CONFIRM_WINDOW_SEC
    )


def _prosody_disarmed_result(btn_label, status):
    """Non-arm branch shape: full disarmed state, base button label, status,
    announcer, and bare no-change updates on all three dropdowns."""
    return (
        dict(_PROSODY_DISARMED_STATE),
        gr.update(value=btn_label),
        status,
        generation._announce_status(status),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def _prosody_apply_dropdown_reset(choice, affected, merged_choices):
    """Conditional reset: value=NONE only when THIS dropdown's own input
    names the affected preset (a stale "name - old text" label must not
    survive an overwrite; unrelated selections untouched — resetting them
    re-fires .change and double-appends). None guards to no reset."""
    current = voice_helpers.prosody_choice_to_name(
        choice if isinstance(choice, str) else None
    )
    if current == affected:
        return gr.update(choices=merged_choices, value=NONE_CHOICE)
    return gr.update(choices=merged_choices)


def _on_save_prosody_preset(
    state, preset_name, instruct_text, custom_choice, design_choice
):
    """Save (or overwrite) a user-defined prosody preset. The name is
    stripped once, up front, and the stripped form is used everywhere —
    validation, existence, armed_name, and the confirm comparison."""
    now = time.time()
    name = preset_name.strip() if isinstance(preset_name, str) else ""
    text = instruct_text.strip() if isinstance(instruct_text, str) else ""

    error = core_config.validate_prosody_preset_name(name)
    if error is None and not text:
        # The writer owns the empty-instruct copy (never a UI-local literal).
        _ok, error = core_config.save_user_prosody_preset(name, instruct_text)
    if error is not None:
        return _prosody_disarmed_result(_PROSODY_SAVE_BTN_BASE, error)

    if (
        not _prosody_arm_is_fresh(state, name, now)
        and name in core_config.get_user_prosody_presets()
    ):
        if state.get("armed") and state.get("armed_name") != name:
            status = f"Changed to '{name}' — click again within 5s to overwrite it."
        elif state.get("armed"):
            status = (
                f"Your confirmation timed out. Click again within 5s to "
                f"overwrite '{name}'."
            )
        else:
            status = (
                f"A preset named '{name}' already exists. "
                "Click again within 5s to overwrite it."
            )
        return (
            {"armed": True, "ts": now, "armed_name": name},
            gr.update(value=_PROSODY_SAVE_BTN_ARM),
            status,
            generation._announce_status(status),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    ok, msg = core_config.save_user_prosody_preset(name, text)
    if not ok:
        return _prosody_disarmed_result(_PROSODY_SAVE_BTN_BASE, msg)
    merged = voice_helpers.get_prosody_choices()
    return (
        dict(_PROSODY_DISARMED_STATE),
        gr.update(value=_PROSODY_SAVE_BTN_BASE),
        msg,
        generation._announce_status(msg),
        _prosody_apply_dropdown_reset(custom_choice, name, merged),
        _prosody_apply_dropdown_reset(design_choice, name, merged),
        gr.update(choices=voice_helpers.get_user_prosody_choices()),
    )


def _on_delete_prosody_preset(state, selection, custom_choice, design_choice):
    """Delete a user-defined prosody preset. Membership misses render the
    data-layer classification message verbatim (never a UI-local literal)."""
    now = time.time()
    name = (
        voice_helpers.prosody_choice_to_name(selection)
        if isinstance(selection, str)
        else NONE_CHOICE
    )
    if name == NONE_CHOICE or not name:
        return _prosody_disarmed_result(
            _PROSODY_DELETE_BTN_BASE, "Select one of your presets to delete."
        )

    fresh = _prosody_arm_is_fresh(state, name, now)
    if not fresh and name not in core_config.get_user_prosody_presets():
        _ok, msg = core_config.delete_user_prosody_preset(name)
        return _prosody_disarmed_result(_PROSODY_DELETE_BTN_BASE, msg)

    if not fresh:
        if state.get("armed") and state.get("armed_name") != name:
            status = f"Now deleting '{name}' — click again within 5s to confirm."
        elif state.get("armed"):
            status = (
                f"Your confirmation timed out. Click again within 5s to "
                f"delete '{name}'."
            )
        else:
            status = f"Delete preset '{name}'? Click again within 5s to confirm."
        return (
            {"armed": True, "ts": now, "armed_name": name},
            gr.update(value=_PROSODY_DELETE_BTN_ARM),
            status,
            generation._announce_status(status),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    ok, msg = core_config.delete_user_prosody_preset(name)
    if not ok:
        return _prosody_disarmed_result(_PROSODY_DELETE_BTN_BASE, msg)
    merged = voice_helpers.get_prosody_choices()
    return (
        dict(_PROSODY_DISARMED_STATE),
        gr.update(value=_PROSODY_DELETE_BTN_BASE),
        msg,
        generation._announce_status(msg),
        _prosody_apply_dropdown_reset(custom_choice, name, merged),
        _prosody_apply_dropdown_reset(design_choice, name, merged),
        gr.update(choices=voice_helpers.get_user_prosody_choices(), value=NONE_CHOICE),
    )


def _generation_preset_disarmed_result(btn_label, status):
    """Non-arm branch shape for the 8-slot generation-preset flows: full
    disarmed state, base button label, status, announcer, and bare
    no-change updates on all four dropdowns."""
    return (
        dict(_PROSODY_DISARMED_STATE),
        gr.update(value=btn_label),
        status,
        generation._announce_status(status),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def _generation_preset_dropdown_reset(choice, affected, merged_choices):
    """Conditional reset: value=NONE only when THIS dropdown's own value is
    the affected preset name (bare names — no formatter here, unlike
    prosody). None / non-str guards to no reset."""
    current = choice if isinstance(choice, str) else ""
    if current == affected:
        return gr.update(choices=merged_choices, value=NONE_CHOICE)
    return gr.update(choices=merged_choices)


def _on_save_generation_preset(
    state,
    preset_name,
    temp,
    top_k,
    top_p,
    rep,
    clone_choice,
    design_choice,
    custom_choice,
):
    """Save (or overwrite) a user-defined generation preset from the Clone
    tab's sampling sliders. The name is stripped once, up front (prosody
    handler precedent); all writes go through the data-layer writer."""
    now = time.time()
    name = preset_name.strip() if isinstance(preset_name, str) else ""

    error = core_config.validate_generation_preset_name(name)
    if error is None:
        # Defense-in-depth: sliders are naturally in range, but the apply
        # path is a blind dict update, so params re-validate at the boundary.
        params = {
            "temperature": temp,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": rep,
        }
        error = core_config.validate_generation_preset_params(params)
    if error is not None:
        return _generation_preset_disarmed_result(_PROSODY_SAVE_BTN_BASE, error)

    if (
        not _prosody_arm_is_fresh(state, name, now)
        and name in core_config.get_user_generation_presets()
    ):
        status = f"A preset named '{name}' exists — click Save again to overwrite."
        return (
            {"armed": True, "ts": now, "armed_name": name},
            gr.update(value=_PROSODY_SAVE_BTN_ARM),
            status,
            generation._announce_status(status),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    ok, msg = core_config.save_user_generation_preset(name, params)
    if not ok:
        return _generation_preset_disarmed_result(
            _PROSODY_SAVE_BTN_BASE, f"Could not save: {msg}"
        )
    merged = shared.get_presets()
    return (
        dict(_PROSODY_DISARMED_STATE),
        gr.update(value=_PROSODY_SAVE_BTN_BASE),
        msg,
        generation._announce_status(msg),
        # Save refreshes CHOICES only — never a value reset (spec §6). Labels
        # are bare names here, unlike prosody's "name - text", so an overwrite
        # leaves every selection valid; clearing one would silently deselect
        # the preset on a tab the user never touched, whose own sliders hold
        # different values.
        gr.update(choices=merged),
        gr.update(choices=merged),
        gr.update(choices=merged),
        gr.update(choices=shared.get_user_generation_preset_choices()),
    )


def _on_delete_generation_preset(
    state, selection, clone_choice, design_choice, custom_choice
):
    """Delete a user-defined generation preset. Classification (factory /
    membership) happens BEFORE arming and never writes; the confirm path
    goes through the data-layer writer."""
    now = time.time()
    name = selection.strip() if isinstance(selection, str) else ""

    if not name or name == NONE_CHOICE:
        return _generation_preset_disarmed_result(
            _PROSODY_DELETE_BTN_BASE, "Select one of your presets to delete."
        )
    error = core_config.factory_generation_preset_error(name)
    if error is not None:
        # Factory classification ONLY — never arms, never writes. The full
        # name validator does not belong here: the name came from the
        # dropdown, not the user, and the reader applies no name filter, so
        # charset/length rules would strand a hand-edited junk entry as
        # visible-but-undeletable.
        return _generation_preset_disarmed_result(_PROSODY_DELETE_BTN_BASE, error)

    fresh = _prosody_arm_is_fresh(state, name, now)
    if not fresh and name not in core_config.get_user_generation_presets():
        return _generation_preset_disarmed_result(
            _PROSODY_DELETE_BTN_BASE, f"Preset '{name}' no longer exists."
        )

    if not fresh:
        status = f"Click Delete again to delete '{name}'."
        return (
            {"armed": True, "ts": now, "armed_name": name},
            gr.update(value=_PROSODY_DELETE_BTN_ARM),
            status,
            generation._announce_status(status),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    ok, msg = core_config.delete_user_generation_preset(name)
    if not ok:
        return _generation_preset_disarmed_result(_PROSODY_DELETE_BTN_BASE, msg)
    merged = shared.get_presets()
    return (
        dict(_PROSODY_DISARMED_STATE),
        gr.update(value=_PROSODY_DELETE_BTN_BASE),
        msg,
        generation._announce_status(msg),
        _generation_preset_dropdown_reset(clone_choice, name, merged),
        _generation_preset_dropdown_reset(design_choice, name, merged),
        _generation_preset_dropdown_reset(custom_choice, name, merged),
        gr.update(
            choices=shared.get_user_generation_preset_choices(), value=NONE_CHOICE
        ),
    )


def _new_gen_guard_state() -> dict:
    """Fresh per-tab generate-guard state.

    Each tab owns its own ``gen_guard_state`` (double-submit guard); returning a
    new dict per call keeps the tabs from sharing one mutable default.
    """
    return {"generating": False, "armed": False, "ts": 0.0}


def _sanitize_voice_name(raw: str) -> tuple[str, str | None]:
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


def _build_clone_tab(status_html, history_state, progress_timer=None):
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
    clone_model_indicator = gr.HTML(
        value=model_management.get_model_status_html("clone")
    )

    with gr.Row():
        with gr.Column(scale=2):
            clone_text = gr.Textbox(
                label="Text Input",
                placeholder="Enter text to synthesize...",
                lines=3,
                info="Long text is automatically split into ~500-char chunks to avoid mid-sentence cutoff (a single unsplit generation is capped at ~170 s of audio)",
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
                label="Preset", choices=shared.get_presets(), value=NONE_CHOICE
            )

            with gr.Accordion("My generation presets", open=False):
                gr.Markdown(
                    "Save the current Temperature / Top-K / Top-P / "
                    "Repetition Penalty sliders as a named preset, or delete "
                    "presets you created. Factory presets can't be "
                    "overwritten. Your presets appear in the Preset dropdown "
                    "on every tab."
                )
                generation_preset_name = gr.Textbox(
                    label="Preset name",
                    placeholder="Preset name (max 40 characters)",
                )
                with gr.Row():
                    generation_preset_save_btn = gr.Button(
                        _PROSODY_SAVE_BTN_BASE, variant="secondary", size="sm"
                    )
                    generation_preset_delete_btn = gr.Button(
                        _PROSODY_DELETE_BTN_BASE, variant="stop", size="sm"
                    )
                generation_preset_save_status = gr.Textbox(
                    label="",
                    show_label=False,
                    interactive=False,
                    max_lines=2,
                    container=False,
                )
                generation_preset_delete_status = gr.Textbox(
                    label="",
                    show_label=False,
                    interactive=False,
                    max_lines=2,
                    container=False,
                )
                generation_preset_delete_dropdown = gr.Dropdown(
                    label="Preset to delete",
                    info="Only presets you created appear here.",
                    choices=shared.get_user_generation_preset_choices(),
                    value=NONE_CHOICE,
                )
                # sr-only announcer — never visible=False (Gradio 6 drops it
                # from the DOM); mirrors both status boxes for SR users.
                generation_preset_announcer = gr.HTML(generation._announce_status(""))
                generation_preset_save_state = gr.State(dict(_PROSODY_DISARMED_STATE))
                generation_preset_delete_state = gr.State(dict(_PROSODY_DISARMED_STATE))

        with gr.Column(scale=1):
            clone_ctrls = generation._build_common_controls()
            clone_no_transcript = gr.Checkbox(
                label="Speaker embedding only",
                value=False,
                info="Speaker embedding only (x-vector): clones without a transcript, lower fidelity",
            )

    clone_btns = generation._build_generate_buttons_and_output("clone")
    gen_guard_state = gr.State(_new_gen_guard_state())

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
        progress=clone_btns["progress"],
        progress_timer=progress_timer,
        config_handler=clone_config_handler,
        api_name="generate_clone",
        history_state=history_state,
        audio_url_converter=clone_btns["audio_url_converter"],
        gen_guard_state=gen_guard_state,
    )
    # Cross-tab refs for the facade-wired save/delete clicks (D1a): the dict
    # avoids a 13-tuple; sliders ride along because the wiring needs them as
    # positional inputs.
    preset_builder = {
        "name": generation_preset_name,
        "save_btn": generation_preset_save_btn,
        "delete_btn": generation_preset_delete_btn,
        "save_state": generation_preset_save_state,
        "delete_state": generation_preset_delete_state,
        "save_status": generation_preset_save_status,
        "delete_status": generation_preset_delete_status,
        "delete_dropdown": generation_preset_delete_dropdown,
        "announcer": generation_preset_announcer,
        "clone_preset": clone_preset,
        "temp": clone_ctrls["temp"],
        "top_k": clone_ctrls["top_k"],
        "top_p": clone_ctrls["top_p"],
        "rep": clone_ctrls["rep"],
    }
    return (
        clone_prompt,
        clone_model_indicator,
        clone_chain,
        clone_ctrls["seed"],
        preset_builder,
    )


def _build_design_tab(status_html, history_state, clone_prompt, progress_timer=None):
    """Build Design Mode tab components and wiring.

    Returns (design_model_indicator, design_chain, design_seed) for cross-tab references.
    design_chain is the final event chain object; callers may append .then() steps to it
    (e.g. to update a history dataframe rendered outside this tab).
    """
    gr.Markdown("Generate a voice from a text description.")
    design_model_indicator = gr.HTML(
        value=model_management.get_model_status_html("design")
    )

    with gr.Row():
        with gr.Column(scale=2):
            design_text = gr.Textbox(
                label="Text Input",
                placeholder="Enter text to synthesize...",
                lines=3,
                info="Long text is automatically split into ~500-char chunks to avoid mid-sentence cutoff (a single unsplit generation is capped at ~170 s of audio)",
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
                    value=NONE_CHOICE,
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
                _none_opt = [NONE_CHOICE]
                with gr.Row():
                    db_gender = gr.Dropdown(
                        label="Gender",
                        choices=_none_opt
                        + core_config.VOICE_DESCRIPTION_ATTRIBUTES["gender"],
                        value=NONE_CHOICE,
                    )
                    db_age = gr.Dropdown(
                        label="Age",
                        choices=_none_opt
                        + core_config.VOICE_DESCRIPTION_ATTRIBUTES["age"],
                        value=NONE_CHOICE,
                    )
                with gr.Row():
                    db_tone = gr.Dropdown(
                        label="Tone",
                        choices=_none_opt
                        + core_config.VOICE_DESCRIPTION_ATTRIBUTES["tone"],
                        value=NONE_CHOICE,
                    )
                    db_texture = gr.Dropdown(
                        label="Texture",
                        choices=_none_opt
                        + core_config.VOICE_DESCRIPTION_ATTRIBUTES["texture"],
                        value=NONE_CHOICE,
                    )
                with gr.Row():
                    db_pace = gr.Dropdown(
                        label="Pace",
                        choices=_none_opt
                        + core_config.VOICE_DESCRIPTION_ATTRIBUTES["pace"],
                        value=NONE_CHOICE,
                    )
                    db_accent = gr.Dropdown(
                        label="Accent",
                        choices=_none_opt
                        + core_config.VOICE_DESCRIPTION_ATTRIBUTES["accent"],
                        value=NONE_CHOICE,
                    )
                db_compose_btn = gr.Button(
                    "Compose Description", size="sm", variant="secondary"
                )

            design_preset = gr.Dropdown(
                label="Preset", choices=shared.get_presets(), value=NONE_CHOICE
            )

        with gr.Column(scale=1):
            design_ctrls = generation._build_common_controls()

    design_btns = generation._build_generate_buttons_and_output("design")
    gen_guard_state = gr.State(_new_gen_guard_state())

    # Save as Voice Prompt (Design-then-Clone pipeline)
    with gr.Accordion("Save as Voice Prompt", open=False):
        gr.Markdown(
            "Save the generated audio as a reusable voice clone prompt. Saved "
            "prompts use speaker-embedding-only (x-vector) mode, so fidelity is "
            "lower than a transcript-based clone."
        )
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
        progress=design_btns["progress"],
        progress_timer=progress_timer,
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
                    safe_audio_path = core_config.safe_path_join(
                        os.getcwd(), audio_expanded
                    )

                # Verify path is under home directory. `resolved` is the
                # canonical, symlink-free path — everything downstream uses it
                # rather than safe_audio_path, so the path that was validated is
                # the exact path that gets opened (no validate-one/use-another
                # gap in the absolute-path branch above).
                home = os.path.realpath(os.path.expanduser("~"))
                resolved = os.path.realpath(safe_audio_path)
                if not (resolved == home or resolved.startswith(home + os.sep)):
                    return (
                        f"History path must be under home directory: {audio_path}",
                        gr.update(),
                    )

                if os.path.exists(resolved):
                    try:
                        from qwen3_tts.tools.create_voice import (
                            create_and_save_voice_prompt,
                        )

                        backend = core_config.get_backend()
                        mlx_only = (backend == "mlx") or core_config.IN_COLAB
                        create_and_save_voice_prompt(
                            resolved,
                            "",
                            voice_name,
                            test_generation=False,
                            mlx_only=mlx_only,
                            x_vector_only_mode=True,
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
    return (
        design_model_indicator,
        design_chain,
        design_ctrls["seed"],
        design_prosody,
        design_preset,
    )


def _build_custom_tab(status_html, history_state, design_prosody, progress_timer=None):
    """Build Custom Mode tab components and wiring.

    Returns (custom_model_indicator, custom_chain, custom_seed) for cross-tab references.
    custom_chain is the final event chain object; callers may append .then() steps to it
    (e.g. to update a history dataframe rendered outside this tab).
    """
    gr.Markdown("Use premium pre-trained speakers.")
    custom_model_indicator = gr.HTML(
        value=model_management.get_model_status_html("custom")
    )

    with gr.Row():
        with gr.Column(scale=2):
            custom_text = gr.Textbox(
                label="Text Input",
                placeholder="Enter text to synthesize...",
                lines=3,
                info="Long text is automatically split into ~500-char chunks to avoid mid-sentence cutoff (a single unsplit generation is capped at ~170 s of audio)",
            )
            custom_text_info = gr.Textbox(
                label="",
                show_label=False,
                interactive=False,
                max_lines=1,
                container=False,
            )
            custom_speaker = gr.Dropdown(
                label="Speaker",
                choices=shared.SPEAKER_CHOICES,
                value=shared.SPEAKER_CHOICES[0],
            )
            custom_prosody = gr.Dropdown(
                label="Style Preset",
                choices=voice_helpers.get_prosody_choices(),
                value=NONE_CHOICE,
                info=(
                    "Select a preset to fill the instruction field, or type "
                    "your own below, then save it under 'My prosody presets'."
                ),
            )
            custom_instruct = gr.Textbox(
                label="Style Instruction (optional)",
                placeholder="e.g., 'Speak with enthusiasm' or 'Read slowly and clearly'",
                lines=1,
            )
            custom_preset = gr.Dropdown(
                label="Preset", choices=shared.get_presets(), value=NONE_CHOICE
            )

            with gr.Accordion("My prosody presets", open=False):
                gr.Markdown(
                    "Save the current Style Instruction as a named preset, or "
                    "delete presets you created. Built-in presets can't be "
                    "changed. The preset saves exactly what's in the Style "
                    "Instruction box — including any preset text appended via "
                    "the Style Preset dropdown."
                )
                prosody_preset_name = gr.Textbox(
                    label="Preset name",
                    placeholder="e.g., 'storyteller'",
                    info=(
                        "Letters, numbers, dashes, underscores, and dots only. "
                        f"Max {core_config.PROSODY_NAME_MAX_LEN} characters."
                    ),
                )
                with gr.Row():
                    prosody_preset_save_btn = gr.Button(
                        "Save as preset", variant="secondary", size="sm"
                    )
                    prosody_preset_delete_btn = gr.Button(
                        "Delete preset", variant="stop", size="sm"
                    )
                prosody_preset_save_status = gr.Textbox(
                    label="",
                    show_label=False,
                    interactive=False,
                    max_lines=2,
                    container=False,
                )
                prosody_preset_delete_status = gr.Textbox(
                    label="",
                    show_label=False,
                    interactive=False,
                    max_lines=2,
                    container=False,
                )
                prosody_preset_delete_dropdown = gr.Dropdown(
                    label="Preset to delete",
                    info="Only presets you created appear here.",
                    choices=voice_helpers.get_user_prosody_choices(),
                    value=NONE_CHOICE,
                )
                # sr-only announcer — never visible=False (Gradio 6 drops it
                # from the DOM); mirrors both status boxes for SR users.
                prosody_preset_announcer = gr.HTML(generation._announce_status(""))
                prosody_preset_save_state = gr.State(dict(_PROSODY_DISARMED_STATE))
                prosody_preset_delete_state = gr.State(dict(_PROSODY_DISARMED_STATE))

        with gr.Column(scale=1):
            custom_ctrls = generation._build_common_controls()

    custom_btns = generation._build_generate_buttons_and_output("custom")
    gen_guard_state = gr.State(_new_gen_guard_state())

    # Prosody preset save/delete. The two apply dropdowns must be INPUTS —
    # the conditional-reset rule reads each dropdown's own value; the delete
    # dropdown is an OUTPUT of both flows (choices refresh on every success).
    prosody_preset_save_btn.click(
        fn=_on_save_prosody_preset,
        inputs=[
            prosody_preset_save_state,
            prosody_preset_name,
            custom_instruct,
            custom_prosody,
            design_prosody,
        ],
        outputs=[
            prosody_preset_save_state,
            prosody_preset_save_btn,
            prosody_preset_save_status,
            prosody_preset_announcer,
            custom_prosody,
            design_prosody,
            prosody_preset_delete_dropdown,
        ],
    )
    prosody_preset_delete_btn.click(
        fn=_on_delete_prosody_preset,
        inputs=[
            prosody_preset_delete_state,
            prosody_preset_delete_dropdown,
            custom_prosody,
            design_prosody,
        ],
        outputs=[
            prosody_preset_delete_state,
            prosody_preset_delete_btn,
            prosody_preset_delete_status,
            prosody_preset_announcer,
            custom_prosody,
            design_prosody,
            prosody_preset_delete_dropdown,
        ],
    )

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
        progress=custom_btns["progress"],
        progress_timer=progress_timer,
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
    return (
        custom_model_indicator,
        custom_chain,
        custom_ctrls["seed"],
        custom_preset,
    )
