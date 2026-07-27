#!/usr/bin/env python3
"""Create Voice / Manage Voices / Manage Models tab builders for the Gradio UI.

These tabs own voice-prompt CRUD and model load/unload wiring, including the
two-step ConfirmButton flows for the destructive delete and unload actions.
Collaborators are imported module-style (``model_management.get_model_table_data()``,
not ``get_model_table_data()``) so tests can patch them at their definition site
and so the handlers can never accidentally resolve a name from the wrong module.
"""

import time

import gradio as gr

from qwen3_tts.interface.ui import model_management, shared, voice_management
from qwen3_tts.interface.ui.components import ConfirmButton


def _build_create_voice_tab(clone_prompt):
    """Build Create Voice tab components and wiring."""
    gr.Markdown("Create a new voice clone from reference audio.")

    with gr.Row():
        with gr.Column(scale=2):
            create_audio = gr.Audio(
                label="Reference Audio",
                type="filepath",
                sources=["upload", "microphone"],
            )
            create_transcript = gr.Textbox(
                label="Transcript",
                placeholder="Enter the exact words spoken in the audio...",
                lines=3,
            )
            with gr.Row():
                auto_transcribe_btn = gr.Button("Auto-Transcribe", size="sm")

        with gr.Column(scale=1):
            create_name = gr.Textbox(
                label="Voice Name",
                placeholder="e.g., my_voice",
                info="Will create my_voice.pt + .wav + .txt",
            )
            create_no_transcript = gr.Checkbox(
                label="Speaker embedding only",
                value=False,
                info="Create without transcript (x-vector only, lower fidelity)",
            )
            create_btn = gr.Button("Create Voice Prompt", variant="primary")
            create_status = gr.Textbox(label="Status", interactive=False)
            voice_list = gr.Dropdown(
                label="Available Voices",
                choices=shared.get_voice_prompts(),
                interactive=False,
            )

    auto_transcribe_btn.click(
        fn=voice_management.auto_transcribe_audio,
        inputs=[create_audio],
        outputs=[create_transcript],
    )
    create_btn.click(
        fn=voice_management.create_voice_prompt,
        inputs=[create_audio, create_transcript, create_name, create_no_transcript],
        outputs=[create_status, voice_list, clone_prompt],
    )


def _build_manage_voices_tab(clone_prompt):
    """Build Manage Voices tab components and wiring."""
    gr.Markdown("View, preview, rename, and delete voice prompts.")

    with gr.Row():
        with gr.Column(scale=2):
            manage_table = gr.Dataframe(
                headers=["Name", "Format", "Default"],
                value=voice_management.get_prompt_table_data(),
                interactive=False,
                wrap=True,
            )
            manage_refresh_btn = gr.Button("Refresh List", size="sm")

        with gr.Column(scale=1):
            manage_preview_audio = gr.Audio(label="Preview", visible=True)
            manage_selected = gr.Textbox(
                label="Selected Voice", interactive=False, max_lines=1, container=True
            )
            manage_new_name = gr.Textbox(
                label="New Name (for rename)",
                placeholder="Enter new name...",
                max_lines=1,
            )
            with gr.Row():
                manage_preview_btn = gr.Button("Preview", size="sm", interactive=False)
                manage_default_btn = gr.Button(
                    "Set Default", size="sm", interactive=False
                )
            with gr.Row():
                manage_rename_btn = gr.Button(
                    "Rename", size="sm", variant="secondary", interactive=False
                )
                manage_delete_btn = gr.Button(
                    "Delete", size="sm", variant="stop", interactive=False
                )
            delete_confirm_state = gr.State({"armed": False, "ts": 0.0})
            manage_status = gr.Textbox(
                label="",
                show_label=False,
                interactive=False,
                max_lines=2,
                container=False,
            )

    def on_table_select(evt: gr.SelectData, table_data):
        active = gr.update(interactive=True)
        inactive = gr.update(interactive=False)
        try:
            row_idx = evt.index[0]
            if hasattr(table_data, "iloc"):
                if row_idx < len(table_data):
                    return (
                        str(table_data.iloc[row_idx, 0]),
                        active,
                        active,
                        active,
                        active,
                    )
            elif table_data and row_idx < len(table_data):
                return table_data[row_idx][0], active, active, active, active
        except (IndexError, TypeError, KeyError):
            pass
        return "", inactive, inactive, inactive, inactive

    manage_table.select(
        fn=on_table_select,
        inputs=[manage_table],
        outputs=[
            manage_selected,
            manage_preview_btn,
            manage_default_btn,
            manage_rename_btn,
            manage_delete_btn,
        ],
    )
    manage_refresh_btn.click(
        fn=voice_management.get_prompt_table_data, outputs=[manage_table]
    )
    manage_preview_btn.click(
        fn=voice_management.preview_voice,
        inputs=[manage_selected],
        outputs=[manage_preview_audio],
    )
    manage_default_btn.click(
        fn=voice_management.set_voice_default,
        inputs=[manage_selected],
        outputs=[manage_status, manage_table],
    )
    manage_rename_btn.click(
        fn=voice_management.rename_voice,
        inputs=[manage_selected, manage_new_name],
        outputs=[manage_status, manage_table, clone_prompt],
    )

    # ConfirmButton for delete voice action
    delete_confirm_btn = ConfirmButton(
        arm_label="Confirm Delete? (click again)",
        original_label="Delete",
        timeout_s=5.0,
        status_message="Click again within 5s to confirm deletion.",
    )

    def on_delete_click(state, selected):
        # First click: show metadata
        if not state.get("armed", False):
            metadata = shared.get_voice_metadata(selected)
            if "error" in metadata:
                return (
                    state,
                    gr.update(),
                    f"Error: {metadata['error']}",
                    gr.update(),
                    gr.update(),
                )

            duration = metadata.get("duration", "N/A")
            formats = ", ".join(metadata.get("formats", []))
            size_mb = metadata.get("size_mb", "N/A")
            created = metadata.get("created")

            # Check if recently created (<5 minutes)
            recent_warning = ""
            if created:
                age_seconds = time.time() - created
                if age_seconds < 300:
                    recent_warning = "\n⚠️ Recently created!"

            banner_msg = (
                f"Delete '{selected}'?\n"
                f"Duration: {duration}\n"
                f"Format: {formats}\n"
                f"Size: {size_mb} MB"
                f"{recent_warning}"
            )

            new_state = delete_confirm_btn.click(state)
            return (
                new_state,
                gr.update(value="Confirm Delete? (click again)"),
                banner_msg,
                gr.update(),
                gr.update(),
            )

        # Second click: proceed with deletion
        new_state, btn_update, status_update, confirmed = delete_confirm_btn.click(
            state
        )
        if not confirmed:
            return new_state, btn_update, status_update, gr.update(), gr.update()
        status, table, prompt = voice_management.delete_voice(selected)
        return new_state, btn_update, status, table, prompt

    manage_delete_btn.click(
        fn=on_delete_click,
        inputs=[delete_confirm_state, manage_selected],
        outputs=[
            delete_confirm_state,
            manage_delete_btn,
            manage_status,
            manage_table,
            clone_prompt,
        ],
    )


def _build_manage_models_tab(
    status_html, clone_model_indicator, design_model_indicator, custom_model_indicator
):
    """Build Manage Models tab components and wiring."""
    gr.Markdown("Load/unload models, configure startup defaults, and audio loader.")

    with gr.Row():
        with gr.Column(scale=2):
            model_table = gr.Dataframe(
                headers=["Model", "Status", "Memory", "Startup"],
                value=model_management.get_model_table_data(),
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
            unload_confirm_state = gr.State({"armed": False, "ts": 0.0})
            model_manage_status = gr.Textbox(
                label="",
                show_label=False,
                interactive=False,
                max_lines=2,
                container=False,
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
                label="",
                show_label=False,
                interactive=False,
                max_lines=1,
                container=False,
            )

            gr.Markdown("### Audio Loader")
            audio_loader_select = gr.Dropdown(
                label="Audio Loader",
                choices=["torchaudio", "librosa"],
                value=model_management.get_audio_loader_setting(),
                info="torchaudio: faster C++ | librosa: broader format support",
            )
            audio_loader_save_btn = gr.Button("Save Audio Loader", size="sm")
            audio_loader_status = gr.Textbox(
                label="",
                show_label=False,
                interactive=False,
                max_lines=1,
                container=False,
            )

    model_refresh_btn.click(
        fn=model_management.get_model_table_data, outputs=model_table
    )

    model_load_btn.click(
        fn=lambda mt: model_management.toggle_model(mt, "load"),
        inputs=[model_type_select],
        outputs=[model_manage_status, model_table, status_html],
    ).then(
        fn=lambda mt: (
            model_management.get_model_status_html("clone") if mt == "clone" else gr.update(),
            model_management.get_model_status_html("design") if mt == "design" else gr.update(),
            model_management.get_model_status_html("custom") if mt == "custom" else gr.update(),
        ),
        inputs=[model_type_select],
        outputs=[clone_model_indicator, design_model_indicator, custom_model_indicator],
    )

    # ConfirmButton for unload model action
    unload_confirm_btn = ConfirmButton(
        arm_label="Confirm Unload? (click again)",
        original_label="Unload",
        timeout_s=5.0,
        status_message="Click again within 5s to confirm unload.",
    )

    def on_unload_click(state, mt):
        # First click: show metadata. Resolve the table data through
        # model_management — .shared does NOT provide get_model_table_data,
        # and importing it from there was the source of a past crash.
        if not state.get("armed", False):
            models = model_management.get_model_table_data()
            model = next((m for m in models if m[0] == mt), None)
            if not model:
                return (
                    state,
                    gr.update(),
                    f"Model '{mt}' not found",
                    gr.update(),
                    gr.update(),
                )

            model_type = model[0]
            memory_mb = model[2] if len(model) > 2 else "N/A"
            startup = model[3] if len(model) > 3 else "unknown"

            # Warning if model is set to load at startup (table emits "Yes"/"No")
            startup_warning = ""
            if startup == "Yes":
                startup_warning = (
                    "\n⚠️ Loaded at startup - will reload on server restart!"
                )

            banner_msg = (
                f"Unload {model_type.upper()} model?\n"
                f"Current memory: {memory_mb}\n"
                f"Startup config: {startup}"
                f"{startup_warning}"
            )

            new_state = unload_confirm_btn.click(state)
            return (
                new_state,
                gr.update(value="Confirm Unload? (click again)"),
                banner_msg,
                gr.update(),
                gr.update(),
            )

        # Second click: proceed with unload
        new_state, btn_update, status_update, confirmed = unload_confirm_btn.click(
            state
        )
        if not confirmed:
            return new_state, btn_update, status_update, gr.update(), gr.update()
        status, table, status_h = model_management.toggle_model(mt, "unload")
        return new_state, btn_update, status, table, status_h

    model_unload_btn.click(
        fn=on_unload_click,
        inputs=[unload_confirm_state, model_type_select],
        outputs=[
            unload_confirm_state,
            model_unload_btn,
            model_manage_status,
            model_table,
            status_html,
        ],
    ).then(
        fn=lambda mt: (
            model_management.get_model_status_html("clone") if mt == "clone" else gr.update(),
            model_management.get_model_status_html("design") if mt == "design" else gr.update(),
            model_management.get_model_status_html("custom") if mt == "custom" else gr.update(),
        ),
        inputs=[model_type_select],
        outputs=[clone_model_indicator, design_model_indicator, custom_model_indicator],
    )

    asr_load_btn.click(
        fn=lambda: model_management.toggle_asr("load"),
        outputs=[model_manage_status, status_html],
    )
    asr_unload_btn.click(
        fn=lambda: model_management.toggle_asr("unload"),
        outputs=[model_manage_status, status_html],
    )
    startup_save_btn.click(
        fn=model_management.update_startup_defaults,
        inputs=[startup_clone, startup_design, startup_custom],
        outputs=startup_status,
    )
    audio_loader_save_btn.click(
        fn=model_management.set_audio_loader_setting,
        inputs=[audio_loader_select],
        outputs=audio_loader_status,
    )
