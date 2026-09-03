#!/usr/bin/env python3
"""Facade module for Qwen3-TTS Gradio UI.

This module contains the main entry points:
- build_ui(): Build the Gradio interface
- main(): CLI entry point
- stop_server(): Stop the TTS server
- _find_available_port(): Find an available port

The tab builders and the Recent Generations handlers live in sibling modules
(tabs_generation, tabs_management, history_panel) and are re-exported below so
``from qwen3_tts.interface.ui._facade import X`` keeps working for every caller.
"""

import logging
import subprocess  # nosec B404  # hardcoded "pm2 stop <name>"; name comes from pm2's own jlist, not user input
import sys
import time

import gradio as gr

from qwen3_tts.core.config import (
    IN_COLAB,
    VALID_MLX_QUANTIZATIONS,
    VALID_MODEL_SIZES,
    load_config,
    pm2_owner_of_port,
)

# confirm_step is re-exported here as part of the UI confirm-pattern wiring
# contract verified by tests/test_ui_confirm_patterns.py.
from qwen3_tts.interface.ui.components import (  # noqa: F401
    ConfirmButton,
    StatusBanner,
    confirm_step,
)

# Recent Generations handlers — re-exported for callers and tests.
from qwen3_tts.interface.ui.history_panel import (  # noqa: F401
    HISTORY_COL_DELETE,
    HISTORY_COL_TEXT_PREVIEW,
    extract_seed_from_history,
    on_clear_history_click,
    on_history_select,
)
from qwen3_tts.interface.ui.model_management import (
    get_all_model_status_html,
    get_model_status_html,
    get_model_table_data,
)

# Import from sibling modules
from qwen3_tts.interface.ui.shared import (
    apply_model_settings,
    format_status_display,
    get_current_model_settings,
)

# Tab builders — re-exported for callers and tests.
from qwen3_tts.interface.ui.tabs_generation import (  # noqa: F401
    _build_clone_tab,
    _build_custom_tab,
    _build_design_tab,
    _sanitize_voice_name,
)
from qwen3_tts.interface.ui.tabs_management import (  # noqa: F401
    _build_create_voice_tab,
    _build_manage_models_tab,
    _build_manage_voices_tab,
)
from qwen3_tts.interface.wavesurfer_js import (
    get_clear_player_js,
    get_copy_transcript_js,
    get_load_into_player_js,
    get_player_html,
    get_script_reexecutor_fn,
    get_streaming_player_js,
    get_wavesurfer_loader_js,
)
from qwen3_tts.server.client import TTSClient

logger = logging.getLogger("tts.ui")

# Human description for each model size, shown in the Model Settings info
# tooltip. Keys must cover every entry in VALID_MODEL_SIZES (asserted in
# tests/test_ui_facade_model_sizes.py) so the tooltip text can't go stale when
# the canonical size list changes.
_MODEL_SIZE_DESCRIPTIONS = {
    "1.7B": "higher quality",
    "0.6B": "~40% faster, lower memory",
}

# How often the UI polls the server for the status bar and the per-mode model
# badges. Both are driven by the same gr.Timer so a tick costs two requests.
STATUS_POLL_SECONDS = 5


def _stop_server_via_pm2(name, client):
    """Stop a PM2-managed server via `pm2 stop <name>`.

    Mirrors `qwen3_tts.cli_server._stop_via_pm2`: POSTing /shutdown (or
    any other direct kill) is indistinguishable to PM2 from a crash, so
    its `autorestart: true` (ecosystem.config.cjs) respawns the server
    within `restart_delay` -- before this function's own poll loop could
    ever observe it as stopped.
    """
    logger.info("Server is managed by PM2 (app '%s') — using `pm2 stop %s`.", name, name)
    try:
        result = subprocess.run(  # nosec B603, B607
            ["pm2", "stop", name],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("pm2 stop failed to run: %s", e)
        return format_status_display()

    if result.returncode != 0:
        logger.warning(
            "pm2 stop failed: %s", (result.stderr or result.stdout).strip()
        )
        return format_status_display()

    for _ in range(10):
        time.sleep(0.5)
        if not client.is_server_running():
            break

    return format_status_display()


def stop_server():
    """Stop the TTS server from the UI."""
    config = load_config()
    port = config.get("server", {}).get("port", 5123)
    client = TTSClient()

    # A PM2-supervised server must be stopped through PM2 -- see
    # _stop_server_via_pm2.
    pm2_name = pm2_owner_of_port(port)
    if pm2_name:
        return _stop_server_via_pm2(pm2_name, client)

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

    bind_addr = "0.0.0.0" if IN_COLAB else "127.0.0.1"  # nosec B104  # Colab only
    for offset in range(max_tries):
        port = preferred + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((bind_addr, port))
                return port
        except OSError:
            continue
    return None


def _load_initial_history(current_history):
    """Load persistent history from disk on UI startup.

    ``demo.load()`` and a Generate button's chain are separate Gradio events
    with no guaranteed delivery order. If a generation completes and pushes a
    fresh entry into ``history_state``/``history_df`` before this disk scan's
    result is delivered to the browser, an unconditional overwrite here would
    clobber that fresh entry with the scan's now-stale view (the scan reflects
    disk state as of whenever *this* call started, which can predate the new
    generation's sidecar file). Comparing top-entry timestamps and keeping
    whichever is newer makes the outcome independent of delivery order.

    Args:
        current_history: The browser session's current ``history_state`` value
            (empty list on a fresh page load, the input Gradio wires here).

    Returns:
        tuple: (clone_status_html, design_status_html, custom_status_html,
                history_list, history_df_data)
    """
    from qwen3_tts.interface.ui.shared import (
        get_history_data,
        load_history_from_disk_for_config,
    )

    config = load_config()
    disk_history = load_history_from_disk_for_config(config)

    history = disk_history
    if (
        isinstance(current_history, list)
        and current_history
        and disk_history
        and current_history[0].get("timestamp", 0) > disk_history[0].get("timestamp", 0)
    ):
        history = current_history

    return (
        get_model_status_html("clone"),
        get_model_status_html("design"),
        get_model_status_html("custom"),
        history,
        get_history_data(history),
    )


def build_ui():
    """Build the Gradio interface."""
    # Covers installs that skip install.sh (Colab, Docker, pip-only).
    # Idempotent — exist_ok=True.
    from qwen3_tts.interface.ui.shared import ensure_history_dirs

    ensure_history_dirs(load_config())

    with gr.Blocks(title="Qwen3-TTS Web Interface") as demo:
        gr.Markdown("# Qwen3-TTS Web Interface")

        # Inject WaveSurfer.js and StreamingPlayer class
        gr.HTML(
            value=get_wavesurfer_loader_js()
            + "<script type='module'>"
            + get_streaming_player_js()
            + "</script>"
        )

        # Status bar
        status_html = gr.HTML(value=format_status_display())
        with gr.Row():
            refresh_btn = gr.Button("Refresh Status", size="sm")
            stop_btn = gr.Button("Stop Server", size="sm", variant="stop")
        refresh_btn.click(fn=format_status_display, outputs=status_html)
        status_timer = (
            gr.Timer(value=STATUS_POLL_SECONDS) if hasattr(gr, "Timer") else None
        )
        if status_timer is not None:
            status_timer.tick(fn=format_status_display, outputs=status_html)
        stop_btn.click(fn=stop_server, outputs=status_html)

        # Model Settings (MLX-first architecture)
        current_size, current_quant, current_backend = get_current_model_settings()
        with gr.Accordion("Model Settings", open=False):
            gr.Markdown(
                "Change model size or quantization. Applying reloads any loaded "
                "models immediately — this can take a few minutes."
            )
            with gr.Row():
                model_size_dropdown = gr.Dropdown(
                    label="Model Size",
                    choices=list(VALID_MODEL_SIZES),
                    value=current_size,
                    info=" | ".join(
                        f"{size}: {_MODEL_SIZE_DESCRIPTIONS[size]}"
                        for size in VALID_MODEL_SIZES
                    ),
                )
                mlx_quant_dropdown = gr.Dropdown(
                    label="MLX Quantization",
                    choices=list(VALID_MLX_QUANTIZATIONS),
                    value=current_quant,
                    info="4bit/5bit/6bit: progressively higher quality, more memory | 8bit: balanced (default) | bf16: highest quality, largest",
                    visible=(current_backend == "mlx"),
                )
            with gr.Row():
                apply_settings_btn = gr.Button(
                    "Apply Settings", variant="secondary", size="sm"
                )
                settings_status = gr.Textbox(
                    label="",
                    show_label=False,
                    interactive=False,
                    max_lines=1,
                    container=False,
                    scale=3,
                )
            apply_settings_btn.click(
                fn=apply_model_settings,
                inputs=[model_size_dropdown, mlx_quant_dropdown],
                outputs=[settings_status, status_html],
            )

        # Per-session history state (shared across tabs)
        history_state = gr.State([])

        # Tabs for different modes
        with gr.Tabs():
            with gr.Tab("Clone Mode"):
                clone_prompt, clone_model_indicator, clone_chain, clone_seed = (
                    _build_clone_tab(status_html, history_state)
                )
            with gr.Tab("Design Mode"):
                design_model_indicator, design_chain, design_seed = _build_design_tab(
                    status_html, history_state, clone_prompt
                )
            with gr.Tab("Custom Mode"):
                custom_model_indicator, custom_chain, custom_seed = _build_custom_tab(
                    status_html, history_state
                )

            with gr.Tab("Create Voice"):
                _build_create_voice_tab(clone_prompt)
            with gr.Tab("Manage Voices"):
                _build_manage_voices_tab(clone_prompt)
            with gr.Tab("Manage Models"):
                model_table = _build_manage_models_tab(
                    status_html,
                    clone_model_indicator,
                    design_model_indicator,
                    custom_model_indicator,
                )

        # Keep the per-mode model badges fresh.
        #
        # These used to be refreshed by a ``select`` listener on each ``gr.Tab``.
        # That is not safe: with a ``select`` listener attached to a ``gr.Tab``,
        # Gradio 6.14+ recurses infinitely inside the Dataframe frontend
        # ("RangeError: Maximum call stack size exceeded" in
        # ``Object.get [as groupedColumnMode]``) as soon as a tab containing a
        # ``gr.Dataframe`` is opened afterwards, which kills the whole page.
        # The crash follows the listener, not its outputs, and ``gr.Tabs.select``
        # never fires, so polling on the existing status timer is the only
        # wiring that keeps the badges live. See tests/test_ui_tab_select_wiring.py.
        if status_timer is not None:
            status_timer.tick(
                fn=get_all_model_status_html,
                outputs=[
                    clone_model_indicator,
                    design_model_indicator,
                    custom_model_indicator,
                ],
            )
            # Keep the Manage Models table fresh too. The Load/Unload handlers
            # already return refreshed table data, but that Dataframe re-render
            # is occasionally dropped by the Gradio frontend; polling on the
            # timer self-heals it within one tick (see I4).
            status_timer.tick(
                fn=get_model_table_data,
                outputs=model_table,
            )

        # History panel below tabs (renders after tabs in the page layout)
        gr.Markdown("### Recent Generations")
        gr.Markdown(
            "*Click a row to replay it and reuse its seed · ✕ permanently "
            "deletes the row's file · ⭳ copies it to Manual Downloads*"
        )
        with gr.Row():
            clear_all_btn = gr.Button("Clear All", size="sm", variant="stop")
        # Two-step confirm state for Clear All (mirrors voice-delete wiring).
        clear_history_confirm_state = gr.State({"armed": False, "ts": 0.0})
        # Path-keyed so a generation arriving between the two clicks (which
        # shifts every row index) can't redirect the per-row Remove confirm at
        # another row. ``ts`` bounds the arm to DELETE_CONFIRM_TIMEOUT_S.
        delete_confirm_state = gr.State({"armed_path": None, "ts": 0.0})
        # Mirror state for the per-row Download action: arms only on a name
        # collision in Manual Downloads, so the second click overwrites.
        download_confirm_state = gr.State({"armed_path": None, "ts": 0.0})
        history_df = gr.Dataframe(
            headers=[
                "Time",
                "Mode",
                "Text Preview",
                "Seed",
                "Chunks",
                "Remove",
                "Download",
            ],
            value=[],
            interactive=False,
            wrap=True,
        )
        gr.HTML(value=get_player_html("history"))
        history_audio_url = gr.Audio(elem_classes=["gr-hidden"])
        # Hidden bridge for copy-to-clipboard: on_history_select writes the
        # action payload ({"action": "copy"|"delete"|"replay", ...}) here, and
        # get_copy_transcript_js reads it in the browser. Kept in the DOM
        # (gr-hidden, not visible=False) so the JS<->Python chain stays live.
        history_select_payload = gr.JSON(elem_classes=["gr-hidden"])
        # Visible status surface for "Copied" / "Entry removed." / clear-all
        # flashes. StatusBanner carries role=status + aria-live=polite, so it
        # is announced to screen readers AND visible to sighted users.
        history_status_html = gr.HTML(value=StatusBanner().render(""))

        history_df.select(
            fn=on_history_select,
            inputs=[history_state, delete_confirm_state, download_confirm_state],
            outputs=[
                history_audio_url,
                clone_seed,
                design_seed,
                custom_seed,
                history_df,
                history_state,
                history_select_payload,
                history_status_html,
                delete_confirm_state,
                download_confirm_state,
            ],
        ).then(
            fn=lambda x: x,
            js=get_load_into_player_js("history"),
            inputs=[history_audio_url],
            outputs=[history_audio_url],
        ).then(
            # Copy the transcript to the clipboard (JS side-effect) when
            # payload.action === 'copy'. The visible "Copied" status is set
            # optimistically by on_history_select; this .then only performs the
            # clipboard write. NOTE: fn=lambda p: p is required — a js-only
            # .then (fn=None) is NOT executed by Gradio 6.14. fn's return
            # (passthrough payload) is what writes the output; js is side-effect.
            fn=lambda p: p,
            js=get_copy_transcript_js(),
            inputs=[history_select_payload],
            outputs=[history_select_payload],
        ).then(
            # Clear the waveform when a row is removed (payload action
            # 'delete'); no-op for replay/copy. fn=lambda required (see above).
            fn=lambda p: p,
            js=get_clear_player_js("history"),
            inputs=[history_select_payload],
            outputs=[history_select_payload],
        )

        clear_all_btn.click(
            fn=on_clear_history_click,
            inputs=[clear_history_confirm_state, history_state],
            outputs=[
                clear_history_confirm_state,
                clear_all_btn,
                history_df,
                history_state,
                history_audio_url,
                history_status_html,
                history_select_payload,
            ],
        ).then(
            # Clear the waveform on confirmed Clear All (payload action 'clear').
            # fn=lambda required — a js-only .then is not executed by Gradio 6.14.
            fn=lambda p: p,
            js=get_clear_player_js("history"),
            inputs=[history_select_payload],
            outputs=[history_select_payload],
        )

        # Wire history_df updates from each tab's generation chain.
        #
        # Each chain's final step re-derives BOTH history_state and history_df
        # from disk rather than from the in-memory history_state. demo.load()'s
        # preload and a generation chain's refresh are independent Gradio events
        # with no guaranteed delivery order; deriving the table from
        # history_state let a stale list win and render an unrelated row
        # (test_13's render race). Re-reading disk makes the outcome
        # order-independent — the sidecar is written before this .then fires, so
        # the fresh entry is always present. Safe only because Remove
        # hard-deletes the file.
        from qwen3_tts.interface.ui import shared as _shared

        def _refresh_history(history_list):
            # Module-style call so unittest.mock.patch targets the definition
            # site (see CLAUDE.md on moved-module patch seams).
            return _shared.refresh_history_from_disk(history_list, load_config())

        for _chain in (clone_chain, design_chain, custom_chain):
            _chain.then(
                fn=_refresh_history,
                inputs=[history_state],
                outputs=[history_state, history_df],
            )

        demo.load(
            fn=_load_initial_history,
            js=get_script_reexecutor_fn(),
            inputs=[history_state],
            outputs=[
                clone_model_indicator,
                design_model_indicator,
                custom_model_indicator,
                history_state,
                history_df,
            ],
        )

        # Size/quantization lists in the tips text are derived from the
        # canonical constants so they can't drift from the actual choices.
        model_size_choices = "/".join(sorted(VALID_MODEL_SIZES))
        mlx_quant_choices = "/".join(VALID_MLX_QUANTIZATIONS)
        gr.Markdown(f"""
        ---
        **Tips:**
        - Start the TTS server first: `tts server start`
        - Models load at server startup per the startup config — load the others in **Manage Models**
        - Use **Model Settings** above to switch between model sizes ({model_size_choices}) and MLX quantizations ({mlx_quant_choices}); applying reloads loaded models now
        - Clone mode uses a voice prompt (.pt for PyTorch, .wav+.txt for MLX)
        - Design mode creates voices from text descriptions
        - Custom mode uses premium pre-trained speakers
        - Generations are saved to `~/Downloads/Qwen3-TTS Output/Automated Output/`; **Clear All** and the ✕ cell permanently delete those files, while ⭳ **copies** a file to `Manual Downloads` (it is not a browser download)
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
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Port to run on (default: {default_port})",
    )
    parser.add_argument("--share", action="store_true", help="Create public URL")
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )

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
