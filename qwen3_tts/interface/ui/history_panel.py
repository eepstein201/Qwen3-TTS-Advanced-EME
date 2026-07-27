#!/usr/bin/env python3
"""Recent Generations panel event handlers.

The history panel is a ``gr.Dataframe`` rendered below the mode tabs in
``_facade.build_ui()``. This module owns its click routing and the two-step
Clear All confirm; the panel's components and JS wiring stay in the facade.

Collaborators are imported module-style (``shared.get_history_data()``, not
``get_history_data()``) so tests can patch them at their definition site.
"""

import logging
import os
import shutil
import tempfile

import gradio as gr

from qwen3_tts.core import config as core_config
from qwen3_tts.interface.ui import shared
from qwen3_tts.interface.ui.components import StatusBanner, confirm_step

logger = logging.getLogger("tts.ui")

# Column indices in the Recent Generations dataframe. Routing in
# on_history_select keys off evt.index[1] (the clicked column).
HISTORY_COL_TEXT_PREVIEW = 2  # "Text Preview" — click copies the transcript
HISTORY_COL_DELETE = 5  # "Remove" (✕) — click removes the row (list-only)


def extract_seed_from_history(evt: gr.SelectData, history_list):
    """Extract seed value from clicked history row for reuse.

    Returns the seed as a string (for Textbox), or "" if unavailable.
    """
    if not (isinstance(history_list, list) and history_list):
        return ""
    if not (
        hasattr(evt, "index")
        and isinstance(evt.index, (list, tuple))
        and len(evt.index) >= 1
        and 0 <= evt.index[0] < len(history_list)
    ):
        return ""
    seed = history_list[evt.index[0]].get("seed")
    return str(seed) if seed is not None else ""


def on_history_select(evt: gr.SelectData, history_list):
    """Handle click on a history row — column-aware router.

    Routes by ``evt.index[1]`` (the clicked column):
      - HISTORY_COL_TEXT_PREVIEW (2): copy the full transcript to the clipboard
        via the copy .then(js=...) chain (payload action ``'copy'``). No audio
        replay, no seed change (copy-only).
      - HISTORY_COL_DELETE (5): remove the row from history (list-only; the
        .wav/.json files on disk are untouched) and re-render the table.
      - any other column, or a legacy ``[row]``-only event: today's behavior —
        load the row's audio into the WaveSurfer player and broadcast its seed
        to all three tab seed textboxes.

    Returns an 8-tuple mapped to outputs:
      [history_audio_url, clone_seed, design_seed, custom_seed,
       history_df, history_state, history_select_payload, history_status_html]

    Defense-in-depth: validates path against safe roots and copies to tempdir
    so Gradio can always serve it (tempdir is always in allowed_paths).
    """
    update = gr.update
    replay_payload = {"action": "replay"}

    if not (isinstance(history_list, list) and history_list):
        return None, "", "", "", update(), [], replay_payload, update()
    if not (
        hasattr(evt, "index")
        and isinstance(evt.index, (list, tuple))
        and len(evt.index) >= 1
        and 0 <= evt.index[0] < len(history_list)
    ):
        return None, "", "", "", update(), list(history_list), replay_payload, update()

    row = evt.index[0]
    col = evt.index[1] if len(evt.index) >= 2 else None
    entry = history_list[row]

    # --- Remove column: delete the row (list-only; disk files untouched) ---
    if col == HISTORY_COL_DELETE:
        with shared.history_lock:
            new_list = shared.remove_history_row(history_list, row)
        # Clear audio so the player doesn't keep replaying the removed entry.
        # NOTE: must be None, not "" — gr.Audio.postprocess turns "" into a
        # FileData(path="") whose abspath is the CWD, and Gradio's
        # move_files_to_cache then tries hash_file(CWD) → IsADirectoryError,
        # which discards the whole handler output (row never removed, no
        # status). None short-circuits postprocess (returns None) safely.
        return (
            None,
            update(),
            update(),
            update(),
            shared.get_history_data(new_list),
            new_list,
            {"action": "delete"},
            StatusBanner().render("Entry removed.", "info"),
        )

    seed_str = extract_seed_from_history(evt, history_list)

    # --- Text Preview column: copy full transcript to clipboard (copy-only) ---
    if col == HISTORY_COL_TEXT_PREVIEW:
        full_text = entry.get("full_text") or entry.get("text", "")
        payload = {
            "action": "copy",
            "text": full_text,
        }
        # Optimistic visible "Copied" banner. Gradio's fn-return→output model
        # means a JS return value cannot drive this output, so the status is set
        # here and the clipboard write happens as a JS side-effect in the chained
        # .then(get_copy_transcript_js). (127.0.0.1 is a secure context, so the
        # clipboard write essentially always succeeds.)
        return (
            update(),
            update(),
            update(),
            update(),
            update(),
            update(),
            payload,
            StatusBanner().render("Copied transcript to clipboard.", "success"),
        )

    # --- Default: replay audio + broadcast seed (today's behavior) ---
    path = entry.get("path", "")
    if not path:
        return None, seed_str, seed_str, seed_str, update(), update(), replay_payload, update()
    resolved = os.path.realpath(path)
    # Containment check: only serve files from known-safe directories
    config = core_config.load_config()
    output_dir = shared._resolve_output_dir(config)
    safe_roots = {
        os.path.realpath(tempfile.gettempdir()),
        os.path.realpath(os.path.expanduser("~/Downloads")),
        output_dir,
    }
    if not any(resolved == r or resolved.startswith(r + os.sep) for r in safe_roots):
        return None, seed_str, seed_str, seed_str, update(), update(), replay_payload, update()
    if not os.path.exists(resolved):
        return None, seed_str, seed_str, seed_str, update(), update(), replay_payload, update()
    # Copy to temp for Gradio compatibility (tempdir always in allowed_paths)
    temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(resolved))
    if not os.path.exists(temp_path):
        shutil.copy2(resolved, temp_path)
    return temp_path, seed_str, seed_str, seed_str, update(), update(), replay_payload, update()


def on_clear_history_click(clear_state, history_list):
    """Two-step confirm to clear the Recent Generations list (list-only).

    First click arms the button (status: "Click again within 5s…"); second
    click within the timeout clears ``history_state`` to ``[]`` and re-renders
    the table. Disk files (``.wav``/``.json`` sidecars) are never touched, so
    entries re-appear after an app restart.

    Returns a 7-tuple mapped to:
      [clear_history_confirm_state, clear_all_btn, history_df, history_state,
       history_audio_url, history_status_html, history_select_payload]
    The payload carries action "clear" on confirm so the shared
    get_clear_player_js chain also resets the waveform.
    """
    if not isinstance(clear_state, dict):
        clear_state = {"armed": False, "ts": 0.0}
    new_state, btn_update, confirmed = confirm_step(
        clear_state,
        "Confirm Clear All? (click again)",
        "Clear All",
    )
    if not confirmed:
        return (
            new_state,
            btn_update,
            gr.update(),  # history_df unchanged
            gr.update(),  # history_state unchanged
            gr.update(),  # audio unchanged
            StatusBanner().render("Click again within 5s to clear all generations.", "warning"),
            {"action": "replay"},  # no waveform clear on arm
        )
    with shared.history_lock:
        new_list = shared.clear_history(history_list)
    return (
        new_state,
        btn_update,
        shared.get_history_data(new_list),  # empty rows
        new_list,  # []
        None,  # clear player (None is safe; "" crashes Audio postprocess)
        StatusBanner().render("Recent generations cleared.", "success"),
        {"action": "clear"},  # triggers get_clear_player_js
    )
