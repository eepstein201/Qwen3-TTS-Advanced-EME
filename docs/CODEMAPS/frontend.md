<!-- Generated: 2026-09-24 | Files scanned: 11 .py (6.2k LOC) | Token estimate: ~1650 -->

# Frontend — Gradio Web UI (`interface/ui/`)

Gradio web UI, launched via `tts ui`. Pin gradio `!=6.14.*` (6.14.x recurses on Dataframe).

## Tab tree (`_facade.py::build_ui()`)
```
gr.Blocks("Qwen3-TTS Web Interface")
├── status_html + Refresh Status / Stop Server      ← status_timer (gr.Timer, STATUS_POLL_SECONDS=5)
├── Accordion "Model Settings" (size / MLX quant → apply_model_settings)
├── gr.Tabs
│   ├── Clone Mode   (_build_clone_tab)  + Accordion "My generation presets" (T1b #327)
│   ├── Design Mode  (_build_design_tab) + "Description Builder", "Save as Voice Prompt"
│   ├── Custom Mode  (_build_custom_tab) + Accordion "My prosody presets" (T1 #322)
│   ├── Create Voice / Manage Voices / Manage Models   (tabs_management.py)
└── "Recent Generations" Dataframe [Time, Mode, Text Preview, Seed, Chunks, Remove, Download] + Clear All
```
**Timers:** `status_timer` (5 s) ticks `format_status_display`, `get_all_model_status_html` (one `/models` call → 3 mode badges) and `get_model_table_data`. `progress_timer` (`PROGRESS_POLL_SECONDS=2.0`, built `active=False`) is shared by the 3 generation tabs, armed/disarmed around a generation; each tab's tick polls authed `/generation-status` only while its `gen_guard_state` says generating (hard stop `PROGRESS_POLL_MAX_MINUTES=20`).

**State:** per-session UI state is `gr.State` — `history_state` (list), `gen_guard_state` per tab, confirm states (`clear_history_confirm_state`, path-keyed `delete_confirm_state` / `download_confirm_state`, `cancel_confirm_state`, preset save/delete states). Persistent data lives server-side or in config.json / the output dirs (see data.md).

## Module map (LOC)
- **shared.py** 1102 — collaborators (`get_presets` etc.), `format_status_display`, display formatters `fmt_duration`/`fmt_size`/`fmt_eta`/`fmt_memory_mb`, `empty_history_rows()`, `resolve_history_output_dir(config)`, `get_gradio_launch_kwargs(config, *, share=False)` (share-auth single source; also passes `css=theme.UI_CSS`); referenced module-style so `mock.patch` targets the definition site
- **tabs_generation.py** 1017 — `_build_{clone,design,custom}_tab(...)` + preset-builder handlers `_on_{save,delete}_{prosody,generation}_preset`
- **generation.py** 869 — generation wiring via `TTSClient`; `status_update(message, severity=None)` + `_with_status_severity(fn, index)` (severity at the wiring layer); progress-timer tick handlers
- **_facade.py** 661 — `build_ui()` / `main()` / `stop_server()` (PM2-aware, #248) + re-exports
- **tabs_management.py** 502 — `_build_create_voice_tab`, `_build_manage_voices_tab` (Rename + Delete are two-click confirms), `_build_manage_models_tab`
- **components.py** 494 — `ConfirmButton`, `confirm_step`, `ProgressIndicator`, `StatusBanner`, `status_badge(message, severity="info")`, `poll_model_loading_state`
- **voice_management.py** 489 — voice CRUD handlers; MLX create routes through the engine writer (0G #282)
- **history_panel.py** 436 — Recent Generations (click routing, Clear All; `DELETE_CONFIRM_TIMEOUT_S=5.0`)
- **model_management.py** 383 — model CRUD handlers; loading badge shows "(last load M:SS)", never an ETA (08c2d83)
- **theme.py** 90 — stdlib-only design tokens `TOKENS`, `SEVERITY_CLASS`, `var(token)`, `UI_CSS` (reaches Gradio 6 via `launch(css=)`, never `gr.Blocks()`)

## Share auth + allowed_paths (0E #276)
- Every launch site (`ui/_facade.py` `main()` AND `generate_server.build_ui_and_launch`) routes kwargs through `get_gradio_launch_kwargs(config, share=...)` (`shared.py`) — share ⇒ auth, fail-closed: `TTS_UI_USERNAME`+`TTS_UI_PASSWORD` when BOTH env vars are set, else generated (username printed once, password written as the sole line of a 0600 file `~/.config/qwen3-tts/.ui_share_credentials` — the password string never reaches any log sink); a half-set pair or a credential/write failure raises RuntimeError instead of launching unauthenticated
- `allowed_paths` narrowed to the history output root (`resolve_history_output_dir`, default `~/Downloads/Qwen3-TTS Output`) + the system tempdir — the old blanket `~/Downloads` entry handed a public URL the whole tree (and the legacy `output_directory` resolver would have made the narrowing a no-op)
- Colab forces share at both sites → the Colab path is authenticated too (notebook mirrors the helper inline)

## Voice create via engine writer (0G #282)
`voice_management.py`'s MLX create path calls engine `save_voice_prompt_mlx` DIRECTLY (module-style import from the definition site) — the same guarded writer as the server's `/create-voice-prompt` MLX route, including reference-path containment; the UI adds only the `gr.Error` mapping + preset/voice cache clears. All four create surfaces now converge on one writer.

## User prosody presets (T1 #322)
Custom tab's "My prosody presets" accordion — save/delete of user-defined prosody presets. Handlers `_on_save_prosody_preset` / `_on_delete_prosody_preset` (module-level): target-keyed two-step confirm (5 s window; mismatch/expired re-arm for the current target), conditional `value=` resets that read each dropdown's OWN input (custom/design dropdowns are handler INPUTS — that's why the wiring keeps them in `inputs=`), sr-only aria-live announcer on every branch. Data layer lives in `core/config/presets.py` (raw-base writers; factory names reserved). Guarded by `tests/test_prosody_preset_builder.py` (89 tests). Generation presets (Clone tab, T1b #327) follow the same pattern; their save/delete refresh the Preset dropdown on all three tabs (`tests/test_generation_preset_builder.py`).

## Critical constraints
- **NEVER** attach `select` to a `gr.Tab` → infinite Dataframe recursion on 6.14.x (kills Manage tabs). Model badges are refreshed by the status timer instead (`tests/test_ui_tab_select_wiring.py`).
- `gr.Audio` output: clear with `None`, **never** `""` (returns "" discards the whole handler result).
- `.then(fn=, js=)`: js-only `.then(fn=None, js=...)` never runs at runtime.

## HTTP only
UI never imports torch/mlx; all generation goes through the local server (`TTSClient`), including the long-running paths (create-voice-prompt on torch, transcribe, unload-asr — leaf-locked; create-voice-prompt on MLX calls the engine writer `save_voice_prompt_mlx` directly in-process — inference-free and unlocked, #236/#282 — see backend.md) which use extended client timeouts.
