<!-- Generated: 2026-09-01 | Files scanned: interface/ui/ (5.1k LOC) | Token estimate: ~440 -->

# Frontend — Gradio Web UI (`interface/ui/`)

Gradio web UI, launched via `tts ui`. Pin gradio `!=6.14.*` (6.14.x recurses on Dataframe).

## Module map (LOC)
- **shared.py** 879 — collaborators (`get_presets` etc.); referenced module-style so `mock.patch` targets the definition site
- **generation.py** 726 — generation wiring (server calls via TTSClient)
- **_facade.py** 540 — `build_ui` / `main` / `stop_server` + re-exports (every moved name re-exported so `from _facade import X` still works)
- **voice_management.py** 513 — voice CRUD handlers; enforces `ensure_min_sample_rate` at write time
- **tabs_generation.py** 502 — Clone / Design / Custom tab builders
- **components.py** 484 — `ConfirmButton`, `confirm_step`, `ProgressIndicator`, `StatusBanner`, `status_badge`, `poll_model_loading_state`
- **tabs_management.py** 460 — Create Voice / Manage Voices / Manage Models
- **history_panel.py** 420 — Recent Generations (click routing + Clear All)
- **model_management.py** 367 — model CRUD handlers; ETA badge during load

## Recent fixes (#218, #195)
- **Confirm-flow repair** — Stop, Delete Voice, and Unload Model each use `ConfirmButton`/`confirm_step`'s two-step path correctly; a prior wiring gap let the second click no-op.
- **`tts ui --port`** honored end-to-end; low-rate voice-prompt warning now surfaces to browser users, not just server logs.
- **ETA badge** — model-load ETA is surfaced in the UI instead of discarded (`44f844b`).

## Critical constraints
- **NEVER** attach `select` to a `gr.Tab` → infinite Dataframe recursion on 6.14.x (kills Manage tabs). Model badges refreshed via shared `gr.Timer` (5 s, one `/models` call) instead.
- `gr.Audio` output: clear with `None`, **never** `""` (returns "" discards the whole handler result).
- `.then(fn=, js=)`: js-only `.then(fn=None, js=...)` never runs at runtime.

## HTTP only
UI never imports torch/mlx; all generation goes through the local server (`TTSClient`), including the long-running leaf-locked paths (create-voice-prompt, transcribe, unload-asr — see backend.md) which now use extended client timeouts.
