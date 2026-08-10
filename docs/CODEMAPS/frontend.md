<!-- Generated: 2026-08-10 | Files scanned: interface/ui/ (4.8k LOC) | Token estimate: ~400 -->

# Frontend — Gradio Web UI (`interface/ui/`)

Gradio web UI, launched via `tts ui`. Pin gradio `!=6.14.*` (6.14.x recurses on Dataframe).

## Module map (LOC)
- **_facade.py** 534 — `build_ui` / `main` / `stop_server` + re-exports (every moved name re-exported so `from _facade import X` still works)
- **tabs_generation.py** 489 — Clone / Design / Custom tab builders
- **tabs_management.py** 438 — Create Voice / Manage Voices / Manage Models
- **generation.py** 693 — generation wiring (server calls via TTSClient)
- **voice_management.py** 416 · **model_management.py** 348 — CRUD handlers
- **history_panel.py** 416 — Recent Generations (click routing + Clear All)
- **shared.py** 803 — collaborators (`get_presets` etc.); referenced module-style so `mock.patch` targets the definition site
- **components.py** 484 — `ConfirmButton`, `confirm_step`, `ProgressIndicator`, `StatusBanner`, `status_badge`, `poll_model_loading_state`

## Critical constraints
- **NEVER** attach `select` to a `gr.Tab` → infinite Dataframe recursion on 6.14.x (kills Manage tabs). Model badges refreshed via shared `gr.Timer` (5 s, one `/models` call) instead.
- `gr.Audio` output: clear with `None`, **never** `""` (returns "" discards the whole handler result).
- `.then(fn=, js=)`: js-only `.then(fn=None, js=...)` never runs at runtime.

## HTTP only
UI never imports torch/mlx; all generation goes through the local server.
