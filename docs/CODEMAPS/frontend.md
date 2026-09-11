<!-- Generated: 2026-09-11 | Files scanned: interface/ui/ (5.2k LOC) | Token estimate: ~560 -->

# Frontend — Gradio Web UI (`interface/ui/`)

Gradio web UI, launched via `tts ui`. Pin gradio `!=6.14.*` (6.14.x recurses on Dataframe).

## Module map (LOC)
- **shared.py** 982 — collaborators (`get_presets` etc.) + `get_gradio_launch_kwargs` (share-auth single source, 0E #276); referenced module-style so `mock.patch` targets the definition site
- **generation.py** 712 — generation wiring (server calls via TTSClient; #260 pruned 14 dead lines)
- **_facade.py** 585 — `build_ui` / `main` / `stop_server` (PM2-aware via `pm2_owner_of_port`, #248) + re-exports (every moved name re-exported so `from _facade import X` still works)
- **tabs_generation.py** 502 — Clone / Design / Custom tab builders
- **components.py** 484 — `ConfirmButton`, `confirm_step`, `ProgressIndicator`, `StatusBanner`, `status_badge`, `poll_model_loading_state`
- **voice_management.py** 494 — voice CRUD handlers; MLX create routes through the engine writer (0G #282)
- **tabs_management.py** 460 — Create Voice / Manage Voices / Manage Models
- **history_panel.py** 420 — Recent Generations (click routing + Clear All)
- **model_management.py** 385 — model CRUD handlers; ETA badge during load

## Share auth + allowed_paths (0E #276)
- Every launch site (`ui/_facade.py` `main()` AND `generate_server.build_ui_and_launch`) routes kwargs through `get_gradio_launch_kwargs(config, share=...)` (`shared.py`) — share ⇒ auth, fail-closed: `TTS_UI_USERNAME`+`TTS_UI_PASSWORD` when BOTH env vars are set, else generated (username printed once, password written as the sole line of a 0600 file `~/.config/qwen3-tts/.ui_share_credentials` — the password string never reaches any log sink); a half-set pair or a credential/write failure raises RuntimeError instead of launching unauthenticated
- `allowed_paths` narrowed to the history output root (`resolve_history_output_dir`, default `~/Downloads/Qwen3-TTS Output`) + the system tempdir — the old blanket `~/Downloads` entry handed a public URL the whole tree (and the legacy `output_directory` resolver would have made the narrowing a no-op)
- Colab forces share at both sites → the Colab path is authenticated too (notebook mirrors the helper inline)

## Voice create via engine writer (0G #282)
`voice_management.py`'s MLX create path calls engine `save_voice_prompt_mlx` DIRECTLY (module-style import from the definition site) — the same guarded writer as the server's `/create-voice-prompt` MLX route, including reference-path containment; the UI adds only the `gr.Error` mapping + preset/voice cache clears. All four create surfaces now converge on one writer.

## Recent fixes (#218, #195)
- **Confirm-flow repair** — Stop, Delete Voice, and Unload Model each use `ConfirmButton`/`confirm_step`'s two-step path correctly; a prior wiring gap let the second click no-op.
- **`tts ui --port`** honored end-to-end; low-rate voice-prompt warning now surfaces to browser users, not just server logs.
- **ETA badge** — model-load ETA is surfaced in the UI instead of discarded (`44f844b`).

## Critical constraints
- **NEVER** attach `select` to a `gr.Tab` → infinite Dataframe recursion on 6.14.x (kills Manage tabs). Model badges refreshed via shared `gr.Timer` (5 s, one `/models` call) instead.
- `gr.Audio` output: clear with `None`, **never** `""` (returns "" discards the whole handler result).
- `.then(fn=, js=)`: js-only `.then(fn=None, js=...)` never runs at runtime.

## HTTP only
UI never imports torch/mlx; all generation goes through the local server (`TTSClient`), including the long-running paths (create-voice-prompt on torch, transcribe, unload-asr — leaf-locked; create-voice-prompt on MLX calls the engine writer `save_voice_prompt_mlx` directly in-process — inference-free and unlocked, #236/#282 — see backend.md) which use extended client timeouts.
