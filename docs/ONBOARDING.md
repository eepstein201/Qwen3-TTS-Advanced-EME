# New Teammate Onboarding

A sequenced first-week path through this repo. Nothing here duplicates the
reference docs — each step links to the authoritative source. The shape of the
week: **Day 1** you generate audio, **Day 2** you ship a PR, by end of **Week 1**
you know where everything lives.

> **Already use Claude Code?** This repo ships [`CLAUDE.md`](../CLAUDE.md) plus
> repo-level guard hooks (`.claude/`), so an agent session starts with full,
> maintained context and the same gates this guide describes. This document is
> the human-readable version of that ramp-up.

## Day 1 — install and generate

1. Install for your platform (details: [README "Local Installation Paths"](../README.md#local-installation-paths)):
   - **macOS (Apple Silicon):** `./install.sh` — creates the `qwen3-tts-mlx`
     conda env and `pip install -e`s the CLI into it
   - **Linux / NVIDIA:** Docker (`docker compose up -d`) or `./install.sh` —
     creates the `qwen3-tts` (torch) env
2. Health check, from inside the activated env:
   ```bash
   conda activate qwen3-tts-mlx   # or qwen3-tts on Linux/torch
   tts doctor
   ```
3. Generate your first audio (server models take 30–60 s to load on first boot):
   ```bash
   tts server start
   tts "Hello, world!" -o hello
   tts ui        # Gradio web UI
   tts server stop   # free the memory when done
   ```
4. Skim the five maps in [`docs/CODEMAPS/`](CODEMAPS/architecture.md) (~10 min).

**Day-1 mental model:** the CLI and web UI are *clients*; a persistent FastAPI
server (port 5123) owns the models; `core/engine` dispatches to the backend
(mlx/torch) chosen in `config.json`. Everything heavy is lazy-imported, so no
module ever loads a model at import time.

## Day 2 — tests, gates, first PR

1. Test environment — any Python ≥3.10, no conda needed:
   ```bash
   pip install -e ".[test]"
   python -m pytest tests/ -m "not e2e"
   ```
2. Static gates (CI runs all of these; there is no pre-commit config):
   ```bash
   ruff check qwen3_tts tests
   mypy qwen3_tts/{core,server,interface}
   bandit -r qwen3_tts -c pyproject.toml
   make check-config-docs
   ```
3. Branch and commit: feature branches only (never commit to `main`),
   conventional-commit messages (`feat:` / `fix:` / `docs:` / `chore:` / …),
   and no AI authorship attribution anywhere. The full workflow including PR
   rules: [CONTRIBUTING.md](CONTRIBUTING.md).
4. Read [`CLAUDE.md`](../CLAUDE.md) end to end once. It reads like a style
   guide; it is actually incident history — every "never do X" marks a bug
   that already happened and was measured.

## Week 1 — reading order

| # | Doc | Why |
|---|-----|-----|
| 1 | [CODEMAPS/](CODEMAPS/architecture.md) | orientation in 10 minutes |
| 2 | [CLAUDE.md](../CLAUDE.md) | the landmine list |
| 3 | [00-Foundations/ARCHITECTURE.md](00-Foundations/ARCHITECTURE.md) | deep dive: config schema, security model, platform matrix |
| 4 | [CONTRIBUTING.md](CONTRIBUTING.md) | env setup, testing procedures, PR workflow |
| 5 | [RUNBOOK.md](RUNBOOK.md) | PM2 services, deployment, troubleshooting |
| 6 | [COMMANDS.md](COMMANDS.md) · [CONFIG.md](CONFIG.md) | full CLI + config reference |
| 7 | [plans/consolidated-roadmap.md](plans/consolidated-roadmap.md) | what is done, what is open |

## The five landmines

1. **Lazy imports everywhere** — `torch`, `mlx`, `transformers` are never
   imported at module scope, in any file. This is what lets the CLI start
   instantly and lets tests run without model deps.
2. **Two conda envs** — `qwen3-tts` (torch) and `qwen3-tts-mlx` (MLX) exist
   because their `transformers` versions conflict. `requirements.lock` is for
   standalone test/CI envs only — never install it into a platform env.
3. **Restart after code changes** — `tts server stop && tts server start`.
   The server is long-lived; it will not pick up your edits otherwise.
4. **Register new test modules** in `BATCHES` in `tests/run_batches.py` —
   the list is explicit, not discovery-based, so an unregistered module
   silently never runs in the batch gates.
5. **`docs/CONFIG.md` is drift-checked** — changing a config default without
   updating the doc fails `make check-config-docs` in CI.

## Working in this repo

- **With Claude Code:** the repo carries its own context — `CLAUDE.md`, tracked
  hooks (direct-push-to-main guard, pre-push local gates, CLAUDE.md length
  guard), and the batch test infrastructure. Open a session and ask; the same
  guards apply to the agent.
- **Without Claude Code:** follow [CONTRIBUTING.md](CONTRIBUTING.md) — same
  gates, same PR flow.
- **Where things live:** `qwen3_tts/core` (config package + engine),
  `server/` (FastAPI app and clients), `interface/` (CLI, Gradio UI, HTTP
  clients), `tools/` (voice creation, cache, health, uninstall),
  `tests/` (flat, batch-registered). Full annotated tree:
  [ARCHITECTURE.md](00-Foundations/ARCHITECTURE.md).
