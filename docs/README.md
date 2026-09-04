# Qwen3-TTS Documentation

Index of the project's reference documentation. For a product overview and quick start, see the [root README](../README.md). For the primary engineering context file, see [`../CLAUDE.md`](../CLAUDE.md).

## Reference guides

| Doc | What it covers |
|-----|----------------|
| [ONBOARDING.md](ONBOARDING.md) | **Start here** — sequenced first-week path for new teammates through the docs below. |
| [COMMANDS.md](COMMANDS.md) | Full `tts` CLI command reference — generation, server, voice, config, cache, testing. |
| [CONFIG.md](CONFIG.md) | Every `config.json` key and environment variable, with defaults. Hand-maintained (no live generator since `config.py` became the `core/config/` package); its defaults are drift-checked against `get_default_config()` by `make check-config-docs`. |
| [rate-limiting.md](rate-limiting.md) | Rate-limiting architecture (slowapi), strategies, and the `security.rate_limits` format. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev environment setup, scripts, testing procedures, code style, PR workflow. |
| [RUNBOOK.md](RUNBOOK.md) | Deployment, PM2, health checks, model management, monitoring, troubleshooting. |
| [00-Foundations/ARCHITECTURE.md](00-Foundations/ARCHITECTURE.md) | Architecture deep dive — full config schema, security model, platform matrix, internals. |
| [CODEMAPS/](CODEMAPS/) | Token-lean maps (~2k tokens total) — architecture, backend, frontend, data, and dependency views. Refreshed on a 90-day cadence. |

## Roadmaps (live)

| Doc | What it covers |
|-----|----------------|
| [plans/consolidated-roadmap.md](plans/consolidated-roadmap.md) | Authoritative status tracker: completed + open work, reconciled against source. |
| [plans/development-roadmap.md](plans/development-roadmap.md) | Companion tracker for `R-*` items. |

## History

| Location | What it holds |
|----------|---------------|
| [plans/archive/](plans/archive/) | Completed implementation plans and research/review reports (point-in-time records). |
| [reviews/](reviews/) | Dated audit snapshots (e.g. E2E/static-gate reviews). |
