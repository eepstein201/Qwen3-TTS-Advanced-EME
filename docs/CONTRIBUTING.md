# Contributing to Qwen3-TTS

> **AUTO-GENERATED** from project structure and development setup. Do not edit manual sections.

## Development Environment Setup

### Prerequisites

- **Python**: 3.10 or later (3.12 recommended)
- **Conda**: Miniforge or Anaconda (recommended for environment management)
- **Git**: For version control
- **macOS**: Apple Silicon M1/M2/M3 for MLX backend (recommended)
- **Linux**: Any modern x86_64 distribution
- **Windows**: Not supported natively (no Windows platform in the support matrix) — use WSL2 with Ubuntu

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/eepstein201/Qwen3-TTS-Advanced-EME.git
cd Qwen3-TTS-Advanced-EME
```

#### 2. Create Conda Environments

Two conda environments exist because the torch and mlx backends conflict on
their `transformers` version requirements — they cannot coexist in one env.
You normally need only the one matching your backend; the other is optional
(and the torch env is required for `tts voice rebuild` on MLX Macs, see
[`COMMANDS.md`](COMMANDS.md)).

**For MLX backend (Apple Silicon only):**
```bash
conda create -n qwen3-tts-mlx python=3.11 -y
conda activate qwen3-tts-mlx
pip install -e ".[mlx,server,ui,dev]"
```

**For Torch backend (Linux/WSL2/Intel Mac):**
```bash
conda create -n qwen3-tts python=3.11 -y
conda activate qwen3-tts
pip install -e ".[torch,server,ui,dev]"
```

#### 3. Test-Only Setup (Any Python Environment)

Running the test suite does **not** require conda. In any Python 3.10+
environment:

```bash
pip install -e ".[test]"
python -m pytest tests/ -v
```

`requirements.lock` pins the full `test+ui+dev` dependency tree for
standalone test/CI environments (regenerate it after changing dependencies in
`pyproject.toml`). **Never install the lock into the platform conda envs**
(`qwen3-tts` / `qwen3-tts-mlx`): the transformers/huggingface-hub conflict is
env-specific — in the torch env, transformers 4.57.3 requires
`huggingface-hub<1.0` while the lock's gradio 6.20 requires `>=1.2.0`, so the
lock would break `pip check` there. The lock is for standalone test/CI envs
only; platform extras (`mlx`, `torch`) are deliberately excluded from it.

#### 4. Verify Installation

```bash
tts doctor
```

Expected output: All checks pass ✓

## Available Scripts

### Core Commands

```bash
# Generate audio (default mode)
tts "Hello, world!" -o hello.wav

# Start server
tts server start
tts server status

# Launch web UI
tts ui

# Run tests
make test
python -m pytest tests/ -v

# Format and lint code (ruff)
make format          # ruff format qwen3_tts/ tests/
make lint            # ruff check qwen3_tts/ tests/
```

### Development Scripts

| Script | Description |
|--------|-------------|
| `make install` | Install all dependencies |
| `make test-batch` | Run all test batches (1-6) |
| `make test-quick` | Run quick subset of tests |
| `make lint` | Run ruff linter (`ruff check`) |
| `make format` | Format code with `ruff format` |
| `make coverage` | Run test coverage analysis |
| `make solid-score` | SOLID-compliance analyzer report |

See [`docs/COMMANDS.md`](COMMANDS.md) for complete command reference.

## Project Structure

```
qwen3-tts/
├── qwen3_tts/              # Main package
│   ├── core/              # Core engine (text/audio/voice/model/inference)
│   ├── server/            # FastAPI server and client
│   ├── interface/         # CLI and Gradio UI
│   └── tools/             # Utilities (healthcheck, cache, voice)
├── tests/                  # Test suite (2000+ tests)
├── docs/                   # Documentation
├── config.json             # Configuration file
└── pyproject.toml          # Package metadata
```

## Testing Procedures

### Test Suite Overview

The project has **2000+ tests** across 100+ modules, organized into 6 batches
(module counts from the `BATCHES` dict in `tests/run_batches.py`):

| Batch | Name | Modules | Description | Server Required |
|-------|------|---------|-------------|-----------------|
| 1 | Core | 22 | Core utilities, config, validation | No |
| 2 | Voice | 23 | Voice prompts, CLI commands | No |
| 3 | Server | 51 | Server infrastructure, API endpoints | No |
| 4 | Engine | 36 | Engine components, UI logic | No |
| 5 | Optional | 16 | Optional-dependency features | No |
| 6 | E2E | 1 | End-to-end Playwright tests | **Yes** |

Note: batch 6 runs only `tests.test_e2e_playwright`. The rest of the
`test_e2e_*` suite is deliberately excluded from batches and runs under pytest
only (`pytest -m e2e`); `tests/test_rate_limiting.py` is excluded too (the
batch runner disables rate limiting, so it would pass hollowly).

### Running Tests

**Run all batches** (no server needed for batches 1-5; the runner sets
`TTS_DISABLE_RATE_LIMITING=1` for its children):
```bash
# Run all batches
python tests/run_batches.py

# Or use Makefile
make test-batch
```

**Run the full test suite via pytest** — no server needed. `pytest.ini`
deselects e2e by default (`-m "not e2e"`), so a plain full run hangs on
nothing:
```bash
python -m pytest tests/ -v --tb=short   # skips E2E by default
python -m pytest tests/ -m e2e          # opt-in: run E2E (needs live server)
```

Run `pytest -m "not e2e"` locally before pushing: CI's coverage job runs the
full pytest discovery, which is broader than the batch runner — a test outside
`BATCHES` can pass every batch gate yet fail CI.

**Run specific batch:**
```bash
python tests/run_batches.py --batch 1  # Core utilities
python tests/run_batches.py --batch 2  # Voice & CLI
python tests/run_batches.py --batch 3  # Server infrastructure
python tests/run_batches.py --batch 4  # Engine & UI
python tests/run_batches.py --batch 5  # Optional features
python tests/run_batches.py --batch 6  # E2E Playwright
```

**Run specific test file:**
```bash
pytest tests/test_core_config.py -v
pytest tests/test_server_app_generation.py -v
```

**Run with coverage:**
```bash
pytest --cov=qwen3_tts --cov-report=term-missing --cov-report=html
open htmlcov/index.html
```

### Test Organization

Tests are organized by module and use pytest marks (all registered in
`pytest.ini`, enforced with `--strict-markers`):
- `@pytest.mark.unit`: Unit tests (no external dependencies)
- `@pytest.mark.integration`: Integration tests (requires server)
- `@pytest.mark.e2e`: End-to-end tests (requires server and browser); deselected by default
- `@pytest.mark.slow`: Slow tests (deselect with `-m "not slow"`)
- `@pytest.mark.requires_server`: Needs a running server

### Mandatory Test Workflow Rules

1. **Register new test modules in `BATCHES`** (`tests/run_batches.py`). The
   list is explicit, not discovery-based — an unregistered module silently
   never runs in the batch gates. Enforced by
   `tests/test_batches_coverage.py` (an `INTENTIONALLY_UNBATCHED` allowlist
   with a written reason exists for genuine exceptions).
2. **Never put `async def test_*` on a plain `unittest.TestCase`** — unittest
   does not await it, so the body never runs yet the test reports *ok* (a
   false green). Use `unittest.IsolatedAsyncioTestCase`, or
   `@pytest.mark.asyncio` outside a TestCase (pytest-asyncio is in `strict`
   mode; unmarked coroutine tests are skipped). Guarded statically by
   `tests/test_async_test_hygiene.py`.
3. **Never attach a `select` listener to a `gr.Tab`** in the Gradio UI —
   gradio 6.14.x recurses infinitely in the Dataframe frontend and kills the
   page. Fixed upstream in 6.20.0, but the ban stays as defense-in-depth (and
   Timer polling is cheaper); it is **not** a reason to avoid gradio > 6.14.
   Guarded by `tests/test_ui_tab_select_wiring.py`.

### E2E Testing Requirements

Batch 6 (E2E Playwright) requires:
1. **Server running**: `tts server start` (the batch setup preloads the clone, design, and custom models via `/load-model`, but start the server with `TTS_DISABLE_RATE_LIMITING=1` or a raised generate limit to avoid false 429 skips)
2. **Models loaded**: all three (clone, design, custom)
3. **Browser**: Playwright browser driver

**Setup for E2E tests:**
```bash
# Start server
tts server start

# Verify server and models
tts server status
tts list models

# Run E2E tests
python tests/run_batches.py --batch 6
```

## Code Style Enforcement

### Formatting & Linting

The project uses **ruff** for both formatting and linting (config in `.ruff.toml`):

```bash
make format          # ruff format qwen3_tts/ tests/
make lint            # ruff check qwen3_tts/ tests/
# Or directly:
ruff format qwen3_tts/ tests/
ruff check --fix qwen3_tts/ tests/
```

### Type Checking

Static typing is checked with **mypy** (config in `pyproject.toml`; FastAPI
`app.py` and vLLM modules are excluded):

```bash
mypy qwen3_tts/{core,server,interface}
```

### Security Scan

```bash
bandit -r qwen3_tts -c pyproject.toml   # target: 0 HIGH findings
```

All three tools (`ruff`, `mypy`, `bandit`) ship in the `dev` extra
(`pip install -e ".[dev]"`). There is no pre-commit config; run these gates
manually (or wire your own local hooks) before pushing.

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following the style guide
- Add tests for new functionality (TDD approach recommended)
- Format code: `make format`
- Run tests: `make test`

### 3. Test Your Changes

```bash
# Quick test
make test-quick

# Full test suite
make test-batch

# E2E tests (if applicable)
tts server start
python tests/run_batches.py --batch 6
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code refactoring
- `docs:` Documentation changes
- `test:` Test changes
- `chore:` Maintenance tasks

### 5. Push and Create PR

```bash
git push -u origin feature/your-feature-name
# Create PR via GitHub or command line
```

## Troubleshooting

### Common Issues

**Issue: Tests fail with "ModuleNotFoundError"**
```bash
# Ensure you're in correct conda environment
conda activate qwen3-tts-mlx  # or qwen3-tts
pip install -e ".[dev]"
```

**Issue: Server won't start**
```bash
# Check port 5123 is available
lsof -i :5123

# Check server log
tts server log

# Restart server
tts server stop && tts server start
```

**Issue: E2E tests fail**
```bash
# Ensure server is running
tts server status

# Ensure clone model is loaded
curl -H "Authorization: Bearer $(cat ~/.config/qwen3-tts/.voice_server_token)" http://127.0.0.1:5123/models

# Check models
tts list models
```

**Issue: Import errors for torch/mlx**
```bash
# Ensure correct environment
conda activate qwen3-tts-mlx  # for MLX
conda activate qwen3-tts     # for Torch

# Reinstall dependencies
pip install -e ".[mlx,dev]"  # or ".[torch,dev]"
```

### Getting Help

- **Documentation**: See `docs/` directory
- **Health check**: Run `tts doctor`
- **Logs**: Check `tts server log` for server issues
- **Issues**: Report via GitHub Issues with:
  - Environment details (OS, Python version)
  - Error messages and stack traces
  - Steps to reproduce

## Architecture Overview

The system is organized into layers:

1. **Core Engine** (`qwen3_tts/core/`): Text processing, audio processing, voice prompts, model loading, inference
2. **Server Layer** (`qwen3_tts/server/`): FastAPI server, WebSocket streaming, client library
3. **Interface Layer** (`qwen3_tts/interface/`): CLI commands, Gradio web UI
4. **Tools** (`qwen3_tts/tools/`): Utilities for health checks, cache management, voice creation

**Key design principles:**
- Lazy imports: Heavy ML libraries imported only when needed
- Backend abstraction: Unified API for torch/mlx backends
- Client-server separation: Clean boundary between CLI and server
- Configuration-driven: Behavior controlled via config.json

See [`docs/00-Foundations/ARCHITECTURE.md`](00-Foundations/ARCHITECTURE.md) for deep-dive architecture documentation.
