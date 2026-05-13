# Contributing to Qwen3-TTS

> **AUTO-GENERATED** from project structure and development setup. Do not edit manual sections.

## Development Environment Setup

### Prerequisites

- **Python**: 3.9 or later
- **Conda**: Miniforge or Anaconda (recommended for environment management)
- **Git**: For version control
- **macOS**: Apple Silicon M1/M2/M3 for MLX backend (recommended)
- **Linux**: Any modern x86_64 distribution
- **Windows**: WSL2 with Ubuntu recommended (native support limited)

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/your-org/Qwen3-TTS.git
cd Qwen3-TTS
```

#### 2. Create Conda Environments

**For MLX backend (Apple Silicon only):**
```bash
conda create -n qwen3-tts-mlx python=3.9 -y
conda activate qwen3-tts-mlx
pip install -e ".[mlx,server,ui,dev]"
```

**For Torch backend (Linux/Windows/Intel Mac):**
```bash
conda create -n qwen3-tts python=3.9 -y
conda activate qwen3-tts
pip install -e ".[torch,server,ui,dev]"
```

#### 3. Verify Installation

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

# Format code
make format
black . ruff check --fix .
```

### Development Scripts

| Script | Description |
|--------|-------------|
| `make install` | Install all dependencies |
| `make test-batch` | Run all test batches (1-6) |
| `make test-quick` | Run quick subset of tests |
| `make lint` | Run ruff linter |
| `make format` | Format code with black and ruff |
| `make coverage` | Run test coverage analysis |

See [`docs/COMMANDS.md`](COMMANDS.md) for complete command reference.

## Project Structure

```
qwen3-tts/
├── qwen3_tts/              # Main package
│   ├── core/              # Core engine (text/audio/voice/model/inference)
│   ├── server/            # FastAPI server and client
│   ├── interface/         # CLI and Gradio UI
│   └── tools/             # Utilities (healthcheck, cache, voice)
├── tests/                  # Test suite (1970+ tests)
├── docs/                   # Documentation
├── config.json             # Configuration file
└── pyproject.toml          # Package metadata
```

## Testing Procedures

### Test Suite Overview

The project has **1970+ tests** across 83 modules, organized into 6 batches:

| Batch | Name | Tests | Description | Server Required |
|-------|------|-------|-------------|-----------------|
| 1 | Core | ~300 | Core utilities, config, validation | No |
| 2 | Voice | ~400 | Voice prompts, CLI commands | No |
| 3 | Server | ~500 | Server infrastructure, API endpoints | No |
| 4 | Engine | ~400 | Engine components, UI logic | No |
| 5 | Optional | ~200 | Pytest-dependent features | No |
| 6 | E2E | ~170 | End-to-end Playwright tests | **Yes** |

### Running Tests

**Run all tests (requires server for Batch 6):**
```bash
# Ensure server is running
tts server start

# Run all batches
python tests/run_batches.py

# Or use Makefile
make test-batch
```

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
pytest --cov=qwen3_tss --cov-report=term-missing --cov-report=html
open htmlcov/index.html
```

### Test Organization

Tests are organized by module and use pytest marks:
- `@pytest.mark.unit`: Unit tests (no external dependencies)
- `@pytest.mark.integration`: Integration tests (requires server)
- `@pytest.mark.e2e`: End-to-end tests (requires server and browser)

### E2E Testing Requirements

Batch 6 (E2E Playwright) requires:
1. **Server running**: `tts server start`
2. **Models loaded**: Clone model must be loaded
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

### Formatting

The project uses:
- **black**: Code formatting (line length 100)
- **isort**: Import sorting
- **ruff**: Fast Python linter

**Format code before committing:**
```bash
make format
# Or manually
black qwen3_tss/
ruff check --fix qwen3_tss/
```

### Type Checking

The project uses **mypy** for static type checking:

```bash
make type-check
# Or manually
mypy qwen3_tts/
```

### Pre-commit Hooks (Recommended)

Install pre-commit hooks for automatic formatting:

```bash
pip install pre-commit
pre-commit install
```

## Development Workflow

### Poltergeist Build System

This project uses **Poltergeist** (polter) for automated build management. **Follow these rules to avoid build conflicts:**

**⚠️ NEVER manually run build commands when Poltergeist is running**
- Poltergeist automatically detects code changes and rebuilds
- Manual builds interfere with Poltergeist's file watching and state management

**✅ ALWAYS use "polter <target>" for builds**
```bash
polter tts          # Build and run TTS target
polter test         # Build and run tests
polter ui           # Build and launch UI
```

**🔄 How Poltergeist Works**
- Monitors source files for changes (automatic rebuild on save)
- Maintains clean build artifacts and cache management
- Provides consistent, reproducible builds

**Note for AI Agents:** When suggesting build commands, always use `polter <target>` syntax rather than direct build commands.

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
curl -H "Authorization: Bearer $(cat ~/.voice_server_token)" http://127.0.0.1:5123/models

# Check models
tts list models
```

**Issue: Import errors for torch/mlx**
```bash
# Ensure correct environment
conda activate qwen3-tts-mlx  # for MLX
conda activate qwen3-tss     # for Torch

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
