# Tests Directory

This directory contains the test suite for Qwen3-TTS.

## Installation

Tests work in any Python environment with the test dependencies installed — **no conda environment required**:

```bash
pip install -e ".[test]"
```

The platform backends (`mlx`, `torch`) live in separate conda envs and are not needed for the suite; optional tests auto-skip when optional deps are missing.

## Test Organization

- `test_*.py` - Unit and integration tests
- `test_e2e_*.py` - End-to-end tests (opt-in; see below). Many of these drive the **live server directly** over HTTP (security, performance, and stress suites); only some use Playwright to drive the Gradio UI.
- `conftest.py` - Shared fixtures and configuration

## Running Tests

Run all tests (excluding E2E — see the warning below):
```bash
python -m pytest tests/
```

> **E2E tests are deselected by default.** `pytest.ini` sets `addopts = -m "not e2e"`, so a plain `pytest` silently skips every `test_e2e_*.py` module. They are opt-in because they make real `/generate` calls and would hang (or take minutes per case) when a server is live:
> ```bash
> python -m pytest tests/ -m e2e   # requires a running server
> ```
>
> **Rate-limit hazard:** the live `/generate` limit (code default 10/minute; check your `config.json` — it may be configured higher) is shared across all e2e modules, so suites that fire many requests starve each other and false-skip with 429s. Start the test server with rate limiting disabled or raised:
> ```bash
> TTS_DISABLE_RATE_LIMITING=1 tts server start          # test/CI servers only
> # or
> TTS_RATE_LIMIT_GENERATE=120/minute tts server start
> ```

Run specific test categories:
```bash
pytest -m ai_regression
pytest -m unit
pytest -m integration
pytest -m e2e
```

Run specific test file:
```bash
pytest tests/test_ai_regression.py
```

## Batch Runner

The batch runner executes the suite in hang-safe batches with stack dumps on timeout (it sets `TTS_DISABLE_RATE_LIMITING=1`; E2E modules are deliberately excluded because it ignores pytest markers):

```bash
python tests/run_batches.py            # All batches
python tests/run_batches.py --batch 2  # One batch (Batch 2: Voice & CLI)
python tests/run_batches.py --list     # List batches
```

> **New test modules MUST be registered in `BATCHES` in `tests/run_batches.py`.** The list is explicit, not discovery-based — an unregistered module silently never runs in the batch gates. This is enforced by `tests/test_batches_coverage.py`, which fails if a module is missing from both `BATCHES` and its allowlist.
>
> Note that CI's `coverage` job runs full `pytest -m "not e2e"` (ALL tests), which discovers more than the batch runner — run `python -m pytest tests/ -m "not e2e"` locally before pushing, not just the batches.

## Test Markers

- `ai_regression` - AI regression tests (prevent systematic AI-introduced bugs)
- `unit` - Unit tests
- `integration` - Integration tests
- `e2e` - End-to-end tests

## AI Regression Tests

The `test_ai_regression.py` module contains specialized tests to prevent systematic AI-introduced bugs:

1. **Backend Consistency** - Verify MLX and Torch backends return identical response shapes
2. **Model State Edge Cases** - Test graceful failures when models not loaded
3. **API Response Contracts** - Ensure all API responses include required fields

These tests target patterns where AI self-review has systematic blind spots.
