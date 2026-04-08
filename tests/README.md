# Tests Directory

This directory contains the test suite for Qwen3-TTS.

## Test Organization

- `test_*.py` - Unit and integration tests
- `test_e2e_*.py` - End-to-end tests (Playwright)
- `conftest.py` - Shared fixtures and configuration

## Running Tests

Run all tests:
```bash
pytest
```

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
