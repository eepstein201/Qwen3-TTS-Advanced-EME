# Qwen3-TTS Makefile
# Provides convenient shortcuts for common development tasks

.PHONY: help install test test-batch test-core test-voice test-server test-engine test-e2e clean lint format solid-score coverage

# Default target
help:
	@echo "Qwen3-TTS Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install          Install package in editable mode"
	@echo "  install-mlx      Install with MLX backend dependencies"
	@echo "  test             Run all tests (may hang, use test-batch)"
	@echo "  test-batch       Run tests in isolated batches (recommended)"
	@echo "  test-quick       Quick test: core utilities only"
	@echo "  test-core        Batch 1: Core utilities"
	@echo "  test-voice       Batch 2: Voice & CLI"
	@echo "  test-server      Batch 3: Server infrastructure"
	@echo "  test-engine      Batch 4: Engine & UI"
	@echo "  test-e2e         Batch 6: E2E Playwright browser tests"
	@echo "  clean            Remove cache and runtime files"
	@echo "  lint             Run linters (if installed)"
	@echo "  format           Format code with black (if installed)"
	@echo "  solid-score      Analyze SOLID principle compliance"
	@echo "  coverage         Run tests with coverage report"

# Installation
install:
	pip install -e .

install-mlx:
	pip install -e .[mlx]

# Testing - Batch runner (recommended)
test-batch:
	python tests/run_batches.py

test-batch-continue:
	python tests/run_batches.py --continue

test-quick:
	python tests/run_batches.py --batch 1

# Individual batches
test-core:
	python -m unittest tests.test_audio_utils tests.test_text_processing tests.test_package_metadata tests.test_deprecated_refs tests.test_config -v

test-optional:
	python -m unittest tests.test_flash_attn_install -v

test-voice:
	python -m unittest \
		tests.test_voice_config tests.test_voice_server tests.test_voice_prompts \
		tests.test_voice_streaming tests.test_voice_engine tests.test_voice_generation \
		tests.test_voice_ui tests.test_voice_features \
		tests.test_cli_daemonization tests.test_caching tests.test_server_helpers -v

test-server:
	python -m unittest tests.test_fastapi_server tests.test_fastapi_endpoints tests.test_client -v

test-engine:
	python -m unittest tests.test_engine tests.test_generate_server_fallback tests.test_ui_headless -v

# E2E browser tests (requires playwright + running server)
test-e2e:
	python tests/run_batches.py --batch 6

# Full test suite (may hang)
test:
	python -m unittest discover -v tests/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .voice_server.*
	rm -rf ~/.voice_server_token
	rm -rf ~/.voice_history.jsonl
	rm -rf ~/.voice_last_text

# Linting (optional)
lint:
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check qwen3_tts/ tests/; \
	elif command -v pylint >/dev/null 2>&1; then \
		pylint qwen3_tts/ tests/; \
	else \
		echo "No linter found. Install ruff: pip install ruff"; \
	fi

# Formatting (optional)
format:
	@if command -v black >/dev/null 2>&1; then \
		black qwen3_tts/ tests/; \
	else \
		echo "black not found. Install: pip install black"; \
	fi

# SOLID Score Analysis
solid-score:
	python -m qwen3_tts.tools.solid_analyzer qwen3_tts/

solid-score-fail:
	python -m qwen3_tts.tools.solid_analyzer qwen3_tts/ --fail-below 35

# Coverage
coverage:
	coverage run -m pytest tests/
	coverage report --include="qwen3_tts/*"
