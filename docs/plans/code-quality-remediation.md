# Code Quality Remediation Plan

## Overview

This plan addresses DRY and SOLID violations identified in the codebase analysis (2026-03-12). Work is organized by priority (P0 → P1 → P2) with each item as a separate task for parallel execution.

## Summary

| Priority | Items | Estimated LOC | Risk |
|----------|-------|---------------|------|
| P0 (Critical) | 2 | ~50 | Low - cleanup only |
| P1 (High) | 4 | ~200 | Medium - some refactoring |
| P2 (Medium) | 5 | ~300 | Medium - structural changes |

---

## P0: Critical Issues (Do First)

### P0-1: Remove Duplicate Pydantic Models

**Problem:** 6 Pydantic models defined twice in same file
**Location:** `qwen3_tts/server/validation.py:49-81` and `120-152`
**Impact:** 6 duplicate class definitions (~30 lines)

**Files:**
- Modify: `qwen3_tts/server/validation.py`

**Changes:**
```python
# DELETE lines 120-152 (duplicate definitions)
# These models are already defined at lines 49-81:
# - LoadModelRequest
# - UnloadModelRequest
# - UpdateModelConfigRequest
# - UpdateStartupConfigRequest
# - DeletePromptRequest
# - RenamePromptRequest
```

**Testing:**
```bash
# Verify no imports break
python -c "from qwen3_tts.server.validation import LoadModelRequest, UnloadModelRequest, UpdateModelConfigRequest, UpdateStartupConfigRequest, DeletePromptRequest, RenamePromptRequest"
# Run server tests
python -m pytest tests/test_fastapi_server.py tests/test_fastapi_endpoints.py -v
```

**Commit:** `refactor: remove duplicate Pydantic model definitions in validation.py`

---

### P0-2: Consolidate Print Helpers

**Problem:** Duplicated print helper functions in tools
**Locations:**
- `qwen3_tts/tools/uninstall.py:27-46`
- `qwen3_tts/tools/healthcheck.py:38-62`

**Files:**
- Modify: `qwen3_tts/tools/_shared.py` (add functions)
- Modify: `qwen3_tts/tools/uninstall.py` (import from _shared)
- Modify: `qwen3_tts/tools/healthcheck.py` (import from _shared)

**Changes to `_shared.py`:**
```python
# Add after _format_size():

def print_header(text: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print('=' * 60)

def print_success(text: str) -> None:
    """Print a success message."""
    print(f"  ✓ {text}")

def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"  ⚠ {text}")

def print_info(label: str, status: str = "", details: str = "") -> None:
    """Print an info line with optional status and details."""
    line = f"  {label}"
    if status:
        line += f" [{status}]"
    if details:
        line += f" - {details}"
    print(line)

def print_check(label: str, status: bool, details: str = "") -> None:
    """Print a check result with status indicator."""
    status_str = "✓" if status else "✗"
    print_info(label, status_str, details)
```

**Changes to `uninstall.py`:**
```python
# At top:
from qwen3_tts.tools._shared import print_header, print_success, print_warning, print_info

# Remove local definitions of _print_header, _print_success, _print_warning, _print_info
```

**Changes to `healthcheck.py`:**
```python
# At top:
from qwen3_tts.tools._shared import print_header, print_info, print_check

# Remove local definitions of _print_header, _print_info, _print_check
```

**Testing:**
```bash
python -m pytest tests/ -v -k "healthcheck or uninstall"
# Manual verification:
python -m qwen3_tts.tools.healthcheck
python -m qwen3_tts.tools.uninstall --help
```

**Commit:** `refactor: consolidate print helpers into tools/_shared.py`

---

## P1: High Priority Issues

### P1-1: Create @require_server Decorator

**Problem:** `is_server_running()` check repeated 13+ times in client.py
**Location:** `qwen3_tts/server/client.py` (lines 222, 243, 274, 309, 336, 367, 382, 423, 446, 468, 493)

**Files:**
- Modify: `qwen3_tts/server/client.py`

**Changes:**
```python
# Add at top after imports:
import functools

def _require_server(func):
    """Decorator that checks server is running before method execution."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_server_running():
            raise ConnectionError(
                "TTS server is not running. Start it with: tts server start"
            )
        return func(self, *args, **kwargs)
    return wrapper

# Apply to all methods that currently have the check:
@_require_server
def load_model(self, model_type: str) -> dict:
    # Remove the is_server_running check from body
    ...

@_require_server
def unload_model(self, model_type: str) -> dict:
    ...
# (apply to all 13 methods)
```

**Testing:**
```bash
python -m pytest tests/test_client.py -v
```

**Commit:** `refactor: add @require_server decorator to eliminate repeated checks`

---

### P1-2: Extract Config Value Helper

**Problem:** 3 nearly identical config loading functions
**Location:** `qwen3_tts/core/config.py:500-548`

**Files:**
- Modify: `qwen3_tts/core/config.py`

**Changes:**
```python
# Add helper function:
def _get_config_value(key_path: list[str], default, validator=None):
    """Get a nested config value with fallback."""
    try:
        config = load_config()
        val = config
        for key in key_path:
            val = val.get(key, {})
        if val == {}:
            return default
        if validator and not validator(val):
            return default
        return val if val is not None else default
    except (json.JSONDecodeError, OSError, KeyError):
        return default

# Refactor existing functions:
def get_voice_prompt_cache_max() -> int:
    return _get_config_value(["cache", "voice_prompt_max"], 10, lambda x: isinstance(x, int) and x > 0)

def get_generation_cache_max() -> int:
    return _get_config_value(["cache", "generation_max"], 5, lambda x: isinstance(x, int) and x > 0)

def get_eta_cache_ttl() -> int:
    return _get_config_value(["cache", "eta_ttl_seconds"], 30, lambda x: isinstance(x, int) and x >= 0)
```

**Testing:**
```bash
python -m pytest tests/test_config.py -v
```

**Commit:** `refactor: extract _get_config_value helper for config loading`

---

### P1-3: Extract Text Chunking Helper

**Problem:** Similar text chunking logic in two inference functions
**Location:** `qwen3_tts/core/engine/inference.py:542-554` and `712-727`

**Files:**
- Modify: `qwen3_tts/core/engine/inference.py`

**Changes:**
```python
# Add helper function:
def _prepare_text_chunks(text: str, language: str, model, max_chunk_chars: int) -> list[str]:
    """Normalize and chunk text for inference."""
    text = _normalize_text(text, language)
    tokenizer = getattr(model, "tokenizer", None)
    max_tokens = _get_max_chunk_tokens() if tokenizer is not None else None

    if tokenizer is not None and max_tokens is not None:
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count > max_tokens:
            return _split_text(text, max_chars=max_chunk_chars, tokenizer=tokenizer, max_tokens=max_tokens)
        return [text]
    elif max_chunk_chars > 0 and len(text) > max_chunk_chars:
        return _split_text(text, max_chars=max_chunk_chars)
    return [text]

# Update run_inference() and run_inference_streaming() to use:
chunks = _prepare_text_chunks(text, language, model, max_chunk_chars)
```

**Testing:**
```bash
python -m pytest tests/test_engine.py -v -k "chunk"
```

**Commit:** `refactor: extract _prepare_text_chunks helper`

---

### P1-4: Define Audio Constants

**Problem:** Magic numbers scattered in audio_processing.py
**Location:** `qwen3_tts/core/engine/audio_processing.py`

**Files:**
- Modify: `qwen3_tts/core/engine/audio_processing.py`

**Changes:**
```python
# Add at top of file after imports:
# Audio processing constants
SILENCE_THRESHOLD_DB = -40
NORMALIZATION_TARGET_DB = -3.0
LUFS_TARGET = -16.0
VOICE_EMBEDDING_MAX_DURATION = 15  # seconds
DEFAULT_SAMPLE_RATE = 24000

# Update usages:
# Line 69: max_duration=VOICE_EMBEDDING_MAX_DURATION
# Line 101: threshold_db=SILENCE_THRESHOLD_DB
# Line 121: target_db=NORMALIZATION_TARGET_DB
# Line 170: target_lufs=LUFS_TARGET
```

**Testing:**
```bash
python -m pytest tests/test_audio_utils.py -v
```

**Commit:** `refactor: define audio processing constants`

---

## P2: Medium Priority Issues

### P2-1: Split ui.py into Modules

**Problem:** 2100-line monolithic UI file
**Location:** `qwen3_tts/interface/ui.py`

**Files:**
- Create: `qwen3_tts/interface/ui/__init__.py`
- Create: `qwen3_tts/interface/ui/clone_tab.py`
- Create: `qwen3_tts/interface/ui/design_tab.py`
- Create: `qwen3_tts/interface/ui/custom_tab.py`
- Create: `qwen3_tts/interface/ui/voice_management.py`
- Create: `qwen3_tts/interface/ui/shared.py`
- Delete: `qwen3_tts/interface/ui.py` (or keep as re-export facade)

**Structure:**
```
qwen3_tts/interface/ui/
├── __init__.py      # Re-exports, build_ui()
├── clone_tab.py     # Clone mode tab (~300 lines)
├── design_tab.py    # Design mode tab (~300 lines)
├── custom_tab.py    # Custom mode tab (~300 lines)
├── voice_management.py  # Voice prompt management tab (~400 lines)
└── shared.py        # Shared components, helpers (~500 lines)
```

**Testing:**
```bash
python -m pytest tests/test_ui_headless.py -v
# Manual verification:
tts ui
```

**Commit:** `refactor: split ui.py into modular package`

---

### P2-2: Split generate.py into Modules

**Problem:** 2400-line CLI file mixing concerns
**Location:** `qwen3_tts/interface/generate.py`

**Files:**
- Create: `qwen3_tts/interface/cli/__init__.py`
- Create: `qwen3_tts/interface/cli/parser.py`
- Create: `qwen3_tts/interface/cli/generation.py`
- Create: `qwen3_tts/interface/cli/batch.py`
- Create: `qwen3_tts/interface/cli/srt.py`
- Create: `qwen3_tts/interface/cli/dialogue.py`

**Structure:**
```
qwen3_tts/interface/cli/
├── __init__.py      # Re-exports, main()
├── parser.py        # Argument parsing (~400 lines)
├── generation.py    # Core generation logic (~600 lines)
├── batch.py         # Batch processing (~400 lines)
├── srt.py           # SRT subtitle generation (~300 lines)
└── dialogue.py      # Multi-speaker dialogue (~400 lines)
```

**Testing:**
```bash
python -m pytest tests/ -v -k "cli or generate"
```

**Commit:** `refactor: split generate.py into modular CLI package`

---

### P2-3: Split TTSClient into Focused Interfaces

**Problem:** TTSClient has 30+ methods (fat interface)
**Location:** `qwen3_tts/server/client.py`

**Files:**
- Create: `qwen3_tts/server/client/__init__.py`
- Create: `qwen3_tts/server/client/generator.py`
- Create: `qwen3_tts/server/client/models.py`
- Create: `qwen3_tts/server/client/voices.py`
- Create: `qwen3_tts/server/client/config.py`
- Modify: `qwen3_tts/server/client.py` → keep as facade

**Structure:**
```python
# generator.py - TTSGenerator
class TTSGenerator:
    def generate(...)
    def generate_streaming(...)
    def generate_dialogue(...)

# models.py - ModelManager
class ModelManager:
    def load_model(...)
    def unload_model(...)
    def get_models(...)
    def update_model_config(...)

# voices.py - VoiceManager
class VoiceManager:
    def list_prompts(...)
    def delete_prompt(...)
    def rename_prompt(...)
    def preview_prompt(...)

# config.py - ConfigFetcher
class ConfigFetcher:
    def list_presets(...)
    def list_aliases(...)
    def get_stats(...)

# __init__.py - TTSClient facade
class TTSClient(TTSGenerator, ModelManager, VoiceManager, ConfigFetcher):
    """Facade combining all client capabilities."""
    pass
```

**Testing:**
```bash
python -m pytest tests/test_client.py -v
```

**Commit:** `refactor: split TTSClient into focused interfaces`

---

### P2-4: Add Dependency Injection for Config

**Problem:** Direct config loading scattered throughout codebase
**Location:** Multiple files

**Files:**
- Modify: `qwen3_tts/core/config.py`
- Modify: `qwen3_tts/core/engine/inference.py`
- Modify: `qwen3_tts/server/app.py`

**Changes:**
```python
# config.py - add Protocol for dependency injection:
from typing import Protocol

class ConfigProvider(Protocol):
    """Protocol for config providers."""
    def get(self, key: str, default=None): ...
    def load(self) -> dict: ...

# inference.py - accept config provider:
def run_inference(..., config_provider: ConfigProvider = None):
    config = config_provider.load() if config_provider else load_config()
    ...

# app.py - inject config provider:
class AppState:
    def __init__(self, config_provider: ConfigProvider = None):
        self.config_provider = config_provider or DefaultConfigProvider()
```

**Testing:**
```bash
python -m pytest tests/ -v -k "config"
```

**Commit:** `refactor: add ConfigProvider protocol for dependency injection`

---

### P2-5: Create Error Response Helper

**Problem:** Repeated error extraction pattern in client
**Location:** `qwen3_tts/server/client.py` (multiple locations)

**Files:**
- Modify: `qwen3_tts/server/client.py`

**Changes:**
```python
# Add helper:
def _extract_error_message(resp: requests.Response, default: str = "Unknown error") -> str:
    """Extract error message from HTTP response."""
    try:
        return resp.json().get("error", default)
    except (ValueError, requests.exceptions.JSONDecodeError):
        return f"Server returned HTTP {resp.status_code}"

# Replace repeated patterns:
# Before:
if resp.status_code != 200:
    try:
        error_msg = resp.json().get("error", "Unknown error")
    except (ValueError, requests.exceptions.JSONDecodeError):
        error_msg = f"Server returned HTTP {resp.status_code}"
    raise SomeError(error_msg)

# After:
if resp.status_code != 200:
    raise SomeError(_extract_error_message(resp))
```

**Testing:**
```bash
python -m pytest tests/test_client.py -v
```

**Commit:** `refactor: add _extract_error_message helper`

---

## Execution Order

### Phase 1: P0 Items (Parallel)
Tasks P0-1 and P0-2 can be executed in parallel - no dependencies.

```
┌─────────────┐  ┌─────────────┐
│   P0-1      │  │   P0-2      │
│ Remove      │  │ Consolidate │
│ Duplicates  │  │ Print       │
└─────────────┘  │ Helpers     │
                 └─────────────┘
```

### Phase 2: P1 Items (Sequential or Parallel)
P1-1, P1-2, P1-3, P1-4 can run in parallel after P0 completes.

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   P1-1      │  │   P1-2      │  │   P1-3      │  │   P1-4      │
│ @require_   │  │ Config      │  │ Text        │  │ Audio       │
│ server      │  │ Helper      │  │ Chunking    │  │ Constants   │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

### Phase 3: P2 Items (Sequential - Large Refactors)
P2-1, P2-2, P2-3 should be done one at a time due to file moves.

```
P2-1 (ui.py split) → P2-2 (generate.py split) → P2-3 (TTSClient split) → P2-4 (DI) → P2-5 (error helper)
```

---

## Verification

After all tasks complete:
```bash
# Run full test suite
python -m pytest tests/ -v --tb=short

# Verify imports work
python -c "from qwen3_tts.server.client import TTSClient"
python -c "from qwen3_tts.interface.ui import build_ui"
python -c "from qwen3_tts.interface.generate import main"

# Manual smoke test
tts server start
tts "Hello world"
tts server stop
```

---

## Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| DRY Score | 6.5/10 | 8.5/10 |
| SOLID Score | 6.4/10 | 8.0/10 |
| Duplicate LOC | ~500 | ~100 |
| Files > 500 lines | 4 | 0 |
| Max file size | 2400 lines | 600 lines |
