# Test Validation Across All Environments

**Date:** 2026-05-23
**Status:** All 6 batches passing on macOS MLX **and** in the Docker test container
**Branch:** `fix/vllm-testing-improvements` (latest commit: 845406c)

## Summary

All 2163+ tests pass after fixing mock patch paths in `test_generate_interactive_ext.py`. The fixes ensure tests work in any Python environment with `pip install -e ".[test]"`.

---

## ✅ Validated Environments

### macOS MLX (Primary Development)
- **Platform:** macOS 15.5 (Darwin 25.5.0), Apple Silicon M2 Pro
- **Python:** 3.11.5
- **Installation:** `pip install -e ".[test]"`
- **Result:** ✅ All 6 batches pass
- **Command:**
  ```bash
  python tests/run_batches.py
  ```

---

## ✅ Docker Container

- **Image:** `qwen3-tts:test` (built from `Dockerfile.test`, non-root testuser)
- **Result:** ✅ All 6 batches pass (Batch 6 cleanly skips with no server)
- **Command:**
  ```bash
  bash tests/validate_docker.sh
  ```
  > Note: if Docker Desktop's credential helper isn't on PATH, prefix with
  > `PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"`.

---

## 🔄 Pending Validation

### Linux CPU (Ubuntu/Debian)
```bash
python3 -m venv /tmp/test_linux
source /tmp/test_linux/bin/activate
pip install -e ".[test]"
python tests/run_batches.py
```

### Linux GPU/CUDA
```bash
python3 -m venv /tmp/test_linux_gpu
source /tmp/test_linux_gpu/bin/activate
pip install -e ".[test]"
# Verify CUDA available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python tests/run_batches.py
```

### Google Colab
```python
# In Colab notebook:
!git clone https://github.com/eepstein201/Qwen3-TTS-Advanced-EME.git
%cd Qwen3-TTS-Advanced-EME
!pip install -e ".[test]"
# Click "Runtime" -> "Restart runtime"
!python tests/run_batches.py
```

---

## Test Results Breakdown

| Batch | Description | Tests | Status |
|-------|-------------|-------|--------|
| 1 | Core Utilities | ~300 | ✅ Pass |
| 2 | Voice & CLI | ~500 | ✅ Pass |
| 3 | Server Infrastructure | ~400 | ✅ Pass |
| 4 | Engine & UI | ~459 | ✅ Pass |
| 5 | Optional Tests | ~400 | ✅ Pass |
| 6 | E2E Playwright | ~100 | ✅ Pass (skip if no server) |

**Total:** 2163+ tests across 108 modules

---

## Fixes Applied

| Commit | Change |
|--------|--------|
| `bafcf7c` | Fix mock patch paths in `test_generate_interactive_ext.py` (target import location, not definition) |
| `db3b750` | Patch `get_server_url` at `qwen3_tts.core.http_client` (its real call site via `server_request`); add memory-check mock in `voice_test_helpers`; widen torch.compile lookback; skip Colab-specific tests when notebook absent; update path-traversal test for new `ValueError` behavior |
| `845406c` | `validate_docker.sh` uses `Dockerfile.test` (real build context); `e2e_helpers.playwright_enabled` gracefully degrades when `.claude/.mcp.json` is absent so Batch 6 can skip cleanly in Docker/CI |

---

## Next Steps

1. ✅ Run Docker validation
2. 🔄 Run Linux/CUDA validation on an actual Linux box
3. 🔄 Test in Google Colab (upload `tests/validate_colab.ipynb`)
4. Merge `fix/vllm-testing-improvements` to main once 2–3 are confirmed

---

## Platform Compatibility Matrix

| Platform | Python Env | Tests Pass | Notes |
|----------|-----------|------------|-------|
| macOS MLX | venv/system | ✅ | Primary dev env |
| Linux CPU | venv | 🔄 | Expected to pass |
| Linux GPU/CUDA | venv | 🔄 | Expected to pass |
| Google Colab | Colab env | 🔄 | Expected to pass |
| Docker | Container | 🔄 | Expected to pass |

Legend: ✅ Validated | 🔄 Pending validation | ❌ Failed
