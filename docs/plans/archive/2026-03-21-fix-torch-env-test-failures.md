# Plan: Fix Torch Environment Test Failures

## Problem Summary

When running `python tests/run_full_suite.py --full --env torch`, 5 tests fail:

| Test | Failure Reason |
|------|----------------|
| `test_speaker_similarity` | Missing `torchcodec` dependency |
| `test_01_clone_generation` | Server uses MLX backend but `mlx-audio` not installed |
| `test_07_concurrent_generation` | Same as above |
| `test_08_load_model` | Timeout - model never loaded due to backend mismatch |
| `test_10_load_unload_cycle` | Timeout - model never loaded due to backend mismatch |

## Root Cause

The torch conda environment (`qwen3-tts`) has `config.json` with `backend: "mlx"`, but MLX dependencies aren't installed in that environment. The server fails to load models because it tries to use MLX backend without `mlx-audio`.

## Solution

### Option A: Per-Environment Config (Recommended)

Create environment-specific config files that are activated based on which conda env is running.

**Changes:**

1. **Create `config.torch.json`** - Default config for torch environment:
   ```json
   {
     "models": { ... },
     "advanced": {
       "backend": "torch",
       ...
     }
   }
   ```

2. **Create `config.mlx.json`** - Default config for MLX environment:
   ```json
   {
     "models": { ... },
     "advanced": {
       "backend": "mlx",
       ...
     }
   }
   ```

3. **Modify `config.py`** to detect environment and load appropriate config:
   ```python
   def get_config_path() -> Path:
       """Return config path, preferring environment-specific config."""
       base_path = Path(__file__).parent.parent / "config.json"

       # Check for environment-specific config
       env = os.environ.get("CONDA_DEFAULT_ENV", "")
       if "mlx" in env:
           env_config = base_path.parent / "config.mlx.json"
           if env_config.exists():
               return env_config
       elif "torch" in env or "qwen3-tts" in env:
           env_config = base_path.parent / "config.torch.json"
           if env_config.exists():
               return env_config

       return base_path
   ```

4. **Update test runner** to ensure correct backend before starting server:
   ```python
   def ensure_backend_config(env: str) -> None:
       """Ensure config.json has correct backend for environment."""
       config_path = PROJECT_ROOT / "config.json"
       with open(config_path) as f:
           config = json.load(f)

       correct_backend = "mlx" if "mlx" in env else "torch"
       if config.get("advanced", {}).get("backend") != correct_backend:
           config["advanced"]["backend"] = correct_backend
           with open(config_path, "w") as f:
               json.dump(config, f, indent=2)
   ```

### Option B: Environment Variable Override

Simpler approach - use environment variable to override backend.

**Changes:**

1. **Modify `config.py`** to check for `QWEN3_TTS_BACKEND` env var:
   ```python
   def load_config() -> dict:
       config = _load_from_file()
       # Allow env var override
       if "QWEN3_TTS_BACKEND" in os.environ:
           config.setdefault("advanced", {})["backend"] = os.environ["QWEN3_TTS_BACKEND"]
       return config
   ```

2. **Update test runner** to set env var when starting server:
   ```python
   def start_server(env: str, dry_run: bool = False) -> bool:
       backend = "mlx" if "mlx" in env else "torch"
       env_vars = f"export QWEN3_TTS_BACKEND={backend} && "
       # ... rest of server start logic
   ```

### Option C: Fix `test_speaker_similarity` Dependency

Separate issue - `torchcodec` is needed for speaker similarity test.

**Changes:**

1. **Add to optional deps in `run_full_suite.py`**:
   ```python
   OPTIONAL_DEPS = {
       "torch": {
           "evaluation": ["openai-whisper", "jiwer"],
           "speaker_similarity": ["torchaudio", "transformers", "torchcodec"],
           "e2e": ["playwright"],
       },
   }
   ```

2. **Or skip test gracefully** if dependency missing (current behavior is correct).

## Recommended Approach

**Combine Option B + C:**

1. Add `QWEN3_TTS_BACKEND` env var override to `config.py`
2. Update test runner to set this env var based on environment
3. Add `torchcodec` to optional deps for torch env

This is minimal, non-breaking, and doesn't require managing multiple config files.

## Implementation Steps

1. [ ] Add `QWEN3_TTS_BACKEND` env var support to `qwen3_tts/core/config.py`
2. [ ] Update `tests/run_full_suite.py` to set env var when starting server
3. [ ] Add `torchcodec` to optional deps for torch env
4. [ ] Test: `python tests/run_full_suite.py --full --env torch`
5. [ ] Test: `python tests/run_full_suite.py --full --env mlx` (verify no regression)

## Files to Modify

- `qwen3_tts/core/config.py` - Add env var override
- `tests/run_full_suite.py` - Set env var, add torchcodec dep
