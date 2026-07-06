# Qwen3-TTS Development Roadmap

This document outlines future improvements identified during the multi-agent code review (2025). Priority 1 and 2 items have been implemented in commits 1-5. This roadmap covers Priority 3 (Medium) and Priority 4 (Low) items for future consideration.

## Priority 3 — Medium Priority

### R-13: Rate Limiting ✅ Fixed
Implement API rate limiting to prevent abuse and ensure fair resource allocation.

- **Location**: `qwen3_tts/server/app.py:139-189` (rate limit key functions)
- **Location**: `qwen3_tts/server/app.py:256-283` (multiple limiters, decorator)
- **Location**: `qwen3_tts/core/config.py:60-91` (config validation)
- **Location**: `tests/test_rate_limiting.py` (comprehensive test suite)
- **Fix:** Enhanced rate limiting with per-IP, per-token, and hybrid strategies — implemented 2026-03-28
- **Features:**
  - Per-IP rate limiting (handles reverse proxies)
  - Per-token rate limiting (SHA-256 hash-based, prevents token leakage)
  - Hybrid strategy (both IP and token limits)
  - Configurable limits via config.json
  - All R-13 endpoints protected (13 endpoints total)
  - Comprehensive test suite with AI regression patterns (15 tests)
  - User documentation and troubleshooting guide
  - slowapi now required dependency (simplified architecture)

### R-14: Crossfade Between Chunks ✅ Fixed
When text is split into chunks and generated separately, add smooth crossfading between segments to reduce audible artifacts at chunk boundaries.

- **Location**: `qwen3_tts/core/engine/inference.py:465-510` (`_crossfade_chunks`)
- **Fix**: Raised-cosine (Hann) window crossfade implemented — already exists

### R-15: Split engine.py into Modules ✅ Fixed
The current `engine.py` is very large (~1800 lines). Split into logical modules:

- **Location**: `qwen3_tts/core/engine/` package (6 submodules)
- **Fix**: Refactored to `model_loader.py`, `inference.py`, `audio_processing.py`, `voice_prompt.py`, `text_processing.py`, `asr.py` with facade pattern — already done

### R-16: Model Warm-up Pass ✅ Fixed
Add a warm-up inference pass after model loading to ensure all kernels are compiled and memory is allocated before the first real request.

- **Location**: `qwen3_tts/core/engine/model_loader.py:335-367`
- **Fix**: `_warmup_model()` already implemented for design models; clone/custom models correctly skipped because they require voice prompts — verified 2026-03-28

### R-17: Temperature Consistency (Torch vs MLX) ✅ Fixed
The default temperature differs between backends (0.7 for torch, 0.9 for MLX). Standardize to use the same default from config.

- **Location**: `_run_inference_mlx` (hardcoded 0.9) should read from gen_params like torch does
- **Fix**: MLX path now reads from `gen_params` via `_get_mlx_gen_params()` consistently with torch — verified 2026-03-28

### R-18: Respect Explicit torch_quantization on Turing GPUs ✅ Fixed
Currently, the CUDA optimization code overrides `torch_quantization` to "8-bit" on Turing GPUs (T4) even if user explicitly sets a different value.

- **Location**: `qwen3_tts/core/engine/model_loader.py:186-198` (Turing override logic)
- **Fix**: Already implemented — override only applies when config key missing, not when explicitly set — verified 2026-03-28

### R-19: Thread-Safe request_queue ✅ Fixed
The `state.request_queue` is a plain `set()` with no locking. Add thread-safe protection.

- **Location**: `qwen3_tts/server/app_lifespan.py:181-182`
- **Fix**: `threading.Lock()` (`request_queue_lock`) already implemented with proper `with` usage — verified 2026-03-28

### R-20: Symlink Resolution in /preview-prompt ✅ Fixed
The `/preview-prompt` endpoint should resolve symlinks in the voice prompt path to prevent potential security issues.

- **Location**: `qwen3_tts/server/app_prompts.py:221-225`
- **Fix**: Uses `os.path.realpath()` with security check — already implemented (covered by R-40)

## Priority 4 — Low Priority

### R-21: Cache pysbd.Segmenter per Language ✅ Fixed
The sentence segmentation parser is created on each chunk. Cache it per language to reduce overhead.

- **Location**: `qwen3_tts/core/engine/text_processing.py:18,341-343`
- **Fix**: `_SEGMENTER_CACHE = {}` dict with language-code caching — already implemented

### R-22: Cache num2words Import ✅ Fixed
The num2words import happens inside the text normalization function. Lazy-load and cache the import.

- **Location**: `qwen3_tts/core/engine/text_processing.py:16,233-241`
- **Fix**: Module-level `_n2w_cached` lazy import with `_n2w_loaded` flag — already implemented

### R-23: LUFS Normalization Option
Add optional LUFS (loudness) normalization to audio post-processing for broadcast-quality output.

- **Library**: `pyloudnorm` or similar
- **Config**: `generation.lufs_target` (e.g., -16.0 for EBU R128)
- **Default**: Disabled (current behavior)

### R-24: Pagination for /prompts ✅ Fixed
The `/prompts` endpoint returns all voice prompts at once. Add pagination for large prompt collections.

- **Location**: `qwen3_tts/server/app_prompts.py:55-71` (handle_list_prompts)
- **Fix**: Pagination with offset/limit query params already implemented — verified 2026-03-28
- **Query params**: `?offset=0&limit=50`
- **Response**: Includes total count, offset, and limit metadata

### R-25: Document Streaming Wire Format ✅ Fixed
The `/generate-stream` endpoint returns raw float32 chunks, but this format is not documented.

- **Location**: `CLAUDE.md` after Server API table
- **Fix**: Added wire format spec with Python and JavaScript consumption examples — implemented 2026-03-28

### R-26: Audit Logging for Auth Failures ✅ Fixed
Add audit logging for authentication failures to detect potential brute-force attacks.

- **Location**: `qwen3_tts/server/app.py:166-189`
- **Fix**: Enhanced `verify_auth()` to log failure reason (missing_token, invalid_token) — implemented 2026-03-28

### R-27: Configurable Silence Gap
The silence gap inserted between chunks is currently hardcoded. Make it configurable.

- **Config key**: `generation.silence_gap_seconds` (float)
- **Default**: 0.0 (current behavior)
- **Implementation**: Apply in chunk concatenation logic

### R-33: _validate_prompt_name Return Type Annotation ✅ Fixed
The annotation says `Optional[tuple]` but the actual return type is `Optional[tuple[dict, int]]`.

- **Location**: `qwen3_tts/server/validation.py:181`
- **Fix**: Updated annotation to `Optional[tuple[dict, int]]` — implemented 2026-03-28

### R-34: X-Queue-Position Read Without Lock ✅ Fixed (documented tradeoff)
`state.pending_requests` length is read to set the `X-Queue-Position` header without holding `pending_lock`.

- **Location**: `qwen3_tts/server/app_generation.py:482`
- **Fix**: Deliberate tradeoff comment already present — "Approximate: read without lock since response is already committed. Exact position available via /queue-status endpoint." — verified 2026-03-28

### R-35: Streaming chunk_total Not Populated ✅ Fixed
`/generate-stream` never updates `chunk_total` in `generation_state`. Callers polling `/generation-status` for progress see `chunk_total=0` throughout streaming.

- **Location**: `qwen3_tts/server/app_generation.py` (streaming path), `qwen3_tts/core/engine/inference.py`
- **Fix**: Added `progress_callback` parameter to streaming chain; MLX path calls with `chunk_total=0` during streaming then `chunk_total=final_count` at completion; torch path calls with known `chunk_total` upfront — implemented 2026-03-28
- **Follow-up R-35b** ✅ Fixed: `/generate` (non-streaming) endpoint also omitted `chunks` from its response, so the Gradio UI history panel always displayed 0. Fixed by adding `"chunks": state.generation_state.get("chunk_total", 0)` to each result dict in `app_generation.py`; client stores `last_chunk_count` attribute; UI reads it via `getattr(client, "last_chunk_count", 0)` and passes to `add_to_history()` and the JSON metadata sidecar — implemented 2026-04-10

### R-36: generate_dialogue Uses list.extend() on NumPy Arrays ✅ Fixed
`list.extend()` on numpy arrays iterates scalar-by-scalar, which is very slow for large audio arrays.

- **Location**: `qwen3_tts/server/client/generator.py:446`
- **Fix**: Replaced with `np.concatenate()` for O(n) single-allocation — implemented in Python review remediation

### R-37: Fix _validate_prompt_name Return Type ✅ Fixed (same as R-33)
`_validate_prompt_name` in `validation.py:181` has an inconsistent return type (`Optional[tuple[dict, int]]` but returns `None` or a tuple). Add type annotation and Optional return type.

- **Location**: `qwen3_tts/server/validation.py:181`
- **Fix**: `-> Optional[tuple[dict, int]]` annotation added — implemented 2026-03-28 (R-33)

### R-38: Lock eta_cache Read-Modify-Write in app_lifespan.py ✅ Fixed
`eta_cache` in `app_lifespan.py:61-95` has an unprotected read-modify-write sequence that could race under concurrent requests.

- **Location**: `qwen3_tts/server/app_lifespan.py:61-95`
- **Fix**: `threading.Lock()` added alongside `eta_cache`; update block wrapped with `with` — implemented 2026-03-28

### R-39: Make write_pid_file Atomic via Temp-File + os.replace() ✅ Fixed
`write_pid_file` in `config.py:472-473` writes the PID file non-atomically; a crash mid-write leaves a partial PID.

- **Location**: `qwen3_tts/core/config.py:472-473`
- **Fix**: Write to a temp file, then `os.replace()` for atomic rename — implemented 2026-03-28

### R-40: Serve realpath in FileResponse for preview_prompt ✅ Fixed
`handle_preview_prompt` resolves the real path for security checks but then passes the original `wav_path` to `FileResponse`. Should serve `real_path`.

- **Location**: `qwen3_tts/server/app_prompts.py:214-222`
- **Fix**: `FileResponse(real_path, media_type="audio/wav")` — implemented 2026-03-28

### R-41: Wrap os.remove in try/except with Partial-Failure Reporting in delete_prompt ✅ Fixed
`handle_delete_prompt` calls `os.remove()` without per-file error handling. A partial delete should report which files failed.

- **Location**: `qwen3_tts/server/app_prompts.py:93-97`
- **Fix**: Wrap each `os.remove` in `try/except OSError` and collect failures — implemented 2026-03-28

### R-42: Add json.JSONDecodeError Handling in load_config ✅ Fixed
`load_config` in `config.py:137` does not catch `json.JSONDecodeError`, so a corrupt config.json causes an unhandled exception.

- **Location**: `qwen3_tts/core/config.py:137`
- **Fix**: Raises `ValueError` with clear message; all call sites updated to also catch `ValueError` — implemented 2026-03-28

### R-43: Refactor create_voice.main() to Accept Args Directly (PARTIAL)
`create_voice.main()` uses argparse (standard pattern) but still reads from `sys.argv` via `parser.parse_args()`. Can be made more directly testable by accepting `args` parameter.

- **Location**: `qwen3_tts/tools/create_voice.py:218-240`
- **Current**: Uses `argparse.ArgumentParser()` with `parser.parse_args()` (reads sys.argv by default)
- **Improvement**: Accept optional `args` parameter for easier testing without subprocess/mocking
- **Note**: Core logic already separated into `create_and_save_voice_prompt()` function (testable)

### R-44: Add Cancellation Check in Non-Streaming Batch Loop ✅ Fixed
`handle_generate` in `app.py:462-464` processes all chunks in a batch loop without checking for cancellation. A cancelled request continues generating.

- **Location**: `qwen3_tts/server/app.py:462-464`
- **Fix**: Check `generation_state["cancelled"]` inside the loop — implemented 2026-03-28

### R-45: Apply sanitize_log to model_name Consistently in app_lifespan.py ✅ Fixed
`app_lifespan.py:262` logs `model_name` without `sanitize_log`, inconsistent with other log calls.

- **Location**: `qwen3_tts/server/app_lifespan.py:262`
- **Fix**: `logger.info("...", sanitize_log(model_name))` — implemented 2026-03-28

### R-46: Narrow dtype-restore except in inference.py ✅ Fixed
`inference.py:189` uses broad `except Exception` for dtype restore. Should narrow to `except (RuntimeError,)`.

- **Location**: `qwen3_tts/core/engine/inference.py:189`
- **Fix**: `except (RuntimeError, TypeError) as e:` — implemented 2026-03-28

### R-47: Change VOICE_PROMPTS_DIR to pathlib.Path ✅ Fixed
`config.py:41` defines `VOICE_PROMPTS_DIR` as a string, but `uninstall.py` uses `.exists()` and `.glob()` on it (pathlib methods). Make it a `pathlib.Path`.

- **Location**: `qwen3_tts/core/config.py:41`
- **Fix**: `VOICE_PROMPTS_DIR = Path(...)` — implemented 2026-03-28 (with caller boundary casts for `str` ops)

### R-48: Remove f-prefix from plain string in cli_config.py ✅ Fixed (was already clean)
`cli_config.py:116` uses an f-string with no interpolation (`f"default_voice_description updated"`).

- **Location**: `qwen3_tts/cli_config.py:116`
- **Fix**: ruff F541 check found no violations — already resolved in an earlier session

### R-49: Move Imports to Top of uninstall.py ✅ Fixed
`uninstall.py:15,25,26` has non-top-level imports (ruff E402).

- **Location**: `qwen3_tts/tools/uninstall.py:15,25,26`
- **Fix**: Moved imports above `logger = ...` declaration — implemented 2026-03-28

### R-50: Concurrent Generation Race Condition ✅ Fixed
`list index out of range` error occurred during simultaneous generation requests from multiple browser tabs.

- **Location**: `qwen3_tts/interface/ui/generation.py`
- **Root cause**: Gradio's `history_state` shared across all tabs without thread-safe concurrent access protection
- **Fix**: Added `threading.Lock()`, defensive type checking, list copying on input/output, lock-protected operations — implemented 2026-03-29
- **Tests**: E2E concurrent generation test passes (2 tabs clicking Generate simultaneously)
`uninstall.py:15,25,26` has non-top-level imports (ruff E402).

- **Location**: `qwen3_tts/tools/uninstall.py:15,25,26`
- **Fix**: Moved imports above `logger = ...` declaration — implemented 2026-03-28

## Priority 5 — Future / Upstream-Dependent

### R-29: Unconstrained TranscribeRequest.language Field
The `language` field in `TranscribeRequest` is an unconstrained string passed directly to whisper's `generate_kwargs`. A malformed value could cause unexpected ASR behavior.

- **Location**: `qwen3_tts/server/validation.py:86`
- **Fix**: Add `Field(pattern=r'^[a-z]{2,3}$')` or an enum of supported ISO 639-1 codes

### R-30: Unbounded base64 Audio Payload ✅ Fixed
`TranscribeRequest.audio_base64` and `CreateVoicePromptRequest.audio_base64` have no `max_length` constraint. Large payloads are buffered entirely into memory before Pydantic validation.

- **Location**: `qwen3_tts/server/validation.py:86, 92`
- **Fix**: Added an ASGI `limit_request_body_size` middleware in `app.py` that rejects over-limit requests with HTTP 413 by `Content-Length` before parsing (`MAX_REQUEST_BODY_BYTES = 2 * MAX_AUDIO_BASE64_BYTES`) — implemented 2026-07-06

### R-31: Speaker Validation Case Normalization Gap ✅ Fixed
`"RYAN"` fails validation even though `"Ryan"` and `"ryan"` both pass. The lowercase key check runs first but the fallback `_VALID_SPEAKER_NAMES` check uses the raw `req.speaker` value.

- **Location**: `qwen3_tts/server/validation.py:166`
- **Fix**: Lowercase `req.speaker` before the `_VALID_SPEAKER_NAMES` fallback check — implemented 2026-03-28

### R-32: cleanup_pid TOCTOU Race ✅ Fixed
`os.path.exists()` followed by `os.remove()` in the shutdown path has a race window where the file can be deleted between the two calls.

- **Location**: `qwen3_tts/server/app_lifespan.py:327`
- **Fix**: Replace with `try: os.remove(TOKEN_FILE) except FileNotFoundError: pass` — implemented 2026-03-28

### R-28: Speculative Decoding for Inference Acceleration
Integrate speculative decoding to achieve 1.5-3x inference speedup by using the 0.6B model as a draft for the 1.7B model. Requires upstream library support.

- **Research:** See [speculative-decoding-research.md](2026-03-23-speculative-decoding-research.md) for full feasibility analysis
- **Key finding:** EAGLE-3 is not directly applicable (designed for text LLMs, not audio codecs). The most viable path is vanilla speculative decoding with PCG-style acoustic verification.
- **Upstream prerequisites:**
  - `qwen_tts` library adds `draft_model` or `speculative_config` parameter to `generate()`
  - `mlx_audio` exposes hook points in autoregressive loop for draft/verify
  - HF Transformers adds assisted decoding support for non-CausalLM architectures
- **Phased approach:**
  1. Monitor upstream releases (current phase)
  2. Prototype 0.6B-as-draft with PCG verification when upstream supports it
  3. Integrate via `config.json` option and model_loader changes

---

## Implementation Notes

- Priority 3 items are medium priority and should be considered for the next minor release.
- Priority 4 items are low priority and should be considered as time permits.
- All changes should maintain backward compatibility with existing configurations.
- Each item should have corresponding tests added before implementation.
