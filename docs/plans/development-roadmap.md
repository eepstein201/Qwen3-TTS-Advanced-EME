# Qwen3-TTS Development Roadmap

This document outlines future improvements identified during the multi-agent code review (2025). Priority 1 and 2 items have been implemented in commits 1-5. This roadmap covers Priority 3 (Medium) and Priority 4 (Low) items for future consideration.

## Priority 3 — Medium Priority

### R-13: Rate Limiting
Implement API rate limiting to prevent abuse and ensure fair resource allocation.

- **Tool**: `slowapi` (recommended) or similar
- **Scope**: Per-IP and per-token rate limits
- **Configurable limits**: Via config.json
- **Endpoints to protect**: `/generate`, `/generate-stream`, `/load-model`, `/update-model-config`

### R-14: Crossfade Between Chunks
When text is split into chunks and generated separately, add smooth crossfading between segments to reduce audible artifacts at chunk boundaries.

- **Location**: `qwen3_tts/core/engine.py` post-processing
- **Duration**: Configurable crossfade duration (default 50ms)
- **Implementation**: Overlap and mix audio at chunk boundaries

### R-15: Split engine.py into Modules
The current `engine.py` is very large (~1800 lines). Split into logical modules:

- `engine/model_loader.py` — Model loading logic (torch/mlx)
- `engine/inference.py` — Core inference functions
- `engine/audio_processing.py` — Audio utilities (trim, normalize, validate)
- `engine/voice_prompt.py` — Voice prompt loading and caching
- `engine/text_processing.py` — Text normalization and chunking

### R-16: Model Warm-up Pass
Add a warm-up inference pass after model loading to ensure all kernels are compiled and memory is allocated before the first real request.

- **Benefit**: More consistent latency for first generation
- **Implementation**: Run a short dummy inference after model load completes

### R-17: Temperature Consistency (Torch vs MLX)
The default temperature differs between backends (0.7 for torch, 0.9 for MLX). Standardize to use the same default from config.

- **Location**: `_run_inference_mlx` (hardcoded 0.9) should read from gen_params like torch does
- **Fix**: Remove hardcoded `temperature=0.9` default

### R-18: Respect Explicit torch_quantization on Turing GPUs
Currently, the CUDA optimization code overrides `torch_quantization` to "8-bit" on Turing GPUs (T4) even if user explicitly sets a different value.

- **Location**: `qwen3_tts/core/config.py` in `get_optimal_torch_quant()`
- **Fix**: Only apply override if config value is None/auto, not if explicitly set

### R-19: Thread-Safe request_queue
The `state.request_queue` is a plain `set()` with no locking. Add thread-safe protection.

- **Current**: `app.state.request_queue = set()`
- **Fix**: Use `threading.Semaphore` or wrap access with lock

### R-20: Symlink Resolution in /preview-prompt
The `/preview-prompt` endpoint should resolve symlinks in the voice prompt path to prevent potential security issues.

- **Location**: `qwen3_tts/server/app.py` `/preview-prompt` endpoint
- **Fix**: Use `os.path.realpath()` before returning audio

## Priority 4 — Low Priority

### R-21: Cache pysbd.Segmenter per Language
The sentence segmentation parser is created on each chunk. Cache it per language to reduce overhead.

- **Location**: Text chunking code in `engine.py`
- **Implementation**: `dict` mapping language code → cached Segmenter instance

### R-22: Cache num2words Import
The num2words import happens inside the text normalization function. Lazy-load and cache the import.

- **Location**: `_normalize_text()` in `engine.py`
- **Implementation**: Module-level `_NUM2WORDS = None` cache

### R-23: LUFS Normalization Option
Add optional LUFS (loudness) normalization to audio post-processing for broadcast-quality output.

- **Library**: `pyloudnorm` or similar
- **Config**: `generation.lufs_target` (e.g., -16.0 for EBU R128)
- **Default**: Disabled (current behavior)

### R-24: Pagination for /prompts
The `/prompts` endpoint returns all voice prompts at once. Add pagination for large prompt collections.

- **Query params**: `?offset=0&limit=50`
- **Response**: Include total count and pagination metadata

### R-25: Document Streaming Wire Format
The `/generate-stream` endpoint returns raw float32 chunks, but this format is not documented.

- **Action**: Add documentation explaining the binary format
- **Include**: Sample code for consuming the stream in Python and JavaScript

### R-26: Audit Logging for Auth Failures
Add audit logging for authentication failures to detect potential brute-force attacks.

- **Log level**: WARNING
- **Include**: IP address, timestamp, failure reason (rate-limited vs invalid token)
- **Retention**: Configurable via `security.audit_log_retention_days`

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

### R-35: Streaming chunk_total Not Populated
`/generate-stream` never updates `chunk_total` in `generation_state`. Callers polling `/generation-status` for progress see `chunk_total=0` throughout streaming.

- **Location**: `qwen3_tts/server/app_generation.py` (streaming path)
- **Fix**: Update `generation_state["chunk_total"]` when chunk count is known, matching the non-streaming path

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

### R-43: Refactor create_voice.main() to Accept Args Directly
`create_voice.main()` reads from `sys.argv` directly, making it untestable without subprocess calls.

- **Location**: `qwen3_tts/cli_voice.py:62-68`
- **Fix**: Accept `args` parameter instead of reading from `sys.argv`

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

## Priority 5 — Future / Upstream-Dependent

### R-29: Unconstrained TranscribeRequest.language Field
The `language` field in `TranscribeRequest` is an unconstrained string passed directly to whisper's `generate_kwargs`. A malformed value could cause unexpected ASR behavior.

- **Location**: `qwen3_tts/server/validation.py:86`
- **Fix**: Add `Field(pattern=r'^[a-z]{2,3}$')` or an enum of supported ISO 639-1 codes

### R-30: Unbounded base64 Audio Payload
`TranscribeRequest.audio_base64` and `CreateVoicePromptRequest.audio_base64` have no `max_length` constraint. Large payloads are buffered entirely into memory before Pydantic validation.

- **Location**: `qwen3_tts/server/validation.py:86, 92`
- **Fix**: Add a server-level body size limit (uvicorn `--limit-max-requests`, FastAPI `Body(max_length=...)`, or a middleware check)

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
