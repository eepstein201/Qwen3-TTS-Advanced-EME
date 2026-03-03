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

---

## Implementation Notes

- Priority 3 items are medium priority and should be considered for the next minor release.
- Priority 4 items are low priority and should be considered as time permits.
- All changes should maintain backward compatibility with existing configurations.
- Each item should have corresponding tests added before implementation.
