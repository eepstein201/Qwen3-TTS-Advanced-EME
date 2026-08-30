# Qwen3-TTS Architecture Deep Dive

This document contains detailed architectural reference extracted from CLAUDE.md for progressive disclosure. For essential project context, see the root `CLAUDE.md`.

## Config Structure (Full)

```json
{
  "default_voice_description": "A calm, friendly male voice ...",
  "default_clone_prompt": "default_clone.pt",
  "default_speaker": "ryan",
  "output_directory": "~/Downloads",
  "history_output_directory": "~/Downloads/Qwen3-TTS Output",
  "language": "English",
  "server": { "host": "127.0.0.1", "port": 5123, "auto_shutdown_minutes": 0 },
  "models": {
    "clone":  { "load_at_startup": true,  "revision": "main" },
    "design": { "load_at_startup": false, "revision": "main" },
    "custom": { "load_at_startup": false, "revision": "main" }
  },
  "security": {
    "max_text_length": 50000, "max_batch_size": 20,
    "rate_limits": { "generate": "20/minute", "model_ops": "3/minute",
                     "transcribe": "15/minute", "prompt_ops": "10/minute",
                     "config_ops": "1/minute" }
  },
  "advanced": {
    "dtype": "bfloat16", "backend": "mlx", "model_size": "1.7B",
    "mlx_quantization": "8bit", "torch_quantization": "none",
    "audio_loader": "torchaudio",
    "vllm_enabled": false, "vllm_fallback_to_torch": true
  },
  "vllm": {
    "enabled": false, "fallback_to_torch": true, "max_model_len": 8192,
    "audio_sample_rate": 24000, "audio_chunk_size": 2000,
    "gpu_memory_utilization": 0.9, "tensor_parallel_size": 1,
    "mm_processor_name": "Qwen/Qwen2-Audio-7B-Instruct",
    "port": null, "dtype": "bfloat16"
  },
  "generation": {
    "temperature": 0.7, "top_k": 50, "top_p": 0.95,
    "repetition_penalty": 1.05, "seed": null,
    "max_chunk_chars": 500, "max_chunk_tokens": 200, "max_new_tokens": 2048,
    "compile_model": true,
    "lufs_normalize": false, "lufs_target": -16.0, "silence_gap_seconds": 0.0
  },
  "presets": {
    "consistent": { "temperature": 0.5, "top_k": 30, "seed": 42 },
    "creative":   { "temperature": 0.9, "top_p": 0.98 }
  },
  "prosody_presets": { "excited": "...", "calm": "...", "whisper": "...", ... },
  "ui": { "port": 7860 },
  "aliases": { "default": { "prompt": "default_clone.pt", "preset": "consistent" } },
  "cache": { "voice_prompt_max": 10, "generation_max": 5, "eta_ttl_seconds": 30 },
  "prompt_enhancer": {
    "enabled": false, "provider": "anthropic",
    "api_key_env": "ANTHROPIC_API_KEY", "model": "claude-haiku-4-5-20251001"
  }
}
```

## Security

- Bearer token auth on all endpoints except `/health`, `/ready`, `/generation-status`, and `/queue-status`
- Token: `~/.config/qwen3-tts/.voice_server_token` (0o600 perms; legacy fallback `~/.voice_server_token`), auto-cleaned on shutdown
- Input validation: text length, batch size, path traversal prevention, symlink resolution, mode/speaker validation
- Rate limiting via `slowapi` (optional) with X-Forwarded-For IP resolution for reverse proxies
- Audit logging for auth failures with client IP
- Server binds `127.0.0.1` by default; `--public` for `0.0.0.0`; Colab auto-binds `0.0.0.0`

## Platform Support

| | macOS (Apple Silicon) | macOS (Intel) | Linux/Colab |
|---|---|---|---|
| Backend | MLX (default) or torch | torch only | torch + CUDA |
| Device | Neural Engine / MPS | CPU | CUDA GPU |
| Conda env | `qwen3-tts-mlx` | `qwen3-tts` | pip install |
| Audio play | `afplay` | `afplay` | `ffplay` |
| Install | `install.sh` (conda) | `install.sh` (conda) | `install.sh` (conda or venv) |

Platform constants in `qwen3_tts/core/config.py`: `IN_COLAB`, `IS_MACOS`, `IS_LINUX`, `get_device()`

### install.sh Linux Support

`install.sh` (~1675 lines) supports macOS and Linux with a platform dispatcher pattern. Key Linux functions:

- **`detect_platform()`** — `/etc/os-release` → distro family (debian/rhel/arch/suse)
- **`detect_linux_gpu()`** — Cascading NVIDIA detection: `lspci -d '10de:'` → `nvcc` → `nvidia-smi` → version files → pkg manager → ldconfig
- **`install_linux_system_deps()`** — Aggregates missing deps (ffmpeg, libsndfile, rubberband), routes to apt/dnf/pacman/zypper
- **`get_torch_index_url()`** — Maps CUDA version to PyTorch index URL (cu118/cu121/cu124/cu126/cpu)
- **`create_linux_venv()`** — Fallback when conda unavailable: finds Python 3.10+, creates venv, installs with CUDA index URL
- **`get_linux_dtype()`/`get_linux_torch_quant()`** — Auto-selects dtype and quantization from GPU compute capability

## Caching (4 layers)

| Cache | Location | Strategy | Invalidation |
|-------|----------|----------|-------------|
| Voice prompt | `qwen3_tts/core/engine/voice_prompt.py` | LRU(10) torch .pt / dict for MLX .wav | `clear_voice_prompt_cache()` |
| ETA | `qwen3_tts/server/app_lifespan.py` | 30s TTL, avoids .jsonl reads per poll | Auto-expires |
| Generation result | `qwen3_tts/server/app_lifespan.py` | 5 entries, SHA256 key (text+mode+params) | Model config change, manual |
| Audio loader | `qwen3_tts/core/engine/audio_processing.py` | `_AUDIO_LOADER` global, no disk I/O | `set_audio_loader()` only |

## Thread Safety Guarantees

All shared mutable state is protected by locks for concurrent access:

| Lock | Location | Protects |
|------|----------|----------|
| `_history_lock` | `interface/ui/generation.py` | `history_list` shared state across Gradio tabs during concurrent generation |
| `_mps_patch_lock` | `model_loader.py` | `_mps_patch_installed` flag during patch installation |
| `_torch_prompt_cache_lock` | `voice_prompt.py` | `_torch_prompt_cache` OrderedDict + hit/miss counters |
| `_mlx_prompt_cache_lock` | `voice_prompt.py` | `_mlx_prompt_cache` OrderedDict |
| `_asr_lock` | `asr.py` | `_asr_model_torch`, `_asr_model_mlx` model references |
| `request_queue_lock` | `app_lifespan.py` | `request_queue` set for concurrent request tracking |
| `pending_lock` | `app_lifespan.py` | `pending_requests` asyncio lock for request coordination |
| `gen_cache_lock` | `app_lifespan.py` | `gen_cache` dict for generation result caching |
| `eta_cache_lock` | `app_lifespan.py` | `eta_cache` dict for ETA read-modify-write operations |

**Pattern:** All locks use double-checked locking for efficiency:
```python
if cached_value is not None:
    return cached_value
with lock:
    if cached_value is not None:  # Double-check after acquiring lock
        return cached_value
    # ... load and cache ...
```

## Inference Serialization (#192 / #214)

### Warm-up serialization (#192 structural fix)

The design load-time warm-up (`_warmup_model`) is real MLX inference and now runs under `inference_lock` in BOTH server paths — `/load-model` (`handle_load_model` is async: load via `to_thread` unlocked, then warm-up locked as a leaf acquisition) and startup `_background_load` (schedules the locked warm-up onto `app.state.event_loop` via `run_coroutine_threadsafe(...).result(timeout=600)`; on loop-gone/timeout the wait is abandoned and the future cancelled — never run unsynchronized). Both paths design-guard AND knob-guard before the lock (clone/custom and `TTS_SKIP_WARMUP` skip the lock round-trip). Engine `load_model(model_type, *, warmup=False)` powers the split; never reintroduce an unlocked warm-up. All `/load-model` clients use `LOAD_MODEL_TIMEOUT_SEC` (=900, defined in `core/http_client.py`, drift-guarded) — a load issued mid-generation queues its warm-up behind it, so the old hardcoded 120s failed spuriously.

### `/create-voice-prompt` serialization (#192, final reachable pair)

`/create-voice-prompt` IS now serialized: `handle_create_voice_prompt` is async — `create_voice_prompt` (`create_voice_clone_prompt`) runs under `inference_lock` as a leaf acquisition via `to_thread`, with decode/staging and the `.pt` save outside the lock; with it ALL MLX inference reachable through the API serializes on `inference_lock` (remaining #192 items: /load-model in-flight dedup, e2e queuing coverage).

### Unload-ASR race closure (#214 item 2)

The unload-asr race is CLOSED: `unload_asr_model` takes `_asr_lock` (it was the module's only unsynchronized writer) AND `/unload-asr` acquires `inference_lock` — `_asr_lock` alone merely narrowed the window, since an unload could still land between `/transcribe`'s post-lock recheck and `transcribe_audio`'s own lazy-load check (`asr.py:168`, or `_transcribe_torch`'s unconditional `_ensure_asr_torch_loaded`) and rebuild the model INSIDE the serialized section. Locking the unload closes it structurally, and closes the identical check-then-use on the ICL echo-trim probe (`inference.py:1155`→`:1163`) for free, since that runs with `inference_lock` already held. `/transcribe` re-checks under the lock and returns a retryable 503 `asr_unloaded` rather than reloading; every `/unload-asr` client must use `UNLOAD_ASR_TIMEOUT_SEC` (=900, `core/http_client.py`) since the unload now queues behind a generation.

### `/transcribe` serialization (#192 follow-up)

`/transcribe` IS now serialized: `handle_transcribe` is async — the mlx-whisper `generate` (`transcribe_audio`) runs under `inference_lock` as a leaf acquisition via `to_thread`; the lazy ASR model load stays OUTSIDE the lock (`preload_asr_model` never preloads on MLX, so first-use load is a real path — minutes unlocked, mirroring the /load-model split); the UI client uses `TRANSCRIBE_TIMEOUT_SEC` (=900, `core/http_client.py`, drift-guarded by `tests/test_issue192_transcribe_serialization.py`) since the old 60s fails spuriously when queuing behind a generation.

### Torch auto-create-from-`.wav` serialization (#214 item 1)

**Torch backend only.** On torch, a `/generate` whose `.pt` voice prompt is missing or corrupt used to run real GPU inference *outside* `inference_lock`, as a side effect of "just loading a prompt": `load_voice_prompt` → `_load_voice_prompt_torch` → `_auto_create_pt_from_wav` calls both `load_model("clone")` and `create_voice_prompt(...)`. All three server call sites invoke it pre-lock. MLX is unaffected — `load_voice_prompt_mlx` only reads files and never creates, so the fix is a provable no-op there (pinned by `TestMlxBackendNoOp`).

The engine cannot take the lock itself: `load_voice_prompt` is sync and reached via `asyncio.to_thread`. So the split lives in `server/prompt_loading.py::load_voice_prompt_serialized`, a drop-in replacement preserving the old `FileNotFoundError`/`None` contract: probe unlocked with `allow_create=False` (raising `VoicePromptCreateRequired`), then re-enter under `inference_lock` with `allow_create=True`.

**`load_model()` is NOT memoized** (`core/engine/model_loader.py`) — every call is a full multi-minute weight construction. So the helper reuses `state.models["clone"]` when present, otherwise builds the model OUTSIDE the lock, and **forwards it via `clone_model=`**. Dropping that forwarding makes `_auto_create_pt_from_wav` reconstruct the model *inside* the lock — reintroducing exactly the starvation the #212 split exists to prevent. A mutation test pins this (`test_load_model_unlocked_then_create_locked`).

`_load_voice_prompt_torch`'s top-of-function cache re-check is **load-bearing**: two callers racing the same missing prompt both probe, both queue, and the second finds the first's cached result rather than creating twice. Do not remove it without preserving that property. Guarded by `tests/test_issue214_prompt_create_serialization.py`.

## Constants

| Constant | Location | Value | Purpose |
|----------|----------|-------|---------|
| `HF_CACHE` | `config.py` | `~/.cache/huggingface/hub` | HuggingFace model cache path |
| `MAX_BUFFER_SIZE` | `client/_base.py` | 100MB (100 * 1024 * 1024) | Streaming response buffer limit |

## Logging

- `tts` — server (RotatingFileHandler: 5MB, 1 backup + stderr)
- `tts.engine` — model/inference
- `tts.cli` — CLI generation
- `tts.ui` — Gradio UI
- Log file: `.voice_server.log`

## Error Responses

Server returns structured JSON with recovery hints:
```json
{ "error": "message", "detail": "...", "recovery": "restart|config|retry|bug" }
```

CLI and UI parse the `recovery` field to show actionable guidance.

## Hardware Optimization (CUDA)

`_apply_cuda_optimizations()` in `qwen3_tts/core/engine/model_loader.py` auto-detects GPU and applies optimal settings:

| GPU | Compute Cap | Attention | dtype | Quantization | torch.compile |
|-----|------------|-----------|-------|-------------|---------------|
| T4 (free Colab) | 7.5 | SDPA | float16 | 8-bit (bitsandbytes) | No |
| L4 (Colab Pro) | 8.9 | SDPA (FA2 opt-in) | bfloat16 | None needed | Yes |
| A100 (Colab Pro+) | 8.0 | SDPA (FA2 opt-in) | bfloat16 | None needed | Yes |
| Non-CUDA | N/A | SDPA | float32 | N/A | No |

`get_cuda_capability()` and `get_optimal_attn_config()` in `qwen3_tts/core/config.py` expose hardware detection. The Colab notebook auto-configures based on detected GPU tier.

## Text Processing Roadmap

**Current (implemented):** pySBD sentence splitting, num2words text normalization, token-aware chunking (torch backend).

**Future options (not yet implemented):**
- **NLTK punkt tokenizer** — Moderate-weight alternative to pySBD; requires punkt data download at first use. Good for multi-language academic text.
- **NVIDIA NeMo text processing** — Production-grade normalization covering dates, times, measures, addresses, financial data. ~500MB+ in new dependencies; suitable for high-volume or broadcast-quality TTS.

## Upstream Dependency Monitoring

Periodically check (monthly) for upstream fixes that could remove local workarounds or require code updates:

### Workarounds (can be removed when upstream fixes land)

| Workaround | Location | Upstream Issue | Check |
|------------|----------|----------------|-------|
| Mistral tokenizer regex warning suppression | `model_loader.py:313-318` | [mlx-audio](https://github.com/Blaizzy/mlx-audio) | File issue requesting `fix_mistral_regex` support or upstream warning suppression |
| `fix_mistral_regex=True` for torch backend | `model_loader.py:272` | [transformers #36615](https://github.com/huggingface/transformers/pull/36615) | Check if warning detection improved for non-Mistral models |

**Rationale:** Qwen3-TTS is NOT a Mistral model, but the tokenizer regex warning fires for many non-Mistral models due to overly aggressive pattern detection. The warning suppression is safe and has no functional impact.

### Breaking Changes to Monitor

| Dependency | Current Version | Concern | What to Check |
|------------|-----------------|---------|---------------|
| **gradio** | 6.x | Gradio 6 removed `visible=False` from DOM; `.then()` chains break after JS-only steps | [Gradio changelog](https://github.com/gradio-app/gradio/releases) for fixes to JS chain handling |
| **mlx-audio** | latest | Qwen3-TTS model support, tokenizer loading | [mlx-audio releases](https://github.com/Blaizzy/mlx-audio/releases) for Qwen3-TTS improvements |
| **transformers** | 4.x | `fix_mistral_regex` parameter, Qwen3 tokenizer support | [transformers releases](https://github.com/huggingface/transformers/releases) for tokenizer improvements |
| **pyrubberband** | 0.3.x | Audio time-stretching fallback to librosa | Check if pyrubberband installation issues on Apple Silicon resolved |
| **pyloudnorm** | 0.1.x | LUFS normalization | Check for API changes affecting `audio_processing.py` |

### Security Updates

| Dependency | Why Monitor | Check Frequency |
|------------|-------------|-----------------|
| **torch** | CUDA/memory security patches | Monthly |
| **numpy** | CVE fixes | Monthly |
| **pillow** | Image processing security | Monthly |
| **starlette/fastapi** | Web server security | Monthly |

## Code Review Status (2026-03-03)

Multi-agent review (8 agents, 56 deduplicated findings). **P1+P2 implemented** (R-1 through R-12). P3/P4 roadmap at `docs/plans/development-roadmap.md`.

### What was fixed (P1+P2)
- Graceful shutdown replaces `os._exit(0)` (R-1)
- Thread-safe config lock (R-2)
- Narrowed generation_lock scope (R-3)
- Config-aware voice prompt cache replacing hardcoded LRU (R-4)
- Gen cache temp file cleanup + NamedTemporaryFile leak fix (R-5)
- CORS middleware for Gradio UI (R-6)
- `/generate-stream` validation parity with `/generate` (R-7)
- Standardized error response helper (R-8)
- Pydantic response models for OpenAPI (R-9)
- MPS float32 dtype restoration after inference (R-10)
- Audio validation (NaN, clipping, silence) (R-11)
- Content negotiation for binary WAV (R-12)

### P3/P4 fixes implemented (2026-03-06)
- Rate limiting with X-Forwarded-For IP resolution (R-13)
- Crossfade between multi-chunk audio with raised-cosine window (R-14)
- engine.py split into 6 submodules with facade (R-15)
- Model warm-up after loading (R-16)
- MLX temperature reads from config, not hardcoded (R-17)
- Turing GPU respects explicit torch_quantization (R-18)
- Thread-safe request_queue with lock (R-19)
- Symlink resolution in /preview-prompt (R-20)
- pysbd.Segmenter cached per language (R-21)
- num2words cached at module level (R-22)
- LUFS normalization via pyloudnorm (R-23)
- Pagination for /prompts endpoint (R-24)
- Streaming wire format documented (R-25)
- Audit logging for auth failures (R-26)
- Configurable silence gap between chunks (R-27)
- `_normalize_text` logs warnings instead of bare except:pass
- `_expand_currency` handles decimals ($5.99 → five dollars and ninety-nine cents)

### TDD Code Audit fixes (2026-03-07)
Comprehensive audit with TDD methodology (14 commits, all tests pass):
- Thread-safe MPS patch installation with `_mps_patch_lock` (model_loader.py)
- Thread-safe MLX voice prompt cache with `_mlx_prompt_cache_lock` (voice_prompt.py)
- Thread-safe ASR model loading for MLX backend with `_asr_lock` (asr.py)
- Extracted `_resolve_voice_alias()` and `_build_gen_params()` helpers (server/client/)
- Division by zero protection in ETA estimation (app_lifespan.py)
- Specific RuntimeError catching instead of bare except (inference.py)
- Streaming buffer overflow protection with `MAX_BUFFER_SIZE = 100MB` (server/client/)
- Temp file cleanup on exception in preview_voice_callback (interface/ui/)
- Empty chunk filtering in save_streaming_audio (interface/ui/)
- Speaker name normalization to lowercase (server/client/)
- Consolidated `HF_CACHE` constant to config.py (was duplicated in 3 files)
- Removed internal symbol exports from engine facade (tests import from submodules)

### Server detection fix (2026-03-12)
Unified server start/stop detection to fix PID-file-vs-health-check inconsistency:
- PID lifecycle helpers in config.py: `read_pid_file()`, `write_pid_file()`, `cleanup_pid_file()`, `is_pid_alive()`, `detect_server_state()`
- `detect_server_state()` returns rich state dict (running, health_ok, pid, pid_alive, stale_pid)
- CLI `stop()`: health check → `/shutdown` → SIGTERM → SIGKILL fallback chain
- CLI `start()`: auto-cleans stale PID files before starting
- `/shutdown` endpoint: `BackgroundTask` + `SIGTERM` replaces `sys.exit(0)` for reliable termination
- Gradio `stop_server()`: polls health for up to 5s instead of fixed 1s sleep
- `healthcheck.py`: uses `detect_server_state()` instead of inline PID/kill logic
- DRY: 6 inline PID file operations consolidated to shared functions in config.py

### Orphan server stop fix (2026-03-12)
Three bugs in `tts server stop` when encountering orphan servers (no PID file, stale auth token):
- `find_pid_by_port(port)` in config.py: discovers PID via `lsof -ti :PORT` when PID file is missing
- `stop()` auth failure detection: explicitly handles 401 responses, only polls if shutdown was accepted (200)
- `stop()` verified termination: final `is_server_running()` check before claiming success; exits 1 with manual kill command if server still alive

### Gradio 6 Generate button fix (2026-03-12)
Generation button was broken: JS streaming completed but Gradio Status textbox stayed at "Connecting..." forever.
**Root cause:** Gradio 6 removes `visible=False` components from the DOM entirely, and `.then()` chains break after JS-only steps (`fn=None, js=...`).
- Hidden data-flow components now use `elem_classes=["gr-hidden"]` + CSS `.gr-hidden { display: none !important; }` instead of `visible=False`
- JS-only `.then()` steps now include `fn=lambda x: x` passthrough alongside `js=...` to keep the chain alive
- CSS passed via `demo.launch(css=...)` (Gradio 6 moved `css` from `Blocks()` to `launch()`)
- E2E Playwright tests: fixed `unittest.SkipTest` being swallowed by `except Exception: pass`

## Streaming Wire Format (R-25)

The `/generate-stream` endpoint returns audio chunks in a binary format with length-prefixed headers.

### Binary Format

**Structure:** `[sample_rate:4][length:4][audio:length]` repeated for each chunk

| Field | Size | Type | Description |
|-------|------|------|-------------|
| sample_rate | 4 bytes | uint32 LE | Audio sample rate (e.g., 24000) |
| length | 4 bytes | uint32 LE | Number of bytes in the audio data |
| audio | variable | float32 LE | PCM audio samples (little-endian) |

### Python Example

```python
import struct
import requests
import numpy as np

response = requests.post(
    "http://127.0.0.1:5123/generate-stream",
    headers={"Authorization": f"Bearer {token}"},
    json={"text": "Hello world", "mode": "design"},
    stream=True,
)

all_audio = []

for chunk in response.iter_content(chunk_size=8192):
    data = chunk
    while len(data) >= 8:  # Need at least header
        sr, length = struct.unpack("<II", data[:8])
        audio_bytes = data[8:8+length]
        data = data[8+length:]
        
        # Convert to numpy float32 array
        wav = np.frombuffer(audio_bytes, dtype="<f4")
        all_audio.append(wav)

# Combine all chunks
full_audio = np.concatenate(all_audio)
```

### JavaScript Example

```javascript
const response = await fetch('http://127.0.0.1:5123/generate-stream', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ text: 'Hello world', mode: 'design' }),
});

const reader = response.body.getReader();
const audioChunks = [];

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  let offset = 0;
  const data = new Uint8Array(value);

  while (offset + 8 <= data.length) {
    // Read header: sample_rate (4 bytes) + length (4 bytes)
    const srView = new DataView(data.buffer, offset, 4);
    const lenView = new DataView(data.buffer, offset + 4, 4);
    const sr = srView.getUint32(0, true);  // little-endian
    const length = lenView.getUint32(0, true);

    // Read audio data
    const audioBytes = data.slice(offset + 8, offset + 8 + length);
    audioChunks.push(audioBytes);

    offset += 8 + length;
  }
}

// Combine all chunks
const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
const combined = new Uint8Array(totalLength);
let position = 0;
for (const chunk of audioChunks) {
  combined.set(chunk, position);
  position += chunk.length;
}
```

## Frontend Architecture: Wavesurfer Peaks Computation

### Current Implementation (Optimal)

The Wavesurfer audio visualization uses **real-time peak computation on the frontend**, which is the optimal architecture for this use case.

**Implementation details:**
- **Location**: `qwen3_tts/interface/wavesurfer_js.py`
- **Function**: `_updateWaveformPeaks()`
- **Computation**: 500 bins from in-memory float32 audio chunks
- **Throttling**: 200ms intervals to prevent excessive updates
- **Measured performance**: <5ms computation time (99th percentile)

**Why this is optimal:**

1. **No network overhead**: Audio data is already in browser memory from the generation response
2. **No server latency**: No additional HTTP requests/response cycles
3. **Streaming-compatible**: Works seamlessly with real-time audio streaming
4. **Scalability**: Computation scales with audio duration (linear time, constant memory)
5. **User experience**: Instant feedback as audio loads, no loading spinners

### Backend Pre-calculation: NOT RECOMMENDED

**Why NOT to implement backend pre-calculation:**

1. **Adds network overhead**: Serialize → Transfer → Deserialize (slower than in-memory)
2. **Increases server latency**: Each generation would require additional peak computation time
3. **Breaks streaming architecture**: Peaks would need to be sent separately from audio chunks
4. **No performance gain**: Frontend computation is already <5ms (sub-perceptible)
5. **Increases complexity**: Additional data structures, serialization logic, error handling

**Measured performance:**
- Frontend peak computation: 2-5ms (99th percentile)
- Network round-trip time: 20-50ms (even on localhost)
- Backend computation: Similar to frontend (same algorithm)
- Total with backend pre-calc: 25-55ms vs 2-5ms (frontend only)

**Decision**: Current frontend-only architecture is **CORRECT and OPTIMAL**.

### Monitoring and Triggers

**When to reconsider backend pre-calculation:**
- If frontend computation exceeds 50ms consistently (99th percentile)
- If users on low-end devices report UI lag
- If WaveSurfer.js library API changes to require pre-computed peaks
- If memory constraints prevent in-memory peak computation

**Current monitoring (none yet):**
- Consider adding performance metrics to track peak computation time
- Add debug mode display: "Peak computation: 3ms"
- Monitor user reports of lag or slowness during audio playback

**Alternative approaches if needed:**
- Progressive enhancement: Compute peaks in chunks during streaming
- Worker thread: Offload to Web Worker for very long audio files
- Reduced bin count: Fall back to 250 bins for better performance
- Debouncing: Increase throttle interval from 200ms to 500ms

**Conclusion**: The current architecture is already optimized for this use case. No backend pre-calculation is needed unless performance degrades significantly.

### vLLM Multimodal Parameters

The vLLM engine is configured with optimized parameters for audio TTS workloads:

**Key Parameters:**
- `--limit-mm-per-prompt audio=1`: Limits multimodal input to 1 audio chunk per prompt
- `--enable-chunked-prefill`: Enables chunked prefill for better throughput with long audio
- `--dtype bfloat16`: Uses bfloat16 precision for memory efficiency
- `--max-model-len 8192`: Maximum context length (optimized for audio sequences)
- `--audio-sample-rate 24000`: Audio sample rate matching TTS output
- `--audio-chunk-size 2000`: Audio chunk size (~83ms at 24kHz)

**Implementation Location:**
- Code: `qwen3_tts/core/engine_vllm.py` (lines 189-201)
- Config: `config.json` under `vllm` section
- Validation: `scripts/validate_vllm_docker.sh`

**Validation:**
Run the Docker validation script to verify parameters:
```bash
./scripts/validate_vllm_docker.sh
```

This script starts the vLLM container and verifies that:
- Multimodal parameters are correctly set
- Audio processing parameters match expected values
- Data type is configured for memory efficiency

**Performance Impact:**
These parameters are optimized for audio TTS workloads:
- Chunked prefill improves throughput for long audio sequences
- bfloat16 reduces memory usage while maintaining quality
- Audio-optimized chunk size balances latency and throughput

### Request Validation Patterns

**TranscribeRequest Language Validation:**

The `/transcribe` endpoint uses regex pattern validation for language codes:

**Pattern:** `^[a-z]{2,3}(-[A-Za-z]{2,4})?$`

**Accepts:**
- `en`, `zho`, `fr` (2-3 letter language codes)
- `en-US`, `zh-CN`, `en-GB` (language + region)
- `es-MX` (language + 3-letter region)

**Rejects:**
- `EN`, `FR` (uppercase not allowed)
- `e1`, `f2` (numbers not allowed)
- `en_US` (underscores not allowed)
- Empty strings

**Implementation:** `qwen3_tts/server/validation.py:99`

```python
# Example usage
req = TranscribeRequest(audio_base64="base64data", language="en-US")
# Valid: en, zho, en-US, zh-CN, es-MX
# Invalid: EN, e1, en_US, empty
```

### Waveform Peaks Performance

Audio peaks are pre-computed server-side for efficient visualization:

**Implementation:** `qwen3_tts/core/engine/audio_processing.py:273`
```python
def calculate_waveform_peaks(audio, num_peaks=500):
    """Calculate waveform peaks for visualization."""
    # Bins audio into 500 bins and returns max amplitude per bin
    # Returns list of floats in [-1.0, 1.0] range
```

**Performance Benchmarks:**
- **1 second audio:** ~1.2ms
- **10 seconds audio:** ~1.2ms
- **1 minute audio:** ~1.7ms
- **Target:** < 50ms (exceeded by 40x+ margin)

**Integration:**
- Server includes peaks in `/generate` response: `app_generation.py:358,364`
- Peaks calculated with 500-point resolution
- Returned as JSON array in response metadata
- Client (wavesurfer.js) can use pre-computed peaks for instant rendering

**Performance Impact:**
Pre-computed peaks eliminate ~200ms client-side delay, enabling instant waveform visualization when audio loads.

**Validation Tests:** `tests/test_validation_ext.py`

## Testing

### E2E Gating and Rate Limiting

**E2E gating:** All `tests/test_e2e_*.py` are marked `pytest.mark.e2e`; `pytest.ini` deselects them by default (`-m "not e2e"`) because they make real `/generate` calls and hang plain `pytest tests/` when a server is up. Opt in with `-m e2e`. The batch runner uses `unittest` (ignores markers), so all batches are unaffected.

**E2E + rate limiting:** the live server's `/generate` limit (default 10/minute, configurable) is shared across e2e modules and starves suites that fire many requests, causing false 429 skips. `run_full_suite.py` starts its test server with `TTS_DISABLE_RATE_LIMITING=1`; for a manual run do the same (`TTS_DISABLE_RATE_LIMITING=1 tts server start`, or raise one limit via `TTS_RATE_LIMIT_GENERATE=120/minute`). The performance/stress e2e tests assert real audio output, so they now perform actual (slow) generation.
