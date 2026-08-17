# Qwen3-TTS Configuration Reference

> **AUTO-GENERATED** from the `qwen3_tts/core/config/` package (`get_default_config()` / `validate_config()`) and `config.json`. The doc currently ships **without a live generator** — tables are curated by hand, and the *default-value* column is drift-checked against `get_default_config()` by `make check-config-docs` (`python -m qwen3_tts.tools.check_config_docs`), which exits non-zero when a documented default disagrees with the code. When adding a key that exists in `get_default_config()`, the default column must match exactly.

## Configuration File Location

The main configuration file lives at the repository root: `config.json`.

The on-disk `config.json` is a **sparse override** — it only needs the keys you want to change. Any key it omits falls back to the built-in default from `get_default_config()`. On load, `validate_config()` fills in missing sections (for example, it adds a default `security.rate_limits` block).

> **Note:** There is intentionally **no** environment-variable override for the config-file path. `_resolve_config_path()` resolves the path in code and deliberately ignores any `QWEN3_TTS_CONFIG`-style override (a path-injection hardening choice).

## Environment Variables

These are the environment variables actually read by the TTS code.

### Backend / model overrides

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `TTS_BACKEND` | No | Override `advanced.backend` for a single run (set by the `--backend` CLI flag; also forced to `torch` during `tts voice rebuild`). | `torch` |
| `TTS_MODEL_SIZE` | No | Override `advanced.model_size` for a single run (set by `--model-size`). | `0.6B` |
| `QWEN3_TTS_BACKEND` | No | Test-runner-only backend override, distinct from `TTS_BACKEND`. | `mlx` |
| `CUDA_VISIBLE_DEVICES` | No | Standard CUDA device-selection signal used during device detection. | `0` |

### Server

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `TTS_LOG_LEVEL` | No | Server/log verbosity (default `INFO`). | `DEBUG` |
| `TTS_TRUSTED_PROXIES` | No | Comma-separated IP allowlist; `X-Forwarded-For` is honored only when the direct peer is in this list (default: loopback). Set this behind a reverse proxy so per-IP rate limiting sees the real client. | `10.0.0.1,10.0.0.2` |
| `TTS_RATE_LIMIT_GENERATE` | No | Override `security.rate_limits.generate` (`/generate`, `/generate-stream`). | `120/minute` |
| `TTS_RATE_LIMIT_MODEL_OPS` | No | Override `security.rate_limits.model_ops` (model load/unload/config endpoints). | `5/minute` |
| `TTS_RATE_LIMIT_TRANSCRIBE` | No | Override `security.rate_limits.transcribe` (`/transcribe`). | `30/minute` |
| `TTS_RATE_LIMIT_PROMPT_OPS` | No | Override `security.rate_limits.prompt_ops` (prompt create/delete/rename). | `30/minute` |
| `TTS_RATE_LIMIT_CONFIG_OPS` | No | Override `security.rate_limits.config_ops` (`/update-startup-config`). | `5/minute` |
| `TTS_RATE_LIMIT_GLOBAL` | No | Override the global pre-auth ceiling (`security.rate_limits.global`), applied to **all** routes. | `240/minute` |
| `TTS_DISABLE_RATE_LIMITING` | No | Test/CI kill-switch: `1` makes every rate-limit decorator and the global limiter a no-op. For local E2E/CI servers only — production leaves it unset. | `1` |

All `TTS_RATE_LIMIT_*` and `TTS_DISABLE_RATE_LIMITING` values are read **once at server import** (`qwen3_tts/server/app.py`) — restart the server to apply a change.

### Web UI

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `TTS_UI_PORT` | No | Gradio UI port override. | `8080` |
| `TTS_UI_SHARE` | No | Enable a public Gradio share link. | `1` |
| `TTS_UI_NO_BROWSER` | No | Suppress auto-opening the browser when launching the UI. | `1` |

### Prompt enhancer

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | No* | API key for the AI voice-description enhancer. The variable *name* is itself configurable via `prompt_enhancer.api_key_env`. | `sk-ant-...` |

*Required only when `prompt_enhancer.enabled` is `true`.

### HuggingFace cache (external)

`HF_HOME` and `HUGGINGFACE_HUB_CACHE` are **not** read by the TTS code, but they are honored transitively by the `huggingface_hub` library that downloads models. Set them to relocate the model cache. They are documented here only for completeness.

## Configuration Structure

Every table below reflects `get_default_config()` defaults.

### Top-level keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_voice_description` | string | `"A calm, friendly male voice with clear articulation and moderate pace."` | Default description used in Design mode. |
| `default_clone_prompt` | string/null | `null` | Default voice-clone prompt filename. `null` = auto-scan `voice_prompts/` for the first usable prompt. |
| `default_speaker` | string | `"ryan"` | Default premium speaker for Custom mode. |
| `output_directory` | string | `"~/Downloads"` | Default output directory for **CLI**-generated audio. |
| `history_output_directory` | string | `"~/Downloads/Qwen3-TTS Output"` | Parent for **web-UI** output. Generations land in its `Automated Output/` subfolder (with `.json` sidecars); per-row Download copies land in `Manual Downloads/`. Distinct from `output_directory`. |
| `language` | string | `"auto"` | Language conditioning + text processing. `"auto"` lets the model infer the language from the text — the safe default, since a concrete value conditions generation on a language the text may not be in. A named value (`"English"`, `"Chinese"`, …) forces conditioning; torch raises `NotImplementedError` on an unrecognized name, MLX warns and generates without conditioning. Text normalization maps anything unrecognized (including `"auto"`) to English rules. |

### `server`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `server.host` | string | `"127.0.0.1"` | Bind address (localhost only by default). |
| `server.port` | integer | `5123` | Server port. |
| `server.auto_shutdown_minutes` | integer | `0` | Auto-shutdown after N idle minutes. `0` disables auto-shutdown. |

### `models`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `models.clone.load_at_startup` | boolean | `true` | Load the Clone model at server start. |
| `models.design.load_at_startup` | boolean | `false` | Load the Design model at server start. |
| `models.custom.load_at_startup` | boolean | `false` | Load the Custom model at server start. |
| `models.<type>.revision` | string | *(unset → `"main"`)* | Pin a HuggingFace branch/tag/SHA per model (`clone`/`design`/`custom`). Falls back to the model's `MODEL_INFO` revision, then `"main"`. |

### `security`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `security.max_text_length` | integer | `50000` | Max characters accepted per generation request. |
| `security.max_batch_size` | integer | `20` | Max number of texts per batch request. |
| `security.rate_limits.generate` | string | `"10/minute"` | Rate limit for `/generate`, `/generate-stream`. |
| `security.rate_limits.model_ops` | string | `"5/minute"` | Rate limit for model load/unload/config endpoints. |
| `security.rate_limits.transcribe` | string | `"10/minute"` | Rate limit for `/transcribe`. |
| `security.rate_limits.prompt_ops` | string | `"10/minute"` | Rate limit for prompt create/delete/rename. |
| `security.rate_limits.config_ops` | string | `"2/minute"` | Rate limit for `/update-startup-config`. |
| `security.rate_limits.global` | string | `"120/minute"` | Global pre-auth ceiling on **all** routes (IP-keyed middleware). Deliberately decoupled from the per-route limits above — it must stay above the Gradio UI's ~24/min `/health`+`/models` polling, or `/health` 429s and the UI reports "Disconnected / Server not running". Override via `TTS_RATE_LIMIT_GLOBAL`. |

Rate-limit values use slowapi's `"<count>/<unit>"` format (`second`/`minute`/`hour`/`day`). See [`rate-limiting.md`](rate-limiting.md) for strategy details. `security.rate_limits.global` is not part of the `validate_config()` default block — the server supplies the `120/minute` fallback at import (`qwen3_tts/server/app.py`).

### `advanced`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `advanced.backend` | string | `"mlx"` (Apple Silicon), `"torch"` elsewhere | Inference backend: `"mlx"`, `"torch"`, or `"vllm"`. |
| `advanced.model_size` | string | `"1.7B"` | `"1.7B"` (full) or `"0.6B"` (fast/light). |
| `advanced.mlx_quantization` | string | `"8bit"` | MLX quantization: `"4bit"`, `"5bit"`, `"6bit"`, `"8bit"`, or `"bf16"`. |
| `advanced.torch_quantization` | string | `"none"` | Torch quantization: `"none"`, `"8bit"`, or `"4bit"`. |
| `advanced.dtype` | string | `"bfloat16"` | Torch compute dtype. |
| `advanced.audio_loader` | string | `"torchaudio"` | Audio I/O library: `"torchaudio"` or `"librosa"`. |
| `advanced.attn_implementation` | string | `"auto"` | Attention kernel: `"auto"`, `"sdpa"`, `"flash_attention_2"`, or `"eager"`. `"auto"` resolves to SDPA; FA2 is opt-in only — upstream #333 reports NaN logits with `flash_attention_2` on Ampere+ GPUs. |
| `advanced.vllm_enabled` | boolean | `false` | Mirror of `vllm.enabled` (see below). |
| `advanced.vllm_fallback_to_torch` | boolean | `true` | Fall back to torch if vLLM init fails. |

### `vllm`

vLLM-Omni backend settings (Linux/NVIDIA). Only relevant when `advanced.backend` is `"vllm"`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `vllm.enabled` | boolean | `false` | Enable the vLLM backend. |
| `vllm.fallback_to_torch` | boolean | `true` | Fall back to torch on vLLM failure. |
| `vllm.max_model_len` | integer | `8192` | Max model context length. |
| `vllm.audio_sample_rate` | integer | `24000` | Output sample rate (Hz). |
| `vllm.audio_chunk_size` | integer | `2000` | Audio chunk size for streaming. |
| `vllm.gpu_memory_utilization` | float | `0.9` | Fraction of VRAM to reserve (0.1–1.0). |
| `vllm.tensor_parallel_size` | integer | `1` | Number of GPUs for tensor parallelism. |
| `vllm.mm_processor_name` | string | `"Qwen/Qwen2-Audio-7B-Instruct"` | Multimodal processor model id. |
| `vllm.port` | integer/null | `null` | Optional dedicated vLLM port. |
| `vllm.dtype` | string | `"bfloat16"` | vLLM compute dtype. |

### `generation`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `generation.temperature` | float | `0.7` | Sampling temperature (higher = more varied). |
| `generation.top_k` | integer | `50` | Top-k sampling. |
| `generation.top_p` | float | `0.95` | Top-p (nucleus) sampling. |
| `generation.repetition_penalty` | float | `1.05` | Repetition penalty. |
| `generation.seed` | integer/null | `null` | Fixed RNG seed (null = random per request). |
| `generation.max_chunk_chars` | integer | `500` | Max characters per chunk before splitting (`0` disables chunking). |
| `generation.max_chunk_tokens` | integer | `200` | Max tokens per chunk (torch backend). |
| `generation.max_new_tokens` | integer | `2048` | Hard cap on generated tokens per `model.generate()` call. **MLX clamps this to 4096** (`_MLX_MAX_TOKENS_CEILING`) regardless of the value here or the request schema's `le=8192`: PRF-9 measured ≥8192 as unstable on 16 GB (EOS-failure runaway loops + memory exhaustion). Torch honors the full range. |
| `generation.compile_model` | boolean | `true` | Enable model compilation (torch). |
| `generation.lufs_normalize` | boolean | `false` | Apply EBU R128 loudness normalization. |
| `generation.lufs_target` | float | `-16.0` | Target loudness in LUFS (used when `lufs_normalize` is true). |
| `generation.silence_gap_seconds` | float | `0.0` | Silence between chunks (0–5s). `0.0` uses a 50 ms crossfade instead. |
| `generation.clone_speed` | float | *(unset)* | Clone-only post-hoc time-stretch rate (0.5–2.0). The model's native rate control is broken for cloning (upstream #290), so `run_inference` time-stretches after generation via `process_audio(speed=…)`. `gen_params["speed"]` overrides this key; out-of-range values are clamped. Design/Custom keep native `instruct` rate control and are never stretched. |
| `generation.trim_icl_echo` | boolean | `true` | Clone-only: clip the ICL echo of the reference transcript (upstream #341) from the head of cloned output. Only fires when ASR is **already loaded** (never force-loads) **and** a reference transcript is resolvable; `x_vector_only_mode` prompts carry no transcript and are never probed. |

Note: `generation.clone_speed` and `generation.trim_icl_echo` are **not keys of `get_default_config()`** — the engine reads them with `.get()` fallbacks (`core/engine/inference.py`), so they are absent from a fresh `config.json` and the defaults above describe the engine's fallback behavior. They are documented here for reference; the drift checker skips keys that are not present in the code defaults.

### `presets`

`config.json`'s `presets` are **merged over** 8 built-in presets at runtime (`{**DEFAULT_GENERATION_PRESETS, **user_presets}`). Built-ins: `stable`, `natural`, `expressive`, `audiobook`, `conversational`, `broadcast`, `dramatic`, `whisper`. The default config also ships two user presets:

```json
{
  "presets": {
    "consistent": { "temperature": 0.5, "top_k": 30, "seed": 42 },
    "creative":   { "temperature": 0.9, "top_p": 0.98 }
  }
}
```

Use a preset with `tts "..." --preset creative`. List them with `tts list presets`.

### `prosody_presets`

A dict of **plain instruction strings** (not parameter dicts) injected into Custom/Design generation. Defaults:

```json
{
  "prosody_presets": {
    "excited": "Speak with excitement and high energy",
    "calm": "Speak in a calm, soothing, relaxed manner",
    "whisper": "Speak in a soft whisper",
    "authoritative": "Speak in a confident, authoritative tone",
    "slow": "Speak slowly and deliberately with clear enunciation",
    "fast": "Speak quickly with urgency",
    "dramatic": "Speak with dramatic flair and emotional intensity",
    "conversational": "Speak in a casual, natural conversational style"
  }
}
```

Use with `tts "..." --prosody excited`. List them with `tts list prosody`.

### `aliases`

Named shortcuts bundling a voice prompt (and optional preset) to a single name,
invoked with `-v`/`--voice`. Ships **empty** (`{}`) — no alias is seeded, so an
unconfigured `-v` can never resolve to a dangling prompt. Add your own:

```json
{
  "aliases": {
    "default": { "prompt": "my_voice.pt", "preset": "consistent" }
  }
}
```

Use with `tts "..." -v default`. List them with `tts list aliases`.

### `cache`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cache.voice_prompt_max` | integer | `10` | Max voice prompts held in the in-memory LRU cache. |
| `cache.generation_max` | integer | `5` | Max generation results cached (keyed by SHA-256 of output-affecting fields). |
| `cache.eta_ttl_seconds` | integer | `30` | ETA cache TTL in seconds. |

### `ui`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ui.port` | integer | `7860` | Gradio UI port (overridable via `TTS_UI_PORT` or `tts ui --port`). |

### `prompt_enhancer`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `prompt_enhancer.enabled` | boolean | `false` | Enable AI voice-description enhancement. |
| `prompt_enhancer.provider` | string | `"anthropic"` | AI provider (anthropic). |
| `prompt_enhancer.model` | string | `"claude-haiku-4-5-20251001"` | Model used for enhancement. |
| `prompt_enhancer.api_key_env` | string | `"ANTHROPIC_API_KEY"` | Name of the env var holding the API key. |

**Install:** `pip install -e ".[prompt-enhancer]"` (adds the `anthropic` SDK).

## Config Validation

`validate_config()` runs on load: it returns a **corrected copy** (the input is never mutated) and fills in any missing `security.rate_limits` block. It only corrects these fields:

- `advanced.backend` ∈ `{"mlx", "torch", "vllm"}` — else the platform default
- `advanced.model_size` ∈ `{"1.7B", "0.6B"}` — else `"1.7B"`
- `advanced.vllm_gpu_memory_utilization` in `(0.0, 1.0]` — else `0.7`
- `advanced.vllm_port` an int in `[1024, 65535]` — else `null`
- `generation.temperature` in `[0.0, 2.0]` — else the built-in default
- `security.max_text_length` a positive int — else `50000`
- `security.rate_limits.*` must match `<count>/<unit>` with a positive count — else the per-key default; a missing `rate_limits` block is added.

These related values are **not** handled by `validate_config()` — they are resolved elsewhere:

- `advanced.mlx_quantization` / `advanced.torch_quantization`: read through `get_mlx_quantization()` / `get_torch_quantization()`, which fall back to the default (`"8bit"` / `"none"`) when unset or invalid.
- `advanced.audio_loader` ∈ `{"torchaudio", "librosa"}`: validated by `set_audio_loader()` when changed via `/update-model-config` (raises on any other value), not at config load.
- `generation.max_chunk_chars`: no config-time clamp; `0` disables chunking. The intended input range is `0`–`10000`.

## Runtime Config Overrides

Some settings can be overridden per-generation via CLI flags without persisting to `config.json`:

```bash
tts "Hello" --backend torch --model-size 0.6B --temperature 0.9
```

Environment-variable overrides (`TTS_BACKEND`, `TTS_MODEL_SIZE`) apply for the duration of the process only.
