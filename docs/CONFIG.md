# Qwen3-TTS Configuration Reference

> **AUTO-GENERATED** from `config.json` and `qwen3_tts/core/config.py`. Do not edit manually.

## Configuration File Location

The main configuration file is located at:
- **Default**: `~/Qwen3-TTS_UserFiles/config.json`
- **Custom**: Set `QWEN3_TTS_CONFIG` environment variable to override

## Environment Variables

### Server Authentication

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `QWEN3_TTS_CONFIG` | No | Path to custom config.json | `/path/to/config.json` |
| `VOICE_SERVER_TOKEN` | No | Path to server auth token file | `~/.voice_server_token` (deprecated) |
| `VOICE_CONFIG_DIR` | No | Directory for server config | `~/.config/qwen3-tts/` |

### Model Cache

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `HF_HOME` | No | HuggingFace cache directory | `~/.cache/huggingface` |
| `HUGGINGFACE_HUB_CACHE` | No | Alternative cache directory | `~/cache/huggingface` |

### Prompt Enhancer

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | No* | API key for AI description enhancement | `sk-ant-xxx` |

*Required only if `prompt_enhancer.enabled=true` in config

### Development

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `TTS_LOG_LEVEL` | No | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

## Configuration Structure

### Models Section

```json
{
  "models": {
    "clone": {
      "load_at_startup": true
    },
    "design": {
      "load_at_startup": false
    },
    "custom": {}
  }
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `models.clone.load_at_startup` | boolean | `true` | Load clone model at server start |
| `models.design.load_at_startup` | boolean | `false` | Load design model at server start |
| `models.custom` | object | `{}` | Custom model configuration (reserved) |

### Advanced Section

```json
{
  "advanced": {
    "model_size": "1.7B",
    "mlx_quantization": "8bit",
    "backend": "mlx",
    "audio_loader": "librosa"
  }
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `advanced.backend` | string | `"mlx"` (Apple Silicon), `"torch"` elsewhere | Inference backend |
| `advanced.model_size` | string | `"1.7B"` | Model size: `"1.7B"` (full) or `"0.6B"` (fast) |
| `advanced.mlx_quantization` | string | `"8bit"` | MLX quantization: `"4bit"`, `"8bit"`, `"bf16"` |
| `advanced.torch_quantization` | string | `"none"` | Torch quantization: `"none"`, `"8bit"`, `"4bit"` |
| `advanced.audio_loader` | string | `"librosa"` | Audio library: `"torchaudio"`, `"librosa"` |

**Backend Selection:**
- **MLX**: Apple Silicon only, faster inference, lower memory
- **Torch**: Cross-platform, more features, slower inference
- **VLLM**: Production server with vLLM backend (experimental)

### Generation Section

```json
{
  "generation": {
    "silence_gap_seconds": 0.0,
    "max_chunk_chars": 500,
    "lufs_normalize": false,
    "lufs_target": -16.0
  }
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `generation.silence_gap_seconds` | float | `0.0` | Silence gap between chunks (0-5 seconds) |
| `generation.max_chunk_chars` | integer | `500` | Max characters per chunk (0=disable chunking) |
| `generation.lufs_normalize` | boolean | `false` | Apply EBU R128 loudness normalization |
| `generation.lufs_target` | float | `-16.0` | Target loudness in LUFS (used when `lufs_normalize` is true) |

### Language Setting

```json
{
  "language": "English"
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `language` | string | `"English"` | Language for text processing (affects tokenization) |

### Rate Limiting (Server)

```json
{
  "rate_limiting": {
    "enabled": true,
    "requests_per_minute": 10,
    "burst": 2
  }
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rate_limiting.enabled` | boolean | `true` | Enable rate limiting on server endpoints |
| `rate_limiting.requests_per_minute` | integer | `10` | Max requests per minute per client |
| `rate_limiting.burst` | integer | `2` | Burst allowance for rate limiter |

### Server Settings

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 5123,
    "auto_shutdown_minutes": 30
  }
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `server.host` | string | `"127.0.0.1"` | Server bind address |
| `server.port` | integer | `5123` | Server port |
| `server.auto_shutdown_minutes` | integer | `30` | Auto-shutdown after N minutes idle |

### Prompt Enhancer (AI Description Enhancement)

**Install:** `pip install -e ".[prompt-enhancer]"` (adds the `anthropic` SDK).

```json
{
  "prompt_enhancer": {
    "enabled": false,
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "api_key_env": "ANTHROPIC_API_KEY"
  }
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `prompt_enhancer.enabled` | boolean | `false` | Enable AI voice description enhancement |
| `prompt_enhancer.provider` | string | `"anthropic"` | AI provider (anthropic only) |
| `prompt_enhancer.model` | string | `"claude-haiku-4-5-20251001"` | Model to use for enhancement |
| `prompt_enhancer.api_key_env` | string | `"ANTHROPIC_API_KEY"` | Environment variable containing API key |

### Generation Presets

```json
{
  "generation_presets": {
    "stable": {
      "temperature": 0.7,
      "top_k": 20,
      "top_p": 0.8,
      "repetition_penalty": 1.0
    }
  }
}
```

Presets define reusable generation parameter sets. See `CLAUDE.md` for default presets.

### Prosody Presets

```json
{
  "prosody_presets": {
    "neutral": {},
    "energetic": {
      "speed": 1.1
    }
  }
}
```

Presets for voice prosody adjustments (speed, pitch, etc.).

### Cache Settings

```json
{
  "cache": {
    "voice_prompts_dir": "voice_prompts",
    "models_unused_days": 30
  }
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cache.voice_prompts_dir` | string | `"voice_prompts"` | Directory for voice clone prompts |
| `cache.models_unused_days` | integer | `30` | Days before pruning unused models |

### UI Settings

```json
{
  "ui": {
    "theme": "soft",
    "share": false,
    "allowed_paths": ["~/Downloads"]
  }
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ui.theme` | string | `"soft"` | Gradio theme: `soft`, `default`, `glass` |
| `ui.share` | boolean | `false` | Enable public sharing via Gradio link |
| `ui.allowed_paths` | array | `["~/Downloads"]` | Paths Gradio can access for file I/O |

## Default Configuration

A minimal `config.json` with defaults:

```json
{
  "models": {
    "clone": {
      "load_at_startup": true
    },
    "design": {
      "load_at_startup": false
    },
    "custom": {}
  },
  "advanced": {
    "model_size": "1.7B",
    "mlx_quantization": "8bit",
    "backend": "mlx",
    "audio_loader": "librosa"
  },
  "language": "English",
  "generation": {
    "silence_gap_seconds": 0.0,
    "max_chunk_chars": 500,
    "lufs_normalize": false,
    "lufs_target": -16.0
  }
}
```

## Config Validation

The config is validated on load. Invalid values will raise errors with specific guidance.

**Validation rules:**
- `backend` must be `"mlx"`, `"torch"`, or `"vllm"`
- `model_size` must be `"1.7B"` or `"0.6B"`
- `mlx_quantization` must be `"4bit"`, `"8bit"`, or `"bf16"`
- `torch_quantization` must be `"none"`, `"8bit"`, or `"4bit"`
- `max_chunk_chars` must be between `0` and `10000`
- `rate_limiting.requests_per_minute` must be positive

## Runtime Config Overrides

Some settings can be overridden per-generation via CLI flags:

```bash
tts "Hello" --backend torch --model-size 0.6B --temperature 0.9
```

These overrides do NOT persist to `config.json`.
