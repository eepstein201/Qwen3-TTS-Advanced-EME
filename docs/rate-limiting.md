# Rate Limiting Guide

## Architecture

Rate limiting is implemented using **slowapi** with support for multiple strategies. The server creates **four** separate limiters at startup:

1. **Global pre-auth ceiling** (`limiter_global`): an IP-keyed default limit enforced by `SlowAPIMiddleware` at the ASGI layer, **before routing and auth**, on ALL routes. This exists because the per-route `@limiter.limit` decorators run *after* `Depends(verify_auth)` — without a pre-auth ceiling, an unauthenticated flood 401s on every request before any limiter fires and bypasses rate limiting entirely.
2. **Hybrid Rate Limiting** (`limiter_hybrid`): Combines IP + token for strictest enforcement
3. **IP-Based Rate Limiting** (`limiter_ip`): Uses `_get_real_client_ip()` to handle reverse proxies
4. **Token-Based Rate Limiting** (`limiter_token`): Hashes auth tokens for per-user limits (SHA-256, first 16 hex chars)

## Configuration

### Rate Limit String Format

`<count>/<unit>` where:
- `count`: Integer number of requests (must be > 0)
- `unit`: `second`, `minute`, `hour`, or `day`

Examples:
- `"20/minute"` - 20 requests per minute
- `"100/hour"` - 100 requests per hour
- `"5/minute"` - 5 requests per minute

### Default Limits

| Limit | Default | Rationale |
|-------|---------|-----------|
| Generation (`generate`) | 10/minute | Long-text chunks take ~40-70 s each on MLX; 10/min is ample headroom for interactive use |
| Model Operations (`model_ops`) | 5/minute | Load/unload are expensive GPU/memory operations |
| Global pre-auth ceiling (`global`) | 120/minute | Flood backstop only — must stay well above the Gradio UI's ~24/min `/health` + `/models` polling (see below) |

**Operational caveat — the global ceiling vs. the Gradio UI:** the web UI polls `/health` + `/models` every 5 s (~24 requests/min) against a single IP. If you lower `security.rate_limits.global` below that, `/health` starts returning 429 and the UI banner shows "Disconnected / Server not running" even though the server is up. Keep the global ceiling comfortably above ~24/min; tighten abuse with the per-endpoint limits, not the global ceiling.

### Actual Endpoint Wiring

Every config key maps to real endpoints — each category has its own decorator (guarded by `tests/test_rate_limiting.py::TestEndpointLimiterWiring`, which also fails on any reintroduced hardcoded limit string):

| Endpoint | Effective limit (default) |
|----------|--------------------------|
| `/generate`, `/generate-stream` | `generate` (10/minute) |
| `/load-model`, `/unload-model`, `/update-model-config`, `/load-asr`, `/unload-asr` | `model_ops` (5/minute) |
| `/transcribe` | `transcribe` (10/minute) |
| `/create-voice-prompt`, `/delete-prompt`, `/rename-prompt` | `prompt_ops` (10/minute) |
| `/update-startup-config` | `config_ops` (2/minute) |
| All routes (pre-auth) | `global` (120/minute) |

### Environment Variables

All rate-limit values are resolved once at **server import time** — restart the server (`tts server stop && tts server start`) to apply any change (config or env):

- `TTS_DISABLE_RATE_LIMITING=1` — makes every limiter a no-op. For local test/CI servers only; never set in production.
- `TTS_RATE_LIMIT_GENERATE`, `TTS_RATE_LIMIT_MODEL_OPS`, `TTS_RATE_LIMIT_TRANSCRIBE`, `TTS_RATE_LIMIT_PROMPT_OPS`, `TTS_RATE_LIMIT_CONFIG_OPS`, `TTS_RATE_LIMIT_GLOBAL` — override the corresponding limit (env wins over `config.json`).

### Custom Configuration

Edit `config.json`:

```json
{
  "security": {
    "rate_limits": {
      "generate": "50/hour",
      "model_ops": "10/hour",
      "transcribe": "100/hour",
      "prompt_ops": "30/hour",
      "config_ops": "5/hour",
      "global": "240/minute"
    }
  }
}
```

## Rate Limit Strategies

### Hybrid Strategy (Default)

Combines both IP and token limits for strictest enforcement:

```python
@_rate_limit("10/minute", strategy="hybrid")
```

Rate limit key format: `{client_ip}:{token_hash}`

### IP-Only Strategy

Rate limits based on client IP address only:

```python
@_rate_limit("10/minute", strategy="ip")
```

Rate limit key format: `{client_ip}`

### Token-Only Strategy

Rate limits based on authentication token only:

```python
@_rate_limit("10/minute", strategy="token")
```

Rate limit key format: `{token_hash}` (SHA-256[:16])

**Use cases:**
- **Hybrid**: Default for most endpoints (strictest enforcement)
- **IP-only**: Shared environments where multiple users share one IP
- **Token-only**: Per-user limits regardless of IP (NAT, corporate proxies)

## Implementation Details

### Token Hashing

Authentication tokens are hashed using SHA-256 (first 16 hex characters) to prevent sensitive data from appearing in rate limit keys:

```python
token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
```

This ensures:
- Same token always produces same hash (deterministic)
- Hash cannot be reversed to reveal original token
- Different tokens produce different hashes (collision resistant)

### Proxy Handling

The server honors `X-Forwarded-For` **only when the direct TCP peer is in the trusted-proxy allowlist** (`TTS_TRUSTED_PROXIES`, comma-separated IPs; loopback — `127.0.0.1`, `::1`, `localhost` — by default):

```python
def _get_real_client_ip(request: Request) -> str:
    direct_host = request.client.host if request.client else "127.0.0.1"
    if direct_host in TRUSTED_PROXIES:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return direct_host
```

This prevents rate-limit bypass: a client connecting **directly** on a public or Colab bind cannot spoof `X-Forwarded-For` to rotate its rate-limit key — the header is only read when the connection actually came from a trusted proxy. **Set `TTS_TRUSTED_PROXIES` when running behind a reverse proxy** so per-IP limits see the real client; unset it (or leave loopback-only) otherwise.

## Testing

### Unit Tests

Run rate limiting tests:

```bash
python -m pytest tests/test_rate_limiting.py -v
```

Tests include:
- IP key extraction (with and without proxy)
- Token hashing consistency
- Decorator strategy support
- Config validation
- 429 error handling

### Manual Testing

Test rate limits with curl:

```bash
# 1. Set auth token
export TOKEN="your-auth-token"

# 2. Send repeated requests (should hit 429 after limit)
for i in {1..25}; do
  curl -X POST http://localhost:5123/generate \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"text": "test"}'
  echo "Request $i complete"
done
```

Expected output: First 10 requests succeed, next 15 return HTTP 429 (the 10/minute generate default).

## Troubleshooting

### Rate Limits Not Working

1. **Verify slowapi is installed:**
   ```bash
   pip list | grep slowapi
   ```

2. **Check config.json has `security.rate_limits` section:**
   ```bash
   cat config.json | grep -A 10 "rate_limits"
   ```

3. **Check server logs for warnings:**
   ```bash
   tts server log
   ```

### Adjusting Limits

If you're seeing 429 errors frequently:

1. **Increase limits in config.json:**
   ```json
   {
     "security": {
       "rate_limits": {
         "generate": "50/minute"
       }
     }
   }
   ```

2. **Restart server to apply changes:**
   ```bash
   tts server stop && tts server start
   ```

3. **Use multiple auth tokens** for different applications

4. **Implement client-side retry with exponential backoff**

### Testing Different Strategies

To test rate limiting behavior with different strategies, modify the decorator in `qwen3_tts/server/app.py`:

```python
# Test IP-only strategy
@_rate_limit("10/minute", strategy="ip")
async def generate_endpoint(...):
    ...

# Test token-only strategy
@_rate_limit("10/minute", strategy="token")
async def generate_endpoint(...):
    ...

# Test hybrid strategy (default)
@_rate_limit("10/minute", strategy="hybrid")
async def generate_endpoint(...):
    ...
```

## Security Considerations

### Token Hashing

Tokens are hashed to prevent sensitive data from appearing in logs or rate limit keys:

- **Hash**: SHA-256 digest (first 16 hex characters)
- **Deterministic**: Same token always produces same hash
- **Irreversible**: Hash cannot be reversed to reveal original token

### Proxy Security

`X-Forwarded-For` headers are only trusted when the direct TCP peer is in the **`TTS_TRUSTED_PROXIES` allowlist** (loopback by default). A client connecting directly — including on a public or Colab bind — cannot spoof the header to rotate its rate-limit key, which prevents rate limit bypass attacks. Only add proxies you actually front the server with.

### Rate Limit Bypass Prevention

The hybrid strategy (IP + token) provides the strongest protection against:
- IP spoofing
- Token theft
- Distributed abuse attempts

## Monitoring

### Check Rate Limit Status

Server stats include rate limit information:

```bash
curl http://localhost:5123/stats \
  -H "Authorization: Bearer $TOKEN"
```

### Log Analysis

Rate limit events are logged with client IP (sanitized) for security auditing:

```python
logger.warning(
    "Auth failure: %s from %s on %s %s",
    failure_reason,
    sanitize_log(client_ip),
    request.method,
    request.url.path,
)
```

## Best Practices

1. **Start with conservative limits** (10/minute for generation)
2. **Monitor 429 error rates** to detect abuse vs. legitimate high usage
3. **Use hybrid strategy** for public-facing endpoints
4. **Adjust limits based on actual usage patterns**
5. **Document custom limits** for your deployment environment
6. **Test rate limiting before production** deployment
