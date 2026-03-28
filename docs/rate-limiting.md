# Rate Limiting Guide

## Architecture

Rate limiting is implemented using **slowapi** with support for multiple strategies. The server creates three separate limiters at startup, each using a different key function:

### Key Components

1. **IP-Based Rate Limiting**: Uses `_get_real_client_ip()` to handle reverse proxies
2. **Token-Based Rate Limiting**: Hashes auth tokens for per-user limits (SHA-256, first 16 hex chars)
3. **Hybrid Rate Limiting**: Combines both IP + token for strictest enforcement

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

| Endpoint Type | Default Limit | Rationale |
|--------------|--------------|-----------|
| Generation | 20/minute | Streaming allows ~3 req/min with headroom |
| Model Operations | 3/minute | Expensive GPU operations |
| Transcription | 15/minute | ASR is faster than TTS |
| Prompt Operations | 10/minute | I/O bound operations |
| Config Changes | 1/minute | Should be rare |

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
      "config_ops": "5/hour"
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

The server respects `X-Forwarded-For` headers **only when not on loopback**:

```python
def _get_real_client_ip(request: Request) -> str:
    direct_host = request.client.host if request.client else "127.0.0.1"
    is_loopback = direct_host in ("127.0.0.1", "::1", "localhost")
    if not is_loopback:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return direct_host
```

This security feature prevents rate limit bypass when the server is bound to localhost.

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

Expected output: First 20 requests succeed, next 5 return HTTP 429.

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

`X-Forwarded-For` headers are only trusted when the server is **not bound to loopback**. This prevents rate limit bypass attacks.

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

1. **Start with conservative limits** (20/minute for generation)
2. **Monitor 429 error rates** to detect abuse vs. legitimate high usage
3. **Use hybrid strategy** for public-facing endpoints
4. **Adjust limits based on actual usage patterns**
5. **Document custom limits** for your deployment environment
6. **Test rate limiting before production** deployment
