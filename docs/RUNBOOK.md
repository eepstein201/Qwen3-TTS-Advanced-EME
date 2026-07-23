# Qwen3-TTS Operations Runbook

> **AUTO-GENERATED** from deployment and operational procedures. Do not edit manual sections.

## Deployment

### Local Development Setup

#### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/eepstein201/Qwen3-TTS-Advanced-EME.git
cd Qwen3-TTS-Advanced-EME

# Create conda environment
conda create -n qwen3-tts-mlx python=3.11 -y
conda activate qwen3-tts-mlx

# Install with MLX backend (Apple Silicon)
pip install -e ".[mlx,server,ui]"

# OR install with Torch backend (Linux/Intel Mac)
pip install -e ".[torch,server,ui]"
```

#### 2. Configuration

```bash
# Run configuration wizard
tts config

# Or edit config.json directly
tts config edit
```

**Key settings for production:**
```json
{
  "models": {
    "clone": { "load_at_startup": true },
    "design": { "load_at_startup": false }
  },
  "advanced": {
    "backend": "mlx",
    "model_size": "1.7B",
    "mlx_quantization": "8bit"
  },
  "server": {
    "port": 5123,
    "auto_shutdown_minutes": 0
  },
  "security": {
    "rate_limits": {
      "generate": "20/minute",
      "model_ops": "3/minute"
    }
  }
}
```

Rate limits use slowapi's `"<count>/<unit>"` string form under `security.rate_limits`
(see [rate-limiting.md](rate-limiting.md)). `auto_shutdown_minutes: 0` disables idle
auto-shutdown — set a positive value for ephemeral/cloud runs.

#### 3. Verify Installation

```bash
tts doctor
```

Expected output: All checks pass ✓

### Server Deployment

#### Starting the Server

**Development mode (auto-reload on code changes):**
```bash
tts server start
```

**Production mode with PM2 (recommended):**

The repo ships an `ecosystem.config.cjs` defining the `tts-server-5123` app (runs `start.cjs`).

```bash
# Install PM2
npm install -g pm2

# First time: start from the ecosystem file
pm2 start ecosystem.config.cjs

# Subsequent starts / lifecycle
pm2 start tts-server-5123
pm2 restart tts-server-5123
pm2 stop tts-server-5123

# Check status and logs
pm2 status
pm2 logs tts-server-5123

# Persist / restore the process list across reboots
pm2 save
pm2 resurrect
```

#### Server Health Checks

**Check server status:**
```bash
tts server status
```

**Health endpoint (for monitoring):**
```bash
curl http://127.0.0.1:5123/health
```

Expected response (once models are loaded):
```json
{
  "status": "ok",
  "backend": "mlx",
  "model_size": "1.7B",
  "clone_model_loaded": true,
  "design_model_loaded": false,
  "custom_model_loaded": false,
  "model_load_times": {},
  "model_load_errors": {},
  "mlx_quantization": "8bit"
}
```

While models are still loading, `/health` returns `503` with `{"status": "loading", ...}`.

**Readiness probe (for load balancers):**
```bash
curl http://127.0.0.1:5123/ready
```

Returns `503` while loading, `200` when ready.

#### Server Logs

**Tail server log:**
```bash
tts server log
```

**Or directly:**
```bash
tail -f .voice_server.log
```

**PM2 logs:**
```bash
pm2 logs tts-server
pm2 logs tts-server --lines 100
```

### Model Management

#### Loading Models

Models load automatically at startup based on `models.<type>.load_at_startup` in
`config.json`, and on demand when a request needs them. To load a model
explicitly, use the API (there is no `tts server load` CLI subcommand):

```bash
curl -X POST http://127.0.0.1:5123/load-model \
  -H "Authorization: Bearer $(cat ~/.config/qwen3-tts/.voice_server_token)" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "clone"}'
```

#### Unloading Models

**Unload models to free memory (API only):**
```bash
curl -X POST http://127.0.0.1:5123/unload-model \
  -H "Authorization: Bearer $(cat ~/.config/qwen3-tts/.voice_server_token)" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "clone"}'
```

#### Checking Model Status

**List loaded models:**
```bash
tts list models
```

**Get detailed model info:**
```bash
curl http://127.0.0.1:5123/models \
  -H "Authorization: Bearer $(cat ~/.config/qwen3-tts/.voice_server_token)"
```

### Backup and Recovery

#### Configuration Backup

**Backup config.json:**
```bash
cp config.json config.json.backup
```

**Voice prompts backup:**
```bash
# Voice prompts are stored in voice_prompts/
tar -czf voice_prompts_backup.tar.gz voice_prompts/
```

#### Model Cache Backup

**List cached models:**
```bash
tts cache list
```

**Backup cache directory:**
```bash
# Cache is typically at ~/.cache/huggingface/
tar -czf hf_cache_backup.tar.gz ~/.cache/huggingface/
```

## Monitoring

### Server Metrics

**Get server statistics:**
```bash
tts stats
```

**Via API:**
```bash
curl http://127.0.0.1:5123/stats \
  -H "Authorization: Bearer $(cat ~/.config/qwen3-tts/.voice_server_token)"
```

Returns:
```json
{
  "memory": {
    "mlx_memory_active_mb": 2500.5
  },
  "models": {
    "clone_model_loaded": true,
    "design_model_loaded": false,
    "custom_model_loaded": false
  },
  "backend": "mlx",
  "model_size": "1.7B",
  "mlx_quantization": "8bit",
  "generation_history": []
}
```

### Performance Monitoring

**Memory usage:**
```bash
# Check server memory
tts server status

# Check system memory
vm_stat | compress_pages=0
```

**Disk usage:**
```bash
# Model cache size
tts cache size

# Voice prompts size
du -sh voice_prompts/
```

### Alerting

**Critical alerts to monitor:**

1. **Server down**: Health check returns non-200
2. **Model unloaded**: Clone model not loaded when expected
3. **Memory high**: Active memory > available RAM
4. **Rate limiting**: High rate of 429 responses

**Example monitoring script:**
```bash
#!/bin/bash
# Simple health check monitoring

while true; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5123/health)
  if [ "$STATUS" != "200" ]; then
    echo "ALERT: Server health check failed (status: $STATUS)"
    # Send alert (email, Slack, etc.)
  fi
  sleep 60
done
```

## Troubleshooting

### Common Issues and Fixes

#### Issue: Server Won't Start

**Symptoms:** Port 5123 already in use

**Diagnosis:**
```bash
lsof -i :5123
```

**Fix:**
```bash
# Kill process using port
kill -9 $(lsof -ti :5123)

# Or use server stop
tts server stop

# Restart server
tts server start
```

#### Issue: Model Loading Fails

**Symptoms:** "Failed to load model" error

**Diagnosis:**
```bash
# Check disk space
df -h

# Check HuggingFace cache
tts cache list

# Check server log
tts server log
```

**Fix:**
```bash
# Clear cache and retry (models re-download and reload on next request)
tts cache clear
tts server restart
```

#### Issue: Generation Fails with 429

**Symptoms:** "Rate limit exceeded" errors

**Diagnosis:**
```bash
# Check rate limit status
curl http://127.0.0.1:5123/stats \
  -H "Authorization: Bearer $(cat ~/.config/qwen3-tts/.voice_server_token)"
```

**Fix:**
```bash
# Wait for rate limit to reset (60 seconds)
# Or increase rate limit in config.json
```

#### Issue: Memory Exhaustion

**Symptoms:** OOM errors, slow generation

**Diagnosis:**
```bash
# Check memory usage
tts server status

# Check system memory
vm_stat
```

**Fix:**
```bash
# Unload unused models (API)
curl -X POST http://127.0.0.1:5123/unload-model \
  -H "Authorization: Bearer $(cat ~/.config/qwen3-tts/.voice_server_token)" \
  -H "Content-Type: application/json" -d '{"model_type": "design"}'

# Switch to 4-bit quantization
tts config edit
# Set advanced.mlx_quantization to "4bit"

# Restart server
tts server stop && tts server start
```

#### Issue: Audio Quality Problems

**Symptoms:** Distorted audio, wrong speed/pitch

**Diagnosis:**
```bash
# Check audio loader in config
tts config show | grep audio_loader

# Test generation with known-good settings
tts "Test" -o test.wav --preset stable
```

**Fix:**
```bash
# Try different audio loader
# In config.json, set advanced.audio_loader to "torchaudio" or "librosa"

# Restart server
tts server stop && tts server start
```

### Emergency Procedures

#### Force Server Shutdown

**If server is unresponsive:**
```bash
# Find and kill process
kill $(cat .voice_server.pid)

# Or force kill
kill -9 $(cat .voice_server.pid)
```

#### Clean Restart

**Full server reset:**
```bash
# Stop server
tts server stop

# Clear models
tts cache clear

# Restart server (models with load_at_startup=true reload automatically)
tts server start
```

#### Configuration Reset

**Reset to defaults:**
```bash
# Backup current config
cp config.json config.json.backup

# Reset to defaults
tts uninstall config

# Reconfigure
tts config
```

## Maintenance

### Regular Maintenance Tasks

#### Weekly

- Review server logs for errors: `tts server log | grep ERROR`
- Check disk space: `df -h`
- Monitor model cache size: `tts cache size`

#### Monthly

- Prune unused models: `tts cache prune`
- Update dependencies: `pip install --upgrade -e ".[mlx,server,ui]"`
- Review rate limiting effectiveness

#### Quarterly

- Audit voice prompts: `tts voice list`
- Review and update generation presets
- Security audit of dependencies

### Cache Management

**Prune unused models:**
```bash
# Remove models not used in 30 days
tts cache prune

# Or specify custom days
python -c "
from qwen3_tts.tools.model_cache import prune_cache
prune_cache(days=60)
"
```

**Clear all cache:**
```bash
tts cache clear
```

**Rebuild cache:**
```bash
# Models will be re-downloaded on the next generation request or server restart
tts server restart
```

### Log Rotation

**Server log rotation:**
```bash
# Archive old logs
mv .voice_server.log .voice_server.log.old

# Compress old logs
gzip .voice_server.log.old
```

**Configure log rotation (optional):**
```bash
# Add to config.json
{
  "server": {
    "log_level": "INFO",
    "log_rotation": true,
    "log_max_size_mb": 100
  }
}
```

## Upgrade Procedure

### Upgrading Qwen3-TTS

**Safe upgrade process:**

1. **Backup current installation**
   ```bash
   cp config.json config.json.backup
   tar -czf qwen3tts_backup.tar.gz config.json voice_prompts/
   ```

2. **Pull latest changes**
   ```bash
   git fetch origin
   git pull origin main
   ```

3. **Update dependencies**
   ```bash
   conda activate qwen3-tts-mlx
   pip install --upgrade -e ".[mlx,server,ui]"
   ```

4. **Run health check**
   ```bash
   tts doctor
   ```

5. **Test basic functionality**
   ```bash
   tts "Test" -o test.wav
   ```

6. **Restart server if needed**
   ```bash
   tts server stop && tts server start
   ```

### Rollback Procedure

**If upgrade fails:**

1. **Restore backup**
   ```bash
   tar -xzf qwen3tss_backup.tar.gz
   ```

2. **Reinstall previous version**
   ```bash
   git checkout <previous-commit-hash>
   pip install -e ".[mlx,server,ui]"
   ```

3. **Verify functionality**
   ```bash
   tts doctor
   tts "Test" -o test.wav
   ```

## Security Considerations

### Authentication

**Server token location:**
- **Legacy**: `~/.voice_server_token`
- **New**: `~/.config/qwen3-tts/.voice_server_token`

**Token permissions:**
```bash
chmod 600 ~/.config/qwen3-tts/.voice_server_token
```

### Rate Limiting

**Default limits** are per-endpoint-group (slowapi), e.g. generation `20/minute`,
model ops `3/minute`.

**Adjust in config.json** under `security.rate_limits` (values are `"<count>/<unit>"` strings):
```json
{
  "security": {
    "rate_limits": {
      "generate": "20/minute",
      "model_ops": "3/minute",
      "transcribe": "15/minute",
      "prompt_ops": "10/minute",
      "config_ops": "1/minute"
    }
  }
}
```

Behind a reverse proxy, set `TTS_TRUSTED_PROXIES` so per-IP limits see the real
client IP. See [rate-limiting.md](rate-limiting.md) for strategies (per-IP / per-token / hybrid).

**For production deployment:** Increase based on capacity requirements.

### Network Security

**Server binds to:** `127.0.0.1` (localhost only)

**For external access:** Configure reverse proxy (nginx, traefik) with SSL.

**Example nginx config:**
```nginx
location /tts/ {
    proxy_pass http://127.0.0.1:5123/;
    proxy_set_header Authorization $http_authorization;
    proxy_set_header X-Real-IP $remote_addr;
}
```

## Performance Tuning

### Backend Selection

**MLX (Apple Silicon):**
- Pros: Faster inference, lower memory
- Cons: macOS only, limited model support

**Torch (Cross-platform):**
- Pros: Widely supported, more features
- Cons: Slower inference, higher memory

**Recommendation:** Use MLX on Apple Silicon, Torch elsewhere.

### Quantization Trade-offs

| Quantization | Memory | Speed | Quality |
|--------------|--------|-------|--------|
| `bf16` | High | Fast | Best |
| `8bit` | Medium | Medium | Good |
| `4bit` | Low | Slow | Acceptable |

**Recommendation:** Start with `8bit`, switch to `4bit` if memory constrained.

### Chunk Size Optimization

**Default:** 500 characters per chunk

**For longer texts:** Increase `max_chunk_chars` in config.json
**For shorter texts:** Decrease for more granular control

```json
{
  "generation": {
    "max_chunk_chars": 1000
  }
}
```
