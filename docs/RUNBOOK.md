# Qwen3-TTS Operations Runbook

> **AUTO-GENERATED** from deployment and operational procedures. Do not edit manual sections.

## Deployment

### Local Development Setup

#### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/Qwen3-TTS.git
cd Qwen3-TTS

# Create conda environment
conda create -n qwen3-tts-mlx python=3.9 -y
conda activate qwen3-tss-mlx

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
    "auto_shutdown_minutes": 30
  },
  "rate_limiting": {
    "enabled": true,
    "requests_per_minute": 10
  }
}
```

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

```bash
# Install PM2
npm install -g pm2

# Start server with PM2
pm2 start ~/.local/bin/tts --name "tts-server" -- server start

# Check status
pm2 status
pm2 logs tts-server

# Stop server
pm2 stop tts-server
pm2 delete tts-server
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

Expected response:
```json
{
  "status": "ready",
  "clone_model_loaded": true,
  "design_model_loaded": false,
  "custom_model_loaded": false
}
```

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

**Load individual models:**
```bash
# Via CLI
tts server load clone
tts server load design
tts server load custom

# Via API
curl -X POST http://127.0.0.1:5123/load-model \
  -H "Authorization: Bearer $(cat ~/.voice_server_token)" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "clone"}'
```

#### Unloading Models

**Unload models to free memory:**
```bash
# Via CLI
tts server unload clone

# Via API
curl -X POST http://127.0.0.1:5123/unload-model \
  -H "Authorization: Bearer $(cat ~/.voice_server_token)" \
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
  -H "Authorization: Bearer $(cat ~/.voice_server_token)"
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
  -H "Authorization: Bearer $(cat ~/.voice_server_token)"
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
# Clear cache and retry
tts cache clear
tts server load clone
```

#### Issue: Generation Fails with 429

**Symptoms:** "Rate limit exceeded" errors

**Diagnosis:**
```bash
# Check rate limit status
curl http://127.0.0.1:5123/stats \
  -H "Authorization: Bearer $(cat ~/.voice_server_token)"
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
# Unload unused models
tts server unload design

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

# Restart server
tts server start

# Load required models
tts server load clone
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
- Update dependencies: `pip install -e ".[mlx,server,ui] --upgrade"`
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
# Models will be re-downloaded on first use
tts server load clone
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
   conda activate qwen3-tss-mlx
   pip install -e ".[mlx,server,ui] --upgrade"
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

**Default rate limit:** 10 requests per minute

**Adjust in config.json:**
```json
{
  "rate_limiting": {
    "enabled": true,
    "requests_per_minute": 10,
    "burst": 2
  }
}
```

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
