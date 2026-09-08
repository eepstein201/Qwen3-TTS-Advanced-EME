---
description: Stop all services.
version: 1.0.0
---

Stop all services.

```bash
cd "/Users/ericepstein/Qwen3-TTS_UserFiles" && pm2 stop all
```

## Observation

Every app flips to `stopped` in `pm2 status` (registrations kept, processes gone).

## Feedback

Success: `curl http://127.0.0.1:5123/health` refuses the connection and no managed process remains under `pm2 status`. Anything still listening on a service port is NOT PM2-managed — find it with `lsof -i :5123`.

## Rollback

Undo the stop: `/pm2-all` (`pm2 start ecosystem.config.cjs`).
