---
description: Stop TTS server (5123).
version: 1.0.0
---

Stop TTS server (5123).

```bash
cd "/Users/ericepstein/Qwen3-TTS_UserFiles" && pm2 stop tts-server-5123
```

## Observation

The app's status flips to `stopped` in `pm2 status` (process gone, registration kept — `autorestart` cannot revive a stopped app).

## Feedback

Success: `curl http://127.0.0.1:5123/health` refuses the connection. Note the memory (per `tts server status`) drops once the process is gone.

## Rollback

Undo the stop: `/pm2-5123` (`pm2 start ecosystem.config.cjs --only tts-server-5123`). A raw `kill` is NOT the rollback path — it reads as a crash and PM2 respawns the app.
