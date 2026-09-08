---
description: Start all services and open PM2 monitor.
version: 1.0.0
---

Start all services and open PM2 monitor.

```bash
cd "/Users/ericepstein/Qwen3-TTS_UserFiles" && pm2 start ecosystem.config.cjs && pm2 monit
```

## Observation

`pm2 monit` shows a live per-process CPU/memory view of every app declared in `ecosystem.config.cjs` (currently `tts-server-5123`); the dashboard updates in place.

## Feedback

Success: every app in `pm2 status` is `online` and `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5123/health` returns `200`/`503` (loading). Quit `monit` with Ctrl-C. Failure: an app `errored` or absent — it was never registered; start it individually.

## Rollback

Undo the start: `/pm2-all-stop` (`pm2 stop all`). `pm2 monit` is view-only — nothing to undo there.
