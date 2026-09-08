---
description: Restart TTS server (5123).
version: 1.0.0
---

Restart TTS server (5123).

```bash
cd "/Users/ericepstein/Qwen3-TTS_UserFiles" && pm2 restart tts-server-5123
```

## Observation

The app's `↺` restart counter increments and its uptime resets in `pm2 status`; the startup sequence replays in `pm2 logs tts-server-5123` (env-var and code changes only take effect because of this restart).

## Feedback

Success: status returns to `online` and `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5123/health` returns `200` (or `503` while models reload — wait for `/ready`). Failure: `errored` status or restart counter climbing without coming up.

## Rollback

A restart is its own recovery — re-run it if the server wedges. To undo the code/config that motivated the restart: `git revert <commit>`, then restart again. To take the server down instead: `/pm2-5123-stop`.
