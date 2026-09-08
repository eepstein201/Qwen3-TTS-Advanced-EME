---
description: Restart all services.
version: 1.0.0
---

Restart all services.

```bash
cd "/Users/ericepstein/Qwen3-TTS_UserFiles" && pm2 restart all
```

## Observation

Every app's `↺` counter increments and uptime resets in `pm2 status`; each app replays its startup in its own `pm2 logs <app>` stream.

## Feedback

Success: all apps `online` afterward and `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5123/health` returns `200` (or `503` while models reload). Failure: any app `errored` — restart that one individually and check its log.

## Rollback

A restart is its own recovery — re-run it. To undo the change that motivated it: `git revert <commit>`, then restart again. To take everything down instead: `/pm2-all-stop`.
