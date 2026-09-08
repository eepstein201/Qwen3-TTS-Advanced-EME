---
description: View all PM2 logs.
version: 1.0.0
---

View all PM2 logs.

```bash
cd "/Users/ericepstein/Qwen3-TTS_UserFiles" && pm2 logs
```

## Observation

A combined, interleaved tail of every app's stdout/stderr from `~/.pm2/logs/`; lines are prefixed with the app name. The application-level log with timings is `.voice_server.log` in the repo root — `pm2 logs` shows process output, not that file.

## Feedback

Success: the tail streams without error; Ctrl-C exits cleanly. No new lines means the apps are quiet (or stopped — cross-check `pm2 status`).

## Rollback
View-only — nothing to undo. Changes to this command file itself: `git revert <commit>` on the branch that touched it.
