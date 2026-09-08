---
description: View PM2 status.
version: 1.0.0
---

View PM2 status.

```bash
cd "/Users/ericepstein/Qwen3-TTS_UserFiles" && pm2 status
```

## Observation

A table of every managed app: name, status (`online`/`stopped`/`errored`), uptime, `↺` restart count, and memory. This is the authoritative view of what PM2 supervises — an app missing here is not registered.

## Feedback

Success: exit 0 with the expected apps listed (currently `tts-server-5123`) in the expected state. A high `↺` with short uptimes is a crash loop — go to `pm2 logs tts-server-5123`.

## Rollback
View-only — nothing to undo. Changes to this command file itself: `git revert <commit>` on the branch that touched it.
