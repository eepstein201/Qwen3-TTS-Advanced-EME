---
name: mine-patterns
description: Mine observer JSONL logs for recurring patterns and persist 3+ occurrence patterns as instinct files with a hard exploration budget. Use when asked to analyze observer/session logs for patterns.
---

# Mine observer log patterns

- Read the observer JSONL path provided. Budget: max 3 Bash/jq calls, targeted queries only.
- Count recurring project patterns.
- ONLY write instinct files for patterns occurring 3+ times (never 2).
- Update the MEMORY index.
- Write files; keep chat output to a one-line confirmation.

## Headless variant

For scheduled / non-interactive runs (no interruption failure mode):

    claude -p "Analyze <observer.jsonl> for patterns appearing 3+ times per CLAUDE.md rules. Max 5 Bash calls, then write instinct files." --allowedTools "Read,Write,Bash" > ~/logs/mine-output.txt 2>&1

(mkdir -p ~/logs first; substitute the real log path.)
