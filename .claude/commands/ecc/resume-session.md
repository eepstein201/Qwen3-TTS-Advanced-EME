---
description: Load the most recent session file from ~/.claude/session-data/ and resume work with full context from where the last session ended. Project override — discovers .md session files, not just .tmp.
---

# Resume Session Command (project override — finds .md sessions)

> **Project-specific override.** This repository's `ecc:save-session` override
> always writes `*-session.md` files (not `.tmp`) because `.tmp` files here are
> not durable across resets. The upstream `resume-session` command's
> auto-discovery logic only looks for `*-session.tmp`, so without this override
> it would silently find nothing. This version prefers `.md`, falls back to
> `.tmp` for older/legacy files, and is otherwise identical to the upstream
> command.

Load the last saved session state and orient fully before doing any work.
This command is the counterpart to `/save-session`.

## When to Use

- Starting a new session to continue work from a previous day
- After starting a fresh session due to context limits
- When handing off a session file from another source (just provide the file path)
- Any time you have a session file and want Claude to fully absorb it before proceeding

## Usage

```
/resume-session                                                       # loads most recent file in ~/.claude/session-data/
/resume-session 2026-07-23                                            # loads most recent session for that date
/resume-session ~/.claude/session-data/2026-07-23-abc123de-session.md # loads a specific session file directly
```

## Process

### Step 1: Find the session file

If no argument provided:

1. Check `~/.claude/session-data/`
2. Pick the most recently modified file matching `*-session.md`. If none exist,
   fall back to the most recently modified `*-session.tmp` (legacy/upstream
   format) so nothing gets silently missed during the transition.
3. If the folder does not exist or has no matching files, tell the user:
   ```
   No session files found in ~/.claude/session-data/
   Run /save-session at the end of a session to create one.
   ```
   Then stop.

If an argument is provided:

- If it looks like a date (`YYYY-MM-DD`), search `~/.claude/session-data/` for
  files matching `YYYY-MM-DD-session.md`, `YYYY-MM-DD-<shortid>-session.md`
  (current format), or the legacy `.tmp` equivalents, and load the most
  recently modified match for that date
- If it looks like a file path, read that file directly (any extension)
- If not found, report clearly and stop

### Step 2: Read the entire session file

Read the complete file. Do not summarize yet.

### Step 3: Confirm understanding

Respond with a structured briefing in this exact format:

```
SESSION LOADED: [actual resolved path to the file]
════════════════════════════════════════════════

PROJECT: [project name / topic from file]

WHAT WE'RE BUILDING:
[2-3 sentence summary in your own words]

CURRENT STATE:
Working: [count] items confirmed
In Progress: [list files that are in progress]
Not Started: [list planned but untouched]

WHAT NOT TO RETRY:
[list every failed approach with its reason — this is critical]

OPEN QUESTIONS / BLOCKERS:
[list any blockers or unanswered questions]

NEXT STEP:
[exact next step if defined in the file]
[if not defined: "No next step defined — recommend reviewing 'What Has NOT Been Tried Yet' together before starting"]

════════════════════════════════════════════════
Ready to continue. What would you like to do?
```

### Step 4: Wait for the user

Do NOT start working automatically. Do NOT touch any files. Wait for the user to say what to do next.

If the next step is clearly defined in the session file and the user says "continue" or "yes" or similar — proceed with that exact next step.

If no next step is defined — ask the user where to start, and optionally suggest an approach from the "What Has NOT Been Tried Yet" section.

---

## Edge Cases

**Multiple sessions for the same date** (`.md` and legacy `.tmp` both present):
Prefer the `.md` file; only use `.tmp` if no `.md` exists for that date.

**Session file references files that no longer exist:**
Note this during the briefing — "WARNING: `path/to/file.ts` referenced in session but not found on disk."

**Session file is from more than 7 days ago:**
Note the gap — "WARNING: This session is from N days ago (threshold: 7 days). Things may have changed." — then proceed normally.

**User provides a file path directly (e.g., forwarded from a teammate):**
Read it and follow the same briefing process — the format is the same regardless of source or extension.

**Session file is empty or malformed:**
Report: "Session file found but appears empty or unreadable. You may need to create a new one with /save-session."

---

## Notes

- Never modify the session file when loading it — it's a read-only historical record
- The briefing format is fixed — do not skip sections even if they are empty
- "What Not To Retry" must always be shown, even if it just says "None" — it's too important to miss
- After resuming, the user may want to run `/save-session` again at the end of the new session to create a new dated file
- In this repository, prefer `.md` session files over `.tmp` — see `.claude/commands/ecc/save-session.md`
