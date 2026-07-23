---
description: Save current session state to a dated .md file in ~/.claude/session-data/ so work can be resumed in a future session with full context. Project override — always writes .md, never .tmp.
---

# Save Session Command (project override — always .md)

> **Project-specific override.** The upstream `ecc:save-session` command (and the
> `session-manager.js` it's paired with) defaults to a `*-session.tmp` filename.
> `.tmp` files in this environment can be cleaned up between sessions and are not
> durable. **In this repository, session files must always be written with a
> `.md` extension.** Everything else about the process is unchanged from the
> upstream command.

Capture everything that happened in this session — what was built, what worked, what failed, what's left — and write it to a dated file so the next session can pick up exactly where this one left off.

## When to Use

- End of a work session before closing Claude Code
- Before hitting context limits (run this first, then start a fresh session)
- After solving a complex problem you want to remember
- Any time you need to hand off context to a future session

## Process

### Step 1: Gather context

Before writing the file, collect:

- Read all files modified during this session (use git diff or recall from conversation)
- Review what was discussed, attempted, and decided
- Note any errors encountered and how they were resolved (or not)
- Check current test/build status if relevant

### Step 2: Create the sessions folder if it doesn't exist

```bash
mkdir -p ~/.claude/session-data
```

### Step 3: Write the session file

Create `~/.claude/session-data/YYYY-MM-DD-<short-id>-session.md` — **`.md`, not
`.tmp`** — using today's actual date and a short-id:

- Compatibility characters: letters `a-z` / `A-Z`, digits `0-9`, hyphens `-`, underscores `_`
- Compatibility minimum length: 1 character
- Recommended style for new files: lowercase letters, digits, and hyphens with 8+ characters to avoid collisions

Valid examples: `abc123de`, `a1b2c3d4`, `frontend-worktree-1`, `ChezMoi_2`

Full valid filename example: `2026-07-23-abc123de-session.md`

If an existing `*-session.tmp` file for today is found instead, treat it as a
mistake to correct: write the equivalent `.md` file with the same content and
delete or ignore the `.tmp` (don't silently perpetuate the wrong extension).

### Step 4: Populate the file with all sections below

Write every section honestly. Do not skip sections — write "Nothing yet" or "N/A" if a section genuinely has no content. An incomplete file is worse than an honest empty section.

### Step 5: Show the file to the user

After writing, display the full contents and ask:

```
Session saved to [actual resolved path to the session file]

Does this look accurate? Anything to correct or add before we close?
```

Wait for confirmation. Make edits if requested.

---

## Session File Format

```markdown
# Session: YYYY-MM-DD

**Started:** [approximate time if known]
**Last Updated:** [current time]
**Project:** [project name or path]
**Topic:** [one-line summary of what this session was about]

---

## What We Are Building

[1-3 paragraphs describing the feature, bug fix, or task. Include enough
context that someone with zero memory of this session can understand the goal.
Include: what it does, why it's needed, how it fits into the larger system.]

---

## What WORKED (with evidence)

[List only things that are confirmed working. For each item include WHY you
know it works — test passed, ran in browser, Postman returned 200, etc.
Without evidence, move it to "Not Tried Yet" instead.]

- **[thing that works]** — confirmed by: [specific evidence]

If nothing is confirmed working yet: "Nothing confirmed working yet — all approaches still in progress or untested."

---

## What Did NOT Work (and why)

[This is the most important section. List every approach tried that failed.
For each failure write the EXACT reason so the next session doesn't retry it.]

- **[approach tried]** — failed because: [exact reason / error message]

If nothing failed: "No failed approaches yet."

---

## What Has NOT Been Tried Yet

- [approach / idea]

If nothing is queued: "No specific untried approaches identified."

---

## Current State of Files

| File              | Status         | Notes                      |
| ----------------- | -------------- | -------------------------- |
| `path/to/file.ts` | Complete       | [what it does]             |
| `path/to/file.ts` | In Progress    | [what's done, what's left] |
| `path/to/file.ts` | Broken         | [what's wrong]             |
| `path/to/file.ts` | Not Started    | [planned but not touched]  |

If no files were touched: "No files modified this session."

---

## Decisions Made

- **[decision]** — reason: [why this was chosen over alternatives]

If no significant decisions: "No major decisions made this session."

---

## Blockers & Open Questions

- [blocker / open question]

If none: "No active blockers."

---

## Exact Next Step

[The single most important thing to do when resuming, precise enough that
resuming requires zero thinking about where to start. If not known: "Next step
not determined — review 'What Has NOT Been Tried Yet' and 'Blockers' sections
to decide on direction before starting."]

---

## Environment & Setup Notes

[Only fill this if relevant. Omit the section entirely if none.]
```

---

## Notes

- Each session gets its own file — never append to a previous session's file
- The "What Did NOT Work" section is the most critical — future sessions will blindly retry failed approaches without it
- If the user asks to save mid-session (not just at the end), save what's known so far and mark in-progress items clearly
- Use the canonical global session store: `~/.claude/session-data/`
- **Always `.md`, never `.tmp`, in this repository.**
- Caveat for resuming: the upstream `/resume-session` command's no-argument
  auto-discovery (via `session-manager.js`) only scans for `*.tmp` files, so it
  will **not** find these `.md` files automatically. Resume with an explicit
  path instead: `/resume-session ~/.claude/session-data/<file>.md` (its Step 1
  already supports "if it looks like a file path, read it directly," no
  extension check). See `.claude/commands/ecc/resume-session.md` in this repo
  for a matching override.
