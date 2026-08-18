#!/usr/bin/env python3
"""PreToolUse hook: surface the local gate checklist before any `git push`.

Implements .claude/hookify.prepush-local-gates.local.md (action: warn) as a
permissionDecision "ask": the reminder is shown and the call needs a human
ok, but it is not hard-blocked — the no-direct-push-main hook owns hard
blocks. Three CI failures came from pushing after running only the new test
file, never the matching gates.

Protocol: reads the PreToolUse payload on stdin; prints ask-decision JSON on
stdout for matching commands; always exits 0 (a warn never kills the call on
its own). Unparseable input exits 0 silently.
"""
import json
import re
import sys

_PUSH_RE = re.compile(r"\bgit\s+push\b")

_REASON = (
    "[prepush-local-gates] Before pushing, run the local gates matching the "
    "files you touched — running just the new/changed test file is not enough. "
    "Three CI failures came from skipping these: "
    "(1) ruff check qwen3_tts tests — unused imports pass pytest but fail CI lint; "
    "(2) if CLAUDE.md changed: python3 -c \"print(len(open('CLAUDE.md').readlines()))\" "
    "must be <=300 (hard gate, readlines() semantics); "
    "(3) python tests/run_batches.py --batch N for whichever batch owns the files "
    "you touched (map via tests/run_batches.py). "
    "A broad matrix red is almost always one test, not coverage (gate 80%, repo ~83%)."
)


def main():
    try:
        payload = json.load(sys.stdin)
        command = payload.get("tool_input", {}).get("command", "")
    except Exception:
        return 0
    if _PUSH_RE.search(command):
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": _REASON,
            }
        }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
