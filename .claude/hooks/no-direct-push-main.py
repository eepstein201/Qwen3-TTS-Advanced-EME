#!/usr/bin/env python3
"""PreToolUse hook: block git operations that push, merge into, or delete main.

Repo policy (CLAUDE.md "Git Workflow"): Claude never touches main directly —
feature branches only; the user reviews and merges. Implements the intent of
.claude/hookify.no-direct-push-main.local.md (action: block), which was a
definition with no executor until this hook was wired in
.claude/settings.local.json.

Covers the `/opt/homebrew/bin/git push ...` spelling (permission prefix rules
match on the literal command head) and a bare `git push` while checked out on
main, which pushes TO main even though no refspec names it. Each segment of a
compound command is judged separately, so `git push origin fix/x && git
checkout main` is not a false positive.

Protocol: reads the PreToolUse payload on stdin; exit 2 blocks the call with
stderr shown to the model. Unparseable input or non-matching commands exit 0
(fail-open on payload shape, never on policy).
"""
import json
import re
import subprocess
import sys

_PUSH_RE = re.compile(r"\bgit\s+push\b")
_MERGE_RE = re.compile(r"\bgit\s+merge\b")
_MAIN_RE = re.compile(r"\bmain\b")
_DELETE_RE = re.compile(r"--delete\b")
_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\||\n")


def _current_branch():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _segment_violation(segment):
    """Return a reason string if this single command segment violates policy."""
    if _PUSH_RE.search(segment):
        if _MAIN_RE.search(segment):
            return "push names main"
        if _DELETE_RE.search(segment):
            return "push deletes a remote branch"
        has_refspec = bool(re.search(r"git\s+push\s+\S+\s+\S", segment))
        if not has_refspec and _current_branch() == "main":
            return "bare push while checked out on main"
    if _MERGE_RE.search(segment) and _MAIN_RE.search(segment):
        return "merge names main"
    return None


def _violates(command):
    for segment in _SEGMENT_SPLIT_RE.split(command):
        reason = _segment_violation(segment.strip())
        if reason:
            return reason
    return None


def main():
    try:
        payload = json.load(sys.stdin)
        command = payload.get("tool_input", {}).get("command", "")
    except Exception:
        return 0
    reason = _violates(command)
    if reason:
        print(
            "BLOCKED by no-direct-push-main (%s): repo policy is feature-branch"
            " workflow — Claude never pushes, merges into, or deletes branches"
            " on main directly, even after being told to go ahead. Commit to a"
            " feature branch and hand the exact command to the user to run"
            " personally." % reason,
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
