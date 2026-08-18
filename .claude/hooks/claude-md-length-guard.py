#!/usr/bin/env python3
"""PreToolUse hook: keep CLAUDE.md at <=300 lines (hard CI gate).

tests/test_claude_md.py::test_claude_md_under_300_lines counts readlines(),
so a file with 300 '\\n' plus a final unterminated line counts as 301 — wc -l
under-reports by one. For Write payloads the new content is counted directly;
for Edit payloads the replacement is applied to the current file and the
result counted, so the check sees the post-edit state. Implements
.claude/hookify.claude-md-length-guard.local.md, upgraded from warn to
deny-on-violation: the CI gate is hard, and this hook only fires when the
edit would actually breach it (otherwise it is silent).

Only the project root's CLAUDE.md is guarded — a CLAUDE.md elsewhere (e.g.
/tmp) is out of scope.

Protocol: reads the PreToolUse payload on stdin; exit 2 blocks the call with
stderr shown to the model. Unparseable input or non-matching paths exit 0.
"""
import json
import os
import sys

MAX_LINES = 300


def _line_count(content):
    """Count lines exactly as file.readlines() would."""
    if not content:
        return 0
    return content.count("\n") + (0 if content.endswith("\n") else 1)


def _future_content(tool_name, tool_input, current):
    """Return the file content after the proposed tool call, or None."""
    if tool_name == "Write":
        return tool_input.get("content", "")
    if tool_name == "Edit":
        old = tool_input.get("old_string", "")
        new = tool_input.get("new_string", "")
        if old not in current:
            return None  # Edit will fail on its own; nothing to guard
        if tool_input.get("replace_all"):
            return current.replace(old, new)
        return current.replace(old, new, 1)
    return None


def main():
    try:
        payload = json.load(sys.stdin)
        tool_name = payload.get("tool_name", "")
        tool_input = payload.get("tool_input", {})
        file_path = tool_input.get("file_path", "")
    except Exception:
        return 0

    if os.path.basename(file_path) != "CLAUDE.md":
        return 0
    project_root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    target = os.path.join(project_root, "CLAUDE.md")
    try:
        with open(target) as f:
            current = f.read()
        # samefile, not realpath equality: on macOS's case-insensitive FS the
        # shell $PWD spelling can differ in case from the canonical path, and
        # realpath does not normalize that — string compare silently allows.
        if not os.path.samefile(file_path, target):
            return 0
    except OSError:
        return 0

    future = _future_content(tool_name, tool_input, current)
    if future is None:
        return 0
    count = _line_count(future)
    if count > MAX_LINES:
        print(
            "BLOCKED by claude-md-length-guard: this %s would put CLAUDE.md at"
            " %d lines (hard CI gate: <=%d, checked via readlines() — wc -l"
            " under-reports by one). Prefer folding detail into existing table"
            " rows over adding new paragraphs; move deep-dive content to"
            " docs/00-Foundations/ARCHITECTURE.md. Verify with: python3 -c"
            " \"print(len(open('CLAUDE.md').readlines()))\""
            % (tool_name, count, MAX_LINES),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
