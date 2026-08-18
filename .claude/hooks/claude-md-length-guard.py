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


def _resolve_project_root():
    """Project root as a canonical absolute path.

    CLAUDE_PROJECT_DIR is provided by the harness; cwd is the fallback.
    The environment value is realpath'd before any file operation touches
    a path derived from it — env and stdin are untrusted sources (CodeQL),
    so both are validated rather than used as given.
    """
    env_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_dir:
        return os.path.realpath(env_dir)
    return os.getcwd()


def _is_within(path, root):
    """True when path is root itself or lies underneath it.

    Case-folded on macOS only: its filesystems are case-insensitive and a
    tool-provided spelling can differ in case from the canonical root
    (realpath does not normalize that on darwin). The stat-based
    samefile() comparison in main() remains the authoritative identity
    check; this containment filter only rejects paths that point outside
    the project entirely.
    """
    if sys.platform == "darwin":
        path, root = path.lower(), root.lower()
    return path == root or path.startswith(root + os.sep)


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
    if not os.path.isabs(file_path):
        return 0  # relative paths cannot be anchored to the project root
    project_root = _resolve_project_root()
    resolved = os.path.realpath(file_path)
    if not _is_within(resolved, project_root):
        return 0
    target = os.path.join(project_root, "CLAUDE.md")
    try:
        with open(target) as f:
            current = f.read()
        # samefile on the resolved path, not realpath equality: on macOS's
        # case-insensitive FS the spelling can differ in case from the
        # canonical path, and realpath does not normalize that — string
        # compare silently allows. Resolving first also strips .. segments
        # that could name a nonexistent intermediate directory, which would
        # make stat fail and fail the guard open.
        if not os.path.samefile(resolved, target):
            return 0
    except OSError:
        return 0

    future = _future_content(tool_name, tool_input, current)
    if future is None:
        return 0
    count = _line_count(future)
    if count > MAX_LINES:
        print(
            f"BLOCKED by claude-md-length-guard: this {tool_name} would put"
            f" CLAUDE.md at {count} lines (hard CI gate: <={MAX_LINES},"
            " checked via readlines() — wc -l under-reports by one). Prefer"
            " folding detail into existing table rows over adding new"
            " paragraphs; move deep-dive content to"
            " docs/00-Foundations/ARCHITECTURE.md. Verify with: python3 -c"
            " \"print(len(open('CLAUDE.md').readlines()))\"",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
