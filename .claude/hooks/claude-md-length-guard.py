#!/usr/bin/env python3
"""PreToolUse hook: enforce the documented line budgets for context files.

Two targets, one guard (CLAUDE.md "Rules" documents both limits):

  - the project root's CLAUDE.md, <=300 lines (hard CI gate)
  - this project's memory index MEMORY.md, <=150 lines (memory index guard)

MEMORY.md lives OUTSIDE the repo, under ~/.claude/projects/<slug>/memory/, so
it gets its own containment root rather than the project-root/samefile check
CLAUDE.md uses. It was the documented rule with no executor: the CLAUDE.md
limit had a hook, the MEMORY.md limit had only prose, and prose does not fire.
One guard with two targets keeps the counting logic single-sourced.

The CLAUDE.md gate mirrors CI: tests/test_claude_md.py counts readlines(),
so a file with 300 '\\n' plus a final unterminated line counts as 301 — wc -l
under-reports by one. For Write payloads the new content is counted directly;
for Edit payloads the replacement is applied to the current file and the
result counted, so the check sees the post-edit state. Implements
.claude/hookify.claude-md-length-guard.local.md, upgraded from warn to
deny-on-violation: the CI gate is hard, and this hook only fires when the
edit would actually breach it (otherwise it is silent).

Only the project root's CLAUDE.md is guarded — a CLAUDE.md elsewhere (e.g.
/tmp) is out of scope. Likewise only a MEMORY.md under the Claude projects
memory tree is guarded; an unrelated MEMORY.md is not.

Protocol: reads the PreToolUse payload on stdin; exit 2 blocks the call with
stderr shown to the model. Unparseable input or non-matching paths exit 0.
"""
import json
import os
import sys

CLAUDE_MD_MAX_LINES = 300
MEMORY_MD_MAX_LINES = 150

# Backwards-compatible alias: the CLAUDE.md budget was this module's only
# limit, and the guard tests import it by this name.
MAX_LINES = CLAUDE_MD_MAX_LINES


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


def _resolve_memory_root():
    """Canonical root of the Claude per-project memory tree.

    MEMORY.md sits at ~/.claude/projects/<slug>/memory/MEMORY.md. The slug is
    machine-specific, so containment under this root — not an exact path — is
    what decides scope. realpath'd here so the prefix check below runs on the
    same canonical values the open() uses (CodeQL normalize-then-check).
    """
    return os.path.realpath(os.path.expanduser("~/.claude/projects"))


def _claude_md_target(file_path):
    """Resolve a CLAUDE.md payload path to the guarded project-root file."""
    project_root = _resolve_project_root()
    resolved = os.path.realpath(file_path)
    target = os.path.join(project_root, "CLAUDE.md")
    # macOS only: canonicalize the spelling by case-folding every path.
    # realpath does not normalize case on darwin, and the strict containment
    # guard below would otherwise silently allow a case-variant spelling of
    # the project CLAUDE.md (the original hollow-guard bug). Lowercased
    # paths address the same files on case-insensitive APFS volumes.
    if sys.platform == "darwin":
        project_root = project_root.lower()
        resolved = resolved.lower()
        target = target.lower()
    # Containment guards in the canonical normalize-then-check shape
    # (realpath above, prefix check here, on the exact values used by the
    # caller's open() — CodeQL: env- and stdin-derived paths are validated
    # before any file operation). A nested CLAUDE.md passes containment but
    # is not the guarded file; samefile rejects it, and resolving first also
    # strips .. segments naming a nonexistent intermediate directory, which
    # would otherwise make stat fail and fail the guard open.
    if not resolved.startswith(project_root + os.sep):
        return None  # tool path points outside the project entirely
    if not target.startswith(project_root + os.sep):
        return None  # derived target must pass the same containment guard
    try:
        # Stat-based identity (not string equality) decides scope: only the
        # project ROOT's CLAUDE.md is guarded.
        if not os.path.samefile(resolved, target):
            return None
    except OSError:
        return None
    return resolved


def _memory_md_target(file_path):
    """Resolve a MEMORY.md payload path to the guarded memory index."""
    memory_root = _resolve_memory_root()
    resolved = os.path.realpath(file_path)
    if sys.platform == "darwin":
        memory_root = memory_root.lower()
        resolved = resolved.lower()
    # Same normalize-then-check shape as the CLAUDE.md branch. No samefile
    # here: the guarded file is "whichever project's memory index this is",
    # so containment under the memory tree IS the identity.
    if not resolved.startswith(memory_root + os.sep):
        return None
    if os.sep + "memory" + os.sep not in resolved:
        return None  # inside the projects tree but not a memory index
    return resolved


def _guard_target(file_path):
    """Return (resolved_path, limit, label, advice) for a guarded file.

    None means the payload names a file this hook does not own.
    """
    basename = os.path.basename(file_path)
    if basename not in ("CLAUDE.md", "MEMORY.md"):
        return None
    if not os.path.isabs(file_path):
        return None  # relative paths cannot be anchored to a containment root

    if basename == "CLAUDE.md":
        resolved = _claude_md_target(file_path)
        if resolved is None:
            return None
        return (
            resolved,
            CLAUDE_MD_MAX_LINES,
            "CLAUDE.md",
            "Prefer folding detail into existing table rows over adding new"
            " paragraphs; move deep-dive content to"
            " docs/00-Foundations/ARCHITECTURE.md.",
        )

    resolved = _memory_md_target(file_path)
    if resolved is None:
        return None
    return (
        resolved,
        MEMORY_MD_MAX_LINES,
        "MEMORY.md",
        "Archive the oldest session pointers to memory/sessions-archive.md,"
        " then retire resolved topic entries to memory/archive/ (CLAUDE.md"
        " 'Memory index guard').",
    )


def main():
    try:
        payload = json.load(sys.stdin)
        tool_name = payload.get("tool_name", "")
        tool_input = payload.get("tool_input", {})
        file_path = tool_input.get("file_path", "")
    except Exception:
        return 0

    guarded = _guard_target(file_path)
    if guarded is None:
        return 0
    resolved, max_lines, label, advice = guarded
    try:
        with open(resolved) as f:
            current = f.read()
    except OSError:
        return 0

    future = _future_content(tool_name, tool_input, current)
    if future is None:
        return 0
    count = _line_count(future)
    if count > max_lines:
        # Echo the path as the caller spelled it. `resolved` is case-folded on
        # darwin for the containment comparison, and printing that lowercased
        # form produced a confusing (if still openable) hint.  This value is
        # only ever written to stderr, never used in a file operation.
        print(
            f"BLOCKED by claude-md-length-guard: this {tool_name} would put"
            f" {label} at {count} lines (limit: <={max_lines}, checked via"
            f" readlines() — wc -l under-reports by one). {advice} Verify"
            " with: python3 -c"
            f" \"print(len(open('{file_path}').readlines()))\"",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
