#!/usr/bin/env python3
"""Shared shell-text handling for the Bash PreToolUse hooks.

Two hooks (no-direct-push-main, workspace-hygiene) decide policy by matching
regexes against raw command text. Both need the same two primitives, and both
need them in the same order, so they live here once rather than being forked
per hook — the stream_protocol lesson (CLAUDE.md): a duplicated protocol
drifts, and the drift is invisible until it matters.

Import works without any sys.path handling: the harness invokes each hook by
path, so Python puts .claude/hooks/ at sys.path[0] for the running script.
"""
import re

SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\||\n")


def strip_quoted(command):
    """Drop quoted spans so a prose mention of a command is not read as one.

    Must run BEFORE segment splitting: a quoted span may itself contain the
    &&/||/;/| separators the splitter keys on, and inside quotes those
    characters are data. An unbalanced quote returns the command unmodified
    — an unmatched quote may hide a real invocation past the truncation
    point, so fail toward blocking.
    """
    kept = []
    quote = None
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if quote is not None:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            kept.append(command[i : i + 2])
            i += 2
            continue
        kept.append(ch)
        i += 1
    if quote is not None:
        return command
    return "".join(kept)


def segments(command):
    """Yield each command segment of a compound command, quotes scrubbed.

    Segments are judged separately so `git push origin fix/x && git checkout
    main` is not a false positive on the second clause.
    """
    for segment in SEGMENT_SPLIT_RE.split(strip_quoted(command)):
        yield segment.strip()
