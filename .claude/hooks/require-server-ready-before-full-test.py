#!/usr/bin/env python3
"""PreToolUse hook: confirm the server is up before full-suite test runs.

Implements the hookify rule require-server-ready-before-full-test (action:
warn) as a permissionDecision "ask" — the reminder is shown and the call
needs a human ok, but it is not hard-blocked. The rule's .local.md
definition was removed when this script became the executor. Full-suite
runs (raw pytest over the tests tree, a bare path-less pytest, or
tests/run_full_suite.py) need the server RUNNING with all 3 models loaded
(clone, design, custom), or E2E/health tests report false reds.

tests/run_batches.py is the sanctioned path and stays silent: it owns E2E
gating and the rate-limit env itself. Explicit single-file targets (any
test_*.py filename, including ::node-id spellings and files inside
tests/ subdirectories) and the make test-* targets stay silent; a MENTION
of tests/run_full_suite.py (cat/grep/wc) is a read, not a run, and stays
silent. Tree spellings ./tests, ../tests/, and <root>/tests normalize onto
the bare tree. `make test` itself reaches neither hook (the hook sees only
"make test") — accepted scope edge. Known warn-level edges, accepted: a
bare `pytest` token asks wherever it appears (pip install pytest —
dismissible); a single-quoted test-file path scrubs away and reads as
path-less; a double-quoted "$(pytest tests/)" substitution EXECUTES but
scrubs away as prose (fail-open direction).

Detection mirrors prefer-batch-runner-over-raw-pytest.py (pytest / `python
-m unittest discover` invocation, positional-token scan for a tests-tree
target, bare-invocation counts as suite-level) — duplicated deliberately:
standalone-per-hook is the convention all three sibling hooks share (a
shared module would import fine — sys.path[0] is the hooks dir — but adds
a shared durability surface for one ~30-line helper). Quoted spans are
scrubbed BEFORE segment splitting (no-direct-push-main precedent); an
unbalanced quote fails toward asking. Warn level: false positives cost a
dismissible prompt.

Protocol: reads the PreToolUse payload on stdin; prints ask-decision JSON
on stdout for matching commands; always exits 0 (a warn never kills the
call on its own). Unparseable input exits 0 silently.
"""
import json
import re
import sys

_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\||\n")
_SUBTREE_BOUNDARY_RE = re.compile(r"(?:^|/)tests/")
_FULL_SUITE_SUFFIX = "run_full_suite.py"

_REASON = (
    "[require-server-ready-before-full-test] Full-suite runs need the server"
    " UP with all 3 models loaded (clone, design, custom), or E2E/health"
    " tests report false reds. Confirm before running: (1) tts server stop"
    " && tts server start; (2) load all 3 models; (3) tts server status."
    " Cancel this run if server state is not confirmed."
)


def _is_python(token):
    return token == "python" or token.startswith("python3")


def _normalize_target(token):
    """Strip ./ and ../ prefixes and a trailing slash so ./tests/, ../tests/,
    and <root>/tests spellings normalize onto the bare tree."""
    t = token
    while t.startswith("./"):
        t = t[2:]
    while t.startswith("../"):
        t = t[3:]
    return t.rstrip("/")


def _names_suite_target(token):
    """True when the token names the tests TREE or a subtree of it.

    An explicit test file — any path whose final component (before an
    optional ::node-id) is test_*.py, at tests/ top level or inside a
    subdirectory — is the sanctioned carve-out and returns False. A
    t-prefixed SUBDIRECTORY (tests/testdata/) is still a tree target: only
    test_*.py filenames are exempt."""
    t = _normalize_target(token)
    if t == "tests" or t.endswith("/tests"):
        return True
    m = _SUBTREE_BOUNDARY_RE.search(t)
    if not m:
        return False
    rest = t[m.end():]
    if not rest:
        return True
    base = rest.split("::", 1)[0].rsplit("/", 1)[-1]
    is_test_file = base.startswith("test_") and base.endswith(".py")
    return not is_test_file


def _strip_quoted(command):
    """Drop quoted spans so a prose mention of the rule string is not read
    as an invocation. Runs BEFORE segment splitting: quoted spans may
    themselves contain the separators the splitter keys on. An unbalanced
    quote returns the command unmodified — fail toward asking."""
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


def _suite_level_reason(tokens):
    """Reason string when these tokens are a full-suite run."""
    for i, token in enumerate(tokens):
        # Invocation shape (the verbatim spec: "python tests/run_full_suite.py"):
        # a MENTION of the file — cat/grep/wc — is a read, not a run.
        if token.endswith(_FULL_SUITE_SUFFIX) and (i == 0 or _is_python(tokens[i - 1])):
            return "full-suite run via tests/run_full_suite.py"
    for i, tok in enumerate(tokens):
        if tok == "pytest":
            rest = tokens[i + 1 :]
        elif _is_python(tok) and tokens[i + 1 : i + 3] == ["-m", "pytest"]:
            rest = tokens[i + 3 :]
        elif _is_python(tok) and tokens[i + 1 : i + 3] == ["-m", "unittest"]:
            rest = tokens[i + 3 :]
            if rest and rest[0] == "discover":
                rest = rest[1:]
        else:
            continue
        positional = 0
        for token in rest:
            if token.startswith("-"):
                continue  # flags are skipped, not their values: -s tests counts
            positional += 1
            if _names_suite_target(token):
                return "full-suite run without a confirmed live server"
        if positional == 0:
            return "full-suite run (no path collects the whole suite)"
    return None


def _reason(command):
    scrubbed = _strip_quoted(command)
    for segment in _SEGMENT_SPLIT_RE.split(scrubbed):
        reason = _suite_level_reason(segment.split())
        if reason:
            return reason
    return None


def main():
    try:
        payload = json.load(sys.stdin)
        command = payload.get("tool_input", {}).get("command", "")
    except Exception:
        return 0
    if _reason(command):
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
