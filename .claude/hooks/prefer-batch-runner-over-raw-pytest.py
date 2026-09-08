#!/usr/bin/env python3
"""PreToolUse hook: steer raw suite-level pytest runs to the batch runner.

Implements the hookify rule prefer-batch-runner-over-raw-pytest (action:
warn) as a permissionDecision "ask" — the reminder is shown and the call
needs a human ok, but it is not hard-blocked. The rule's .local.md
definition was removed when this script became the executor; its verbatim
pattern never compiled as a regex, so the rule MESSAGE is the contract:
raw suite-level pytest in one process risks the known hang/OOM (exit 137)
failures here, and tests/run_batches.py exists to isolate batches.

A run counts as suite-level when a pytest / `python -m unittest discover`
invocation is present AND either (a) some positional token after it names
the tests TREE — bare `tests` with or without a slash, or a `tests/`
subtree that is not an explicit tests/test_*.py-style path (the spec's
tests/[^t] carve-out) — or (b) there is NO positional path at all (bare
pytest collects the whole rootdir). Explicit single-file targets,
including `::`-node-id spellings and files inside tests/ subdirectories,
plus tests/run_batches.py and the make test-* targets, are sanctioned
shapes and stay silent. Tree spellings ./tests, ../tests/, and
<root>/tests normalize onto the bare tree. `make test` itself reaches
neither hook (the hook sees only "make test") — accepted scope edge.

Known warn-level edges, accepted: a bare `pytest` token asks wherever it
appears (pip install pytest, pytest --version — dismissible); a
single-quoted test-file path scrubs away and reads as path-less (asks;
placeholder-substitution would trade this for a quoted-tree false
negative); a double-quoted "$(pytest tests/)" command substitution
EXECUTES but scrubs away as prose (inherited from the sibling — fail-open
direction).

Quoted spans are scrubbed BEFORE segment splitting (no-direct-push-main
precedent): a mention of the rule string inside a quoted grep pattern is
data, not an invocation — nagging on prose trains rubber-stamping. An
unbalanced quote leaves the command unscrubbed — fail toward asking.

Warn level, so like prepush-local-gates: false positives cost a
dismissible prompt, never a lost command.

Protocol: reads the PreToolUse payload on stdin; prints ask-decision JSON
on stdout for matching commands; always exits 0 (a warn never kills the
call on its own). Unparseable input exits 0 silently.
"""
import json
import re
import sys

_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\||\n")
_SUBTREE_BOUNDARY_RE = re.compile(r"(?:^|/)tests/")

_REASON = (
    "[prefer-batch-runner-over-raw-pytest] Raw suite-level pytest runs risk"
    " hangs and OOM kills (exit 137) here — 3,200+ tests in one process is a"
    " known NO-GO. Use the batch runner instead: python tests/run_batches.py"
    " (all batches), python tests/run_batches.py --batch N (one batch), or"
    " make test-batch. Explicit single files stay fine:"
    " python -m pytest tests/test_specific.py"
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
    """Reason string when these tokens are a raw suite-level run."""
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
                return "raw suite-level pytest run"
        if positional == 0:
            return "raw suite-level pytest run (no path collects the whole suite)"
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
