"""Behavioral contracts for the two hookify rules wired as PreToolUse hooks.

The durability/wiring guards live in tests/test_claude_hooks.py (ALL_HOOKS
covers the scripts' tracked-ness, single wiring, Bash matcher scope, and
the removal of the dormant .local.md definitions). This module pins what
each script DOES: hookify ``action: warn`` maps to a PreToolUse
permissionDecision "ask" — valid decision JSON on stdout, exit 0 always —
never a block (exit 2) and never a silent plain-text print.

Spec provenance: rule 1's verbatim hookify pattern does not compile as a
regex (``missing )``), so the operative contract for both hooks is the rule
MESSAGE, implemented as:

- ask when a pytest / ``unittest discover`` invocation runs SUITE-LEVEL:
  the tests tree (bare ``tests`` with or without slash, or any ``tests/``
  subtree that is not an explicit tests/test_*.py-style file), or with NO
  positional path at all (bare pytest collects the whole rootdir);
- stay silent on explicit test-file paths (including ``::``-node-id
  spellings), tests/run_batches.py, and the make test-* targets.

Deliberate extensions beyond the verbatim rule text (pin, do not "fix"):
``python3`` spelling, ``tests`` without trailing slash, flag-first shapes
(``-v tests/``, ``discover -v tests/`` — the latter is verbatim the Makefile
``make test`` body), multi-file explicit allows, and node-id single-file
allows. Tree spellings ``./tests`` / ``../tests/`` / ``<root>/tests``
normalize onto the bare tree; the single-file carve-out keys on the FILENAME
(test_*.py, any tests/ subdirectory), so a t-prefixed subdirectory like
``tests/testdata/`` still asks where the verbatim ``tests/[^t]`` rule would
have been silent. ``run_full_suite.py`` requires invocation shape (a
cat/grep mention is a read). Known tradeoffs pinned as tests: a piped
single-file run stays silent (``\\|`` is a segment split), while a
single-quoted test-file path scrubs away and asks as path-less. Known
scope edge, accepted: ``make test`` reaches neither hook (the hook sees
only "make test").

Quoted spans are scrubbed before segment splitting (no-direct-push-main
precedent): a mention of the rule string inside a quoted grep pattern is
data, not an invocation — nagging on prose trains rubber-stamping. An
unbalanced quote fails toward asking.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"

BATCH_RUNNER_HOOK = "prefer-batch-runner-over-raw-pytest.py"
SERVER_READY_HOOK = "require-server-ready-before-full-test.py"

HOOK_TIMEOUT_S = 15


def _run_hook(script, payload=None, raw=None):
    """Run a hook script the way the harness does: payload on stdin.

    Duplicated from tests/test_claude_hooks.py to keep the durability and
    behavior modules independently runnable (the hooks themselves are
    deliberately standalone single files too).
    """
    stdin = raw if raw is not None else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(HOOKS_DIR / script)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=HOOK_TIMEOUT_S,
        cwd=REPO_ROOT,
    )


def _ask_decision(proc):
    """Strict ask contract: parse stdout AS a decision, not a substring.

    A hook that prints the rule message as plain text is inert (PreToolUse
    only acts on decision JSON), so every ask test must parse the JSON and
    check permissionDecision — ``"ask" in stdout`` accepts that hollow
    output.
    """
    decision = json.loads(proc.stdout)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "ask", proc.stdout
    assert decision["hookEventName"] == "PreToolUse", proc.stdout
    return decision


class TestPreferBatchRunnerOverRawPytest(unittest.TestCase):
    def test_asks_on_bare_tests_tree(self):
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m pytest tests/ -x"}},
        )
        self.assertEqual(proc.returncode, 0)
        decision = _ask_decision(proc)
        self.assertIn("run_batches.py", decision["permissionDecisionReason"])

    def test_asks_on_tests_dir_without_trailing_slash(self):
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m pytest tests -q"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_asks_on_tests_subtree(self):
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python3 -m pytest tests/security -q"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_asks_with_leading_flags_before_path(self):
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m pytest -v tests/"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_asks_on_unittest_discover(self):
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m unittest discover tests/"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_asks_on_unittest_discover_with_leading_flags(self):
        """``python -m unittest discover -v tests/`` is verbatim the Makefile
        `make test` body — the spec carried a whole alternation branch for
        this spelling."""
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m unittest discover -v tests/"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_asks_on_bare_pytest_without_path(self):
        """Path-less pytest collects the whole suite from rootdir — the most
        common raw full-suite invocation; a path-required matcher would
        silently allow it."""
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m pytest -q"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_asks_inside_compound_command(self):
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "cd /tmp && python -m pytest tests/"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_unbalanced_quote_fails_toward_asking(self):
        """An unmatched quote may hide a real invocation past the truncation
        point, so the scrubber gives up and matches the raw text."""
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "echo ' && python -m pytest tests/"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_allows_single_explicit_test_file(self):
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m pytest tests/test_engine_streaming.py -x"}},
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_allows_node_id_spelling_of_single_file(self):
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={
                "tool_name": "Bash",
                "tool_input": {
                    "command": (
                        "python -m pytest"
                        " tests/test_engine_streaming.py::TestStreamingCapWarning::test_delta_accumulation"
                    )
                },
            },
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_allows_multiple_explicit_test_files(self):
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={
                "tool_name": "Bash",
                "tool_input": {
                    "command": "python -m pytest tests/test_engine_streaming.py tests/test_claude_md.py"
                },
            },
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_allows_batch_runner(self):
        for command in ("python tests/run_batches.py", "python tests/run_batches.py --batch 3"):
            proc = _run_hook(
                BATCH_RUNNER_HOOK,
                payload={"tool_name": "Bash", "tool_input": {"command": command}},
            )
            self.assertEqual(proc.returncode, 0, command)
            self.assertEqual(proc.stdout, "", command)

    def test_allows_make_targets(self):
        for command in ("make test-batch", "make test-core"):
            proc = _run_hook(
                BATCH_RUNNER_HOOK,
                payload={"tool_name": "Bash", "tool_input": {"command": command}},
            )
            self.assertEqual(proc.returncode, 0, command)
            self.assertEqual(proc.stdout, "", command)

    def test_quoted_grep_mention_of_rule_string_is_allowed(self):
        """Grepping the docs for the rule string is how the rule gets
        looked up — the mention is data, not an invocation."""
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "grep -rn 'python -m pytest tests/' docs/"}},
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_allows_unrelated_command(self):
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "git status"}},
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_fails_open_on_malformed_stdin(self):
        proc = _run_hook(BATCH_RUNNER_HOOK, raw="{not json")
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_allows_piped_single_file(self):
        """`|` is a segment split: the grep RHS must not leak a tests-tree
        target into a sanctioned single-file run's segment."""
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={
                "tool_name": "Bash",
                "tool_input": {
                    "command": "python -m pytest tests/test_engine_streaming.py | grep tests/"
                },
            },
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_allows_subdir_single_test_file(self):
        """10 real test files live under tests/{security,colab,evaluations} —
        every one of them is a sanctioned explicit-file run."""
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m pytest tests/security/test_path_injection.py"}},
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_asks_on_dotted_and_absolute_tree_spellings(self):
        for command in (
            "python -m pytest ./tests/",
            "python -m pytest ../tests/",
            "python -m pytest /repo/tests/",
        ):
            proc = _run_hook(
                BATCH_RUNNER_HOOK,
                payload={"tool_name": "Bash", "tool_input": {"command": command}},
            )
            self.assertEqual(proc.returncode, 0, command)
            _ask_decision(proc)

    def test_asks_on_t_prefixed_subdirectory(self):
        """Only test_*.py FILENAMES are exempt — a t-prefixed subdirectory is
        still a tree target (the verbatim tests/[^t] rule was silent here)."""
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m pytest tests/testdata/"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_asks_on_env_var_prefix(self):
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "TTS_LOG_LEVEL=DEBUG python -m pytest tests/"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_asks_on_quoted_single_file_path_known_edge(self):
        """Pinned tradeoff: the quoted path scrubs away and reads as
        path-less. Placeholder-substitution would trade this for a
        quoted-tree false negative — do not 'fix' blindly."""
        proc = _run_hook(
            BATCH_RUNNER_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m pytest 'tests/test_engine_streaming.py'"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)


class TestRequireServerReadyBeforeFullTest(unittest.TestCase):
    def test_asks_on_run_full_suite(self):
        proc = _run_hook(
            SERVER_READY_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python tests/run_full_suite.py --full --env all"}},
        )
        self.assertEqual(proc.returncode, 0)
        decision = _ask_decision(proc)
        self.assertIn("tts server", decision["permissionDecisionReason"])

    def test_asks_on_raw_pytest_tree(self):
        proc = _run_hook(
            SERVER_READY_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m pytest tests/"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_asks_on_bare_pytest_without_path(self):
        proc = _run_hook(
            SERVER_READY_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m pytest -q"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_allows_single_explicit_test_file(self):
        proc = _run_hook(
            SERVER_READY_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "python -m pytest tests/test_batches_coverage.py"}},
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_allows_batch_runner(self):
        """run_batches.py is the sanctioned path: it owns E2E gating and the
        rate-limit env itself, so the server rule must not nag on it."""
        for command in ("python tests/run_batches.py", "python tests/run_batches.py --batch 6"):
            proc = _run_hook(
                SERVER_READY_HOOK,
                payload={"tool_name": "Bash", "tool_input": {"command": command}},
            )
            self.assertEqual(proc.returncode, 0, command)
            self.assertEqual(proc.stdout, "", command)

    def test_allows_make_targets(self):
        proc = _run_hook(
            SERVER_READY_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "make test-server"}},
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_quoted_grep_mention_of_rule_string_is_allowed(self):
        proc = _run_hook(
            SERVER_READY_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "grep -rn 'python tests/run_full_suite.py' docs/"}},
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_allows_unrelated_command(self):
        proc = _run_hook(
            SERVER_READY_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "ruff check qwen3_tts tests"}},
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_allows_reading_run_full_suite(self):
        """A mention of run_full_suite.py is a read, not a run — only
        invocation shape (preceded by an interpreter, or segment head)
        asks."""
        for command in (
            "cat tests/run_full_suite.py",
            "grep -c def tests/run_full_suite.py",
        ):
            proc = _run_hook(
                SERVER_READY_HOOK,
                payload={"tool_name": "Bash", "tool_input": {"command": command}},
            )
            self.assertEqual(proc.returncode, 0, command)
            self.assertEqual(proc.stdout, "", command)

    def test_unbalanced_quote_fails_toward_asking(self):
        proc = _run_hook(
            SERVER_READY_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "echo ' && python -m pytest tests/"}},
        )
        self.assertEqual(proc.returncode, 0)
        _ask_decision(proc)

    def test_fails_open_on_malformed_stdin(self):
        proc = _run_hook(SERVER_READY_HOOK, raw="{not json")
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")


if __name__ == "__main__":
    unittest.main()
