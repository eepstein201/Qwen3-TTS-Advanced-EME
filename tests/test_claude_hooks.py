"""Guard tests for the repo-durable Claude Code PreToolUse hooks.

The three hook scripts under .claude/hooks/ enforce repo policy at the
harness level (no push/merge/delete on main, pre-push gate reminder,
CLAUDE.md <=300 lines), but they only protect contributors if they actually
ship with the repo:

- the scripts are git-tracked (untracked scripts die with the clone),
- the wiring lives in a TRACKED .claude/settings.json — the file is
  gitignored by default, so wiring there silently stays machine-local,
- settings.local.json does not re-wire the same scripts: hooks from all
  settings levels accumulate, so a duplicate double-fires every call.

The behavioral pipe tests pin the block/allow contract each script promises
in its docstring (exit 2 blocks; exit 0 allows; malformed stdin fails open).
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"
SHARED_SETTINGS = REPO_ROOT / ".claude" / "settings.json"
LOCAL_SETTINGS = REPO_ROOT / ".claude" / "settings.local.json"
GITIGNORE = REPO_ROOT / ".gitignore"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
TEST_DOCKERFILE = REPO_ROOT / "Dockerfile.test"

PUSH_HOOK = "no-direct-push-main.py"
PREPUSH_HOOK = "prepush-local-gates.py"
LENGTH_HOOK = "claude-md-length-guard.py"
ALL_HOOKS = (PUSH_HOOK, PREPUSH_HOOK, LENGTH_HOOK)

HOOK_TIMEOUT_S = 15
CLAUDE_MD_MAX_LINES = 300


def _run_hook(script, payload=None, raw=None):
    """Run a hook script the way the harness does: payload on stdin."""
    stdin = raw if raw is not None else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(HOOKS_DIR / script)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=HOOK_TIMEOUT_S,
        cwd=REPO_ROOT,
    )


def _pretooluse_commands(settings_path):
    """Extract every PreToolUse hook command from a settings file."""
    data = json.loads(settings_path.read_text())
    blocks = data.get("hooks", {}).get("PreToolUse", [])
    return [
        (block.get("matcher", ""), hook.get("command", ""))
        for block in blocks
        for hook in block.get("hooks", [])
    ]


class TestHooksAreRepoDurable(unittest.TestCase):
    """The hooks must ship with the repo, not just this machine."""

    def test_hook_scripts_are_git_tracked(self):
        if not (REPO_ROOT / ".git").exists():
            self.skipTest("no git checkout (docker test image ships no .git)")
        proc = subprocess.run(
            ["git", "ls-files", "--", ".claude/hooks"],
            capture_output=True,
            text=True,
            timeout=HOOK_TIMEOUT_S,
            cwd=REPO_ROOT,
        )
        tracked = {Path(line).name for line in proc.stdout.split() if line}
        for script in ALL_HOOKS:
            self.assertIn(script, tracked, f"{script} is not git-tracked")

    def test_shared_settings_exist_and_wire_all_hooks(self):
        self.assertTrue(
            SHARED_SETTINGS.exists(),
            ".claude/settings.json is missing — hooks are not repo-durable",
        )
        wired = _pretooluse_commands(SHARED_SETTINGS)
        self.assertTrue(wired, "shared settings wire no PreToolUse hooks")
        for script in ALL_HOOKS:
            commands = [cmd for _, cmd in wired if script in cmd]
            self.assertEqual(
                len(commands),
                1,
                f"{script} must be wired exactly once in shared settings",
            )
            self.assertIn(
                "$CLAUDE_PROJECT_DIR",
                commands[0],
                f"{script} wiring must be project-relative, not absolute",
            )

    def test_length_guard_wired_for_write_and_edit(self):
        matchers = [
            matcher
            for matcher, cmd in _pretooluse_commands(SHARED_SETTINGS)
            if LENGTH_HOOK in cmd
        ]
        self.assertEqual(len(matchers), 1)
        self.assertIn("Write", matchers[0])
        self.assertIn("Edit", matchers[0])

    def test_shared_settings_are_hooks_only(self):
        """Secret-bearing sections (env/permissions/mcpServers) stay local.

        .gitignore ignored this file because settings "contain secrets";
        the shared file is un-ignored on the condition that it carries
        hooks and nothing else.
        """
        data = json.loads(SHARED_SETTINGS.read_text())
        self.assertEqual(
            set(data),
            {"hooks"},
            "shared .claude/settings.json must be hooks-only — secrets and"
            " permissions belong in .claude/settings.local.json (gitignored)",
        )

    def test_shared_settings_are_not_gitignored(self):
        if not (REPO_ROOT / ".git").exists():
            self.skipTest("no git checkout (docker test image ships no .git)")
        proc = subprocess.run(
            ["git", "check-ignore", "-q", str(SHARED_SETTINGS)],
            capture_output=True,
            timeout=HOOK_TIMEOUT_S,
            cwd=REPO_ROOT,
        )
        self.assertEqual(
            proc.returncode,
            1,
            ".claude/settings.json is gitignored — its wiring never reaches clones",
        )

    def test_local_settings_do_not_rewire_same_hooks(self):
        if not LOCAL_SETTINGS.exists():
            self.skipTest("no settings.local.json (CI checkout)")
        for _, cmd in _pretooluse_commands(LOCAL_SETTINGS):
            for script in ALL_HOOKS:
                self.assertNotIn(
                    script,
                    cmd,
                    f"{script} is wired in BOTH shared and local settings"
                    " — every call would double-fire",
                )


class TestHooksShipInTestImage(unittest.TestCase):
    """The docker test lane must be able to run these guard tests.

    Dockerfile.test copies explicit paths (there is no blanket ``COPY . .``)
    and .dockerignore excludes all of ``.claude/`` from the build context —
    so the tracked hook scripts and the hooks-only shared settings only
    reach /app when BOTH halves exist: a dockerignore re-include
    (last-match-wins, so it must sit after the .claude exclusion) and an
    explicit COPY line (the .github/workflows precedent). Without them every
    hook test failed or errored in the container on missing files while
    passing on native runners.
    """

    def test_dockerignore_reincludes_hooks_and_shared_settings(self):
        lines = [
            stripped
            for stripped in (ln.strip() for ln in DOCKERIGNORE.read_text().splitlines())
            if stripped and not stripped.startswith("#")
        ]
        claude_excludes = [
            i for i, pattern in enumerate(lines) if pattern in (".claude", ".claude/")
        ]
        self.assertEqual(len(claude_excludes), 1, "expected exactly one .claude exclusion")
        for needed in ("!.claude/hooks", "!.claude/settings.json"):
            self.assertIn(
                needed,
                lines,
                f"{needed} missing — hook files never enter the docker build context",
            )
            self.assertGreater(
                lines.index(needed),
                claude_excludes[0],
                f"{needed} must follow the .claude exclusion"
                " (dockerignore is last-match-wins)",
            )

    def test_dockerfile_test_copies_hooks_and_shared_settings(self):
        copy_lines = [
            ln for ln in TEST_DOCKERFILE.read_text().splitlines() if ln.startswith("COPY")
        ]
        for needed in (".claude/hooks/", ".claude/settings.json"):
            self.assertTrue(
                any(needed in ln for ln in copy_lines),
                f"Dockerfile.test must COPY {needed} — explicit-copy image,"
                " dockerignore alone keeps it out of /app",
            )

    def test_git_checks_skip_without_a_checkout(self):
        """The two git-metadata durability checks must skip, not error.

        The docker test image ships no .git (dockerignored, deliberately —
        like settings.local.json on CI checkouts). Those assertions still
        fully run on the native ubuntu/macos lanes; the docker lane must
        report them skipped.
        """
        module = sys.modules[__name__]
        git_checks = (
            "test_hook_scripts_are_git_tracked",
            "test_shared_settings_are_not_gitignored",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(module, "REPO_ROOT", Path(tmp)):
                for name in git_checks:
                    case = module.TestHooksAreRepoDurable(name)
                    with self.assertRaises(unittest.SkipTest):
                        getattr(case, name)()


class TestNoDirectPushMain(unittest.TestCase):
    def test_blocks_push_that_names_main(self):
        proc = _run_hook(
            PUSH_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "git push origin main"}},
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("push names main", proc.stderr)

    def test_blocks_absolute_spelling_of_git(self):
        proc = _run_hook(
            PUSH_HOOK,
            payload={
                "tool_name": "Bash",
                "tool_input": {"command": "/opt/homebrew/bin/git push origin main"},
            },
        )
        self.assertEqual(proc.returncode, 2)

    def test_blocks_push_that_deletes_remote_branch(self):
        proc = _run_hook(
            PUSH_HOOK,
            payload={
                "tool_name": "Bash",
                "tool_input": {"command": "git push origin --delete feature/x"},
            },
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("deletes a remote branch", proc.stderr)

    def test_blocks_merge_into_main(self):
        proc = _run_hook(
            PUSH_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "git merge main"}},
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("merge names main", proc.stderr)

    def test_allows_feature_branch_push(self):
        proc = _run_hook(
            PUSH_HOOK,
            payload={
                "tool_name": "Bash",
                "tool_input": {"command": "git push origin feature/some-work"},
            },
        )
        self.assertEqual(proc.returncode, 0)

    def test_allows_compound_command_touching_main_after_push(self):
        proc = _run_hook(
            PUSH_HOOK,
            payload={
                "tool_name": "Bash",
                "tool_input": {"command": "git push origin fix/x && git checkout main"},
            },
        )
        self.assertEqual(proc.returncode, 0)

    def test_allows_non_git_command(self):
        proc = _run_hook(
            PUSH_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "echo hi"}},
        )
        self.assertEqual(proc.returncode, 0)

    def test_fails_open_on_malformed_stdin(self):
        proc = _run_hook(PUSH_HOOK, raw="{not json")
        self.assertEqual(proc.returncode, 0)


class TestPrepushLocalGates(unittest.TestCase):
    def test_asks_on_push(self):
        proc = _run_hook(
            PREPUSH_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "git push origin fix/x"}},
        )
        self.assertEqual(proc.returncode, 0)
        decision = json.loads(proc.stdout)["hookSpecificOutput"]
        self.assertEqual(decision["permissionDecision"], "ask")
        self.assertIn("ruff", decision["permissionDecisionReason"])

    def test_silent_on_non_push(self):
        proc = _run_hook(
            PREPUSH_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "pytest tests/"}},
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_fails_open_on_malformed_stdin(self):
        proc = _run_hook(PREPUSH_HOOK, raw="{not json")
        self.assertEqual(proc.returncode, 0)


class TestClaudeMdLengthGuard(unittest.TestCase):
    @staticmethod
    def _write_payload(line_count):
        return {
            "tool_name": "Write",
            "tool_input": {
                "file_path": str(REPO_ROOT / "CLAUDE.md"),
                "content": "line\n" * line_count,
            },
        }

    def test_blocks_write_over_limit(self):
        proc = _run_hook(LENGTH_HOOK, payload=self._write_payload(CLAUDE_MD_MAX_LINES + 1))
        self.assertEqual(proc.returncode, 2)
        self.assertIn(str(CLAUDE_MD_MAX_LINES + 1), proc.stderr)

    def test_allows_write_at_limit(self):
        proc = _run_hook(LENGTH_HOOK, payload=self._write_payload(CLAUDE_MD_MAX_LINES))
        self.assertEqual(proc.returncode, 0)

    def test_blocks_edit_that_grows_past_limit(self):
        current = (REPO_ROOT / "CLAUDE.md").read_text()
        first_line = current.split("\n", 1)[0]
        proc = _run_hook(
            LENGTH_HOOK,
            payload={
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": str(REPO_ROOT / "CLAUDE.md"),
                    "old_string": first_line,
                    "new_string": "line\n" * (CLAUDE_MD_MAX_LINES + 5),
                },
            },
        )
        self.assertEqual(proc.returncode, 2)

    def test_allows_line_neutral_edit(self):
        current = (REPO_ROOT / "CLAUDE.md").read_text()
        first_line = current.split("\n", 1)[0]
        proc = _run_hook(
            LENGTH_HOOK,
            payload={
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": str(REPO_ROOT / "CLAUDE.md"),
                    "old_string": first_line,
                    "new_string": first_line,
                },
            },
        )
        self.assertEqual(proc.returncode, 0)

    def test_ignores_claude_md_outside_project_root(self):
        proc = _run_hook(
            LENGTH_HOOK,
            payload={
                "tool_name": "Write",
                "tool_input": {
                    "file_path": "/tmp/not-the-project/CLAUDE.md",
                    "content": "line\n" * (CLAUDE_MD_MAX_LINES + 50),
                },
            },
        )
        self.assertEqual(proc.returncode, 0)

    def test_fails_open_on_malformed_stdin(self):
        proc = _run_hook(LENGTH_HOOK, raw="{not json")
        self.assertEqual(proc.returncode, 0)


class TestLengthGuardPathValidation(unittest.TestCase):
    """Path-validation contract for the length guard (CodeQL hardening).

    The guard acts on a tool-provided path (stdin payload) and an
    environment-provided project root — both untrusted sources in CodeQL's
    model — so every path is validated in main() (resolved, absolute,
    contained in the project root, inline at the use site) before any file
    operation. These tests pin that validation both rejects what it must
    and never hollows the guard for in-project spellings.
    """

    @staticmethod
    def _guard_module():
        spec = importlib.util.spec_from_file_location(
            "claude_md_length_guard_under_test", HOOKS_DIR / LENGTH_HOOK
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_resolve_project_root_canonicalizes_env_value(self):
        guard = self._guard_module()
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "nested"
            nested.mkdir()
            with mock.patch.dict(os.environ, {"CLAUDE_PROJECT_DIR": str(nested / "..")}):
                self.assertEqual(
                    guard._resolve_project_root(), str(nested.resolve().parent)
                )

    def test_resolve_project_root_falls_back_to_cwd(self):
        guard = self._guard_module()
        with mock.patch.dict(os.environ):
            os.environ.pop("CLAUDE_PROJECT_DIR", None)
            self.assertEqual(guard._resolve_project_root(), os.getcwd())

    def test_relative_tool_path_is_out_of_scope(self):
        """A relative path cannot be anchored to the project root, so it is
        out of scope (exit 0), like every other non-matching payload."""
        proc = _run_hook(
            LENGTH_HOOK,
            payload={
                "tool_name": "Write",
                "tool_input": {
                    "file_path": "CLAUDE.md",
                    "content": "line\n" * (CLAUDE_MD_MAX_LINES + 1),
                },
            },
        )
        self.assertEqual(proc.returncode, 0)

    def test_in_root_traversal_spelling_still_guards(self):
        """Validation must resolve, not reject, in-project traversal
        spellings — the guard still fires through subdir/.. paths."""
        sneaky = REPO_ROOT / "subdir" / ".." / "CLAUDE.md"
        proc = _run_hook(
            LENGTH_HOOK,
            payload={
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(sneaky),
                    "content": "line\n" * (CLAUDE_MD_MAX_LINES + 1),
                },
            },
        )
        self.assertEqual(proc.returncode, 2)

    @unittest.skipUnless(sys.platform == "darwin", "case-insensitive FS only")
    def test_darwin_case_variant_spelling_still_guards(self):
        """Regression pin for the original macOS hollow-guard bug: a path
        spelled with different case than the canonical project root is the
        same file on darwin and must still be guarded (stat-based samefile
        + case-folded containment, never raw string equality)."""
        variant = str(REPO_ROOT).replace("Qwen3", "qwen3", 1) + "/CLAUDE.md"
        if variant == str(REPO_ROOT / "CLAUDE.md"):
            self.skipTest("no case-folding opportunity in this repo path")
        proc = _run_hook(
            LENGTH_HOOK,
            payload={
                "tool_name": "Write",
                "tool_input": {
                    "file_path": variant,
                    "content": "line\n" * (CLAUDE_MD_MAX_LINES + 1),
                },
            },
        )
        self.assertEqual(proc.returncode, 2)

    def test_nested_claude_md_is_out_of_scope(self):
        """Containment alone is not scope: only the project ROOT's CLAUDE.md
        is guarded, so a nested CLAUDE.md passes validation yet exits 0."""
        proc = _run_hook(
            LENGTH_HOOK,
            payload={
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(REPO_ROOT / "docs" / "CLAUDE.md"),
                    "content": "line\n" * (CLAUDE_MD_MAX_LINES + 50),
                },
            },
        )
        self.assertEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
