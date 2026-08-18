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

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"
SHARED_SETTINGS = REPO_ROOT / ".claude" / "settings.json"
LOCAL_SETTINGS = REPO_ROOT / ".claude" / "settings.local.json"
GITIGNORE = REPO_ROOT / ".gitignore"

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


if __name__ == "__main__":
    unittest.main()
