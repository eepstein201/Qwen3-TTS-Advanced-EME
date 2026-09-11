"""Guard tests for the repo-durable Claude Code PreToolUse hooks.

The six hook scripts under .claude/hooks/ enforce repo policy at the
harness level (no push/merge/delete on main, pre-push gate reminder,
CLAUDE.md <=300 lines and MEMORY.md <=150, batch-runner preference over raw
suite pytest, server-ready confirmation before full-suite runs, and no broad
`git add` over untracked credential files), but they only protect
contributors if they actually ship with the repo:

- the scripts are git-tracked (untracked scripts die with the clone),
- the wiring lives in a TRACKED .claude/settings.json — the file is
  gitignored by default, so wiring there silently stays machine-local,
- settings.local.json does not re-wire the same scripts: hooks from all
  settings levels accumulate, so a duplicate double-fires every call.

The behavioral pipe tests pin the block/allow contract each script promises
in its docstring (exit 2 blocks; ask-decision JSON with exit 0 warns;
silent exit 0 allows; malformed stdin fails open).
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
BATCH_RUNNER_HOOK = "prefer-batch-runner-over-raw-pytest.py"
SERVER_READY_HOOK = "require-server-ready-before-full-test.py"
HYGIENE_HOOK = "workspace-hygiene.py"
ALL_HOOKS = (
    PUSH_HOOK,
    PREPUSH_HOOK,
    LENGTH_HOOK,
    BATCH_RUNNER_HOOK,
    SERVER_READY_HOOK,
    HYGIENE_HOOK,
)

HOOK_TIMEOUT_S = 15
CLAUDE_MD_MAX_LINES = 300
MEMORY_MD_MAX_LINES = 150


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

    def test_bash_hooks_wired_in_bash_matcher_block(self):
        """event:bash rules must sit in a Bash matcher block.

        Pipe tests invoke hook scripts directly and bypass settings
        entirely, so a script wired under the wrong matcher passes every
        pipe test while never seeing a real Bash call.
        """
        for script in (
            PUSH_HOOK,
            PREPUSH_HOOK,
            BATCH_RUNNER_HOOK,
            SERVER_READY_HOOK,
            HYGIENE_HOOK,
        ):
            matchers = [
                matcher
                for matcher, cmd in _pretooluse_commands(SHARED_SETTINGS)
                if script in cmd
            ]
            self.assertEqual(len(matchers), 1, script)
            self.assertIn("Bash", matchers[0], script)

    def test_dormant_hookify_rule_definitions_removed(self):
        """The wired executors REPLACE the dormant hookify definitions.

        Both .local.md files were git-tracked; leaving them in place lets a
        dormant duplicate silently drift from the hook that owns the rule
        (the same never-re-fork-the-protocol lesson as stream_protocol).
        """
        for name in (
            "hookify.prefer-batch-runner-over-raw-pytest.local.md",
            "hookify.require-server-ready-before-full-test.local.md",
        ):
            self.assertFalse(
                (REPO_ROOT / ".claude" / name).exists(),
                f"{name} still present — the wired hook owns this rule now",
            )

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
        self.assertEqual(proc.stdout, "")


class TestNoDirectPushMainQuotedText(unittest.TestCase):
    """Quoted text must never read as an invocation.

    The hook matches regexes against raw command text, so a mere MENTION of
    "git push" inside a quoted grep pattern, printf body, or commit message
    blocked harmless commands (observed live: a grep whose pattern contained
    the literal rule string, and a printf appending to the gc log). Shell
    semantics are the contract: quoted spans are data, not commands. The
    scrubber must run BEFORE segment splitting too — quoted spans may
    themselves contain the &&/||/;/| separators the splitter keys on.
    """

    def test_quoted_grep_pattern_mentioning_push_is_allowed(self):
        proc = _run_hook(
            PUSH_HOOK,
            payload={
                "tool_name": "Bash",
                "tool_input": {"command": "grep -n 'git push' src/main.py"},
            },
        )
        self.assertEqual(proc.returncode, 0)

    def test_quoted_printf_body_mentioning_rule_string_is_allowed(self):
        proc = _run_hook(
            PUSH_HOOK,
            payload={
                "tool_name": "Bash",
                "tool_input": {
                    "command": (
                        "printf '%s\\n' '- Bash(/opt/homebrew/bin/git push:*) removed'"
                        " >> ~/.claude/gc_log.md"
                    )
                },
            },
        )
        self.assertEqual(proc.returncode, 0)

    def test_commit_message_mentioning_push_and_main_is_allowed(self):
        proc = _run_hook(
            PUSH_HOOK,
            payload={
                "tool_name": "Bash",
                "tool_input": {"command": 'git commit -m "fix: handle git push on main"'},
            },
        )
        self.assertEqual(proc.returncode, 0)

    def test_quoted_prose_then_real_push_still_blocks(self):
        proc = _run_hook(
            PUSH_HOOK,
            payload={
                "tool_name": "Bash",
                "tool_input": {
                    "command": 'echo "git push is gated" && git push origin main'
                },
            },
        )
        self.assertEqual(proc.returncode, 2)

    def test_unbalanced_quote_fails_toward_blocking(self):
        """An unmatched quote may hide a real invocation past the truncation
        point, so the scrubber must give up and match the raw text."""
        proc = _run_hook(
            PUSH_HOOK,
            payload={
                "tool_name": "Bash",
                "tool_input": {"command": "echo ' && git push origin main"},
            },
        )
        self.assertEqual(proc.returncode, 2)


class TestNoDirectPushMainOnMainBranch(unittest.TestCase):
    """The bare-push-on-main path, tested deterministically.

    Blocking a bare `git push` requires the hook to see itself as running on
    main, which a pipe test cannot control (CI checks out the PR branch) —
    so these load the hook module and force `_current_branch` to return
    "main". Each command here is prose in quotes; the bug being pinned is
    that raw matching made the mention look like a bare push.
    """

    @staticmethod
    def _hook_module():
        spec = importlib.util.spec_from_file_location(
            "no_direct_push_main_under_test", HOOKS_DIR / PUSH_HOOK
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _violates_on_main(self, command):
        module = self._hook_module()
        with mock.patch.object(module, "_current_branch", return_value="main"):
            return module._violates(command)

    def test_bare_push_mention_in_quotes_allowed_on_main(self):
        self.assertIsNone(self._violates_on_main("echo 'git push now'"))

    def test_printf_mention_allowed_on_main(self):
        command = (
            "printf '%s\\n' '- Bash(/opt/homebrew/bin/git push:*) removed'"
            " >> ~/.claude/gc_log.md"
        )
        self.assertIsNone(self._violates_on_main(command))

    def test_separators_inside_quotes_do_not_split_segments(self):
        reason = self._violates_on_main("grep 'git push||main;rm' notes.md && git status")
        self.assertIsNone(reason)


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
        self.assertEqual(proc.stdout, "")


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
        self.assertEqual(proc.stdout, "")


class TestMemoryIndexLengthGuard(unittest.TestCase):
    """The <=150-line MEMORY.md rule must fire, not just be documented.

    MEMORY.md lives OUTSIDE the repo (~/.claude/projects/<slug>/memory/), so
    the guard uses a containment root there instead of the project-root
    samefile check CLAUDE.md uses. These tests build a fake memory tree and
    point the guard at it via HOME, so they never touch the real index.
    """

    def _payload(self, path, line_count):
        return {
            "tool_name": "Write",
            "tool_input": {"file_path": str(path), "content": "line\n" * line_count},
        }

    def _run_in_fake_home(self, home, payload):
        env = dict(os.environ, HOME=str(home))
        return subprocess.run(
            [sys.executable, str(HOOKS_DIR / LENGTH_HOOK)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=HOOK_TIMEOUT_S,
            cwd=REPO_ROOT,
            env=env,
        )

    def _memory_index(self, home):
        memory_dir = home / ".claude" / "projects" / "some-project" / "memory"
        memory_dir.mkdir(parents=True)
        index = memory_dir / "MEMORY.md"
        index.write_text("existing\n")
        return index

    def test_blocks_write_over_memory_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            index = self._memory_index(home)
            proc = self._run_in_fake_home(
                home, self._payload(index, MEMORY_MD_MAX_LINES + 1)
            )
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn(str(MEMORY_MD_MAX_LINES + 1), proc.stderr)
            self.assertIn("MEMORY.md", proc.stderr)

    def test_allows_write_at_memory_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            index = self._memory_index(home)
            proc = self._run_in_fake_home(
                home, self._payload(index, MEMORY_MD_MAX_LINES)
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_memory_limit_is_not_the_claude_md_limit(self):
        """A 200-line MEMORY.md is over its own budget, under CLAUDE.md's.

        Pins that the two targets carry SEPARATE limits — a single shared
        constant would let the memory index grow to 300 lines unchallenged.
        """
        self.assertLess(MEMORY_MD_MAX_LINES, CLAUDE_MD_MAX_LINES)
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            index = self._memory_index(home)
            over = MEMORY_MD_MAX_LINES + 50
            self.assertLess(over, CLAUDE_MD_MAX_LINES)
            proc = self._run_in_fake_home(home, self._payload(index, over))
            self.assertEqual(proc.returncode, 2, proc.stderr)

    def test_ignores_memory_md_outside_the_memory_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self._memory_index(home)
            stray = home / "MEMORY.md"
            stray.write_text("existing\n")
            proc = self._run_in_fake_home(
                home, self._payload(stray, MEMORY_MD_MAX_LINES + 100)
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_claude_md_guard_still_fires_with_memory_target_added(self):
        """Regression: adding the second target must not shadow the first."""
        proc = _run_hook(
            LENGTH_HOOK,
            payload={
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(REPO_ROOT / "CLAUDE.md"),
                    "content": "line\n" * (CLAUDE_MD_MAX_LINES + 1),
                },
            },
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("CLAUDE.md", proc.stderr)


class TestWorkspaceHygiene(unittest.TestCase):
    """Broad staging must be blocked while credential-ish files sit untracked.

    origin is PUBLIC, so `git add -A` over an untracked .gd/credentials.json
    publishes a live OAuth refresh token. Each test runs the hook inside a
    throwaway git repo so the real working tree is never a variable.
    """

    def _run_in_repo(self, repo, command):
        return subprocess.run(
            [sys.executable, str(HOOKS_DIR / HYGIENE_HOOK)],
            input=json.dumps({"tool_name": "Bash", "tool_input": {"command": command}}),
            capture_output=True,
            text=True,
            timeout=HOOK_TIMEOUT_S,
            cwd=str(repo),
        )

    @staticmethod
    def _make_repo(tmp, files=(), gitignore=None):
        repo = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=repo, timeout=HOOK_TIMEOUT_S)
        if gitignore is not None:
            (repo / ".gitignore").write_text(gitignore)
        for rel in files:
            target = repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x")
        return repo

    def test_blocks_add_all_when_credential_file_is_untracked(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo(tmp, files=[".gd/credentials.json"])
            proc = self._run_in_repo(repo, "git add -A")
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn(".gd/credentials.json", proc.stderr)

    def test_blocks_add_dot_and_commit_dash_a(self):
        for command in ("git add .", "git commit -am 'wip'", "git commit -a"):
            with self.subTest(command=command):
                with tempfile.TemporaryDirectory() as tmp:
                    repo = self._make_repo(tmp, files=["secrets.yaml"])
                    proc = self._run_in_repo(repo, command)
                    self.assertEqual(proc.returncode, 2, f"{command}: {proc.stderr}")

    def test_allows_broad_add_once_the_file_is_gitignored(self):
        """The .gitignore fix and the hook agree on what counts as at-risk."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo(
                tmp, files=[".gd/credentials.json"], gitignore=".gd/\n"
            )
            proc = self._run_in_repo(repo, "git add -A")
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_allows_broad_add_on_a_clean_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo(tmp, files=["src/main.py", "README.md"])
            proc = self._run_in_repo(repo, "git add -A")
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_allows_explicit_paths_even_with_credentials_present(self):
        """Naming paths is already explicit; only blind staging is blocked."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo(tmp, files=[".gd/credentials.json", "src/main.py"])
            proc = self._run_in_repo(repo, "git add src/main.py")
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_allows_add_u_which_stages_tracked_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo(tmp, files=[".gd/credentials.json"])
            proc = self._run_in_repo(repo, "git add -u")
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_quoted_mention_of_add_all_is_not_an_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo(tmp, files=[".gd/credentials.json"])
            proc = self._run_in_repo(repo, "echo 'never run git add -A here'")
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_source_file_named_for_tokens_is_not_a_finding(self):
        """A bare token/auth stem in source must not block ordinary work."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo(
                tmp, files=["tests/test_auth_token_write.py", "src/tokenizer.py"]
            )
            proc = self._run_in_repo(repo, "git add -A")
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_fails_open_on_malformed_stdin(self):
        proc = _run_hook(HYGIENE_HOOK, raw="{not json")
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

    def test_non_git_command_is_allowed(self):
        proc = _run_hook(
            HYGIENE_HOOK,
            payload={"tool_name": "Bash", "tool_input": {"command": "ls -la"}},
        )
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
