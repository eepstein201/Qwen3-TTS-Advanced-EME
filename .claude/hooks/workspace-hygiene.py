#!/usr/bin/env python3
"""PreToolUse hook: block broad `git add`/`git commit -a` over credential files.

origin is a PUBLIC repo, so every push is a publication. The working tree
routinely accumulates untracked local agent/tool state — .gd/ (a Google Drive
OAuth refresh_token + client_secret), .codex/, .ua/, scratch security reports.
A .gitignore entry is the primary defense, but ignore rules drift behind the
tools that create the files: the 2026-09-11 workspace audit found all of the
above untracked AND unignored, one `git add -A` away from publication.

This hook is the durable half of that fix. It fires only on commands that
stage indiscriminately (`git add -A/--all/.`, `git commit -a/-am`), and only
blocks when the repo ACTUALLY holds an untracked, unignored path whose name
looks credential-bearing. A clean tree is silent, so the normal workflow is
untouched.

`git add <specific paths>` and `git add -u` (tracked files only) are never
blocked — both are already explicit about what they stage.

Scope note: only UNTRACKED + UNIGNORED paths are scanned. A tracked file named
tests/test_auth_token_write.py is not a finding — it is already public by
decision. This hook guards the accidental first commit, not the archive.

Quoted spans are scrubbed before matching (shared with no-direct-push-main via
_hook_shell), so `echo "git add -A"` is prose, not an invocation.

Protocol: reads the PreToolUse payload on stdin; exit 2 blocks the call with
stderr shown to the model. Unparseable input, non-matching commands, or any
git failure exit 0 (fail-open on shape, never on policy).
"""
import json
import os
import re
import subprocess
import sys

# _hook_shell lives beside this script. Direct execution (how the harness
# invokes hooks) already puts that directory on sys.path, but the guard tests
# import this file via importlib, which does not — so anchor it explicitly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _hook_shell import segments  # noqa: E402

GIT_TIMEOUT_S = 10


# Name fragments that mark a file as credential-bearing. Matched against each
# path segment, case-insensitively.
_SECRET_NAME_RE = re.compile(
    r"(credential|secret|password|passwd|htpasswd|"
    r"api[-_]?key|auth[-_]?token|access[-_]?token|refresh[-_]?token|"
    r"service[-_]?account|id_rsa|id_ed25519|\.netrc|\.npmrc|\.pypirc)",
    re.IGNORECASE,
)
# Extensions that are credential material regardless of their stem.
_SECRET_EXT = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".ppk")
# A bare `token`/`auth` stem is too common in source to block on; restrict the
# looser match to data/config formats and extensionless files.
_LOOSE_NAME_RE = re.compile(r"(token|auth|\.env)", re.IGNORECASE)
_DATA_EXT = (".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", "")
# Source and prose never match on NAME alone. tests/test_auth_token_write.py is
# a test about credential handling, not a credential — blocking it would make
# the hook fire on ordinary work and train the user to work around it.
# Extension-based matches (_SECRET_EXT) still apply: a .pem is a .pem.
_SOURCE_EXT = (
    ".py", ".pyi", ".md", ".rst", ".txt", ".ts", ".tsx", ".js", ".jsx",
    ".go", ".rs", ".java", ".c", ".h", ".cc", ".cpp", ".hpp", ".ipynb",
    ".html", ".css", ".sql", ".lock",
)


def _git_subcommand_args(segment, subcommand):
    """Args following `git <subcommand>` in this segment, or None if absent.

    Token scanning rather than a regex: matching `git commit ... -a` needs a
    quantifier to skip intervening args, and every regex shape for that nests
    quantifiers, which CodeQL reports as polynomial ReDoS (py/polynomial-redos)
    since the command text is model/user-provided.
    """
    tokens = segment.split()
    for index, token in enumerate(tokens):
        if token == "git" and tokens[index + 1 : index + 2] == [subcommand]:
            return tokens[index + 2 :]
    return None


def _has_short_flag(args, letter):
    """True if any clustered short flag (`-a`, `-am`) carries this letter."""
    return any(
        arg.startswith("-") and not arg.startswith("--") and letter in arg[1:]
        for arg in args
    )


def _is_broad_stage(command):
    """True if any segment stages indiscriminately."""
    for segment in segments(command):
        add_args = _git_subcommand_args(segment, "add")
        if add_args is not None and (
            "." in add_args or "--all" in add_args or _has_short_flag(add_args, "A")
        ):
            return True
        commit_args = _git_subcommand_args(segment, "commit")
        if commit_args is not None and (
            "--all" in commit_args or _has_short_flag(commit_args, "a")
        ):
            return True
    return False


def _untracked_unignored():
    """Paths git would newly add: untracked and not covered by any ignore rule."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _looks_secret(path):
    """True if any segment of this path names credential material."""
    for part in path.split("/"):
        if not part:
            continue
        ext = os.path.splitext(part)[1].lower()
        if ext in _SECRET_EXT:
            return True  # credential material by format, whatever it is named
        if ext in _SOURCE_EXT:
            continue  # source/prose: never a finding on its name alone
        if _SECRET_NAME_RE.search(part):
            return True
        if _LOOSE_NAME_RE.search(part) and ext in _DATA_EXT:
            return True
    return False


def main():
    try:
        payload = json.load(sys.stdin)
        command = payload.get("tool_input", {}).get("command", "")
    except Exception:
        return 0

    if not _is_broad_stage(command):
        return 0

    findings = sorted({p for p in _untracked_unignored() if _looks_secret(p)})
    if not findings:
        return 0

    shown = findings[:10]
    more = len(findings) - len(shown)
    listing = "\n".join(f"  - {p}" for p in shown)
    if more:
        listing += f"\n  - ... and {more} more"
    print(
        "BLOCKED by workspace-hygiene: this command stages every untracked"
        " file, and the working tree holds untracked, UNIGNORED paths whose"
        " names look credential-bearing:\n"
        f"{listing}\n"
        "origin is a PUBLIC repo — staging these publishes them. Either add"
        " them to .gitignore first (verify with: git check-ignore -v <path>),"
        " or stage the files you actually mean by name:"
        " git add path/one path/two.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
