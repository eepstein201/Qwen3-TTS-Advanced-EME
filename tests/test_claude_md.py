"""Guard tests: CLAUDE.md progressive-disclosure budget and companion doc."""
import os


def test_claude_md_under_300_lines():
    """CLAUDE.md must be <=300 lines for progressive disclosure (R-15 / Phase 1.1).

    Deep-dive content belongs in docs/00-Foundations/ARCHITECTURE.md.
    """
    with open("CLAUDE.md") as f:
        lines = f.readlines()
    assert len(lines) <= 300, (
        f"CLAUDE.md is {len(lines)} lines; must be <=300. "
        "Move deep-dive content to docs/00-Foundations/ARCHITECTURE.md."
    )


def test_architecture_md_exists():
    """ARCHITECTURE.md must exist as the deep-dive companion to CLAUDE.md."""
    assert os.path.exists("docs/00-Foundations/ARCHITECTURE.md"), (
        "docs/00-Foundations/ARCHITECTURE.md missing — "
        "this is where deep-dive content from CLAUDE.md should live."
    )


def test_memory_index_guard_rule_present():
    """CLAUDE.md must carry the standing MEMORY.md index guard rule."""
    with open("CLAUDE.md") as f:
        content = f.read()
    assert "Memory index guard" in content, (
        "CLAUDE.md missing the memory-index guard rule: "
        "MEMORY.md <=150 lines, archive overflow per the rule text."
    )


def test_claude_md_does_not_cite_missing_local_command_overrides():
    """CLAUDE.md must not point at nonexistent local ecc command overrides.

    `.claude/commands/ecc/` holds only pm2 commands — the ecc commands come
    from the plugin cache. The session-save `.tmp` rule itself lives only in
    the user-scope global CLAUDE.md (enforced by the global
    no-tmp-session-writes hook), so the project file no longer restates it.
    """
    with open("CLAUDE.md") as f:
        content = f.read()
    assert ".claude/commands/ecc/" not in content, (
        "CLAUDE.md cites .claude/commands/ecc/ overrides that do not exist."
    )
