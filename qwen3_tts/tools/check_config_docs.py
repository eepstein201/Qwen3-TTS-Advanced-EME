"""Verify docs/CONFIG.md default values against get_default_config().

``docs/CONFIG.md`` is marked ``AUTO-GENERATED`` but ships without a generator,
so its *default-value* column drifts whenever ``get_default_config()`` changes
(e.g. ``default_clone_prompt`` → ``null``, ``aliases`` → ``{}``). This tool
imports the live defaults, parses the documented defaults out of CONFIG.md's
markdown tables, and reports any key whose documented default no longer matches
the code. It exits non-zero on drift so it can gate CI / pre-push.

Only the **default values** are checked — they are fully derivable from code
and are the part that drifts. Prose descriptions, the environment-variable
tables, and JSON-block examples (``presets`` / ``prosody_presets`` /
``aliases``) are curated by hand and are intentionally out of scope.

Usage::

    python -m qwen3_tts.tools.check_config_docs        # check; exit 1 on drift
    python -m qwen3_tts.tools.check_config_docs --fix   # print corrected rows
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# core.config has no heavy imports — safe at module scope.
from qwen3_tts.core.config import get_default_config

CONFIG_DOC = Path(__file__).resolve().parents[2] / "docs" / "CONFIG.md"

# A markdown table row whose first cell is a `dotted.path` key.
# Captures: key, type, default cell, description.
_TABLE_ROW = re.compile(
    r"^\|\s*`([a-z][a-z0-9_.]*)`\s*\|"  # 1: key path (backtick code span)
    r"\s*([^|]+?)\s*\|"  # 2: type
    r"\s*([^|]+?)\s*\|"  # 3: default cell
    r"\s*([^|]*)\s*\|$",  # 4: description
    re.MULTILINE,
)


@dataclass(frozen=True)
class Drift:
    """A single documented default that disagrees with the code."""

    key: str
    documented: str
    actual: str

    def as_row(self) -> str:
        """Render the corrected default cell for ``--fix`` output."""
        return f"{self.key}: documented `{self.documented}` → should be `{self.actual}`"


def flatten_config(cfg: dict, prefix: str = "") -> dict[str, object]:
    """Flatten a nested config dict into ``dotted.path -> scalar`` entries.

    Nested dicts recurse; scalars (str/int/float/bool/None) become leaves.
    Leaf dicts such as ``aliases`` / ``presets`` that are empty or hold only
    further dicts produce no scalar entries, which is fine — they are not
    documented as scalar table rows.
    """
    out: dict[str, object] = {}
    for key, value in cfg.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_config(value, path))
        else:
            out[path] = value
    return out


def normalize(value: object) -> str:
    """Render a Python value as it appears in CONFIG.md's default column.

    ``True``/``False`` → ``true``/``false``; ``None`` → ``null``; numbers and
    strings are stringified with surrounding quotes/backticks stripped.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value).strip("`").strip('"').strip("'")


def _leading_value(cell: str) -> str:
    """The documented default value at the start of a cell.

    Most cells are a single code span (`` `value` ``); some add prose after a
    platform-dependent value (`` `"mlx"` (Apple Silicon), `"torch"` elsewhere``).
    We take the first quoted string if present, else the first bare token.
    """
    cleaned = cell.replace("`", "").strip().lstrip("*").strip()
    quoted = re.match(r"""["']([^"']+)["']""", cleaned)
    if quoted:
        return quoted.group(1)
    bare = re.match(r"([^\s,()]+)", cleaned)
    return bare.group(1) if bare else cleaned


def _quoted_values(cell: str) -> list[str]:
    """Every quoted alternative documented in the cell (platform options)."""
    return re.findall(r"""["']([^"']+)["']""", cell.replace("`", ""))


def _eq(a: str, b: str) -> bool:
    """Equal strings, or numerically equal (so ``0`` matches ``0.0``)."""
    if a == b:
        return True
    fa, fb = _to_float(a), _to_float(b)
    return fa is not None and fb is not None and fa == fb


def _matches(actual_norm: str, cell: str) -> bool:
    """True if the documented cell lists ``actual_norm`` as the default.

    Accepts the leading value or any quoted alternative (for platform-dependent
    defaults documented with more than one option).
    """
    candidates = [_leading_value(cell), *_quoted_values(cell)]
    return any(_eq(actual_norm, c) for c in candidates)


def _to_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def parse_defaults_from_markdown(text: str) -> dict[str, str]:
    """Extract ``dotted.key -> documented default cell`` from CONFIG.md tables."""
    parsed: dict[str, str] = {}
    for key, _type, default_cell, _desc in _TABLE_ROW.findall(text):
        parsed[key] = default_cell
    return parsed


def check_drift(
    config_md_text: str, actual_defaults: dict[str, object]
) -> list[Drift]:
    """Return every documented default that disagrees with ``actual_defaults``.

    Only keys present in *both* the markdown tables and ``actual_defaults`` are
    compared; keys documented only in prose or only in code are skipped.
    """
    documented = parse_defaults_from_markdown(config_md_text)
    drifts: list[Drift] = []
    for key, actual_value in actual_defaults.items():
        if key not in documented:
            continue
        actual_norm = normalize(actual_value)
        if not _matches(actual_norm, documented[key]):
            drifts.append(
                Drift(
                    key=key,
                    documented=documented[key].strip(),
                    actual=actual_norm,
                )
            )
    drifts.sort(key=lambda d: d.key)
    return drifts


def main(argv: list[str] | None = None) -> int:
    """Entry point: check CONFIG.md against live defaults. Returns exit code."""
    args = sys.argv[1:] if argv is None else argv
    show_fix = "--fix" in args

    if not CONFIG_DOC.is_file():
        print(f"error: {CONFIG_DOC} not found", file=sys.stderr)
        return 2

    text = CONFIG_DOC.read_text(encoding="utf-8")
    actual = flatten_config(get_default_config())
    drifts = check_drift(text, actual)

    if not drifts:
        print(f"OK: {CONFIG_DOC.name} defaults match get_default_config() ({len(actual)} keys).")
        return 0

    print(f"DRIFT: {len(drifts)} documented default(s) disagree with code:\n")
    for d in drifts:
        print(f"  - {d.as_row()}")
    if show_fix:
        print("\nSuggested corrections (update the default column in CONFIG.md):")
        for d in drifts:
            print(f"    {d.key} -> `{d.actual}`")
    print(
        "\nFix: update the listed rows in docs/CONFIG.md "
        "(or run with --fix for the target values)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
