"""Per-module coverage floor gate (ratcheted allowlist).

The CI coverage gate is aggregate-only (``--cov-fail-under=80``): new
well-covered code can raise the total while a critical module rots
underneath it. This tool closes that blind spot by asserting every
module's measured coverage against a checked-in ratcheted allowlist,
``coverage-floors.json`` at the repo root.

Rules (check mode, the CI gate):

* a module in the allowlist must measure ``>=`` its floor — meeting the
  floor passes, dropping below fails regardless of the aggregate;
* a floor of 80 or above is invalid — modules that cross the target
  *exit* the allowlist instead of carrying a floor;
* a listed module measuring ``>= 80`` fails until its entry is removed
  (the gate nudges the exit);
* a sub-80 module missing from the allowlist fails — new low coverage
  must be registered, not silently averaged away;
* an allowlist entry whose module is absent from the coverage report
  fails (stale entry after a rename/deletion).

``--update`` re-seeds floors at the measured values for maintenance:
floors only ever rise (a module measuring below its current floor keeps
the higher floor and emits a note), modules crossing 80 are dropped,
and stale entries are dropped. Seeding the initial floors is the same
mode run against a fresh coverage JSON report.

Usage::

    python -m qwen3_tts.tools.coverage_floors            # check; exit 1 on violation
    python -m qwen3_tts.tools.coverage_floors --update   # re-seed floors (never lower)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Modules measuring at/above this exit the allowlist (the P2 target).
TARGET_COVERAGE = 80

DEFAULT_FLOORS_PATH = Path("coverage-floors.json")
DEFAULT_COVERAGE_JSON = Path("coverage.json")


class CoverageFloorsError(Exception):
    """Malformed input (floors file or coverage JSON) — never a gate failure."""


def load_floors(path: Path) -> dict[str, int]:
    """Load and validate ``coverage-floors.json``; raise on any shape drift."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CoverageFloorsError(f"floors file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CoverageFloorsError(
            f"floors file {path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw, dict) or set(raw) != {"modules"}:
        raise CoverageFloorsError(
            f"floors file {path} must be a JSON object with a single 'modules' key"
        )
    modules = raw["modules"]
    if not isinstance(modules, dict):
        raise CoverageFloorsError(f"'modules' in {path} must be an object")

    floors: dict[str, int] = {}
    for module, floor in modules.items():
        if not isinstance(module, str) or not module:
            raise CoverageFloorsError(
                f"module keys in {path} must be non-empty strings"
            )
        # bool is an int subclass — reject it explicitly.
        if isinstance(floor, bool) or not isinstance(floor, int):
            raise CoverageFloorsError(
                f"floor for {module} in {path} must be an integer percent, got {floor!r}"
            )
        floors[module] = floor
    return floors


def load_measured(path: Path) -> dict[str, float]:
    """Extract ``{module: percent_covered}`` from a coverage.py JSON report."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CoverageFloorsError(f"coverage JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CoverageFloorsError(
            f"coverage JSON {path} is not valid JSON: {exc}"
        ) from exc

    files = raw.get("files")
    if not isinstance(files, dict) or not files:
        raise CoverageFloorsError(f"coverage JSON {path} has no 'files' measurements")

    measured: dict[str, float] = {}
    for module, entry in files.items():
        summary = entry.get("summary") if isinstance(entry, dict) else None
        percent = summary.get("percent_covered") if isinstance(summary, dict) else None
        if isinstance(percent, bool) or not isinstance(percent, (int, float)):
            raise CoverageFloorsError(
                f"coverage JSON {path} entry for {module} lacks summary.percent_covered"
            )
        measured[module] = float(percent)
    return measured


def check_floors(floors: dict[str, int], measured: dict[str, float]) -> list[str]:
    """Return one violation message per broken rule, sorted by module path."""
    violations: list[str] = []
    for module in sorted(set(floors) | set(measured)):
        if module not in floors:
            # Unlisted module: sub-80 coverage must be registered.
            percent = measured[module]
            if percent < TARGET_COVERAGE:
                violations.append(
                    f"{module}: {percent:.1f}% is below {TARGET_COVERAGE}% and not "
                    f"allowlisted — register a floor (--update)"
                )
            continue
        floor = floors[module]
        covered = measured.get(module)
        if covered is None:
            violations.append(
                f"{module}: allowlisted but absent from the coverage report — "
                f"remove the stale entry"
            )
        elif floor >= TARGET_COVERAGE:
            violations.append(
                f"{module}: floor {floor} is >= {TARGET_COVERAGE} — crossed modules "
                f"exit the allowlist instead of carrying a floor"
            )
        elif covered >= TARGET_COVERAGE:
            violations.append(
                f"{module}: {covered:.1f}% crossed {TARGET_COVERAGE}% — remove from "
                f"the allowlist"
            )
        elif covered < floor:
            violations.append(
                f"{module}: {covered:.1f}% is below its floor of {floor}% — add tests "
                f"or raise the floor (--update)"
            )
    return violations


def update_floors(
    existing: dict[str, int], measured: dict[str, float]
) -> tuple[dict[str, int], list[str]]:
    """Re-seed floors at measured values; never lower, drop crossed/stale."""
    updated: dict[str, int] = {}
    notes: list[str] = []
    for module in sorted(measured):
        percent = measured[module]
        if percent >= TARGET_COVERAGE:
            if module in existing:
                notes.append(f"{module}: dropped (crossed {TARGET_COVERAGE}%)")
            continue
        floor = int(percent)  # 0-100 range: int() is floor()
        if existing.get(module, -1) > floor:
            notes.append(
                f"{module}: kept floor {existing[module]}% (measured {percent:.1f}% "
                f"— floors never lower)"
            )
            floor = existing[module]
        updated[module] = floor
    for module in sorted(existing):
        if module not in measured:
            notes.append(f"{module}: dropped (absent from coverage report)")
    return updated, notes


def write_floors(path: Path, floors: dict[str, int]) -> None:
    """Write the allowlist: single write, sorted keys, trailing newline."""
    payload = {"modules": dict(sorted(floors.items()))}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--floors",
        type=Path,
        default=DEFAULT_FLOORS_PATH,
        help="path to coverage-floors.json (default: %(default)s)",
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=DEFAULT_COVERAGE_JSON,
        help="coverage.py JSON report to read (default: %(default)s)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="re-seed floors at measured values instead of checking",
    )
    args = parser.parse_args(argv)

    try:
        measured = load_measured(args.coverage_json)
        if args.update:
            existing = load_floors(args.floors) if args.floors.exists() else {}
            updated, notes = update_floors(existing, measured)
            write_floors(args.floors, updated)
            for note in notes:
                print(note)
            print(f"wrote {len(updated)} floor(s) to {args.floors}")
            return 0
        floors = load_floors(args.floors)
    except CoverageFloorsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    violations = check_floors(floors, measured)
    if violations:
        for violation in violations:
            print(f"FAIL: {violation}")
        print(
            f"{len(violations)} per-module coverage floor violation(s) — the "
            f"aggregate floor cannot see these"
        )
        return 1
    print(
        f"per-module floors OK: {len(floors)} allowlisted module(s) at/above their "
        f"floors; all other modules >= {TARGET_COVERAGE}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
