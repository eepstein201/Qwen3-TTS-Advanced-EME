"""Tests for the per-module coverage floor gate (qwen3_tts.tools.coverage_floors).

unittest.TestCase style throughout: this module is registered in
``tests/run_batches.py`` (batch 5), and the batch runner collects via
unittest, so pytest-only fixtures would make it a hollow gate.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from qwen3_tts.tools import coverage_floors as cf


def write_coverage_json(directory: Path, modules: dict[str, float]) -> str:
    """Write a minimal coverage.py JSON report measuring ``modules``."""
    payload = {
        "files": {
            module: {"summary": {"percent_covered": percent}}
            for module, percent in modules.items()
        }
    }
    path = directory / "coverage.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def write_floors_file(directory: Path, floors: dict[str, int]) -> str:
    """Write a coverage-floors.json allowlist."""
    path = directory / "coverage-floors.json"
    path.write_text(json.dumps({"modules": floors}), encoding="utf-8")
    return str(path)


class TempDirTestCase(unittest.TestCase):
    """File-backed tests get a cleaned-up temp directory."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)


class TestLoadFloors(TempDirTestCase):
    def test_reads_module_map(self):
        path = Path(write_floors_file(self.dir, {"qwen3_tts/cli_server.py": 71}))
        self.assertEqual(cf.load_floors(path), {"qwen3_tts/cli_server.py": 71})

    def test_rejects_malformed_json(self):
        path = self.dir / "coverage-floors.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(cf.CoverageFloorsError, "not valid JSON"):
            cf.load_floors(path)

    def test_rejects_wrong_shape(self):
        # An extra top-level key breaks the strict single-key contract.
        path = self.dir / "coverage-floors.json"
        path.write_text(json.dumps({"modules": {}, "extra": 1}), encoding="utf-8")
        with self.assertRaisesRegex(cf.CoverageFloorsError, "single 'modules' key"):
            cf.load_floors(path)

    def test_rejects_non_integer_floor(self):
        for bad_floor in (71.5, "71", True):
            with self.subTest(bad_floor=bad_floor):
                path = self.dir / "coverage-floors.json"
                path.write_text(
                    json.dumps({"modules": {"m.py": bad_floor}}), encoding="utf-8"
                )
                with self.assertRaisesRegex(cf.CoverageFloorsError, "integer percent"):
                    cf.load_floors(path)

    def test_missing_file_raises(self):
        with self.assertRaisesRegex(cf.CoverageFloorsError, "not found"):
            cf.load_floors(self.dir / "absent.json")


class TestLoadMeasured(TempDirTestCase):
    def test_extracts_percentages(self):
        path = Path(write_coverage_json(self.dir, {"a.py": 71.25, "b.py": 100}))
        self.assertEqual(cf.load_measured(path), {"a.py": 71.25, "b.py": 100.0})

    def test_missing_file_raises(self):
        with self.assertRaisesRegex(cf.CoverageFloorsError, "not found"):
            cf.load_measured(self.dir / "absent.json")

    def test_without_files_measurements_raises(self):
        path = self.dir / "coverage.json"
        path.write_text(json.dumps({"totals": {}}), encoding="utf-8")
        with self.assertRaisesRegex(cf.CoverageFloorsError, "no 'files'"):
            cf.load_measured(path)


class TestCheckFloors(unittest.TestCase):
    def test_meeting_floor_passes(self):
        # Exactly at floor, plus every surrounding rule category clean.
        floors = {"cli.py": 71}
        measured = {"cli.py": 71.0, "healthy.py": 95.0}
        self.assertEqual(cf.check_floors(floors, measured), [])

    def test_below_floor_fails_with_module_and_numbers(self):
        floors = {"cli.py": 71}
        measured = {"cli.py": 70.2}
        violations = cf.check_floors(floors, measured)
        self.assertEqual(len(violations), 1)
        self.assertIn("cli.py", violations[0])
        self.assertIn("70.2%", violations[0])
        self.assertIn("71%", violations[0])

    def test_floor_at_target_rejected(self):
        # Crossed modules exit; they never carry a floor.
        floors = {"cli.py": 80}
        measured = {"cli.py": 80.0}
        violations = cf.check_floors(floors, measured)
        self.assertTrue(any("exit the allowlist" in v for v in violations))

    def test_crossed_target_module_must_exit(self):
        floors = {"cli.py": 71}
        measured = {"cli.py": 82.4}
        violations = cf.check_floors(floors, measured)
        self.assertTrue(any("crossed 80" in v for v in violations))

    def test_crossed_at_exact_target_must_exit(self):
        # The boundary: exactly 80 has crossed, not "still below".
        floors = {"cli.py": 71}
        measured = {"cli.py": 80.0}
        violations = cf.check_floors(floors, measured)
        self.assertTrue(any("crossed 80" in v for v in violations))

    def test_stale_entry_fails(self):
        # An allowlisted module missing from the coverage report is stale.
        floors = {"deleted.py": 71}
        measured = {"other.py": 90.0}
        violations = cf.check_floors(floors, measured)
        self.assertTrue(any("absent from the coverage report" in v for v in violations))

    def test_unlisted_sub_80_module_must_register(self):
        # The new-rot rule: unlisted low coverage is not averaged away.
        floors = {"cli.py": 71}
        measured = {"cli.py": 71.0, "new_module.py": 45.0}
        violations = cf.check_floors(floors, measured)
        self.assertTrue(any("not allowlisted" in v for v in violations))

    def test_unlisted_module_at_target_passes(self):
        floors = {}
        measured = {"healthy.py": 80.0}
        self.assertEqual(cf.check_floors(floors, measured), [])

    def test_zero_floor_zero_percent_passes(self):
        # The __main__.py shape: 2 statements, 0% measured, floor 0.
        floors = {"qwen3_tts/__main__.py": 0}
        measured = {"qwen3_tts/__main__.py": 0.0}
        self.assertEqual(cf.check_floors(floors, measured), [])

    def test_violations_sorted_by_module_path(self):
        floors = {}
        measured = {"z.py": 10.0, "a.py": 20.0}
        violations = cf.check_floors(floors, measured)
        self.assertEqual([v.split(":")[0] for v in violations], ["a.py", "z.py"])


class TestUpdateFloors(unittest.TestCase):
    def test_seeds_floor_below_measured(self):
        # 79.9 floors to 79 so at-floor checks still pass.
        updated, notes = cf.update_floors({}, {"m.py": 79.9})
        self.assertEqual(updated, {"m.py": 79})
        self.assertEqual(notes, [])

    def test_raises_floor_to_measured(self):
        updated, _ = cf.update_floors({"m.py": 71}, {"m.py": 74.9})
        self.assertEqual(updated, {"m.py": 74})

    def test_refuses_to_lower_floor(self):
        # The ratchet: a dip keeps the higher floor.
        updated, notes = cf.update_floors({"m.py": 75}, {"m.py": 71.2})
        self.assertEqual(updated, {"m.py": 75})
        self.assertTrue(any("never lower" in n for n in notes))

    def test_drops_module_crossing_target(self):
        updated, notes = cf.update_floors({"m.py": 71}, {"m.py": 83.0})
        self.assertEqual(updated, {})
        self.assertTrue(any("crossed 80" in n for n in notes))

    def test_drops_module_measuring_exactly_target(self):
        # The boundary: exactly 80 exits, never floors to 80.
        updated, notes = cf.update_floors({"m.py": 71}, {"m.py": 80.0})
        self.assertEqual(updated, {})
        self.assertTrue(any("crossed 80" in n for n in notes))

    def test_drops_stale_entry(self):
        updated, notes = cf.update_floors({"gone.py": 71}, {"m.py": 71.0})
        self.assertEqual(updated, {"m.py": 71})
        self.assertTrue(any("absent from coverage report" in n for n in notes))


class TestMain(TempDirTestCase):
    def test_green_gate_exits_zero(self):
        floors = write_floors_file(self.dir, {"cli.py": 71})
        coverage = write_coverage_json(self.dir, {"cli.py": 71.4, "ok.py": 88.0})
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cf.main(["--floors", floors, "--coverage-json", coverage])
        self.assertEqual(exit_code, 0)
        self.assertIn("per-module floors OK", stdout.getvalue())

    def test_red_gate_exits_one_and_names_module(self):
        floors = write_floors_file(self.dir, {"cli.py": 71})
        coverage = write_coverage_json(self.dir, {"cli.py": 70.1, "ok.py": 88.0})
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cf.main(["--floors", floors, "--coverage-json", coverage])
        self.assertEqual(exit_code, 1)
        self.assertIn("FAIL: cli.py", stdout.getvalue())
        self.assertIn("1 per-module coverage floor violation", stdout.getvalue())

    def test_missing_floors_file_exits_two(self):
        coverage = write_coverage_json(self.dir, {"ok.py": 88.0})
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = cf.main(
                ["--floors", str(self.dir / "none.json"), "--coverage-json", coverage]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("error:", stderr.getvalue())

    def test_update_writes_seeded_file(self):
        floors = write_floors_file(self.dir, {"cli.py": 71})
        coverage = write_coverage_json(self.dir, {"cli.py": 74.9, "new.py": 50.0})
        exit_code = cf.main(
            ["--floors", floors, "--coverage-json", coverage, "--update"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(cf.load_floors(Path(floors)), {"cli.py": 74, "new.py": 50})

    def test_update_keeps_higher_floor(self):
        # Update must not silently lower via the file round-trip.
        floors = write_floors_file(self.dir, {"cli.py": 75})
        coverage = write_coverage_json(self.dir, {"cli.py": 71.2})
        exit_code = cf.main(
            ["--floors", floors, "--coverage-json", coverage, "--update"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(cf.load_floors(Path(floors)), {"cli.py": 75})

    def test_update_seeds_when_floors_absent(self):
        # First-run seeding has no prior file to ratchet against.
        coverage = write_coverage_json(self.dir, {"cli.py": 71.4})
        fresh = self.dir / "fresh.json"
        exit_code = cf.main(
            ["--floors", str(fresh), "--coverage-json", coverage, "--update"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(cf.load_floors(fresh), {"cli.py": 71})


if __name__ == "__main__":
    unittest.main()
