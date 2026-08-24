"""Guard: every test module on disk must be registered in ``run_batches.py``.

``BATCHES`` is an explicit list, not discovery-based, so a new ``tests/test_*.py``
silently never runs in the batch gates — it passes every ``run_batches.py`` run by
being absent. CI's coverage job runs the full pytest suite and *does* pick it up,
so the module only fails once it reaches CI (this is what happened on PR #126).

This test closes that gap: an unregistered module fails locally, before the push.

Exclusions must be declared in ``INTENTIONALLY_UNBATCHED`` with a reason, so that
"not batched" is always a decision someone wrote down rather than an oversight.

``run_batches.py`` is parsed with ``ast`` rather than imported: importing it sets
``TTS_DISABLE_RATE_LIMITING=1`` at module scope, which would leak into whatever
process runs this test and quietly neuter the rate-limiting tests.
"""

import ast
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).parent
_RUN_BATCHES = _TESTS_DIR / "run_batches.py"

# Modules deliberately outside the batch runner. Keep the reason with the entry.
INTENTIONALLY_UNBATCHED = {
    "tests.test_rate_limiting": (
        "run_batches.py sets TTS_DISABLE_RATE_LIMITING=1 at module scope, which "
        "would make these assertions vacuous. Covered under pytest/coverage, which "
        "leaves the flag unset."
    ),
    # The e2e suite is marked pytest.mark.e2e and run via `pytest -m e2e`. The batch
    # runner uses unittest (markers are ignored) and disables rate limiting, so the
    # security tests below would pass hollowly. tests.test_e2e_playwright is the one
    # exception: batch 6 exists to drive it with models preloaded.
    "tests.test_e2e_gradio_guard": "e2e: run via `pytest -m e2e`",
    "tests.test_e2e_history_clear_copy": "e2e: run via `pytest -m e2e`",
    "tests.test_e2e_performance_batch": "e2e: run via `pytest -m e2e`",
    "tests.test_e2e_performance_stress": "e2e: run via `pytest -m e2e`",
    "tests.test_e2e_security_auth": "e2e: run via `pytest -m e2e`",
    "tests.test_e2e_security_rate_limiting": (
        "e2e: run via `pytest -m e2e`; the batch runner disables rate limiting"
    ),
    "tests.test_e2e_security_validation": "e2e: run via `pytest -m e2e`",
    "tests.test_e2e_tab_navigation": "e2e: run via `pytest -m e2e`",
    "tests.test_e2e_wavesurfer_live": "e2e: run via `pytest -m e2e`",
    # Plain pytest-style classes (not unittest.TestCase): python -m unittest's
    # TestLoader.loadTestsFromModule only collects TestCase subclasses, so the
    # batch runner would silently execute these as "Ran 0 tests ... OK" —
    # a hollow pass, not real coverage. Measured empirically 2026-08-24.
    # Covered under pytest/coverage instead.
    "tests.security.test_play_audio": (
        "pytest-style class (TestPlayAudioSafety), not unittest.TestCase; "
        "collects 0 tests under `python -m unittest`"
    ),
    "tests.security.test_seed_bounds": (
        "pytest-style class (TestSeedBounds), not unittest.TestCase; "
        "collects 0 tests under `python -m unittest`"
    ),
    "tests.security.test_voice_name_validation": (
        "pytest-style class (TestValidateVoiceName), not unittest.TestCase; "
        "collects 0 tests under `python -m unittest`"
    ),
}


def _registered_modules() -> set[str]:
    """Every "tests.*" string inside the BATCHES assignment in run_batches.py."""
    tree = ast.parse(_RUN_BATCHES.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "BATCHES" for t in node.targets
        ):
            continue
        return {
            n.value
            for n in ast.walk(node.value)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.startswith("tests.")
        }
    raise AssertionError("No BATCHES assignment found in run_batches.py")


def _modules_on_disk() -> set[str]:
    """Dotted names for every test module under tests/, tests/evaluations/, and tests/security/."""
    modules = {f"tests.{p.stem}" for p in _TESTS_DIR.glob("test_*.py")}
    modules |= {
        f"tests.evaluations.{p.stem}"
        for p in (_TESTS_DIR / "evaluations").glob("test_*.py")
    }
    modules |= {
        f"tests.security.{p.stem}"
        for p in (_TESTS_DIR / "security").glob("test_*.py")
    }
    return modules


class TestBatchesCoverage(unittest.TestCase):
    def test_every_test_module_is_batched_or_declared(self):
        missing = sorted(
            _modules_on_disk() - _registered_modules() - set(INTENTIONALLY_UNBATCHED)
        )
        self.assertEqual(
            missing,
            [],
            "Test modules missing from BATCHES in tests/run_batches.py — they never "
            "run in the batch gates:\n  "
            + "\n  ".join(missing)
            + "\n\nAdd each to the appropriate batch, or to INTENTIONALLY_UNBATCHED "
            "in this file with a reason.",
        )

    def test_declared_exclusions_still_exist(self):
        """A stale allowlist entry hides the next module that takes its name."""
        stale = sorted(set(INTENTIONALLY_UNBATCHED) - _modules_on_disk())
        self.assertEqual(
            stale,
            [],
            f"INTENTIONALLY_UNBATCHED lists modules that no longer exist: {stale}",
        )

    def test_exclusions_are_not_also_batched(self):
        """A module cannot be both declared-excluded and registered."""
        both = sorted(set(INTENTIONALLY_UNBATCHED) & _registered_modules())
        self.assertEqual(
            both,
            [],
            f"Modules are both batched and listed as INTENTIONALLY_UNBATCHED: {both}",
        )
