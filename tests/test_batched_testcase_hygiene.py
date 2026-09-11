"""Guard against batched test classes that the batch runner never collects.

``tests/run_batches.py`` spawns ``python -m unittest tests.<module>`` per batch.
unittest collects ONLY ``unittest.TestCase`` subclasses, so a pytest-style
``class TestFoo:`` in a batched module contributes zero tests to the batch gate
while still reporting **OK** -- the module is registered, the runner says it
ran, and every assertion inside it is dead. Proven on
``tests/test_server_peaks.py``: with ``calculate_waveform_peaks`` sabotaged to
return ``[]``, ``pytest`` reported 1 failed while
``python -m unittest tests.test_server_peaks`` reported "Ran 0 tests ... OK".

This is the same silent-false-green family as
``tests/test_async_test_hygiene.py`` (async tests that never await) and is
guarded the same way: statically, so it cannot recur through review drift.

``KNOWN_HOLLOW`` is a RATCHET, not an exemption list. Every entry is a module
Step 1E converts; entries only ever get deleted. Adding one is not a fix -- the
allowlist exists so this guard can land before the sweep finishes rather than
being blocked behind 14 conversions. A module absent from both the allowlist
and the clean set fails here.

Both files are parsed with ``ast`` rather than imported: importing test modules
pulls in heavy optional dependencies, and importing ``run_batches`` sets
``TTS_DISABLE_RATE_LIMITING=1`` at module scope (see
``tests/test_batches_coverage.py`` for that trap).
"""

import ast
import re
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).parent
_RUN_BATCHES = _TESTS_DIR / "run_batches.py"

# A class is collected by unittest iff some ancestor base name ends in
# "TestCase" (unittest.TestCase, IsolatedAsyncioTestCase, and project-local
# subclasses all qualify) -- either directly or through a chain of
# module-local base classes: unittest collects by real inheritance, not by
# the textual base name, so ``TestX(_ContractTestBase)`` IS collected when
# ``_ContractTestBase`` subclasses ``unittest.TestCase`` (proven 2026-09-11:
# test_response_contracts / test_ui_low_rate_prompt_warning / test_ui_port_flag
# ran identical test counts under both runners while the name-only check
# flagged them). Bases are compared by their final dotted segment so both
# ``unittest.TestCase`` and a bare imported ``TestCase`` match. A base that is
# neither TestCase-suffixed nor defined in the module has unknown lineage and
# does NOT count -- fail closed, the same posture as before.
_TESTCASE_SUFFIX = "TestCase"

# Modules registered in BATCHES whose top-level test classes are still
# pytest-style. Each is hollow under the batch runner TODAY -- Step 1E converts
# them and deletes the entry. Counts are from the 2026-09-11 scan.
KNOWN_HOLLOW = {
    "tests.test_solid_analyzer": "Step 1E: 10 classes",
}


def _registered_modules() -> set[str]:
    """Return every ``tests.*`` module string listed in run_batches.BATCHES."""
    return set(re.findall(r'"(tests\.[A-Za-z0-9_.]+)"', _RUN_BATCHES.read_text()))


def _module_path(dotted: str) -> Path:
    """Map ``tests.sub.mod`` to its file under the tests directory."""
    return _TESTS_DIR.parent / (dotted.replace(".", "/") + ".py")


def _class_bases(tree: ast.Module) -> dict[str, list[str]]:
    """Map every top-level class name to its unparsed base expressions."""
    return {
        node.name: [ast.unparse(base) for base in node.bases]
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


def _resolves_to_testcase(
    name: str,
    bases_by_class: dict[str, list[str]],
    _seen: frozenset[str] = frozenset(),
) -> bool:
    """True if unittest will collect this class (transitively, in-module)."""
    if name in _seen:
        return False  # defensive: an inheritance cycle is not a TestCase chain
    seen = _seen | {name}
    for base in bases_by_class.get(name, []):
        final = base.split(".")[-1]
        if final.endswith(_TESTCASE_SUFFIX):
            return True
        if final in bases_by_class and _resolves_to_testcase(final, bases_by_class, seen):
            return True
    return False


def _uncollected_classes(path: Path) -> list[str]:
    """Return top-level ``Test*`` classes unittest would silently skip."""
    tree = ast.parse(path.read_text(), filename=str(path))
    bases_by_class = _class_bases(tree)
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name.startswith("Test")
        and not _resolves_to_testcase(node.name, bases_by_class)
    ]


class TestBatchedModulesUseTestCase(unittest.TestCase):
    """Every batched module's test classes must actually run under unittest."""

    def test_no_unlisted_batched_module_has_uncollected_test_classes(self):
        """A batched module with a pytest-style test class is a hollow gate."""
        offenders = {}
        for dotted in sorted(_registered_modules()):
            if dotted in KNOWN_HOLLOW:
                continue
            path = _module_path(dotted)
            if not path.exists():
                continue  # tests/test_batches_coverage.py owns existence
            uncollected = _uncollected_classes(path)
            if uncollected:
                offenders[dotted] = uncollected

        self.assertEqual(
            offenders,
            {},
            "batched module(s) define top-level Test* classes that do NOT "
            "subclass unittest.TestCase -- the batch runner collects zero "
            "tests from them and still reports OK. Convert them to "
            f"unittest.TestCase: {offenders}",
        )

    def test_known_hollow_entries_are_still_hollow(self):
        """The ratchet cannot rot: a converted module must leave the list.

        Without this, an entry silently becomes a permanent exemption that
        would let the module regress back to pytest-style unnoticed.
        """
        stale = []
        for dotted in sorted(KNOWN_HOLLOW):
            path = _module_path(dotted)
            if not path.exists():
                stale.append(f"{dotted} (file missing)")
            elif not _uncollected_classes(path):
                stale.append(f"{dotted} (already converted)")

        self.assertEqual(
            stale,
            [],
            "KNOWN_HOLLOW entries that are no longer hollow -- delete them "
            f"from the allowlist (Step 1E ratchet): {stale}",
        )

    def test_transitive_module_local_bases_resolve(self):
        """A Test* class inheriting a module-local TestCase subclass is
        collected by unittest; the guard must not flag it (2026-09-11 false
        positives: test_response_contracts / test_ui_low_rate_prompt_warning /
        test_ui_port_flag, whose helper bases already extend TestCase).
        """
        bases = {
            "_ContractTestBase": ["unittest.TestCase"],
            "TestPublicStatusContracts": ["_ContractTestBase"],
            "TestPlainPytest": [],
            "TestViaDirectMixin": ["_Mixin", "unittest.TestCase"],
            "_Mixin": ["object"],
            "TestUnknownImport": ["SomeImportedMixin"],
        }
        self.assertTrue(_resolves_to_testcase("TestPublicStatusContracts", bases))
        self.assertTrue(_resolves_to_testcase("TestViaDirectMixin", bases))
        self.assertFalse(_resolves_to_testcase("TestPlainPytest", bases))
        # Imported base with unknown lineage fails closed
        self.assertFalse(_resolves_to_testcase("TestUnknownImport", bases))
        # An inheritance cycle is not a TestCase chain
        cyclic = {"TestCycleA": ["TestCycleB"], "TestCycleB": ["TestCycleA"]}
        self.assertFalse(_resolves_to_testcase("TestCycleA", cyclic))

    def test_known_hollow_modules_are_actually_batched(self):
        """An allowlist entry that isn't batched is dead weight."""
        registered = _registered_modules()
        unregistered = sorted(set(KNOWN_HOLLOW) - registered)
        self.assertEqual(
            unregistered,
            [],
            "KNOWN_HOLLOW lists module(s) not registered in BATCHES; this "
            f"guard only covers batched modules: {unregistered}",
        )


if __name__ == "__main__":
    unittest.main()
