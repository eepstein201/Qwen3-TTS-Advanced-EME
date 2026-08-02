"""Guard against async test methods that silently never run.

Two ways an `async def test_*` becomes a false green — reported as passed while
executing zero lines of its body:

1. Declared on a plain `unittest.TestCase`. Calling it returns a coroutine that
   is never awaited; unittest sees a non-``None`` return and reports **ok**.
   The fix is `unittest.IsolatedAsyncioTestCase`.
2. Declared outside a TestCase without `@pytest.mark.asyncio`. pytest-asyncio
   runs in ``strict`` mode here (no ``asyncio_mode`` in pytest.ini), so an
   unmarked coroutine test is skipped rather than executed.

Both are silent: the suite count goes up, the assertions never fire. This is a
recurring failure mode (see docs/plans/repo-audit-2026-07-31.md, P0-2), so it is
guarded statically rather than left to review.

The scan is AST-based on purpose — importing every test module to introspect it
would pull in heavy optional dependencies and run module-level side effects.
"""

import ast
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).parent

# A base spelled like this makes the class a unittest.TestCase subclass whose
# runner does NOT await coroutines. IsolatedAsyncioTestCase is excluded below.
SYNC_TESTCASE_SUFFIX = "TestCase"
ASYNC_TESTCASE_NAME = "IsolatedAsyncioTestCase"

# Decorators that hand a bare coroutine test to an event loop.
ASYNC_RUNNER_MARKS = ("asyncio", "anyio")


def _base_names(node: ast.ClassDef) -> list[str]:
    """Return each base as source text, e.g. ``unittest.TestCase``."""
    return [ast.unparse(base) for base in node.bases]


def _is_async_testcase(node: ast.ClassDef) -> bool:
    """True if the class runs its own event loop (awaits coroutine tests)."""
    return any(
        name.split(".")[-1] == ASYNC_TESTCASE_NAME for name in _base_names(node)
    )


def _is_sync_testcase(node: ast.ClassDef) -> bool:
    """True if the class is a TestCase whose runner will not await coroutines."""
    if _is_async_testcase(node):
        return False
    return any(
        name.split(".")[-1].endswith(SYNC_TESTCASE_SUFFIX)
        for name in _base_names(node)
    )


def _has_async_runner_mark(node: ast.AsyncFunctionDef) -> bool:
    """True if decorated with pytest.mark.asyncio / anyio (call form included)."""
    for decorator in node.decorator_list:
        text = ast.unparse(decorator)
        if any(f"mark.{mark}" in text for mark in ASYNC_RUNNER_MARKS):
            return True
    return False


def _async_tests(body: list[ast.stmt]) -> list[ast.AsyncFunctionDef]:
    """Direct-child async test defs only.

    Deliberately not ``ast.walk`` — a coroutine nested inside a sync test and
    driven by ``asyncio.run(...)`` (the pattern in
    tests/test_vllm_async_nonblocking.py) does run, and is not a violation.
    """
    return [
        stmt
        for stmt in body
        if isinstance(stmt, ast.AsyncFunctionDef) and stmt.name.startswith("test")
    ]


def _scan(path: Path) -> tuple[list[str], list[str]]:
    """Return (unawaited_on_testcase, unmarked_bare) violations for one file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    on_testcase: list[str] = []
    unmarked: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for func in _async_tests(node.body):
                where = f"{path.name}:{func.lineno} {node.name}.{func.name}"
                if _is_async_testcase(node):
                    continue  # IsolatedAsyncioTestCase awaits its own tests
                if _is_sync_testcase(node):
                    on_testcase.append(where)
                elif not _has_async_runner_mark(func):
                    unmarked.append(where)
        elif isinstance(node, ast.AsyncFunctionDef) and node.name.startswith("test"):
            if not _has_async_runner_mark(node):
                unmarked.append(f"{path.name}:{node.lineno} {node.name}")

    return on_testcase, unmarked


class TestAsyncTestHygiene(unittest.TestCase):
    """Static guard — no test module is imported."""

    @classmethod
    def setUpClass(cls):
        cls.test_files = sorted(TESTS_DIR.glob("test_*.py"))

    def test_scan_covers_the_suite(self):
        """A scan that finds no files would pass vacuously."""
        self.assertGreater(
            len(self.test_files),
            50,
            "expected to scan the full tests/ directory; glob found too few files",
        )

    def test_no_async_test_methods_on_sync_testcase(self):
        """`async def test_*` on a plain TestCase is reported ok without running."""
        violations = [v for path in self.test_files for v in _scan(path)[0]]
        self.assertEqual(
            violations,
            [],
            "async test methods on a plain unittest.TestCase never execute "
            "(unittest does not await them) yet are reported as passed. "
            "Use unittest.IsolatedAsyncioTestCase instead:\n  "
            + "\n  ".join(violations),
        )

    def test_no_unmarked_bare_async_tests(self):
        """Outside a TestCase, an unmarked coroutine test is skipped by pytest."""
        violations = [v for path in self.test_files for v in _scan(path)[1]]
        self.assertEqual(
            violations,
            [],
            "async tests outside a TestCase need @pytest.mark.asyncio — "
            "pytest-asyncio runs in strict mode, so unmarked coroutine tests "
            "do not execute:\n  " + "\n  ".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
