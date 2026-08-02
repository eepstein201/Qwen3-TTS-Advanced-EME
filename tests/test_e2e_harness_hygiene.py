"""Guard the E2E harness against silently unauthenticated requests.

The auth token moved to ``~/.config/qwen3-tts/.voice_server_token`` some time
ago; ``~/.voice_server_token`` is the legacy location kept only for backward
compatibility. A harness helper that reads *only* the legacy path returns an
empty token on any normal install, and every authenticated request it makes
then fails with `missing_token` — visible in `.voice_server.log` but not in the
test output, so model-management tests degrade quietly rather than failing
loudly (repo-audit-2026-07-31 follow-up finding).

The invariant is deliberately weak enough to allow the legitimate pattern —
a candidate list that tries canonical first and falls back to legacy, which
several E2E suites use — while catching the legacy-only read that caused the
bug. Prefer delegating to ``config.read_auth_token()``; it already owns this
resolution, and the copies are what let this drift.

Static, like tests/test_async_test_hygiene.py: importing every E2E module would
pull in playwright and launch machinery.
"""

import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).parent

LEGACY_TOKEN = "~/.voice_server_token"
CANONICAL_TOKEN = ".config/qwen3-tts/.voice_server_token"
# The production reader, which resolves canonical-then-legacy itself. A file
# delegating to it needs no literal path at all.
CANONICAL_READER = "read_auth_token"


def _files_referencing_legacy_only():
    """Return files naming the legacy token path with no canonical counterpart."""
    offenders = []
    for path in sorted(TESTS_DIR.glob("*.py")):
        if path.name == Path(__file__).name:
            continue  # this guard names both paths by necessity
        text = path.read_text()
        if LEGACY_TOKEN not in text:
            continue
        if CANONICAL_TOKEN in text or CANONICAL_READER in text:
            continue
        offenders.append(path.name)
    return offenders


class TestE2EHarnessTokenPath(unittest.TestCase):
    """Static guard — no test module is imported."""

    def test_scan_covers_the_suite(self):
        """A scan that finds no files would pass vacuously."""
        self.assertGreater(
            len(list(TESTS_DIR.glob("*.py"))),
            50,
            "expected to scan the full tests/ directory; glob found too few files",
        )

    def test_no_harness_reads_only_the_legacy_token_path(self):
        offenders = _files_referencing_legacy_only()
        self.assertEqual(
            offenders,
            [],
            "these files reference the legacy auth-token path without the "
            "canonical one, so they read an empty token on a normal install and "
            "their authenticated requests fail as `missing_token` without "
            "failing the test. Delegate to "
            "qwen3_tts.core.config.read_auth_token(), or try the canonical path "
            f"first and fall back: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
