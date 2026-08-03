"""Test package init.

Redirects file-based logging away from the real ~/.voice_server.log before any
test module runs. Mirrors the pytest ``_isolate_test_logging`` autouse fixture
in conftest.py — but conftest.py fixtures only apply to pytest collection.
``tests/run_batches.py`` (and ``python -m unittest discover``) run tests via
plain unittest, which imports this package first and never sees pytest
fixtures. Without this, batch/unittest runs that exercise run_server() add a
RotatingFileHandler to the 'tts' logger pointing at the production log file,
polluting it with mock/testclient entries (see docs/plans/repo-audit-2026-07-31.md P3).
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

_test_log_dir = tempfile.mkdtemp(prefix="qwen3-tts-test-logs-")
_test_log_file = Path(_test_log_dir) / ".voice_server.log"

_log_file_patcher = patch("qwen3_tts.core.config.LOG_FILE", _test_log_file)
_log_file_patcher.start()
