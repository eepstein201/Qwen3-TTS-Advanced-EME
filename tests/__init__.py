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

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# CPython <=3.11 import-lock reentrancy guard.
#
# gradio's utils.colab_check() runs `from IPython.core.getipython import
# get_ipython` on every gr.Blocks() construction. IPython is not a test
# dependency, so the import fails — but it still takes a module lock first, so a
# batch that builds the UI many times acquires that lock thousands of times.
# On CPython 3.10/3.11 `importlib._bootstrap._blocking_on` maps a thread id to a
# *single* lock and is not reentrancy-safe: when a GC finalizer runs an import
# while one is already in flight, the inner import consumes the entry and the
# outer import's `finally: del _blocking_on[tid]` raises `KeyError: <thread-id>`
# from inside `_ModuleLock.acquire`. CPython 3.12 reworked `_blocking_on` to
# hold a *list* per thread, which is why 3.12 never fails; 3.10 is vulnerable
# too and merely loses the race less often, so this is not 3.11-specific.
#
# Seeding sys.modules with None makes `_find_and_load` take its sys.modules fast
# path and raise ModuleNotFoundError *without acquiring the module lock* — which
# colab_check() already catches. The full dotted name is required: the lock is
# taken on "IPython.core.getipython" before its parent is resolved, so a
# sentinel on "IPython" alone does not help.
#
# Measured on linux/CPython 3.11: batch 4 failed 9/10 without this, 0/10 with.
for _absent_module in ("IPython", "IPython.core", "IPython.core.getipython"):
    sys.modules.setdefault(_absent_module, None)  # type: ignore[assignment]

_test_log_dir = tempfile.mkdtemp(prefix="qwen3-tts-test-logs-")
_test_log_file = Path(_test_log_dir) / ".voice_server.log"

_log_file_patcher = patch("qwen3_tts.core.config.LOG_FILE", _test_log_file)
_log_file_patcher.start()
