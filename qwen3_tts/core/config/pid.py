#!/usr/bin/env python3
"""PID lifecycle helpers for the local TTS server process.

No torch, numpy, or heavy imports.

PID_FILE is defined in paths.py; functions here read it via a lazy per-call
import from ``qwen3_tts.core.config`` (the package facade) rather than a
static module-level import, so that ``@patch("qwen3_tts.core.config.PID_FILE", ...)``
(the existing test seam) is observed at call time. See
qwen3_tts/core/config/__init__.py for the rationale.
"""

import os
import subprocess


def read_pid_file() -> int | None:
    """Read PID from .voice_server.pid. Returns None if missing/invalid."""
    from qwen3_tts.core.config import PID_FILE

    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None


def write_pid_file(pid: int) -> None:
    """Write PID to .voice_server.pid atomically via temp-file + os.replace()."""
    from qwen3_tts.core.config import PID_FILE

    tmp = PID_FILE.with_suffix(".pid.tmp")
    tmp.write_text(str(pid))
    os.replace(tmp, PID_FILE)


def cleanup_pid_file() -> None:
    """Remove .voice_server.pid if it exists. Idempotent."""
    from qwen3_tts.core.config import PID_FILE

    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def find_pid_by_port(port: int) -> int | None:
    """Discover PID of process listening on a TCP port via lsof.
    Works on macOS and Linux. Returns int PID or None.
    """
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().splitlines()[0]
            return int(first_line)
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None
    except ValueError:
        return None
    return None


def is_pid_alive(pid: int) -> bool:
    """Check if process exists via os.kill(pid, 0). Cross-platform."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Process exists but owned by another user
    except OSError:
        return False


def detect_server_state(config: dict | None = None) -> dict:
    """Unified server state combining health check + PID file + process liveness.

    Returns dict with keys:
        running (bool): True if server is definitely running
        health_ok (bool): True if /health responds
        pid (int|None): PID if known from file
        pid_alive (bool): True if PID process exists
        stale_pid (bool): True if PID file exists but process dead + health fails
    """
    from qwen3_tts.core.config import is_pid_alive as _is_pid_alive
    from qwen3_tts.core.config import is_server_running
    from qwen3_tts.core.config import read_pid_file as _read_pid_file

    health_ok = is_server_running(config)
    pid = _read_pid_file()
    pid_alive = _is_pid_alive(pid) if pid is not None else False

    running = health_ok  # Health check is authoritative
    stale_pid = pid is not None and not pid_alive and not health_ok

    return {
        "running": running,
        "health_ok": health_ok,
        "pid": pid,
        "pid_alive": pid_alive,
        "stale_pid": stale_pid,
    }
