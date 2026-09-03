#!/usr/bin/env python3
"""PM2 process-supervision detection for the local TTS server.

No torch, numpy, or heavy imports.

`tts server stop`/`restart` are written for a bare-daemon lifecycle (their
own `.voice_server.pid` file, direct SIGTERM/SIGKILL). When the server is
instead launched via PM2 (`ecosystem.config.cjs`, `autorestart: true`),
killing the process directly is indistinguishable to PM2 from a crash --
it respawns the server within `restart_delay`, so the stop silently
undoes itself. `pm2_owner_of_port()` lets the CLI detect that case and
delegate to `pm2 stop`/`pm2 restart` instead, which correctly disables
autorestart for an intentional stop.
"""

import json
import subprocess  # nosec B404  # pm2/ps process introspection; hardcoded commands, int args only

_PM2_ANCESTOR_SEARCH_DEPTH = 10


def _parent_pid(pid: int) -> int | None:
    """Return the parent PID of `pid` via `ps -o ppid=`, or None if unavailable."""
    try:
        result = subprocess.run(  # nosec B603, B607  # hardcoded "ps" lookup of an int pid; no user input
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    try:
        return int(text) if text else None
    except ValueError:
        return None


def _pm2_jlist() -> list | None:
    """Return the parsed `pm2 jlist` app list, or None if pm2 is
    unavailable/unusable (not installed, timed out, non-zero exit, or
    malformed output)."""
    try:
        result = subprocess.run(  # nosec B603, B607  # hardcoded "pm2 jlist"; no user input
            ["pm2", "jlist"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        apps = json.loads(result.stdout)
    except ValueError:
        return None
    return apps if isinstance(apps, list) else None


def pm2_owner_of_port(port: int) -> str | None:
    """Return the PM2 app name managing the process listening on `port`, or None.

    The PID bound to the port is typically a grandchild of the PM2-tracked
    process (e.g. this project's `start.cjs` shells out to `zsh -c '... &&
    python -m qwen3_tts.server.app'`), so a direct PID match against
    `pm2 jlist` isn't enough -- this walks the port PID's parent chain
    looking for a match against any online app's own PID.
    """
    from qwen3_tts.core.config import find_pid_by_port

    pid = find_pid_by_port(port)
    if pid is None:
        return None

    apps = _pm2_jlist()
    if not apps:
        return None

    online_pids = {
        app["pid"]: app.get("name")
        for app in apps
        if isinstance(app, dict)
        and app.get("pm2_env", {}).get("status") == "online"
        and app.get("pid")
    }
    if not online_pids:
        return None

    current = pid
    for _ in range(_PM2_ANCESTOR_SEARCH_DEPTH):
        if current in online_pids:
            return online_pids[current]
        parent = _parent_pid(current)
        if parent is None or parent <= 1:
            break
        current = parent
    return None


def pm2_registered_app(port: int) -> str | None:
    """Return the PM2 app name for this server's port if PM2 has ANY
    entry registered under it (any status -- online, stopped, errored),
    or None.

    `pm2_owner_of_port` can only see an app PM2 is *currently running*
    (it walks the listening PID's ancestry, so it needs a live process).
    `tts server start` needs to detect a *stopped* PM2 app too --
    otherwise it spawns a second, PM2-untracked daemon on the same port
    the next time the PM2 app is started. PM2 keeps an app's
    registration (name, cwd, `pid: 0`) in `jlist` even while stopped, so
    this matches by this project's established naming convention
    (`tts-server-<port>`, see CLAUDE.md's PM2 Services table) rather
    than by PID.
    """
    apps = _pm2_jlist()
    if not apps:
        return None

    expected_name = f"tts-server-{port}"
    for app in apps:
        if isinstance(app, dict) and app.get("name") == expected_name:
            return expected_name
    return None
