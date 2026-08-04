"""Server management CLI commands.

Extracted from cli.py to keep each module under 800 lines.
Contains: server group (start, stop, status, log, restart) and stats command.
"""

import os
import subprocess  # nosec B404  # server PID/daemon management (launches app, tails logs via hardcoded commands)
import sys
import time

import click


def _start_server_daemon(public=False):
    """Start the TTS server as a daemon (background subprocess).

    Args:
        public: If True, bind to 0.0.0.0 instead of 127.0.0.1

    Returns:
        subprocess.Popen: The server process object
    """
    from qwen3_tts.core.config import write_pid_file

    cmd = [sys.executable, "-m", "qwen3_tts.server.app"]
    if public:
        cmd.append("--public")

    proc = subprocess.Popen(  # nosec B603  # launches local server via sys.executable; cmd is a hardcoded list, no user input
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    write_pid_file(proc.pid)
    return proc


@click.group()
def server():
    """Manage the TTS server."""
    pass


@server.command()
@click.option("--public", is_flag=True, help="Bind to 0.0.0.0")
@click.option(
    "--foreground", is_flag=True, help="Run in foreground (for Colab/notebooks)"
)
def start(public, foreground):
    """Start the TTS server.

    By default, the server runs in the background as a daemon.
    Use --foreground to run in the foreground (useful for Colab).
    """
    from qwen3_tts.core.config import (
        cleanup_pid_file,
        detect_server_state,
        get_server_url,
        load_config,
    )

    config = load_config()
    state = detect_server_state(config)

    if state["running"]:
        try:
            url = get_server_url(config)
        except ValueError as exc:
            click.echo(f"Invalid server configuration: {exc}")
            sys.exit(1)
        click.echo(f"TTS Server is already running at {url}")
        sys.exit(1)

    if state["stale_pid"]:
        cleanup_pid_file()
        click.echo(f"Cleaned stale PID file (PID {state['pid']} no longer running).")

    if foreground:
        click.echo("Starting TTS server in foreground...")
        import uvicorn

        from qwen3_tts.server.app import app

        host = config.get("server", {}).get("host", "127.0.0.1")
        if public:
            host = "0.0.0.0"  # nosec B104  # intentional --public network bind
        port = config.get("server", {}).get("port", 5123)
        uvicorn.run(app, host=host, port=port, log_level="info")
    else:
        proc = _start_server_daemon(public=public)
        click.echo(f"TTS Server started with PID {proc.pid}")
        from qwen3_tts.core.config import LOG_FILE

        click.echo(f"Logs: {LOG_FILE}")


def _reject_if_not_stoppable(state):
    """Exit early when the server is already down.

    Not running at all: exit(1). Only a stale PID file: clean it up and
    exit(0). Otherwise (server actually running) return normally.
    """
    from qwen3_tts.core.config import cleanup_pid_file

    if not state["running"] and not state["stale_pid"]:
        click.echo("TTS Server is not running.")
        sys.exit(1)

    if state["stale_pid"]:
        cleanup_pid_file()
        click.echo("TTS Server is not running (cleaned stale PID file).")
        sys.exit(0)


def _attempt_graceful_shutdown(state, config):
    """POST /shutdown and poll for the server to exit.

    Exits the process with code 0 if the server confirms it stopped.
    Returns normally (falling through to the SIGTERM/SIGKILL fallback)
    if shutdown wasn't accepted or polling timed out.
    """
    from qwen3_tts.core.config import cleanup_pid_file, is_server_running

    if not state["health_ok"]:
        return

    shutdown_accepted = False
    try:
        import requests

        from qwen3_tts.core.http_client import server_request

        resp = server_request("POST", "/shutdown", timeout=5)
        if resp.status_code == 200:
            shutdown_accepted = True
            click.echo("TTS Server shutdown signal sent.")
        elif resp.status_code == 401:
            click.echo("Shutdown rejected: 401 Unauthorized (token mismatch).")
        else:
            click.echo(f"Shutdown returned HTTP {resp.status_code}.")
    except (
        requests.ConnectionError,
        requests.Timeout,
        requests.RequestException,
        OSError,
    ):
        click.echo("Server did not respond to shutdown request.")

    # Only poll if shutdown was accepted
    if shutdown_accepted:
        for _ in range(10):
            time.sleep(0.5)
            if not is_server_running(config):
                cleanup_pid_file()
                click.echo("TTS Server stopped.")
                sys.exit(0)


def _kill_server_process(state, port):
    """Fallback termination: SIGTERM, then SIGKILL, then clean up the PID file."""
    import signal

    from qwen3_tts.core.config import cleanup_pid_file, find_pid_by_port, is_pid_alive

    pid = state["pid"]

    # Discover PID by port if PID file was missing
    if pid is None:
        pid = find_pid_by_port(port)
        if pid is not None:
            click.echo(f"Discovered server PID {pid} on port {port}.")

    if pid and is_pid_alive(pid):
        click.echo(f"Server still alive (PID {pid}), sending SIGTERM...")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        # Wait up to 3 seconds for termination
        for _ in range(6):
            time.sleep(0.5)
            if not is_pid_alive(pid):
                break
        # Last resort: SIGKILL
        if is_pid_alive(pid):
            click.echo(f"SIGTERM failed, sending SIGKILL to PID {pid}...")
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    cleanup_pid_file()


def _report_stop_result(config, port):
    """Verify the server actually stopped and report the outcome."""
    from qwen3_tts.core.config import is_server_running

    if is_server_running(config):
        click.echo(f"Error: TTS Server is still running on port {port}.")
        click.echo(f"Manual kill: kill -9 $(lsof -ti :{port})")
        sys.exit(1)

    click.echo("TTS Server stopped.")


@server.command()
def stop():
    """Stop the TTS server."""
    from qwen3_tts.core.config import detect_server_state, load_config

    config = load_config()
    port = config.get("server", {}).get("port", 5123)
    state = detect_server_state(config)

    _reject_if_not_stoppable(state)

    # Server is running — attempt graceful shutdown via /shutdown
    _attempt_graceful_shutdown(state, config)

    # Fallback: SIGTERM/SIGKILL if the shutdown request didn't stop it
    _kill_server_process(state, port)

    _report_stop_result(config, port)


@server.command()
@click.option("--public", is_flag=True, help="Bind to 0.0.0.0")
@click.pass_context
def restart(ctx, public):
    """Stop the server, then start it again (daemon mode)."""
    from qwen3_tts.core.config import is_server_running, load_config

    config = load_config()

    if is_server_running(config):
        try:
            ctx.invoke(stop)
        except SystemExit as exc:
            if exc.code != 0:
                click.echo("Error: failed to stop the server.")
                sys.exit(1)
        for _ in range(20):
            time.sleep(0.5)
            if not is_server_running(config):
                break
        if is_server_running(config):
            click.echo("Error: server did not stop in time.")
            sys.exit(1)
    else:
        click.echo("Server was not running.")

    click.echo("Starting server...")
    proc = _start_server_daemon(public=public)
    click.echo(f"TTS Server started with PID {proc.pid}")
    from qwen3_tts.core.config import LOG_FILE

    click.echo(f"Logs: {LOG_FILE}")


@server.command()
def status():
    """Show server health, loaded models, and memory usage."""
    import requests

    from qwen3_tts.core.config import get_server_url, is_server_running, load_config
    from qwen3_tts.core.http_client import server_request

    config = load_config()
    if not is_server_running(config):
        click.echo("TTS Server is not running.")
        sys.exit(1)

    try:
        url = get_server_url(config)
    except ValueError as exc:
        click.echo(f"Invalid server configuration: {exc}")
        sys.exit(1)
    try:
        resp = server_request("GET", "/health", timeout=5)
        health = resp.json()
        click.echo(f"Server: running ({url})")
        click.echo(f"Backend: {health.get('backend', 'unknown')}")

        resp = server_request("GET", "/models", timeout=5)
        models = resp.json().get("models", {})
        click.echo("\nModels:")
        for name, info in models.items():
            status_str = "loaded" if info.get("loaded") else "not loaded"
            click.echo(f"  {name}: {status_str}")

        resp = server_request("GET", "/stats", timeout=5)
        stats = resp.json()
        mem_key = next(
            (k for k in stats if "memory" in k.lower() and "mb" in k.lower()), None
        )
        if mem_key:
            click.echo(f"\nMemory: {stats[mem_key]}MB")
    except (
        requests.ConnectionError,
        requests.Timeout,
        requests.RequestException,
        ValueError,
        KeyError,
    ) as e:
        click.echo(f"Error connecting to server: {e}")
        sys.exit(1)


@server.command()
def log():
    """Tail the server log."""
    from qwen3_tts.core.config import LOG_FILE

    log_file = LOG_FILE

    if not log_file.exists():
        click.echo(f"Log file not found: {log_file}")
        sys.exit(1)

    # Tail the log file
    try:
        # Use tail command if available, otherwise Python fallback
        result = subprocess.run(  # nosec B603, B607  # hardcoded "tail -f" on a validated log path; no user input
            ["tail", "-f", str(log_file)],
            text=True,
        )
        sys.exit(result.returncode)
    except FileNotFoundError:
        # Fallback: read and print new lines
        click.echo(f"Tailing {log_file} (Ctrl+C to stop)...")
        with open(log_file) as f:
            f.seek(0, 2)  # Seek to end
            try:
                while True:
                    line = f.readline()
                    if line:
                        click.echo(line.rstrip())
                    else:
                        time.sleep(0.1)
            except KeyboardInterrupt:
                click.echo("\nStopped tailing log.")


def stats_command():
    """Show server statistics."""
    import json

    from qwen3_tts.core.config import is_server_running, load_config
    from qwen3_tts.core.http_client import server_request

    config = load_config()
    if not is_server_running(config):
        click.echo("Server not running. Start with: tts server start")
        sys.exit(1)
    resp = server_request("GET", "/stats", timeout=10)
    if resp.status_code == 200:
        click.echo(json.dumps(resp.json(), indent=2))
    else:
        click.echo(f"Error: {resp.text}")
        sys.exit(1)
