"""Auth-token write must fail fast, not silently continue.

TTSClient discovers the server token ONLY by reading TOKEN_FILE. If the write
fails (unwritable dir, disk full) the server previously logged an error and kept
running, leaving every authenticated endpoint permanently unreachable with a
misleading "use an alternative auth method" comment (there is none). Startup must
abort instead.

See docs/reviews/e2e-review-2026-07-01.md (Phase 5 silent-failure, HIGH).

TestStartupLock covers a related but distinct failure: two `tts server start`
invocations racing for the same port. uvicorn runs the FastAPI lifespan startup
(which writes a *new random* token to the shared TOKEN_FILE) before it ever
attempts to bind the port — see uvicorn.server.Server.startup(). A losing
process that loses the port-bind race still clobbers the winning process's
still-valid token on disk with a token nobody will ever authenticate against,
then exits — breaking every authenticated endpoint (including the Gradio UI's
own /models poll) against the process that actually won and is still serving.
Reproduced live in production logs 2026-08-13: three near-simultaneous
`tts server start`/`tts ui` invocations left the on-disk token permanently
mismatched with the running server's in-memory token, producing a stream of
"Auth failure: invalid_token" on /models and /stats.
"""

import fcntl
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qwen3_tts.server import app_lifespan


class TestWriteAuthToken(unittest.TestCase):
    def test_raises_when_write_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            token_file = Path(tmp) / "sub" / ".voice_server_token"
            with patch.object(app_lifespan, "TOKEN_FILE", token_file), \
                 patch("builtins.open", side_effect=OSError("read-only file system")):
                with self.assertRaises(RuntimeError):
                    app_lifespan._write_auth_token("secret-token")

    def test_writes_token_with_restricted_perms(self):
        with tempfile.TemporaryDirectory() as tmp:
            token_file = Path(tmp) / "cfg" / ".voice_server_token"
            with patch.object(app_lifespan, "TOKEN_FILE", token_file):
                app_lifespan._write_auth_token("secret-token")
            self.assertEqual(token_file.read_text(), "secret-token")
            # 0o600 file, 0o700 parent
            self.assertEqual(os.stat(token_file).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(token_file.parent).st_mode & 0o777, 0o700)


class TestAuthTokenWriteIsAtomic(unittest.TestCase):
    """The token file must update atomically.

    TTSClient discovers the server token ONLY by reading TOKEN_FILE, and
    ``auth_headers()`` reads it fresh on every call. A truncate-then-write
    (``open(path, "w")``) leaves the file EMPTY on disk between the truncation
    and the write completing; a reader in that window gets ``""`` ->
    ``auth_headers()`` returns ``{}`` -> the request ships with no
    ``Authorization`` header and the server logs ``Auth failure: missing_token``.
    During a ``tts server restart`` the Gradio UI's ``/models`` poll can land in
    exactly that window. Atomic temp-file + ``os.replace`` (POSIX-atomic) keeps
    the previous token fully readable until the new one is in place -- the same
    pattern already used for config (``core/config/io.py``) and the PID file
    (``core/config/pid.py``).
    """

    def test_write_replaces_file_atomically_not_truncates_in_place(self):
        """Atomic os.replace installs a NEW inode; a truncating open keeps the
        same one. Same inode => the old token was destroyed before the new one
        was ready, so a concurrent reader can observe an empty file."""
        with tempfile.TemporaryDirectory() as tmp:
            token_file = Path(tmp) / "cfg" / ".voice_server_token"
            with patch.object(app_lifespan, "TOKEN_FILE", token_file):
                app_lifespan._write_auth_token("first-token")
                inode_before = os.stat(token_file).st_ino

                app_lifespan._write_auth_token("second-token")

                self.assertEqual(token_file.read_text(), "second-token")
                inode_after = os.stat(token_file).st_ino
                self.assertNotEqual(
                    inode_before,
                    inode_after,
                    "token file was truncated in place (same inode) instead of "
                    "atomically replaced -- a concurrent reader can observe an "
                    "empty token",
                )

    def test_concurrent_reader_never_observes_empty_or_partial_token(self):
        """While the token is rewritten repeatedly, a reader must see only whole
        old/new tokens -- never empty or partial -- which is what prevents the
        missing_token auth failures during a restart."""
        import threading

        token_a = "A" * 64
        token_b = "B" * 64
        valid = {token_a, token_b}

        with tempfile.TemporaryDirectory() as tmp:
            token_file = Path(tmp) / "cfg" / ".voice_server_token"
            with patch.object(app_lifespan, "TOKEN_FILE", token_file):
                app_lifespan._write_auth_token(token_a)

                bad: list[str] = []
                stop = threading.Event()

                def reader() -> None:
                    while not stop.is_set():
                        try:
                            content = token_file.read_text()
                        except OSError:
                            continue
                        if content not in valid:
                            bad.append(content)

                def writer() -> None:
                    for i in range(800):
                        app_lifespan._write_auth_token(
                            token_a if i % 2 else token_b
                        )

                t = threading.Thread(target=reader)
                t.start()
                try:
                    writer()
                finally:
                    stop.set()
                    t.join()

                self.assertEqual(
                    bad,
                    [],
                    f"concurrent reader observed non-atomic token state(s): "
                    f"{bad[:5]!r}",
                )


class TestStartupLock(unittest.TestCase):
    def test_acquires_lock_when_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_file = Path(tmp) / ".voice_server.lock"
            with patch.object(app_lifespan, "LOCK_FILE", lock_file):
                fh = app_lifespan._acquire_startup_lock()
                try:
                    self.assertTrue(lock_file.exists())
                finally:
                    fh.close()

    def test_raises_when_already_locked_by_another_process(self):
        """A losing `tts server start` must fail to acquire the lock, not clobber it."""
        with tempfile.TemporaryDirectory() as tmp:
            lock_file = Path(tmp) / ".voice_server.lock"
            holder = open(lock_file, "w")
            try:
                fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(app_lifespan, "LOCK_FILE", lock_file):
                    with self.assertRaises(RuntimeError):
                        app_lifespan._acquire_startup_lock()
            finally:
                fcntl.flock(holder, fcntl.LOCK_UN)
                holder.close()

    def test_losing_lifespan_never_writes_token(self):
        """Regression: a losing instance's lifespan must abort BEFORE
        _write_auth_token runs, so it never clobbers the winner's token."""
        import asyncio

        from qwen3_tts.server.app import app

        with tempfile.TemporaryDirectory() as tmp:
            lock_file = Path(tmp) / ".voice_server.lock"
            holder = open(lock_file, "w")
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

            async def _run():
                with patch.object(app_lifespan, "LOCK_FILE", lock_file), \
                     patch.object(app_lifespan, "_write_auth_token") as mock_write:
                    with self.assertRaises(RuntimeError):
                        async with app_lifespan.lifespan(app):
                            pass
                    mock_write.assert_not_called()

            try:
                asyncio.run(_run())
            finally:
                fcntl.flock(holder, fcntl.LOCK_UN)
                holder.close()


if __name__ == "__main__":
    unittest.main()
