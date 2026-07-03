"""Auth-token write must fail fast, not silently continue.

TTSClient discovers the server token ONLY by reading TOKEN_FILE. If the write
fails (unwritable dir, disk full) the server previously logged an error and kept
running, leaving every authenticated endpoint permanently unreachable with a
misleading "use an alternative auth method" comment (there is none). Startup must
abort instead.

See docs/reviews/e2e-review-2026-07-01.md (Phase 5 silent-failure, HIGH).
"""

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


if __name__ == "__main__":
    unittest.main()
