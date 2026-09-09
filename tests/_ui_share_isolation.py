#!/usr/bin/env python3
"""Shared support: keep launch tests out of the real credentials file.

The legacy launch tests assert helper-PRODUCED launch kwargs (share, auth,
server_name), so the real helper runs — only the 0600 credential writer is
patched out, at its definition site, so no test run can create the real
``~/.config/qwen3-tts/.ui_share_credentials`` under either runner (the
unittest batch runner has no pytest conftest fixtures).
"""

from unittest.mock import patch

CREDENTIALS_WRITER = "qwen3_tts.interface.ui.shared._write_ui_credentials"


class CredentialWriterIsolation:
    """Mixin: patch out the credential writer for the whole test class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        patcher = patch(CREDENTIALS_WRITER)
        patcher.start()
        cls.addClassCleanup(patcher.stop)
