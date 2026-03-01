"""Backward compatibility shim. Import from qwen3_tts.server.app_fastapi instead."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from qwen3_tts.server import app_fastapi as _mod
_sys.modules[__name__] = _mod
