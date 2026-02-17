"""Backward compatibility shim. Import from qwen3_tts.tools.create_voice instead."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from qwen3_tts.tools import create_voice as _mod
_sys.modules[__name__] = _mod
