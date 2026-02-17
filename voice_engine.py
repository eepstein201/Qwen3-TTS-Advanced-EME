"""Backward compatibility shim. Import from qwen3_tts.core.engine instead."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from qwen3_tts.core import engine as _mod
_sys.modules[__name__] = _mod
