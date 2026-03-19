#!/usr/bin/env python3
"""CLI package for Qwen3-TTS.

This package provides the command-line interface components:
- batch: Batch processing from JSON files
- srt: SRT subtitle file processing
- dialogue: Multi-speaker dialogue processing

The main entry point is re-exported from generate.py for backward compatibility.
"""

# Re-export main entry point for backward compatibility
from qwen3_tts.interface.generate import main

# Export submodules
from qwen3_tts.interface.cli import batch
from qwen3_tts.interface.cli import dialogue
from qwen3_tts.interface.cli import srt

__all__ = [
    # Main entry point
    "main",
    # Submodules
    "batch",
    "dialogue",
    "srt",
]
