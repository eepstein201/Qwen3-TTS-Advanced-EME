#!/usr/bin/env python3
"""Argument parsing for Qwen3-TTS CLI.

This module contains the argparse configuration for all CLI options.
"""

import argparse


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Qwen3-TTS Generator")
    parser.add_argument("text", nargs="*", help="Text(s) to synthesize or path to text file")
    parser.add_argument("-o", "--output", help="Output filename or directory for batch")
    parser.add_argument("-m", "--mode", choices=["clone", "design", "custom"], help="Voice mode (clone, design, or custom)")
    parser.add_argument("-p", "--prompt", help="Voice clone prompt filename")
    parser.add_argument("-d", "--description", help="Voice description (for design mode)")
    parser.add_argument("-s", "--speaker", help="Premium speaker name for custom mode (ryan, aiden, vivian, etc.)")
    parser.add_argument("-i", "--instruct", help="Style instruction for custom mode (e.g., 'very happy', 'speak slowly')")
    parser.add_argument("--prosody", metavar="PRESET", help="Prosody preset for custom/design mode (excited, calm, whisper, etc.)")
    parser.add_argument("--no-transcript", action="store_true", dest="no_transcript",
                        help="Clone using speaker embedding only (no transcript needed)")
    parser.add_argument("--batch", help="JSON file with array of texts")
    parser.add_argument("--preset", help="Use named preset from config")

    # Generation parameters
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, dest="top_k", help="Top-k sampling")
    parser.add_argument("--top-p", type=float, dest="top_p", help="Top-p (nucleus) sampling")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--repetition-penalty", type=float, dest="repetition_penalty", help="Repetition penalty")

    # Text chunking
    parser.add_argument("--max-chunk-chars", type=int, dest="max_chunk_chars", metavar="N",
                        help="Max chars per chunk for long text (default: 500 from config, 0 to disable)")

    # Backend and model size override
    parser.add_argument("--backend", choices=["torch", "mlx"], help="Override backend for this run (default: from config.json)")
    parser.add_argument("--model-size", choices=["1.7B", "0.6B"], help="Override model size for this run (default: from config.json)")
    parser.add_argument("--list-backends", action="store_true", help="List available backends and current setting")

    # Utility options
    parser.add_argument("--list-prompts", action="store_true", help="List available voice prompts")
    parser.add_argument("--voices", action="store_true", help="List available voice prompts (alias for --list-prompts)")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    parser.add_argument("--list-aliases", action="store_true", help="List available voice aliases")
    parser.add_argument("--list-speakers", action="store_true", help="List premium CustomVoice speakers")
    parser.add_argument("--list-prosody", action="store_true", help="List available prosody presets")
    parser.add_argument("--list-models", action="store_true", help="List available TTS models and their load status")
    parser.add_argument("--stats", action="store_true", help="Show server statistics")
    parser.add_argument("--edit-config", action="store_true", help="Edit default voice description")
    parser.add_argument("--no-open", action="store_true", help="Don't open the output file")
    parser.add_argument("--local", action="store_true", help="Force local generation (skip server)")
    parser.add_argument("--play", action="store_true", help="Play audio after generation")
    parser.add_argument("--stream", action="store_true", help="Stream audio playback as it generates (MLX backend)")
    parser.add_argument("--clipboard", action="store_true", help="Read text from clipboard")
    parser.add_argument("--trim-silence", action="store_true", help="Trim leading/trailing silence")
    parser.add_argument("--normalize", action="store_true", help="Normalize audio to -3dB peak")
    parser.add_argument("--speed", type=float, metavar="FACTOR", help="Speed factor (1.2 = 20%% faster, 0.8 = 20%% slower)")
    parser.add_argument("--pitch", type=float, metavar="SEMITONES", help="Pitch shift in semitones (+2 = higher, -2 = lower)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without running")

    # Voice alias
    parser.add_argument("-v", "--voice", help="Use a voice alias from config (combines prompt + preset)")

    # Voice prompt management
    parser.add_argument("--delete-prompt", metavar="NAME", help="Delete a voice prompt")
    parser.add_argument("--rename-prompt", nargs=2, metavar=("OLD", "NEW"), help="Rename a voice prompt")
    parser.add_argument("--preview-prompt", metavar="NAME", help="Preview a voice prompt")

    # History
    parser.add_argument("--history", nargs="?", const=10, type=int, metavar="N", help="Show last N generations (default: 10)")

    # Integration features
    parser.add_argument("--repl", action="store_true", help="Start interactive REPL mode")
    parser.add_argument("--watch", metavar="DIR", help="Watch directory for .txt files")
    parser.add_argument("--srt", metavar="FILE", help="Process SRT subtitle file")
    parser.add_argument("--ssml", action="store_true", help="Enable SSML markup parsing")
    parser.add_argument("--dialogue", metavar="FILE", help="Process dialogue JSON file with multiple speakers")
    parser.add_argument("--save-individual", action="store_true", help="Save individual audio files for each dialogue line")
    parser.add_argument("--ui", "--gui", action="store_true", dest="ui", help="Launch the Gradio web interface")

    # Internal flags set by wrapper script
    parser.add_argument("--_server-mode", dest="server_mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--text-override", dest="text_override", help=argparse.SUPPRESS)

    return parser
