"""Click-based unified CLI for Qwen3-TTS.

Usage:
    tts "Hello, world!" -o hello          # Generate audio
    tts server start                       # Start model server
    tts voice list                         # List voice prompts
    tts list speakers                      # List premium speakers
    tts ui                                 # Launch web UI
    tts config                             # Run config wizard
"""

import sys

import click

from qwen3_tts.cli_config import (  # noqa: F401
    cache,
    config,
    doctor_command,
    history_command,
    ui_command,
    uninstall,
)

# Submodule imports — groups and helpers
from qwen3_tts.cli_server import (  # noqa: F401
    _start_server_daemon,
    server,
    stats_command,
)
from qwen3_tts.cli_voice import (  # noqa: F401
    list_group,
    voice,
)

# ---------------------------------------------------------------------------
# Custom Group class — routes bare `tts "Hello"` to generate
# ---------------------------------------------------------------------------


class TTSGroup(click.Group):
    """Routes bare `tts "Hello"` to the generate subcommand."""

    def parse_args(self, ctx, args):
        # Strip --_server-mode before routing (it's a generate-level flag)
        # and stash it so we can re-insert after routing
        server_mode = False
        if "--_server-mode" in args:
            args = [a for a in args if a != "--_server-mode"]
            server_mode = True
        # If first arg isn't a subcommand or flag, prepend 'generate'
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["generate"] + args
        # If no args at all (bare `tts`), route to generate (interactive mode)
        if not args:
            args = ["generate"]
        # Re-insert --_server-mode only for commands that accept it (generation commands)
        if server_mode:
            _generation_commands = {
                "generate",
                "batch",
                "srt",
                "dialogue",
                "repl",
                "watch",
            }
            if args and args[0] in _generation_commands:
                args.insert(1, "--_server-mode")
        return super().parse_args(ctx, args)


# ---------------------------------------------------------------------------
# Shared: translate Click kwargs → argparse sys.argv → call generate main()
# ---------------------------------------------------------------------------

_FLAG_MAP = {
    "mode": ("-m", str),
    "prompt": ("-p", str),
    "description": ("-d", str),
    "speaker": ("-s", str),
    "instruct": ("-i", str),
    "voice": ("-v", str),
    "prosody": ("--prosody", str),
    "no_transcript": ("--no-transcript", bool),
    "output": ("-o", str),
    "play": ("--play", bool),
    "stream": ("--stream", bool),
    "no_open": ("--no-open", bool),
    "speed": ("--speed", float),
    "pitch": ("--pitch", float),
    "trim_silence": ("--trim-silence", bool),
    "normalize": ("--normalize", bool),
    "preset": ("--preset", str),
    "temperature": ("--temperature", float),
    "top_k": ("--top-k", int),
    "top_p": ("--top-p", float),
    "seed": ("--seed", int),
    "repetition_penalty": ("--repetition-penalty", float),
    "max_chunk_chars": ("--max-chunk-chars", int),
    "max_new_tokens": ("--max-new-tokens", int),
    "clipboard": ("--clipboard", bool),
    "ssml": ("--ssml", bool),
    "local": ("--local", bool),
    "dry_run": ("--dry-run", bool),
    "backend": ("--backend", str),
    "model_size": ("--model-size", str),
    "server_mode": ("--_server-mode", bool),
    "text_override": ("--text-override", str),
    # Advanced mode flags
    "batch": ("--batch", str),
    "srt": ("--srt", str),
    "dialogue": ("--dialogue", str),
    "save_individual": ("--save-individual", bool),
    "repl_mode": ("--repl", bool),
    "watch_dir": ("--watch", str),
}


def _call_generate(text=(), **kwargs):
    """Translate Click kwargs to argparse-style sys.argv and call generate main()."""
    argv = []
    for key, (flag, typ) in _FLAG_MAP.items():
        val = kwargs.get(key)
        if val is None or val is False:
            continue
        if typ is bool:
            argv.append(flag)
        else:
            argv.extend([flag, str(val)])
    if text:
        argv.extend(text)

    old_argv = sys.argv
    sys.argv = ["tts"] + argv
    try:
        from qwen3_tts.interface.generate import main as _gen_main

        result = _gen_main()
        if result is True:
            sys.exit(2)
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group(cls=TTSGroup)
@click.version_option(version="3.0.0", prog_name="Qwen3-TTS")
def cli():
    """Qwen3-TTS -- Text to speech with voice cloning."""
    pass


# ---------------------------------------------------------------------------
# Shared generation options decorator
# ---------------------------------------------------------------------------


def _generation_options(f):
    """Apply all generation-related options to a command."""
    decorators = [
        click.option(
            "-m",
            "--mode",
            type=click.Choice(["clone", "design", "custom"]),
            help="Voice mode",
        ),
        click.option("-p", "--prompt", help="Voice clone prompt filename"),
        click.option("-d", "--description", help="Voice description (design mode)"),
        click.option("-s", "--speaker", help="Premium speaker name (custom mode)"),
        click.option("-i", "--instruct", help="Style instruction (custom mode)"),
        click.option("-v", "--voice", help="Voice alias from config"),
        click.option("--prosody", help="Prosody preset (custom/design mode)"),
        click.option(
            "--no-transcript", is_flag=True, help="Clone using speaker embedding only"
        ),
        click.option("-o", "--output", help="Output filename or directory"),
        click.option("--play", is_flag=True, help="Play audio after generation"),
        click.option(
            "--stream", is_flag=True, help="Stream audio playback as it generates"
        ),
        click.option("--no-open", is_flag=True, help="Don't open the output file"),
        click.option(
            "--speed", type=float, help="Speed factor (1.2=faster, 0.8=slower)"
        ),
        click.option("--pitch", type=float, help="Pitch shift in semitones"),
        click.option(
            "--trim-silence", is_flag=True, help="Trim leading/trailing silence"
        ),
        click.option("--normalize", is_flag=True, help="Normalize audio to -3dB peak"),
        click.option("--preset", help="Named preset from config"),
        click.option("--temperature", type=float, help="Sampling temperature"),
        click.option("--top-k", type=int, help="Top-k sampling"),
        click.option("--top-p", type=float, help="Top-p (nucleus) sampling"),
        click.option("--seed", type=int, help="Random seed"),
        click.option("--repetition-penalty", type=float, help="Repetition penalty"),
        click.option(
            "--max-chunk-chars", type=int, help="Max chars per chunk (0=disable)"
        ),
        click.option("--max-new-tokens", type=int, help="Max new tokens per chunk"),
        click.option("--clipboard", is_flag=True, help="Read text from clipboard"),
        click.option("--ssml", is_flag=True, help="Enable SSML markup parsing"),
        click.option(
            "--local", is_flag=True, help="Force local generation (skip server)"
        ),
        click.option(
            "--dry-run",
            is_flag=True,
            help="Show what would be generated without running",
        ),
        click.option(
            "--backend",
            type=click.Choice(["torch", "mlx"]),
            help="Override backend for this run",
        ),
        click.option(
            "--model-size",
            type=click.Choice(["1.7B", "0.6B"]),
            help="Override model size for this run",
        ),
        click.option("--_server-mode", "server_mode", is_flag=True, hidden=True),
        click.option("--text-override", hidden=True),
    ]
    for decorator in reversed(decorators):
        f = decorator(f)
    return f


# ---------------------------------------------------------------------------
# generate — the default command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("text", nargs=-1)
@_generation_options
def generate(text, **kwargs):
    """Generate audio from text.

    This is the default command — you can omit 'generate':

    \b
      tts "Hello, world!" -o hello
    """
    _call_generate(text, **kwargs)


# ---------------------------------------------------------------------------
# Register subgroups from split modules
# ---------------------------------------------------------------------------

cli.add_command(server)
cli.add_command(voice)
cli.add_command(list_group, "list")
cli.add_command(config)
cli.add_command(uninstall)
cli.add_command(cache)


# ---------------------------------------------------------------------------
# Standalone commands from split modules (need Click decorators here)
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--port", type=int, help="Port number (default: 7860)")
@click.option("--share", is_flag=True, help="Create public URL via Gradio")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def ui(port, share, no_browser):
    """Launch the Gradio web interface."""
    ui_command(port, share, no_browser)


@cli.command()
@click.argument("count", default=10, type=int, required=False)
def history(count):
    """Show last N generations (default: 10)."""
    history_command(count)


@cli.command()
def stats():
    """Show server statistics."""
    stats_command()


# ---------------------------------------------------------------------------
# Batch / advanced modes
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@_generation_options
def batch(file, **kwargs):
    """Process batch JSON file with array of texts."""
    _call_generate(batch=file, **kwargs)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@_generation_options
def srt(file, **kwargs):
    """Process SRT subtitle file."""
    _call_generate(srt=file, **kwargs)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--save-individual", is_flag=True, help="Save individual audio files for each line"
)
@_generation_options
def dialogue(file, save_individual, **kwargs):
    """Process dialogue JSON file with multiple speakers."""
    _call_generate(dialogue=file, save_individual=save_individual, **kwargs)


@cli.command()
@_generation_options
def repl(**kwargs):
    """Start interactive REPL mode."""
    _call_generate(repl_mode=True, **kwargs)


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@_generation_options
def watch(directory, **kwargs):
    """Watch directory for .txt files and generate audio."""
    _call_generate(watch_dir=directory, **kwargs)


# ---------------------------------------------------------------------------
# doctor / healthcheck
# ---------------------------------------------------------------------------


@cli.command()
def doctor():
    """Check TTS installation health."""
    doctor_command()


@cli.command("healthcheck", hidden=True)
def healthcheck_cmd():
    """Alias for doctor command."""
    doctor_command()


if __name__ == "__main__":
    cli()
