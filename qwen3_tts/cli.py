"""Click-based unified CLI for Qwen3-TTS.

Usage:
    tts "Hello, world!" -o hello          # Generate audio
    tts server start                       # Start model server
    tts voice list                         # List voice prompts
    tts list speakers                      # List premium speakers
    tts ui                                 # Launch web UI
    tts config                             # Run config wizard
"""

import os
import sys

import click

# ---------------------------------------------------------------------------
# Legacy flag → subcommand rewriting
# ---------------------------------------------------------------------------

LEGACY_REDIRECTS = {
    '--list-prompts': ['voice', 'list'],
    '--voices':       ['voice', 'list'],
    '--list-presets':  ['list', 'presets'],
    '--list-speakers': ['list', 'speakers'],
    '--list-prosody':  ['list', 'prosody'],
    '--list-models':   ['list', 'models'],
    '--list-backends': ['list', 'backends'],
    '--list-aliases':  ['list', 'aliases'],
    '--stats':         ['stats'],
    '--edit-config':   ['config', 'edit'],
    '--ui':            ['ui'],
    '--gui':           ['ui'],
}

LEGACY_VALUE_REDIRECTS = {
    '--delete-prompt':  'voice delete',
    '--rename-prompt':  'voice rename',
    '--preview-prompt': 'voice preview',
}


def _rewrite_legacy_flags(args):
    """Intercept old --flag-style arguments and rewrite to subcommands."""
    if not args:
        return args

    new_args = list(args)

    # Check for value-taking legacy flags first
    for flag, subcmd in LEGACY_VALUE_REDIRECTS.items():
        if flag in new_args:
            idx = new_args.index(flag)
            new_args.pop(idx)
            parts = subcmd.split()
            for i, part in enumerate(parts):
                new_args.insert(i, part)
            click.echo(f"Note: '{flag}' is now 'tts {subcmd}'. See 'tts --help'.", err=True)
            return new_args

    # Check for simple redirects
    for flag, subcmd in LEGACY_REDIRECTS.items():
        if flag in new_args:
            idx = new_args.index(flag)
            new_args.pop(idx)
            for i, part in enumerate(subcmd):
                new_args.insert(i, part)
            click.echo(f"Note: '{flag}' is now 'tts {' '.join(subcmd)}'. See 'tts --help'.", err=True)
            return new_args

    # --history N (optional value)
    if '--history' in new_args:
        idx = new_args.index('--history')
        new_args.pop(idx)
        value = None
        if idx < len(new_args) and not new_args[idx].startswith('-'):
            value = new_args.pop(idx)
        new_args.insert(0, 'history')
        if value:
            new_args.insert(1, value)
        click.echo("Note: '--history' is now 'tts history'. See 'tts --help'.", err=True)
        return new_args

    return new_args


# ---------------------------------------------------------------------------
# Custom Group class — routes bare `tts "Hello"` to generate
# ---------------------------------------------------------------------------

class TTSGroup(click.Group):
    """Routes bare `tts "Hello"` to the generate subcommand."""

    def parse_args(self, ctx, args):
        args = _rewrite_legacy_flags(args)
        # Strip --_server-mode before routing (it's a generate-level flag)
        # and stash it so we can re-insert after routing
        server_mode = False
        if '--_server-mode' in args:
            args = [a for a in args if a != '--_server-mode']
            server_mode = True
        # If first arg isn't a subcommand or flag, prepend 'generate'
        if args and args[0] not in self.commands and not args[0].startswith('-'):
            args = ['generate'] + args
        # If no args at all (bare `tts`), route to generate (interactive mode)
        if not args:
            args = ['generate']
        # Re-insert --_server-mode after the subcommand name
        if server_mode:
            # Find where the subcommand is and insert after it
            args.insert(1, '--_server-mode')
        return super().parse_args(ctx, args)


# ---------------------------------------------------------------------------
# Shared: translate Click kwargs → argparse sys.argv → call generate main()
# ---------------------------------------------------------------------------

_FLAG_MAP = {
    'mode': ('-m', str),
    'prompt': ('-p', str),
    'description': ('-d', str),
    'speaker': ('-s', str),
    'instruct': ('-i', str),
    'voice': ('-v', str),
    'prosody': ('--prosody', str),
    'no_transcript': ('--no-transcript', bool),
    'output': ('-o', str),
    'play': ('--play', bool),
    'stream': ('--stream', bool),
    'no_open': ('--no-open', bool),
    'speed': ('--speed', float),
    'pitch': ('--pitch', float),
    'trim_silence': ('--trim-silence', bool),
    'normalize': ('--normalize', bool),
    'preset': ('--preset', str),
    'temperature': ('--temperature', float),
    'top_k': ('--top-k', int),
    'top_p': ('--top-p', float),
    'seed': ('--seed', int),
    'repetition_penalty': ('--repetition-penalty', float),
    'max_chunk_chars': ('--max-chunk-chars', int),
    'max_new_tokens': ('--max-new-tokens', int),
    'clipboard': ('--clipboard', bool),
    'ssml': ('--ssml', bool),
    'local': ('--local', bool),
    'dry_run': ('--dry-run', bool),
    'backend': ('--backend', str),
    'model_size': ('--model-size', str),
    'server_mode': ('--_server-mode', bool),
    'text_override': ('--text-override', str),
    # Advanced mode flags
    'batch': ('--batch', str),
    'srt': ('--srt', str),
    'dialogue': ('--dialogue', str),
    'save_individual': ('--save-individual', bool),
    'repl_mode': ('--repl', bool),
    'watch_dir': ('--watch', str),
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
    sys.argv = ['tts'] + argv
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
@click.version_option(version='2.0.0', prog_name='Qwen3-TTS')
def cli():
    """Qwen3-TTS -- Text to speech with voice cloning."""
    pass


# ---------------------------------------------------------------------------
# Shared generation options decorator
# ---------------------------------------------------------------------------

def _generation_options(f):
    """Apply all generation-related options to a command."""
    decorators = [
        click.option('-m', '--mode', type=click.Choice(['clone', 'design', 'custom']),
                     help='Voice mode'),
        click.option('-p', '--prompt', help='Voice clone prompt filename'),
        click.option('-d', '--description', help='Voice description (design mode)'),
        click.option('-s', '--speaker', help='Premium speaker name (custom mode)'),
        click.option('-i', '--instruct', help='Style instruction (custom mode)'),
        click.option('-v', '--voice', help='Voice alias from config'),
        click.option('--prosody', help='Prosody preset (custom/design mode)'),
        click.option('--no-transcript', is_flag=True,
                     help='Clone using speaker embedding only'),
        click.option('-o', '--output', help='Output filename or directory'),
        click.option('--play', is_flag=True, help='Play audio after generation'),
        click.option('--stream', is_flag=True,
                     help='Stream audio playback as it generates'),
        click.option('--no-open', is_flag=True, help="Don't open the output file"),
        click.option('--speed', type=float, help='Speed factor (1.2=faster, 0.8=slower)'),
        click.option('--pitch', type=float, help='Pitch shift in semitones'),
        click.option('--trim-silence', is_flag=True, help='Trim leading/trailing silence'),
        click.option('--normalize', is_flag=True, help='Normalize audio to -3dB peak'),
        click.option('--preset', help='Named preset from config'),
        click.option('--temperature', type=float, help='Sampling temperature'),
        click.option('--top-k', type=int, help='Top-k sampling'),
        click.option('--top-p', type=float, help='Top-p (nucleus) sampling'),
        click.option('--seed', type=int, help='Random seed'),
        click.option('--repetition-penalty', type=float, help='Repetition penalty'),
        click.option('--max-chunk-chars', type=int,
                     help='Max chars per chunk (0=disable)'),
        click.option('--max-new-tokens', type=int, help='Max new tokens per chunk'),
        click.option('--clipboard', is_flag=True, help='Read text from clipboard'),
        click.option('--ssml', is_flag=True, help='Enable SSML markup parsing'),
        click.option('--local', is_flag=True, help='Force local generation (skip server)'),
        click.option('--dry-run', is_flag=True,
                     help='Show what would be generated without running'),
        click.option('--backend', type=click.Choice(['torch', 'mlx']),
                     help='Override backend for this run'),
        click.option('--model-size', type=click.Choice(['1.7B', '0.6B']),
                     help='Override model size for this run'),
        click.option('--_server-mode', 'server_mode', is_flag=True, hidden=True),
        click.option('--text-override', hidden=True),
    ]
    for decorator in reversed(decorators):
        f = decorator(f)
    return f


# ---------------------------------------------------------------------------
# generate — the default command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument('text', nargs=-1)
@_generation_options
def generate(text, **kwargs):
    """Generate audio from text.

    This is the default command — you can omit 'generate':

    \b
      tts "Hello, world!" -o hello
    """
    _call_generate(text, **kwargs)


# ---------------------------------------------------------------------------
# server group
# ---------------------------------------------------------------------------

@cli.group()
def server():
    """Manage the TTS server."""
    pass


@server.command()
@click.option('--public', is_flag=True, help='Bind to 0.0.0.0')
def start(public):
    """Start the TTS server."""
    if public:
        os.environ['TTS_SERVER_PUBLIC'] = '1'
    sys.exit(10)


@server.command()
def stop():
    """Stop the TTS server."""
    sys.exit(11)


@server.command()
def status():
    """Show server health, loaded models, and memory usage."""
    from qwen3_tts.core.config import load_config, get_server_url, is_server_running, auth_headers
    import json
    import requests

    config = load_config()
    if not is_server_running(config):
        click.echo("TTS Server is not running.")
        sys.exit(1)

    url = get_server_url(config)
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        health = resp.json()
        click.echo(f"Server: running ({url})")
        click.echo(f"Backend: {health.get('backend', 'unknown')}")

        resp = requests.get(f"{url}/models", headers=auth_headers(), timeout=5)
        models = resp.json()
        click.echo("\nModels:")
        for name, info in models.items():
            status_str = "loaded" if info.get("loaded") else "not loaded"
            click.echo(f"  {name}: {status_str}")

        resp = requests.get(f"{url}/stats", headers=auth_headers(), timeout=5)
        stats = resp.json()
        mem_key = next((k for k in stats if 'memory' in k.lower() and 'mb' in k.lower()), None)
        if mem_key:
            click.echo(f"\nMemory: {stats[mem_key]}MB")
    except Exception as e:
        click.echo(f"Error connecting to server: {e}")
        sys.exit(1)


@server.command()
def log():
    """Tail the server log."""
    sys.exit(12)


# ---------------------------------------------------------------------------
# voice group
# ---------------------------------------------------------------------------

@cli.group()
def voice():
    """Manage voice prompts."""
    pass


@voice.command('list')
def voice_list():
    """List available voice prompts."""
    from qwen3_tts.interface.generate import list_voice_prompts
    from qwen3_tts.core.config import get_default_clone_prompt
    prompts = list_voice_prompts()
    if not prompts:
        click.echo("No voice prompts found.")
        return
    click.echo("Available voice prompts:")
    default = get_default_clone_prompt()
    for p in prompts:
        marker = " (default)" if p == default else ""
        click.echo(f"  {p}{marker}")


@voice.command()
@click.argument('audio', required=False)
@click.option('-n', '--name', help='Name for the voice prompt')
@click.option('-t', '--transcript', help='Transcript text')
@click.option('--mlx-only', is_flag=True, help='Save only MLX format (no torch needed)')
@click.option('--no-transcript', is_flag=True, help='Skip transcript (use speaker embedding only)')
@click.option('--auto-transcribe', is_flag=True, help='Auto-transcribe with ASR')
def create(audio, name, transcript, mlx_only, no_transcript, auto_transcribe):
    """Create a voice clone from reference audio."""
    argv = []
    if audio:
        argv.append(audio)
    if name:
        argv.append(name)
    if transcript:
        argv.extend(['-t', transcript])
    if mlx_only:
        argv.append('--mlx-only')
    if no_transcript:
        argv.append('--no-transcript')
    if auto_transcribe:
        argv.append('--auto-transcribe')

    old_argv = sys.argv
    sys.argv = ['tts-voice-create'] + argv
    try:
        from qwen3_tts.tools.create_voice import main as _create_main
        _create_main()
    finally:
        sys.argv = old_argv


@voice.command()
@click.argument('name')
def delete(name):
    """Delete a voice prompt."""
    from qwen3_tts.interface.generate import delete_voice_prompt
    delete_voice_prompt(name)


@voice.command()
@click.argument('old_name')
@click.argument('new_name')
def rename(old_name, new_name):
    """Rename a voice prompt."""
    from qwen3_tts.interface.generate import rename_voice_prompt
    rename_voice_prompt(old_name, new_name)


@voice.command()
@click.argument('name')
def preview(name):
    """Play a voice prompt sample."""
    from qwen3_tts.core.config import load_config
    from qwen3_tts.interface.generate import preview_voice_prompt
    preview_voice_prompt(name, load_config())


@voice.command()
@click.argument('name')
def info(name):
    """Show voice prompt metadata."""
    from qwen3_tts.core.config import load_config, get_server_url, is_server_running, auth_headers
    import json
    import requests

    config = load_config()
    if not is_server_running(config):
        click.echo("Server not running. Start with: tts server start")
        sys.exit(1)
    url = get_server_url(config)
    resp = requests.get(f"{url}/prompt-details", params={"name": name},
                        headers=auth_headers(), timeout=10)
    if resp.status_code == 200:
        click.echo(json.dumps(resp.json(), indent=2))
    else:
        click.echo(f"Error: {resp.json().get('error', resp.text)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# list group
# ---------------------------------------------------------------------------

@cli.group('list')
def list_group():
    """List resources (speakers, presets, aliases, etc.)."""
    pass


@list_group.command()
def speakers():
    """List premium CustomVoice speakers."""
    from qwen3_tts.core.config import CUSTOM_VOICE_SPEAKERS
    click.echo("Premium CustomVoice speakers:")
    for key, info in CUSTOM_VOICE_SPEAKERS.items():
        click.echo(f"  {key:12s} ({info['lang']}) - {info['desc']}")


@list_group.command()
def presets():
    """List generation presets from config."""
    from qwen3_tts.core.config import load_config
    config = load_config()
    preset_dict = config.get("presets", {})
    if not preset_dict:
        click.echo("No presets configured.")
        return
    click.echo("Generation presets:")
    for name, params in preset_dict.items():
        parts = [f"{k}={v}" for k, v in params.items()]
        click.echo(f"  {name}: {', '.join(parts)}")


@list_group.command()
def aliases():
    """List voice aliases from config."""
    from qwen3_tts.core.config import load_config
    config = load_config()
    alias_dict = config.get("aliases", {})
    if not alias_dict:
        click.echo("No aliases configured.")
        return
    click.echo("Voice aliases:")
    for name, alias_info in alias_dict.items():
        parts = []
        if "prompt" in alias_info:
            parts.append(f"prompt={alias_info['prompt']}")
        if "preset" in alias_info:
            parts.append(f"preset={alias_info['preset']}")
        if "mode" in alias_info:
            parts.append(f"mode={alias_info['mode']}")
        click.echo(f"  {name}: {', '.join(parts)}")


@list_group.command()
def prosody():
    """List prosody presets."""
    from qwen3_tts.core.config import get_prosody_presets
    presets = get_prosody_presets()
    if not presets:
        click.echo("No prosody presets configured.")
        return
    click.echo("Prosody presets:")
    for name, text in sorted(presets.items()):
        click.echo(f"  {name:12s} {text}")


@list_group.command()
def models():
    """List TTS models and their load status."""
    from qwen3_tts.core.config import (
        load_config, get_server_url, is_server_running, auth_headers,
        get_backend, MODEL_INFO, get_model_size
    )
    import requests

    config = load_config()
    backend = get_backend()
    size = get_model_size()
    click.echo(f"Backend: {backend}, Model size: {size}")

    if is_server_running(config):
        url = get_server_url(config)
        try:
            resp = requests.get(f"{url}/models", headers=auth_headers(), timeout=5)
            data = resp.json()
            click.echo("\nServer models:")
            for name, model_info in data.items():
                status_str = "loaded" if model_info.get("loaded") else "not loaded"
                mem = model_info.get("memory_mb")
                mem_str = f" ({mem}MB)" if mem else ""
                click.echo(f"  {name}: {status_str}{mem_str}")
        except Exception:
            click.echo("\nCould not reach server for live status.")
    else:
        click.echo("\nServer not running — showing configured models:")
        for mt in ("clone", "design", "custom"):
            click.echo(f"  {mt}: {MODEL_INFO.get(mt, {}).get('name', 'unknown')}")


@list_group.command()
def backends():
    """List available backends and current setting."""
    from qwen3_tts.core.config import (
        get_backend, VALID_BACKENDS, get_mlx_quantization,
        get_torch_dtype_name, get_mlx_model_name, MODEL_INFO
    )
    current = get_backend()
    click.echo(f"Available backends: {', '.join(VALID_BACKENDS)}")
    click.echo(f"Current backend:    {current}")
    click.echo()
    if current == "mlx":
        quant = get_mlx_quantization()
        click.echo(f"  MLX quantization: {quant}")
        for mt in ("clone", "design", "custom"):
            click.echo(f"  {mt}: {get_mlx_model_name(mt)}")
    else:
        dtype = get_torch_dtype_name()
        click.echo(f"  PyTorch dtype: {dtype}")
        for mt, model_info in MODEL_INFO.items():
            click.echo(f"  {mt}: {model_info['name']}")


# ---------------------------------------------------------------------------
# config group
# ---------------------------------------------------------------------------

@cli.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """Configure TTS settings."""
    if ctx.invoked_subcommand is None:
        import subprocess  # nosec B404
        wizard = os.path.expanduser('~/Qwen3-TTS_UserFiles/install.sh')
        subprocess.run([wizard, '--reconfigure'])  # nosec B603


@config.command()
def show():
    """Show current configuration."""
    from qwen3_tts.core.config import load_config
    import json
    click.echo(json.dumps(load_config(), indent=2))


@config.command()
def edit():
    """Edit default voice description."""
    from qwen3_tts.core.config import load_config, save_config
    config_data = load_config()
    current = config_data.get("default_voice_description", "")
    click.echo(f"Current voice description:\n  {current}\n")
    new_desc = click.prompt("New description (or Enter to keep)", default=current)
    if new_desc != current:
        config_data["default_voice_description"] = new_desc
        save_config(config_data)
        click.echo("Voice description updated.")
    else:
        click.echo("No changes.")


@config.command()
def path():
    """Print the config.json path."""
    from qwen3_tts.core.config import CONFIG_PATH
    click.echo(CONFIG_PATH)


# ---------------------------------------------------------------------------
# ui command
# ---------------------------------------------------------------------------

@cli.command()
@click.option('--port', type=int, help='Port number (default: 7860)')
@click.option('--share', is_flag=True, help='Create public URL via Gradio')
@click.option('--no-browser', is_flag=True, help="Don't open browser automatically")
def ui(port, share, no_browser):
    """Launch the Gradio web interface."""
    from qwen3_tts.core.config import load_config
    from qwen3_tts.interface.generate import launch_gradio_ui
    if port:
        os.environ['TTS_UI_PORT'] = str(port)
    if share:
        os.environ['TTS_UI_SHARE'] = '1'
    if no_browser:
        os.environ['TTS_UI_NO_BROWSER'] = '1'
    launch_gradio_ui(load_config())


# ---------------------------------------------------------------------------
# history / stats
# ---------------------------------------------------------------------------

@cli.command()
@click.argument('count', default=10, type=int, required=False)
def history(count):
    """Show last N generations (default: 10)."""
    from qwen3_tts.interface.generate import show_history
    show_history(count)


@cli.command()
def stats():
    """Show server statistics."""
    from qwen3_tts.core.config import load_config, get_server_url, is_server_running, auth_headers
    import json
    import requests

    config = load_config()
    if not is_server_running(config):
        click.echo("Server not running. Start with: tts server start")
        sys.exit(1)
    url = get_server_url(config)
    resp = requests.get(f"{url}/stats", headers=auth_headers(), timeout=10)
    if resp.status_code == 200:
        click.echo(json.dumps(resp.json(), indent=2))
    else:
        click.echo(f"Error: {resp.text}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Batch / advanced modes
# ---------------------------------------------------------------------------

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@_generation_options
def batch(file, **kwargs):
    """Process batch JSON file with array of texts."""
    _call_generate(batch=file, **kwargs)


@cli.command()
@click.argument('file', type=click.Path(exists=True))
@_generation_options
def srt(file, **kwargs):
    """Process SRT subtitle file."""
    _call_generate(srt=file, **kwargs)


@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--save-individual', is_flag=True,
              help='Save individual audio files for each line')
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
@click.argument('directory', type=click.Path(exists=True))
@_generation_options
def watch(directory, **kwargs):
    """Watch directory for .txt files and generate audio."""
    _call_generate(watch_dir=directory, **kwargs)


if __name__ == '__main__':
    cli()
