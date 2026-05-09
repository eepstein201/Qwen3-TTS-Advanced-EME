"""Voice and list CLI commands.

Extracted from cli.py to keep each module under 800 lines.
Contains: voice group (list, create, delete, rename, preview, info)
and list group (speakers, presets, aliases, prosody, models, backends).
"""

import sys

import click

# ---------------------------------------------------------------------------
# voice group
# ---------------------------------------------------------------------------


@click.group()
def voice():
    """Manage voice prompts."""
    pass


@voice.command("list")
def voice_list():
    """List available voice prompts."""
    from qwen3_tts.core.config import get_default_clone_prompt
    from qwen3_tts.interface.generate import list_voice_prompts

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
@click.argument("audio", required=False)
@click.option("-n", "--name", help="Name for the voice prompt")
@click.option("-t", "--transcript", help="Transcript text")
@click.option("--mlx-only", is_flag=True, help="Save only MLX format (no torch needed)")
@click.option(
    "--no-transcript", is_flag=True, help="Skip transcript (use speaker embedding only)"
)
@click.option("--auto-transcribe", is_flag=True, help="Auto-transcribe with ASR")
def create(audio, name, transcript, mlx_only, no_transcript, auto_transcribe):
    """Create a voice clone from reference audio."""
    argv = []
    if audio:
        argv.append(audio)
    if name:
        argv.extend(["-n", name])
    if transcript:
        argv.extend(["-t", transcript])
    if mlx_only:
        argv.append("--mlx-only")
    if no_transcript:
        argv.append("--no-transcript")
    if auto_transcribe:
        argv.append("--auto-transcribe")

    old_argv = sys.argv
    sys.argv = ["tts-voice-create"] + argv
    try:
        from qwen3_tts.tools.create_voice import main as _create_main

        _create_main()
    finally:
        sys.argv = old_argv


@voice.command()
@click.argument("name")
def delete(name):
    """Delete a voice prompt."""
    from qwen3_tts.interface.generate import delete_voice_prompt

    delete_voice_prompt(name)


@voice.command()
@click.argument("old_name")
@click.argument("new_name")
def rename(old_name, new_name):
    """Rename a voice prompt."""
    from qwen3_tts.interface.generate import rename_voice_prompt

    rename_voice_prompt(old_name, new_name)


@voice.command()
@click.argument("name")
def preview(name):
    """Play a voice prompt sample."""
    from qwen3_tts.core.config import load_config
    from qwen3_tts.interface.generate import preview_voice_prompt

    preview_voice_prompt(name, load_config())


@voice.command()
@click.argument("name")
def info(name):
    """Show voice prompt metadata."""
    import json

    from qwen3_tts.core.config import is_server_running, load_config
    from qwen3_tts.core.http_client import server_request

    config = load_config()
    if not is_server_running(config):
        click.echo("Server not running. Start with: tts server start")
        sys.exit(1)
    resp = server_request("GET", "/prompt-details", params={"name": name}, timeout=10)
    if resp.status_code == 200:
        click.echo(json.dumps(resp.json(), indent=2))
    else:
        click.echo(f"Error: {resp.json().get('error', resp.text)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# list group
# ---------------------------------------------------------------------------


@click.group("list")
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
    import requests

    from qwen3_tts.core.config import (
        MODEL_INFO,
        get_backend,
        get_model_size,
        is_server_running,
        load_config,
    )
    from qwen3_tts.core.http_client import server_request

    config = load_config()
    backend = get_backend()
    size = get_model_size()
    click.echo(f"Backend: {backend}, Model size: {size}")

    if is_server_running(config):
        try:
            resp = server_request("GET", "/models", timeout=5)
            data = resp.json()
            click.echo("\nServer models:")
            for name, model_info in data.items():
                status_str = "loaded" if model_info.get("loaded") else "not loaded"
                mem = model_info.get("memory_mb")
                mem_str = f" ({mem}MB)" if mem else ""
                click.echo(f"  {name}: {status_str}{mem_str}")
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.RequestException,
            ValueError,
            KeyError,
        ):
            click.echo("\nCould not reach server for live status.")
    else:
        click.echo("\nServer not running — showing configured models:")
        for mt in ("clone", "design", "custom"):
            info = MODEL_INFO.get(size, {}).get(mt, {})
            click.echo(f"  {mt}: {info.get('name', 'unknown')}")


@list_group.command()
def backends():
    """List available backends and current setting."""
    from qwen3_tts.core.config import (
        MODEL_INFO,
        VALID_BACKENDS,
        get_backend,
        get_mlx_model_name,
        get_mlx_quantization,
        get_torch_dtype_name,
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
