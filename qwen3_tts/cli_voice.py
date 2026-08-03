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


def _torch_available():
    """Check whether the torch backend (qwen_tts package) is installed.

    Uses find_spec so the check does not import the heavy package.
    """
    import importlib.util

    return importlib.util.find_spec("qwen_tts") is not None


def _is_pt_valid(pt_path):
    """Check whether a .pt voice prompt loads with safe deserialization.

    Registers VoiceClonePromptItem in torch safe globals (same as
    _load_pt_safe in voice_prompt.py) — without it every valid prompt
    fails weights_only loading and gets needlessly rebuilt.
    """
    import torch
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    torch.serialization.add_safe_globals([VoiceClonePromptItem])
    try:
        torch.load(pt_path, weights_only=True, map_location="cpu")
        return True
    except Exception:
        return False


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
    "--force-torch",
    is_flag=True,
    help="Force .pt creation (torch) even when MLX backend is active",
)
@click.option(
    "--no-transcript", is_flag=True, help="Skip transcript (use speaker embedding only)"
)
@click.option("--auto-transcribe", is_flag=True, help="Auto-transcribe with ASR")
def create(audio, name, transcript, mlx_only, force_torch, no_transcript, auto_transcribe):
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
    if force_torch:
        argv.append("--force-torch")
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


def _require_voice_prompts_dir(voice_prompts_dir):
    """Exit if the voice_prompts directory does not exist."""
    import os

    if not os.path.isdir(voice_prompts_dir):
        click.echo(f"Voice prompts directory not found: {voice_prompts_dir}")
        sys.exit(1)


def _resolve_rebuild_targets(name, voice_prompts_dir):
    """Determine which prompt basenames are candidates for rebuild."""
    import os

    if name:
        base = name.removesuffix(".pt").removesuffix(".wav").removesuffix(".txt")
        return [base]
    return sorted(
        {
            f.removesuffix(".wav")
            for f in os.listdir(voice_prompts_dir)
            if f.endswith(".wav")
        }
    )


def _ensure_torch_backend_for_rebuild():
    """Force the torch backend for .pt creation, exiting if torch is unavailable.

    .pt prompts can only be created (or even validated) with the torch backend —
    both create_voice_clone_prompt and the safe-globals registration in
    _is_pt_valid need the qwen_tts package, which has no MLX equivalent. Check
    before scanning so an MLX-only env fails fast instead of misclassifying
    valid prompts as corrupt.

    Returns the prior TTS_BACKEND env value (or None) so it can be restored.
    """
    import os

    from qwen3_tts.core.config import get_backend

    backend_env_before = os.environ.get("TTS_BACKEND")
    if get_backend() != "torch":
        if not _torch_available():
            click.echo(
                "Rebuilding .pt prompts requires the torch backend, but the "
                "qwen_tts package is not installed in this environment.\n"
                "Re-run in the torch environment:\n"
                "  conda run -n qwen3-tts tts voice rebuild",
                err=True,
            )
            sys.exit(1)
        click.echo("Forcing torch backend (.pt prompts cannot be built with MLX).")
        os.environ["TTS_BACKEND"] = "torch"
    return backend_env_before


def _restore_backend_env(backend_env_before):
    """Restore TTS_BACKEND to its pre-rebuild value."""
    import os

    if backend_env_before is None:
        os.environ.pop("TTS_BACKEND", None)
    else:
        os.environ["TTS_BACKEND"] = backend_env_before


def _classify_rebuild_targets(targets, voice_prompts_dir):
    """Partition targets into those needing rebuild vs already-valid/skippable."""
    import os

    to_rebuild = []
    skipped = 0
    for base in targets:
        wav_path = os.path.join(voice_prompts_dir, f"{base}.wav")
        pt_path = os.path.join(voice_prompts_dir, f"{base}.pt")
        if not os.path.exists(wav_path):
            click.echo(f"  {base}: no .wav file, skipping")
            skipped += 1
            continue
        if os.path.exists(pt_path) and _is_pt_valid(pt_path):
            click.echo(f"  {base}: .pt valid, skipping")
            skipped += 1
            continue
        to_rebuild.append(base)
    return to_rebuild, skipped


def _rebuild_voice_prompts(to_rebuild, voice_prompts_dir):
    """Load the clone model once and rebuild each target .pt prompt."""
    import os

    click.echo(f"\nLoading clone model to rebuild {len(to_rebuild)} prompt(s)...")
    from qwen3_tts.core.engine.audio_processing import load_audio_for_cloning
    from qwen3_tts.core.engine.inference import create_voice_prompt
    from qwen3_tts.core.engine.model_loader import load_model

    model = load_model("clone")

    rebuilt = 0
    failed = 0
    for base in to_rebuild:
        wav_path = os.path.join(voice_prompts_dir, f"{base}.wav")
        txt_path = os.path.join(voice_prompts_dir, f"{base}.txt")
        pt_path = os.path.join(voice_prompts_dir, f"{base}.pt")
        transcript = ""
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                transcript = f.read().strip()
        try:
            import torch

            ref_audio, ref_sr = load_audio_for_cloning(wav_path)
            voice_prompt = create_voice_prompt(
                model, ref_audio, ref_sr, transcript, x_vector_only_mode=not transcript
            )
            torch.save(voice_prompt, pt_path)
            click.echo(f"  {base}: rebuilt ({os.path.getsize(pt_path)} bytes)")
            rebuilt += 1
        except Exception as e:
            click.echo(f"  {base}: FAILED — {e}", err=True)
            failed += 1
    return rebuilt, failed


@voice.command()
@click.argument("name", required=False)
def rebuild(name):
    """Rebuild .pt voice prompts from .wav+.txt pairs.

    Loads the clone model once and regenerates all corrupt or missing .pt files.
    Run before uploading voice_prompts/ to Colab to avoid OOM from the server-side
    fallback loading a second model instance.

    With NAME, rebuild only that prompt. Without, rebuild all.
    """
    from qwen3_tts.core.config import VOICE_PROMPTS_DIR

    _require_voice_prompts_dir(VOICE_PROMPTS_DIR)
    targets = _resolve_rebuild_targets(name, VOICE_PROMPTS_DIR)

    if not targets:
        click.echo("No .wav files found in voice_prompts/.")
        return

    backend_env_before = _ensure_torch_backend_for_rebuild()

    try:
        to_rebuild, skipped = _classify_rebuild_targets(targets, VOICE_PROMPTS_DIR)

        if not to_rebuild:
            click.echo(f"All {skipped} prompt(s) already valid.")
            return

        rebuilt, failed = _rebuild_voice_prompts(to_rebuild, VOICE_PROMPTS_DIR)
    finally:
        _restore_backend_env(backend_env_before)

    click.echo(f"\nDone: {rebuilt} rebuilt, {skipped} skipped, {failed} failed.")
    if failed:
        sys.exit(1)


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
