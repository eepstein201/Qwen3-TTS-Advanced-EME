"""Config, uninstall, cache, and utility CLI commands.

Extracted from cli.py to keep each module under 800 lines.
Contains: config group, uninstall group, cache group, doctor, ui, history.
"""

import os
import sys

import click

from qwen3_tts.core.config import (
    VALID_MLX_QUANTIZATIONS as _VALID_MLX_Q,
)
from qwen3_tts.core.config import (
    VALID_MODEL_SIZES as _VALID_MODEL_SIZES,
)

# ---------------------------------------------------------------------------
# config group
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """Configure TTS settings."""
    if ctx.invoked_subcommand is None:
        import subprocess  # nosec B404

        from qwen3_tts.core.config import USER_FILES_DIR

        wizard = os.path.join(USER_FILES_DIR, "install.sh")
        subprocess.run([wizard, "--reconfigure"], timeout=300)  # nosec B603


@config.command()
def show():
    """Show current configuration."""
    import json

    from qwen3_tts.core.config import load_config

    click.echo(json.dumps(load_config(), indent=2))


@config.command()
@click.option(
    "--backend",
    type=click.Choice(["mlx", "torch", "vllm"]),
    help="Set backend (mlx/torch/vllm)",
)
@click.option(
    "--model-size", type=click.Choice(list(_VALID_MODEL_SIZES)), help="Set model size"
)
@click.option(
    "--mlx-quantization",
    type=click.Choice(list(_VALID_MLX_Q)),
    help="Set MLX quantization (4bit/5bit/6bit/8bit/bf16)",
)
@click.option(
    "--torch-quantization",
    type=click.Choice(["none", "4bit", "8bit"]),
    help="Set Torch quantization",
)
@click.option("--language", help="Set default language")
@click.option("--output-dir", help="Set output directory for generated audio")
@click.option("--voice-description", help="Set default voice description")
def edit(
    backend,
    model_size,
    mlx_quantization,
    torch_quantization,
    language,
    output_dir,
    voice_description,
):
    """Edit TTS configuration settings.

    Without options, opens interactive editor for voice description.
    With options, sets specific values directly.

    Examples:
        tts config edit --backend mlx
        tts config edit --model-size 0.6B --mlx-quantization 4bit
        tts config edit --language Spanish
    """
    from qwen3_tts.core.config import load_config, save_config

    config_data = load_config()

    # Track if any changes were made
    changes = []

    # Build immutable updates — never mutate config_data in-place
    adv = dict(config_data.get("advanced", {}))
    top = dict(config_data)

    # Handle backend
    if backend:
        old = adv.get("backend")
        if backend != old:
            adv = {**adv, "backend": backend}
            changes.append(f"backend: {old} → {backend}")

    # Handle model size
    if model_size:
        old = adv.get("model_size")
        if model_size != old:
            adv = {**adv, "model_size": model_size}
            changes.append(f"model_size: {old} → {model_size}")

    # Handle MLX quantization
    if mlx_quantization:
        old = adv.get("mlx_quantization")
        if mlx_quantization != old:
            adv = {**adv, "mlx_quantization": mlx_quantization}
            changes.append(f"mlx_quantization: {old} → {mlx_quantization}")

    # Handle Torch quantization
    if torch_quantization:
        old = adv.get("torch_quantization")
        if torch_quantization != old:
            adv = {**adv, "torch_quantization": torch_quantization}
            changes.append(f"torch_quantization: {old} → {torch_quantization}")

    # Handle language
    if language:
        old = top.get("language")
        if language != old:
            top = {**top, "language": language}
            changes.append(f"language: {old} → {language}")

    # Handle output directory
    if output_dir:
        old = top.get("output_directory")
        if output_dir != old:
            top = {**top, "output_directory": output_dir}
            changes.append(f"output_directory: {old} → {output_dir}")

    # Handle voice description (direct setting or interactive)
    if voice_description:
        old = top.get("default_voice_description")
        if voice_description != old:
            top = {**top, "default_voice_description": voice_description}
            changes.append("default_voice_description updated")

    # If no options provided, fall back to interactive voice description editor
    if not any(
        [
            backend,
            model_size,
            mlx_quantization,
            torch_quantization,
            language,
            output_dir,
            voice_description,
        ]
    ):
        current = config_data.get("default_voice_description", "")
        click.echo(f"Current voice description:\n  {current}\n")
        new_desc = click.prompt("New description (or Enter to keep)", default=current)
        if new_desc != current:
            save_config({**config_data, "default_voice_description": new_desc})
            click.echo("Voice description updated.")
        else:
            click.echo("No changes.")
        return

    # Save if there were changes
    if changes:
        updated = {**top, "advanced": adv}
        save_config(updated)
        click.echo("Configuration updated:")
        for change in changes:
            click.echo(f"  • {change}")
    else:
        click.echo("No changes.")


@config.command()
def path():
    """Print the config.json path."""
    from qwen3_tts.core.config import CONFIG_PATH

    click.echo(CONFIG_PATH)


# ---------------------------------------------------------------------------
# ui command (standalone, registered by cli.py)
# ---------------------------------------------------------------------------


def ui_command(port, share, no_browser):
    """Launch the Gradio web interface."""
    from qwen3_tts.core.config import load_config
    from qwen3_tts.interface.generate import launch_gradio_ui

    if port:
        os.environ["TTS_UI_PORT"] = str(port)
    if share:
        os.environ["TTS_UI_SHARE"] = "1"
    if no_browser:
        os.environ["TTS_UI_NO_BROWSER"] = "1"
    launch_gradio_ui(load_config())


# ---------------------------------------------------------------------------
# history command (standalone, registered by cli.py)
# ---------------------------------------------------------------------------


def history_command(count):
    """Show last N generations (default: 10)."""
    from qwen3_tts.interface.generate import show_history

    show_history(count)


# ---------------------------------------------------------------------------
# uninstall group
# ---------------------------------------------------------------------------


@click.group()
def uninstall():
    """Uninstall and clean up TTS components."""
    pass


@uninstall.command("models")
@click.option("--dry-run", is_flag=True, help="Preview changes without deleting")
def uninstall_models_cmd(dry_run):
    """Remove cached TTS models from HuggingFace cache."""
    from qwen3_tts.tools.uninstall import uninstall_models

    uninstall_models(dry_run=dry_run)


@uninstall.command()
@click.option("--dry-run", is_flag=True, help="Preview changes without deleting")
def voices(dry_run):
    """Remove all voice prompts."""
    from qwen3_tts.tools.uninstall import uninstall_voices

    uninstall_voices(dry_run=dry_run)


@uninstall.command("config")
@click.option("--dry-run", is_flag=True, help="Preview changes without deleting")
def config_cmd(dry_run):
    """Reset config.json to defaults."""
    from qwen3_tts.tools.uninstall import uninstall_config

    uninstall_config(dry_run=dry_run)


@uninstall.command()
def environment():
    """Print conda environment removal commands (does NOT execute)."""
    from qwen3_tts.tools.uninstall import print_environment_instructions

    print_environment_instructions()


@uninstall.command("all")
@click.option("--dry-run", is_flag=True, help="Preview changes without deleting")
def all_cmd(dry_run):
    """Run all uninstall steps except conda environments."""
    from qwen3_tts.tools.uninstall import uninstall_all

    uninstall_all(dry_run=dry_run)


# ---------------------------------------------------------------------------
# cache group
# ---------------------------------------------------------------------------


@click.group()
def cache():
    """Manage TTS model cache."""
    pass


@cache.command(name="list")  # Explicit CLI command name preserves "tts cache list"
def list_cmd():  # Python function name avoids shadowing built-in list()
    """List all cached models."""
    from qwen3_tts.tools.model_cache import list_models_cmd

    list_models_cmd()


@cache.command()
def size():
    """Show total cache size."""
    from qwen3_tts.tools.model_cache import get_size_cmd

    get_size_cmd()


@cache.command()
@click.option(
    "--unused",
    type=int,
    default=30,
    help="Remove models not accessed in N days (default: 30)",
)
def prune(unused):
    """Remove models not used in N days."""
    from qwen3_tts.tools.model_cache import prune_models_cmd

    prune_models_cmd(days=unused, dry_run=False)


@cache.command()
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def clear(force):
    """Remove all cached models."""
    from qwen3_tts.tools.model_cache import clear_cache_cmd

    clear_cache_cmd(force=force)


# ---------------------------------------------------------------------------
# doctor / healthcheck
# ---------------------------------------------------------------------------


def doctor_command():
    """Check TTS installation health."""
    from qwen3_tts.tools.healthcheck import run_healthcheck

    sys.exit(run_healthcheck())
