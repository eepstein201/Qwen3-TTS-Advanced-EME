#!/usr/bin/env python3
"""TTS model cache management utilities.

Provides commands to list, analyze, and clean up cached TTS models
from the HuggingFace cache. Invoked via `tts cache` namespace.
"""

import pathlib
import shutil
import sys
from datetime import datetime, timedelta

import click

from qwen3_tts.core.config import HF_CACHE
from qwen3_tts.tools._shared import _format_size


# Model name patterns to identify TTS models
_TORCH_MODEL_PREFIXES = (
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base",
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base",
    "models--Qwen--Qwen3-TTS-12Hz-0.6B-VoiceDesign",
    "models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice",
)

_MLX_MODEL_PREFIXES = (
    "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-",
    "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-",
    "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-",
    "models--mlx-community--Qwen3-TTS-12Hz-0.6B-Base-",
    "models--mlx-community--Qwen3-TTS-12Hz-0.6B-VoiceDesign-",
    "models--mlx-community--Qwen3-TTS-12Hz-0.6B-CustomVoice-",
)

_MODEL_ALIASES = {
    "Qwen3-TTS-12Hz-1.7B-Base": {"torch": "clone", "mlx": "clone"},
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign": {"torch": "design", "mlx": "design"},
    "Qwen3-TTS-12Hz-1.7B-CustomVoice": {"torch": "custom", "mlx": "custom"},
    "Qwen3-TTS-12Hz-0.6B-Base": {"torch": "clone", "mlx": "clone"},
    "Qwen3-TTS-12Hz-0.6B-VoiceDesign": {"torch": "design", "mlx": "design"},
    "Qwen3-TTS-12Hz-0.6B-CustomVoice": {"torch": "custom", "mlx": "custom"},
}


def _get_model_dir_size(model_dir: pathlib.Path) -> int:
    """Calculate total size of a model directory in bytes."""
    total = 0
    if model_dir.is_dir():
        for item in model_dir.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    pass
    return total


def _get_model_access_time(model_dir: pathlib.Path) -> datetime:
    """Get the last access time of a model directory."""
    # Check various files that might have access time
    for file_name in ("pytorch_model.bin", "model.safetensors", "config.json"):
        file_path = model_dir / file_name
        if file_path.exists():
            return datetime.fromtimestamp(file_path.stat().st_atime)

    # Fallback to directory mtime
    if model_dir.exists():
        return datetime.fromtimestamp(model_dir.stat().st_mtime)

    return datetime.min


def _get_model_info(model_dir: pathlib.Path) -> dict:
    """Get detailed information about a cached model."""
    name = model_dir.name
    size_bytes = _get_model_dir_size(model_dir)
    access_time = _get_model_access_time(model_dir)

    # Determine model type and backend
    backend = None
    model_type = None
    model_size = None
    quant = None

    if name.startswith("models--Qwen--"):
        backend = "torch"
        for alias, info in _MODEL_ALIASES.items():
            if alias in name:
                model_type = info["torch"]
                if "1.7B" in name:
                    model_size = "1.7B"
                elif "0.6B" in name:
                    model_size = "0.6B"
                break
    elif name.startswith("models--mlx-community--"):
        backend = "mlx"
        # Parse quantization from name (4bit, 8bit, bf16)
        if "-4bit-" in name:
            quant = "4bit"
        elif "-bf16-" in name:
            quant = "bf16"
        elif "-8bit-" in name:
            quant = "8bit"
        else:
            quant = "unknown"

        for alias, info in _MODEL_ALIASES.items():
            if alias in name:
                model_type = info["mlx"]
                if "1.7B" in name:
                    model_size = "1.7B"
                elif "0.6B" in name:
                    model_size = "0.6B"
                break

    return {
        "name": name,
        "size_bytes": size_bytes,
        "size_formatted": _format_size(size_bytes),
        "last_access": access_time,
        "backend": backend,
        "model_type": model_type,
        "model_size": model_size,
        "quantization": quant,
        "path": str(model_dir),
    }


def list_models() -> list[dict]:
    """List all cached TTS models with detailed information."""
    models = []
    if not HF_CACHE.exists():
        return models

    for model_dir in HF_CACHE.iterdir():
        # Check if this is a TTS model
        is_tts_model = (
            model_dir.name.startswith(_TORCH_MODEL_PREFIXES) or
            model_dir.name.startswith(_MLX_MODEL_PREFIXES)
        )

        if is_tts_model and model_dir.is_dir():
            info = _get_model_info(model_dir)
            models.append(info)

    # Sort by last access time (oldest first)
    models.sort(key=lambda m: m["last_access"])

    return models


def get_total_size() -> int:
    """Calculate total size of all cached TTS models."""
    total = 0
    models = list_models()
    for model in models:
        total += model["size_bytes"]
    return total


def list_models_cmd() -> None:
    """List all cached models in a formatted table."""
    models = list_models()

    if not models:
        click.echo("  No TTS models found in cache.")
        return

    click.echo()
    click.echo(f"  Found {len(models)} cached TTS model(s):")
    click.echo()

    # Table header
    click.echo(f"  {'Model Type':<12} {'Size':<8} {'Backend':<6} {'Last Accessed':<20} {'Size on Disk':>12}")
    click.echo(f"  {'-'*12:<12} {'-'*8:<8} {'-'*6:<6} {'-'*20:<20} {'-'*12:>12}")

    for model in models:
        model_type = model["model_type"] or "unknown"
        model_size = model["model_size"] or "unknown"
        backend = model["backend"] or "unknown"
        last_access = model["last_access"].strftime("%Y-%m-%d %H:%M")
        if model["last_access"] == datetime.min:
            last_access = "unknown"
        size_formatted = model["size_formatted"]

        type_str = f"{model_type} ({model_size})"

        click.echo(f"  {type_str:<12} {size_formatted:<8} {backend:<6} {last_access:<20} {size_formatted:>12}")

    click.echo()
    click.echo(f"  Total: {_format_size(get_total_size())}")


def get_size_cmd() -> None:
    """Show total cache size and breakdown."""
    models = list_models()

    if not models:
        click.echo("  No TTS models found in cache.")
        return

    total_size = get_total_size()

    click.echo()
    click.echo("  TTS Model Cache Size:")
    click.echo()

    # Group by backend
    torch_size = sum(m["size_bytes"] for m in models if m["backend"] == "torch")
    mlx_size = sum(m["size_bytes"] for m in models if m["backend"] == "mlx")

    click.echo(f"    PyTorch models: {_format_size(torch_size)}")
    click.echo(f"    MLX models:     {_format_size(mlx_size)}")
    click.echo(f"    Total:          {_format_size(total_size)}")
    click.echo()

    # Count by model type
    torch_count = sum(1 for m in models if m["backend"] == "torch")
    mlx_count = sum(1 for m in models if m["backend"] == "mlx")
    click.echo(f"  Model count: {torch_count} PyTorch, {mlx_count} MLX")
    click.echo()


def prune_models_cmd(days: int = 30, dry_run: bool = False) -> None:
    """Remove models that haven't been accessed in N days.

    Args:
        days: Number of days of inactivity before pruning.
        dry_run: If True, preview what would be removed without removing.
    """
    models = list_models()

    cutoff = datetime.now() - timedelta(days=days)
    to_prune = [m for m in models if m["last_access"] < cutoff]

    if not to_prune:
        click.echo(f"  No models found unused for {days}+ days.")
        return

    click.echo()
    click.echo(f"  Models to prune (not accessed in {days}+ days):")
    click.echo("  WARNING: Modern filesystems (macOS APFS, Linux relatime) often do not track")
    click.echo("  accurate file access times. Models listed below may have been used recently.")
    click.echo()

    total_size = 0
    for model in to_prune:
        click.echo(f"    - {model['name']}")
        total_size += model["size_bytes"]

    click.echo(f"  Total size to free: {_format_size(total_size)}")
    click.echo()

    if dry_run:
        click.echo("  Dry run mode: no models were deleted.")
        return

    response = input(f"  Delete {len(to_prune)} model(s)? [y/N]: ").strip().lower()
    if response != "y":
        click.echo("  Cancelled.")
        return

    deleted = 0
    for model in to_prune:
        model_path = pathlib.Path(model["path"])
        try:
            shutil.rmtree(model_path)
            deleted += 1
            click.echo(f"    ✓ Removed: {model['name']}")
        except OSError as e:
            click.echo(f"    ✗ Failed to remove {model['name']}: {e}")

    click.echo()
    click.echo(f"  Deleted {deleted} model(s), freed {_format_size(total_size)}")


def clear_cache_cmd(force: bool = False) -> None:
    """Remove all cached TTS models.

    Args:
        force: Skip confirmation prompt.
    """
    models = list_models()

    if not models:
        click.echo("  No TTS models found in cache.")
        return

    total_size = get_total_size()

    click.echo()
    click.echo(f"  This will delete all {len(models)} cached TTS model(s).")
    click.echo(f"  Total size: {_format_size(total_size)}")
    click.echo()

    if not force:
        response = input("  Are you sure? [y/N]: ").strip().lower()
        if response != "y":
            click.echo("  Cancelled.")
            return

    deleted = 0
    for model in models:
        model_path = pathlib.Path(model["path"])
        try:
            shutil.rmtree(model_path)
            deleted += 1
        except OSError as e:
            click.echo(f"    ✗ Failed to remove {model['name']}: {e}")

    click.echo()
    click.echo(f"  Deleted {deleted} model(s), freed {_format_size(total_size)}")


def main():
    """CLI entry point for model cache commands."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage TTS model cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tts cache list                     List all cached models
  tts cache size                     Show total cache size
  tts cache prune --unused 30d       Remove models not used in 30 days
  tts cache clear --force            Remove all cached models
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Model cache commands")

    # list command
    subparsers.add_parser("list", help="List all cached models")

    # size command
    subparsers.add_parser("size", help="Show total cache size")

    # prune command
    prune_parser = subparsers.add_parser("prune", help="Remove unused models")
    prune_parser.add_argument(
        "--unused",
        type=int,
        default=30,
        metavar="DAYS",
        help="Remove models not accessed in N days (default: 30)",
    )

    # clear command
    clear_parser = subparsers.add_parser("clear", help="Remove all cached models")
    clear_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_models_cmd()
    elif args.command == "size":
        get_size_cmd()
    elif args.command == "prune":
        prune_models_cmd(days=args.unused, dry_run=False)
    elif args.command == "clear":
        clear_cache_cmd(force=args.force)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
