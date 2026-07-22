#!/usr/bin/env python3
"""TTS uninstall and cleanup utilities.

Provides selective cleanup of models, voice prompts, configuration,
and environment management without breaking the running environment.
"""

import logging
import os
import shutil
import sys

from qwen3_tts.core.config import (
    CONFIG_PATH,
    HF_CACHE,
    HISTORY_FILE,
    LOG_FILE,
    PID_FILE,
    TOKEN_FILE,
    USER_FILES_DIR,
    VOICE_PROMPTS_DIR,
)
from qwen3_tts.tools._shared import (
    _format_size,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from qwen3_tts.tools.model_cache import _MLX_MODEL_PREFIXES, _TORCH_MODEL_PREFIXES

logger = logging.getLogger("tts.uninstall")


def _get_models_size() -> int:
    """Calculate total size of cached TTS models in bytes."""
    total = 0
    if HF_CACHE.exists():
        for model_dir in HF_CACHE.iterdir():
            if model_dir.is_dir() and any(
                model_dir.name.startswith(prefix)
                for prefix in _TORCH_MODEL_PREFIXES + _MLX_MODEL_PREFIXES
            ):
                # Calculate directory size
                for item in model_dir.rglob("*"):
                    if item.is_file():
                        try:
                            total += item.stat().st_size
                        except OSError:
                            pass
    return total


def _list_cached_models() -> list[str]:
    """List all cached TTS model directories."""
    models = []
    if HF_CACHE.exists():
        for model_dir in HF_CACHE.iterdir():
            if model_dir.is_dir() and any(
                model_dir.name.startswith(prefix)
                for prefix in _TORCH_MODEL_PREFIXES + _MLX_MODEL_PREFIXES
            ):
                models.append(model_dir.name)
    return sorted(models)


def uninstall_models(dry_run: bool = False) -> None:
    """Remove all cached TTS models from HuggingFace cache.

    Args:
        dry_run: If True, preview what would be deleted without deleting.
    """
    models = _list_cached_models()

    if not models:
        print_info("No TTS models found in cache.")
        return

    size_bytes = _get_models_size()
    print_header(
        f"Models to remove ({len(models)} model(s), {_format_size(size_bytes)})"
    )

    for model in models:
        print(f"  - {model}")

    if dry_run:
        print_info("Dry run mode: no files were deleted.")
        return

    # Confirm removal
    response = input(f"\n  Delete {len(models)} model(s)? [y/N]: ").strip().lower()
    if response != "y":
        print_info("Cancelled.")
        return

    deleted = 0
    for model in models:
        model_path = HF_CACHE / model
        try:
            shutil.rmtree(model_path)
            deleted += 1
            print_success(f"Removed: {model}")
        except OSError as e:
            print_warning(f"Failed to remove {model}: {e}")

    print_success(f"Deleted {deleted} model(s), freed {_format_size(size_bytes)}")


def uninstall_voices(dry_run: bool = False) -> None:
    """Remove all voice prompts from the voice_prompts directory.

    Args:
        dry_run: If True, preview what would be deleted without deleting.
    """
    if not VOICE_PROMPTS_DIR.exists():
        print_info("No voice prompts directory found.")
        return

    # Count files by type
    pt_files = list(VOICE_PROMPTS_DIR.glob("*.pt"))
    wav_files = list(VOICE_PROMPTS_DIR.glob("*.wav"))
    txt_files = list(VOICE_PROMPTS_DIR.glob("*.txt"))

    total_files = len(pt_files) + len(wav_files) + len(txt_files)

    if total_files == 0:
        print_info("No voice prompts found.")
        return

    print_header(f"Voice prompts to remove ({total_files} file(s))")
    print_info(f"  .pt files: {len(pt_files)}")
    print_info(f"  .wav files: {len(wav_files)}")
    print_info(f"  .txt files: {len(txt_files)}")

    if dry_run:
        print_info("Dry run mode: no files were deleted.")
        return

    # Confirm removal
    response = input(f"\n  Delete {total_files} file(s)? [y/N]: ").strip().lower()
    if response != "y":
        print_info("Cancelled.")
        return

    try:
        shutil.rmtree(VOICE_PROMPTS_DIR)
        os.makedirs(VOICE_PROMPTS_DIR, exist_ok=True)
        print_success(f"Deleted all voice prompts ({total_files} file(s))")
    except OSError as e:
        print_warning(f"Failed to remove voice prompts: {e}")


def uninstall_config(dry_run: bool = False) -> None:
    """Reset config.json to default values.

    Args:
        dry_run: If True, preview what would be done without doing it.
    """
    if not CONFIG_PATH.exists():
        print_info("No config.json found (will be created with defaults on next use).")
        return

    print_header("Config reset to defaults")
    print_info("This will reset config.json to default values.")

    if dry_run:
        print_info("Dry run mode: config was not modified.")
        return

    # Import config utilities
    from qwen3_tts.core.config import get_default_config, load_config

    # Backup current config
    backup_path = CONFIG_PATH.with_suffix(".backup")
    try:
        shutil.copy2(CONFIG_PATH, backup_path)
        print_info(f"Backup saved to: {backup_path}")
    except OSError as e:
        print_warning(f"Could not create backup: {e}")

    # Load current config to preserve some settings (backend, model_size)
    try:
        current_config = load_config()
    except Exception as e:
        logger.warning("Could not load existing config before reset: %s", e)
        current_config = {}

    # Build and save the default config
    default_config = get_default_config(current_config)
    import json

    with open(CONFIG_PATH, "w") as f:
        json.dump(default_config, f, indent=2)

    print_success("Config reset to defaults (backup saved)")


def print_environment_instructions() -> None:
    """Print instructions for removing conda environments."""
    print_header("Conda Environment Removal")

    # Detect which environments exist. miniforge_envs must always be a Path —
    # it is later divided with ``/ env_name`` and queried with .exists(), which
    # both fail on a bare str. Pre-fix, the primary miniforge3/miniconda3
    # branches (the documented install paths) returned a str and crashed here
    # with TypeError on the documented primary install path.
    from pathlib import Path

    conda_envs = []
    miniforge_envs = None  # Initialize to prevent undefined variable
    if os.path.exists(os.path.expanduser("~/miniforge3/envs")):
        miniforge_envs = Path(os.path.expanduser("~/miniforge3/envs"))
    elif os.path.exists(os.path.expanduser("~/miniconda3/envs")):
        miniforge_envs = Path(os.path.expanduser("~/miniconda3/envs"))
    else:
        # Try to find conda environments
        home = Path.home()
        for base in ("miniforge3", "miniconda3", "anaconda3", "opt/anaconda3"):
            envs_path = home / base / "envs"
            if envs_path.exists():
                miniforge_envs = envs_path
                break
        else:
            miniforge_envs = None

    if miniforge_envs:
        for env_name in ("qwen3-tts", "qwen3-tts-mlx"):
            env_path = miniforge_envs / env_name
            if env_path.exists():
                conda_envs.append(env_name)

    print_info("To remove the conda environments, run these commands:")
    print()
    print("  # First, deactivate the current environment:")
    print("  conda deactivate")
    print()

    for env in conda_envs:
        print(f"  # Remove the {env} environment:")
        print(f"  conda env remove -n {env} -y")
        print()

    if conda_envs:
        print("  # Then remove user files:")
        print(f"  rm -rf {USER_FILES_DIR}")
        print(f"  rm -f {HISTORY_FILE} {TOKEN_FILE}")
        print(f"  rm -f {PID_FILE} {LOG_FILE}")
    else:
        print_info("No conda environments found.")


def uninstall_all(dry_run: bool = False) -> None:
    """Run all uninstall steps except conda environment removal.

    Args:
        dry_run: If True, preview what would be done without doing it.
    """
    print_header("TTS Uninstall - All Components")

    uninstall_models(dry_run)
    uninstall_voices(dry_run)
    uninstall_config(dry_run)

    print()
    print_environment_instructions()


def main():
    """CLI entry point for uninstall commands."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Uninstall and clean up TTS components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tts uninstall --models           Remove cached models
  tts uninstall --voices           Remove voice prompts
  tts uninstall --config            Reset config to defaults
  tts uninstall --all               Remove all (except conda envs)
  tts uninstall --dry-run           Preview what would be deleted
  tts uninstall --environment       Show conda removal commands
        """,
    )

    parser.add_argument(
        "--models",
        action="store_true",
        help="Remove cached TTS models from HuggingFace cache",
    )
    parser.add_argument(
        "--voices",
        action="store_true",
        help="Remove all voice prompts",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Reset config.json to defaults",
    )
    parser.add_argument(
        "--environment",
        action="store_true",
        help="Print conda environment removal commands (does NOT execute)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all uninstall steps except conda environments",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually deleting anything",
    )

    args = parser.parse_args()

    # If no flags provided, show help
    if not any([args.models, args.voices, args.config, args.environment, args.all]):
        parser.print_help()
        return 0

    # Execute requested actions
    if args.all:
        uninstall_all(dry_run=args.dry_run)
    else:
        if args.environment:
            print_environment_instructions()
        if args.models:
            uninstall_models(dry_run=args.dry_run)
        if args.voices:
            uninstall_voices(dry_run=args.dry_run)
        if args.config:
            uninstall_config(dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
