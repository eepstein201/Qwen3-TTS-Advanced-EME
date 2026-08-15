#!/usr/bin/env python3
"""Interactive mode functions for Qwen3-TTS generation.

Contains REPL, watch mode, interactive mode, progress polling,
and voice prompt management (delete/rename/preview).
"""

import logging
import os
import sys
import threading
import time

logger = logging.getLogger("tts.cli")

from qwen3_tts.core.config import (  # noqa: E402
    VOICE_PROMPTS_DIR,
    get_default_clone_prompt,
    is_server_running,
    safe_path_join,
)
from qwen3_tts.interface.generate_helpers import (  # noqa: E402
    _decode_base64_result,
    _save_base64_result,
    get_voice_alias,
    list_voice_prompts,
    open_file,
    play_audio,
    process_audio_args,
    voice_prompt_exists,
)

# ---------------------------------------------------------------------------
# Voice prompt management
# ---------------------------------------------------------------------------


def delete_voice_prompt(prompt_name):
    """Delete a voice prompt file (supports .pt, .wav+.txt formats)."""
    if ".." in prompt_name or "/" in prompt_name or "\\" in prompt_name:
        print(f"Error: Invalid prompt name: {prompt_name!r}")
        return False

    # Strip known extensions to get the base name
    base = prompt_name
    for ext in (".pt", ".wav", ".txt"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break

    # Find all format files that exist
    pt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.pt")
    wav_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.wav")
    txt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.txt")
    to_delete = [p for p in (pt_path, wav_path, txt_path) if os.path.exists(p)]

    if not to_delete:
        print(f"Error: Voice prompt not found: {prompt_name}")
        return False

    filenames = ", ".join(os.path.basename(p) for p in to_delete)
    confirm = (
        input(f"Delete '{base}' ({filenames})? This cannot be undone. [y/N]: ")
        .strip()
        .lower()
    )
    if confirm != "y":
        print("Cancelled.")
        return False

    for p in to_delete:
        os.remove(p)
    print(f"Deleted: {filenames}")
    return True


def rename_voice_prompt(old_name, new_name):
    """Rename a voice prompt file (supports .pt, .wav+.txt formats)."""
    for _name in (old_name, new_name):
        if ".." in _name or "/" in _name or "\\" in _name:
            print(f"Error: Invalid prompt name: {_name!r}")
            return False

    # Strip known extensions to get base names
    old_base = old_name
    new_base = new_name
    for ext in (".pt", ".wav", ".txt"):
        if old_base.endswith(ext):
            old_base = old_base[: -len(ext)]
        if new_base.endswith(ext):
            new_base = new_base[: -len(ext)]

    # Find all format files that exist for old name
    rename_pairs = []
    for ext in (".pt", ".wav", ".txt"):
        old_path = safe_path_join(VOICE_PROMPTS_DIR, f"{old_base}{ext}")
        new_path = safe_path_join(VOICE_PROMPTS_DIR, f"{new_base}{ext}")
        if os.path.exists(old_path):
            if os.path.exists(new_path):
                print(f"Error: Voice prompt already exists: {new_base}{ext}")
                return False
            rename_pairs.append((old_path, new_path))

    if not rename_pairs:
        print(f"Error: Voice prompt not found: {old_name}")
        return False

    completed = []
    try:
        for old_path, new_path in rename_pairs:
            os.rename(old_path, new_path)
            completed.append((old_path, new_path))
    except OSError as e:
        # Rollback completed renames on failure
        for done_old, done_new in completed:
            try:
                os.rename(done_new, done_old)
            except OSError:
                pass
        print(f"Error: Rename failed: {e}")
        return False

    print(f"Renamed: {old_base} -> {new_base}")
    return True


def preview_voice_prompt(prompt_name, config):
    """Preview a voice prompt by generating a short sample."""
    import requests  # lazy

    if not prompt_name.endswith(".pt"):
        prompt_name += ".pt"

    if not voice_prompt_exists(prompt_name):
        print(f"Error: Voice prompt not found: {prompt_name}")
        return False

    if is_server_running(config):
        payload = {
            "texts": ["This is a preview of the voice prompt."],
            "mode": "clone",
            "prompt_file": prompt_name,
            "language": config.get("language", "English"),
            "temperature": 0.7,
        }
        print(f"Generating preview for '{prompt_name}'...")
        from qwen3_tts.core.http_client import server_request

        resp = server_request("POST", "/generate", json=payload, timeout=60)
        if resp.status_code == 200:
            result = resp.json()["results"][0]
            print("Playing preview...")
            import tempfile

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            _save_base64_result(result, tmp.name)
            play_audio(tmp.name)
            os.remove(tmp.name)
            return True
        else:
            try:
                error_msg = resp.json().get("error", "Unknown error")
            except (ValueError, requests.exceptions.JSONDecodeError):
                error_msg = f"Server returned HTTP {resp.status_code}"
            print(f"Error: {error_msg}")
            return False
    else:
        print(
            "Error: TTS server must be running for preview. Start with 'tts server start'."
        )
        return False


# ---------------------------------------------------------------------------
# Progress poller
# ---------------------------------------------------------------------------


class _ProgressPoller:
    """Background thread that polls /generation-status and displays progress.

    Uses Rich for pretty progress bars if available, falls back to print-based progress.
    """

    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    # Try to import Rich
    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        HAS_RICH = True
    except ImportError:
        HAS_RICH = False

    def __init__(self, batch_total=1):
        self.batch_total = batch_total
        self._stop = threading.Event()
        self._thread = None
        self._rich_progress = None
        self._rich_task_id = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        # Clean up Rich display if used
        if self.HAS_RICH and self._rich_progress:
            self._rich_progress.stop()
        else:
            # Clear the progress line (fallback)
            sys.stderr.write("\r" + " " * 80 + "\r")
            sys.stderr.flush()

    def _run(self):
        # Use Rich if available
        if self.HAS_RICH:
            self._run_rich()
        else:
            self._run_fallback()

    def _run_rich(self):
        """Run with Rich progress bar."""
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        from qwen3_tts.core.http_client import server_request

        console = Console(stderr=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task(
                "Generating audio...",
                total=100 if self.batch_total > 1 else None,
            )

            while not self._stop.is_set():
                try:
                    resp = server_request("GET", "/generation-status", timeout=2)
                    if resp.status_code == 200:
                        state = resp.json()
                        if state.get("active"):
                            elapsed = state.get("elapsed_sec", 0)
                            eta = state.get("eta_sec")

                            # Chunk progress suffix
                            chunk_total = state.get("chunk_total", 0)
                            chunk_suffix = ""
                            if chunk_total > 1:
                                chunk_idx = state.get("chunk_index", 0) + 1
                                chunk_suffix = f" [chunk {chunk_idx}/{chunk_total}]"
                                progress.update(
                                    task_id, description=f"Generating...{chunk_suffix}"
                                )

                            if self.batch_total > 1:
                                idx = state.get("batch_index", 0) + 1
                                progress.update(
                                    task_id,
                                    description=f"[{idx}/{self.batch_total}] Generating...{chunk_suffix}",
                                )
                                if eta is not None:
                                    total_est = elapsed + eta
                                    pct = (
                                        min(95, int(elapsed / total_est * 100))
                                        if total_est > 0
                                        else 0
                                    )
                                    progress.update(task_id, completed=pct)
                except Exception as e:
                    logger.debug("Progress poller (_run_rich) error: %s", e)

                self._stop.wait(1.0)

    def _run_fallback(self):
        """Run with print-based progress (original implementation)."""
        from qwen3_tts.core.http_client import server_request

        tick = 0
        while not self._stop.is_set():
            try:
                resp = server_request("GET", "/generation-status", timeout=2)
                if resp.status_code == 200:
                    state = resp.json()
                    if state.get("active"):
                        elapsed = state.get("elapsed_sec", 0)
                        eta = state.get("eta_sec")
                        spinner = self.SPINNER[tick % len(self.SPINNER)]

                        # Chunk progress suffix
                        chunk_total = state.get("chunk_total", 0)
                        chunk_suffix = ""
                        if chunk_total > 1:
                            chunk_idx = state.get("chunk_index", 0) + 1
                            chunk_suffix = f" [chunk {chunk_idx}/{chunk_total}]"

                        if self.batch_total > 1:
                            idx = state.get("batch_index", 0) + 1
                            if eta is not None:
                                total_est = elapsed + eta
                                pct = (
                                    min(95, int(elapsed / total_est * 100))
                                    if total_est > 0
                                    else 0
                                )
                                bar_filled = pct // 5
                                bar = "=" * bar_filled + ">" + " " * (19 - bar_filled)
                                line = f"\r{spinner} [{idx}/{self.batch_total}] Generating... {elapsed:.0f}s / ~{elapsed + eta:.0f}s [{bar}] {pct}%{chunk_suffix}"
                            else:
                                line = f"\r{spinner} [{idx}/{self.batch_total}] Generating... {elapsed:.0f}s elapsed{chunk_suffix}"
                        else:
                            if eta is not None:
                                line = f"\r{spinner} Generating... {elapsed:.0f}s elapsed (ETA ~{eta:.0f}s){chunk_suffix}"
                            else:
                                line = f"\r{spinner} Generating... {elapsed:.0f}s elapsed{chunk_suffix}"

                        sys.stderr.write(line)
                        sys.stderr.flush()
            except Exception as e:
                logger.debug("Progress poller (_run_fallback) error: %s", e)

            tick += 1
            self._stop.wait(1.0)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


def interactive_mode(use_server, config, gen_params):
    """Run in interactive mode with prompts."""
    import soundfile as sf

    from qwen3_tts.interface.generate_helpers import get_text
    from qwen3_tts.interface.generate_server import (
        generate_local,
        generate_via_server,
        launch_gradio_ui,
    )

    print("\n=== Qwen3-TTS Generator ===\n")

    print("How would you like to generate speech?")
    print("  1. Command Line (enter text here)")
    print("  2. Web Interface (Gradio UI in browser)")
    print()
    mode_choice = input("Select [1/2]: ").strip()

    if mode_choice == "2":
        launch_gradio_ui(config)
        return None

    text_input = input("\nEnter text or file path: ").strip()
    if not text_input:
        print("Error: No text provided")
        sys.exit(1)
    text = get_text(text_input)
    print(
        f"\nText to synthesize ({len(text)} chars):\n{text[:200]}{'...' if len(text) > 200 else ''}\n"
    )

    print("Voice mode:")
    print("  1. Default (VoiceDesign with description)")
    print("  2. Custom (Voice clone prompt)")
    mode_choice = input("Select [1/2]: ").strip()

    if mode_choice == "2":
        prompts = list_voice_prompts()
        if not prompts:
            print("Error: No voice prompts found in", VOICE_PROMPTS_DIR)
            sys.exit(1)

        print("\nAvailable voice prompts:")
        for i, p in enumerate(prompts, 1):
            default_marker = (
                " (default)" if p == config.get("default_clone_prompt") else ""
            )
            print(f"  {i}. {p}{default_marker}")

        choice = input(f"Select [1-{len(prompts)}]: ").strip()
        try:
            prompt_file = prompts[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid selection, using default")
            prompt_file = config.get("default_clone_prompt", prompts[0])

        mode = "clone"
        voice_param = prompt_file
    else:
        mode = "design"
        voice_param = config.get("default_voice_description", "")
        print(f"\nCurrent voice description: {voice_param}")
        custom = input("Use this description? [Y/n]: ").strip().lower()
        if custom == "n":
            voice_param = input("Enter new description: ").strip()

    output_name = input("\nOutput filename (saved to ~/Downloads/): ").strip()
    if not output_name:
        output_name = "tts_output.wav"
    # Strip directory separators to prevent path traversal
    output_name = os.path.basename(output_name)
    if not output_name.endswith(".wav"):
        output_name += ".wav"

    output_dir = os.path.expanduser(config.get("output_directory", "~/Downloads"))
    output_path = safe_path_join(output_dir, output_name)

    print()
    language = config.get("language", "English")

    if use_server:
        print("Using TTS server...")
        if mode == "clone":
            results = generate_via_server(
                [text], mode, config, gen_params, prompt_file=voice_param
            )
        else:
            results = generate_via_server(
                [text], mode, config, gen_params, voice_description=voice_param
            )
        _save_base64_result(results[0], output_path)
    else:
        wav, sr = generate_local(
            text,
            mode,
            gen_params,
            language,
            prompt_file=voice_param if mode == "clone" else None,
            voice_description=voice_param if mode == "design" else None,
        )
        sf.write(output_path, wav, sr)

    print(f"Saved to: {output_path}")
    open_file(output_path)
    return output_path


# ---------------------------------------------------------------------------
# REPL mode
# ---------------------------------------------------------------------------


def run_repl(config, use_server, gen_params=None):
    """Run interactive REPL mode for rapid TTS iteration."""
    import soundfile as sf

    from qwen3_tts.interface.generate_server import generate_local, generate_via_server

    print("\n=== TTS REPL Mode ===")
    print("Commands:")
    print("  Type text to generate speech")
    print("  /voice NAME    - Switch voice alias")
    print("  /preset NAME   - Switch preset")
    print("  /prompt NAME   - Switch voice prompt")
    print("  /play on|off   - Toggle auto-play")
    print("  /speed FACTOR  - Set speed (e.g., 1.2)")
    print("  /pitch SEMI    - Set pitch shift")
    print("  /status        - Show current settings")
    print("  /quit or /q    - Exit REPL")
    print()

    state = {
        "mode": "clone",
        "prompt": get_default_clone_prompt(config),
        "preset": None,
        "auto_play": True,
        "speed": None,
        "pitch": None,
        "counter": 1,
    }

    # Base the REPL's working params on the caller's CLI-merged gen_params so
    # flags like --seed/--temperature are honored. Previously this rebuilt from
    # raw config["generation"], silently dropping every CLI override.
    # base_gen_params is the reset point when switching presets. Fall back to
    # the config block only when no params were supplied (e.g. unit tests).
    if gen_params is None:
        gen_params = config.get("generation", {})
    base_gen_params = gen_params.copy()
    gen_params = base_gen_params.copy()

    while True:
        try:
            user_input = input("tts> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting REPL.")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd in ("/quit", "/q"):
                print("Exiting REPL.")
                break
            elif cmd == "/voice":
                if arg:
                    alias = get_voice_alias(arg, config)
                    if alias:
                        if "prompt" in alias:
                            state["prompt"] = alias["prompt"]
                        if "preset" in alias:
                            state["preset"] = alias["preset"]
                            preset_params = config.get("presets", {}).get(
                                alias["preset"], {}
                            )
                            gen_params = {**base_gen_params, **preset_params}
                        print(f"Switched to voice alias: {arg}")
                    else:
                        print(f"Unknown alias: {arg}")
                else:
                    print("Usage: /voice NAME")
            elif cmd == "/preset":
                if arg:
                    presets = config.get("presets", {})
                    if arg in presets:
                        state["preset"] = arg
                        gen_params = {**base_gen_params, **presets[arg]}
                        print(f"Switched to preset: {arg}")
                    else:
                        print(f"Unknown preset: {arg}")
                else:
                    print("Usage: /preset NAME")
            elif cmd == "/prompt":
                if arg:
                    prompt_name = arg if arg.endswith(".pt") else arg + ".pt"
                    prompt_path = safe_path_join(VOICE_PROMPTS_DIR, prompt_name)
                    if os.path.exists(prompt_path):
                        state["prompt"] = prompt_name
                        print(f"Switched to prompt: {prompt_name}")
                    else:
                        print(f"Prompt not found: {prompt_name}")
                else:
                    print("Usage: /prompt NAME")
            elif cmd == "/play":
                if arg and arg.lower() in ("on", "off"):
                    state["auto_play"] = arg.lower() == "on"
                    print(f"Auto-play: {'on' if state['auto_play'] else 'off'}")
                else:
                    print("Usage: /play on|off")
            elif cmd == "/speed":
                if arg:
                    try:
                        state["speed"] = float(arg)
                        print(f"Speed: {state['speed']}")
                    except ValueError:
                        print("Invalid speed value")
                else:
                    state["speed"] = None
                    print("Speed: reset to default")
            elif cmd == "/pitch":
                if arg:
                    try:
                        state["pitch"] = float(arg)
                        print(f"Pitch: {state['pitch']} semitones")
                    except ValueError:
                        print("Invalid pitch value")
                else:
                    state["pitch"] = None
                    print("Pitch: reset to default")
            elif cmd == "/status":
                print(f"  Mode: {state['mode']}")
                print(f"  Prompt: {state['prompt']}")
                print(f"  Preset: {state['preset'] or 'default'}")
                print(f"  Auto-play: {'on' if state['auto_play'] else 'off'}")
                print(f"  Speed: {state['speed'] or '1.0'}")
                print(f"  Pitch: {state['pitch'] or '0'}")
                print(f"  Server: {'connected' if use_server else 'local'}")
            else:
                print(f"Unknown command: {cmd}")
            continue

        # Generate speech
        text = user_input
        output_dir = os.path.expanduser(config.get("output_directory", "~/Downloads"))
        output_path = safe_path_join(output_dir, f"repl_{state['counter']}.wav")

        try:
            if use_server:
                results = generate_via_server(
                    [text],
                    state["mode"],
                    config,
                    gen_params,
                    prompt_file=state["prompt"],
                )
                wav, sr = _decode_base64_result(results[0])
            else:
                wav, sr = generate_local(
                    text,
                    state["mode"],
                    gen_params,
                    config.get("language", "English"),
                    prompt_file=state["prompt"],
                )

            # Apply audio processing
            if state["speed"] and state["speed"] != 1.0:
                from qwen3_tts.core.engine import adjust_speed

                wav = adjust_speed(wav, sr, state["speed"])
            if state["pitch"] and state["pitch"] != 0:
                from qwen3_tts.core.engine import adjust_pitch

                wav = adjust_pitch(wav, sr, state["pitch"])

            sf.write(output_path, wav, sr)
            print(f"  -> {output_path}")

            if state["auto_play"]:
                play_audio(output_path)

            state["counter"] += 1

        except Exception as e:
            print(f"Error: {e}")


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------


def run_watch_mode(watch_dir, config, args, gen_params, use_server):
    """Watch a directory for new .txt files and generate TTS."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    from qwen3_tts.core.config import safe_path_join
    from qwen3_tts.interface.generate_server import generate_local, generate_via_server

    # Security: validate watch_dir against traversal
    expanded = os.path.expanduser(watch_dir)
    if os.path.isabs(expanded):
        if ".." in expanded:
            raise ValueError(f"Path traversal detected in watch_dir: {watch_dir}")
        safe_watch_dir = expanded
    else:
        safe_watch_dir = safe_path_join(os.getcwd(), expanded)

    if not os.path.isdir(safe_watch_dir):
        print(f"Error: Directory not found: {safe_watch_dir}")
        return

    # Security: validate output_dir against traversal
    output_raw = args.output or config.get("output_directory", "~/Downloads")
    expanded = os.path.expanduser(output_raw)
    if os.path.isabs(expanded):
        if ".." in expanded:
            raise ValueError(f"Path traversal detected in output_dir: {output_raw}")
        safe_output_dir = expanded
    else:
        safe_output_dir = safe_path_join(os.getcwd(), expanded)

    # Verify output_dir is under home directory
    home = os.path.realpath(os.path.expanduser("~"))
    resolved = os.path.realpath(safe_output_dir)
    if not (resolved == home or resolved.startswith(home + os.sep)):
        raise ValueError(f"output_dir must be under home directory: {output_raw}")

    os.makedirs(safe_output_dir, exist_ok=True)

    mode = args.mode or "clone"
    prompt_file = args.prompt or get_default_clone_prompt(config)
    voice_description = args.description or config.get("default_voice_description", "")

    processed_files = set()

    class TTSHandler(FileSystemEventHandler):
        def on_created(self, event):
            import soundfile as sf

            if event.is_directory or not event.src_path.endswith(".txt"):
                return
            if event.src_path in processed_files:
                return

            time.sleep(0.5)

            try:
                with open(event.src_path) as f:
                    text = f.read().strip()

                if not text:
                    return

                basename = os.path.splitext(os.path.basename(event.src_path))[0]
                output_path = safe_path_join(safe_output_dir, f"{basename}.wav")

                print(f"\nProcessing: {event.src_path}")

                if use_server:
                    results = generate_via_server(
                        [text],
                        mode,
                        config,
                        gen_params,
                        prompt_file=prompt_file if mode == "clone" else None,
                        voice_description=voice_description
                        if mode == "design"
                        else None,
                    )
                    wav, sr = _decode_base64_result(results[0])
                else:
                    wav, sr = generate_local(
                        text,
                        mode,
                        gen_params,
                        config.get("language", "English"),
                        prompt_file=prompt_file,
                        voice_description=voice_description,
                    )

                wav = process_audio_args(wav, sr, args)
                sf.write(output_path, wav, sr)
                # Mark processed only after a successful write so a failed
                # generation stays eligible for retry on the next event
                # (previously it was marked before generation and silently
                # dropped forever on failure).
                processed_files.add(event.src_path)

                print(f"  -> {output_path}")

                if args.play:
                    play_audio(output_path)

            except Exception as e:
                print(f"Error processing {event.src_path}: {e}")

    print("\n=== Watch Mode ===")
    print(f"Watching: {watch_dir}")
    print(f"Output to: {safe_output_dir}")
    print("Drop .txt files into the watch directory to generate TTS.")
    print("Press Ctrl+C to stop.\n")

    event_handler = TTSHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nWatch mode stopped.")

    observer.join()
