#!/usr/bin/env python3
"""Headless Gradio UI integration test using gradio_client.

Launches the Gradio UI on an isolated port, connects via gradio_client,
runs clone-mode generation, and verifies valid audio output.

Requirements:
    - TTS server running on port 5123 with clone model loaded
    - qwen3-tts-mlx conda env (has gradio, gradio_client, soundfile)

Usage:
    python tests/test_ui_headless.py
"""

import os
import signal
import subprocess
import sys
import time

UI_PORT = 7865
UI_URL = f"http://127.0.0.1:{UI_PORT}"
PROJECT_DIR = os.path.expanduser("~/Qwen3-TTS_UserFiles")


def wait_for_server(url, timeout=30):
    """Poll until the Gradio server responds, or raise on timeout."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Gradio UI did not respond at {url} within {timeout}s")


def main():
    ui_proc = None
    try:
        # 1. Launch Gradio UI on isolated port
        print(f"Launching Gradio UI on port {UI_PORT}...")
        env = os.environ.copy()
        ui_proc = subprocess.Popen(  # nosec B603
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, '{PROJECT_DIR}'); "
             f"from voice_ui import build_ui; "
             f"demo = build_ui(); "
             f"import os; "
             f"demo.launch(server_name='127.0.0.1', server_port={UI_PORT}, "
             f"share=False, show_error=True, "
             f"allowed_paths=[os.path.expanduser('~/Downloads'), '/tmp'], "
             f"prevent_thread_lock=False)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # 2. Wait for UI to be ready
        print("Waiting for UI to be ready...", end="", flush=True)
        wait_for_server(UI_URL, timeout=30)
        print(" ready!")

        # 3. Connect with gradio_client
        from gradio_client import Client
        client = Client(UI_URL, verbose=False)

        # 4. Get available prompts from the UI's API info
        # The dropdown lists what the UI sees (backend-aware: .wav for MLX, .pt for torch)
        api_info = client.view_api(return_format="dict")
        # Find the generate_clone endpoint and get prompt choices
        prompt_choices = None
        for endpoint in api_info.get("named_endpoints", {}).values():
            if endpoint.get("parameters"):
                for param in endpoint["parameters"]:
                    if param.get("label") == "Voice Prompt" and "enum" in param.get("type", {}):
                        prompt_choices = param["type"]["enum"]
                        break
            if prompt_choices:
                break

        if prompt_choices:
            default_prompt = prompt_choices[0]
        else:
            # Fallback: query server prompts endpoint
            sys.path.insert(0, PROJECT_DIR)
            from voice_config import get_default_clone_prompt, load_config
            config = load_config()
            default_prompt = get_default_clone_prompt(config)
        print(f"Using voice prompt: {default_prompt}")

        # 5. Call clone generation (non-streaming)
        print("Generating audio via clone mode (non-streaming)...")
        result = client.predict(
            "Hello, this is a headless test of the Gradio UI.",  # text
            default_prompt,   # prompt
            "(none)",         # preset
            0.7,              # temperature
            50,               # top_k
            0.95,             # top_p
            1.05,             # repetition_penalty
            "",               # seed
            False,            # trim_silence
            False,            # normalize
            1.0,              # speed
            0,                # pitch (int)
            False,            # streaming
            False,            # no_transcript
            api_name="/generate_clone",
        )

        # 6. Verify output
        # Result is a tuple: (audio_output, status_text, status_html, history_df)
        # audio_output may be a streaming .m3u8 (because gr.Audio has streaming=True),
        # so we extract the actual wav filename from the status text instead.
        status_text = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else "N/A"
        print(f"\nStatus: {status_text}")

        if "Generated:" not in str(status_text):
            print(f"FAIL: Generation did not succeed. Status: {status_text}")
            return 1

        # Extract filename from status like "Generated: voice_ui_49ca4b63.wav"
        import re
        match = re.search(r"Generated:\s*(\S+\.wav)", str(status_text))
        if not match:
            print(f"FAIL: Could not parse wav filename from status: {status_text}")
            return 1

        wav_filename = match.group(1)
        output_dir = os.path.expanduser("~/Downloads")
        audio_path = os.path.join(output_dir, wav_filename)
        print(f"Audio file: {audio_path}")

        if not os.path.exists(audio_path):
            print(f"FAIL: Audio file does not exist: {audio_path}")
            return 1

        # Read and validate audio
        import soundfile as sf
        import numpy as np

        wav, sr = sf.read(audio_path)
        duration = len(wav) / sr
        peak = np.max(np.abs(wav))

        print(f"Sample rate: {sr} Hz")
        print(f"Samples: {len(wav)}")
        print(f"Duration: {duration:.2f}s")
        print(f"Peak amplitude: {peak:.4f}")

        if len(wav) == 0:
            print("FAIL: Audio has zero samples")
            return 1
        if duration < 0.5:
            print("FAIL: Audio too short (< 0.5s)")
            return 1
        if peak < 0.001:
            print("FAIL: Audio is effectively silent")
            return 1

        print("\nPASS: Gradio UI headless test succeeded!")
        return 0

    except Exception as e:
        print(f"\nFAIL: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Clean up UI process
        if ui_proc and ui_proc.poll() is None:
            print("Shutting down Gradio UI...")
            ui_proc.send_signal(signal.SIGTERM)
            try:
                ui_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ui_proc.kill()
                ui_proc.wait()
            print("UI shut down.")


if __name__ == "__main__":
    sys.exit(main())
