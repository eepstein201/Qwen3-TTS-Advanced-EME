#!/usr/bin/env python3
"""Headless Gradio UI integration test using gradio_client.

Launches the Gradio UI on an isolated port, connects via gradio_client,
runs clone-mode generation, and verifies valid audio output. Also carries
the E2E create-from-audio case for the Voice Management tab (Step 0G),
which drives the MLX voice-prompt create through the engine writer.

Requirements:
    - TTS server running on port 5123 with clone model loaded
    - qwen3-tts-mlx conda env (has gradio, gradio_client, soundfile)

Usage:
    python tests/test_ui_headless.py
"""

import os
import shutil
import signal
import subprocess  # nosec B404
import sys
import tempfile
import time
import unittest
import uuid

UI_PORT = 7865
UI_URL = f"http://127.0.0.1:{UI_PORT}"
PROJECT_DIR = os.path.expanduser("~/Qwen3-TTS_UserFiles")
SERVER_URL = "http://127.0.0.1:5123"


def wait_for_server(url, timeout=30):
    """Poll until the Gradio server responds, or raise on timeout."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=2)  # nosec B310
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
             f"from qwen3_tts.interface.ui import build_ui; "
             f"demo = build_ui(); "
             f"import os; "
             f"demo.launch(server_name='127.0.0.1', server_port={UI_PORT}, "
             f"share=False, show_error=True, "
             f"allowed_paths=[os.path.realpath(os.path.expanduser('~/Downloads/Qwen3-TTS Output')), '/tmp'], "
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
            from qwen3_tts.core.config import get_default_clone_prompt, load_config
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
        import numpy as np
        import soundfile as sf

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


class TestUiCreateFromAudioEngineWriter(unittest.TestCase):
    """E2E: the Voice Management create-from-audio flow (Step 0G).

    Drives the real UI subprocess end to end: a temp-dir .wav fixture is
    uploaded through gradio_client, the create endpoint runs, and the engine
    writer's artifacts (.wav+.txt pair, stripped transcript, >=24 kHz mono)
    are asserted under the configured voice-prompts dir. This pins the create
    flow ONLY — the script's generation flow above is stale (consolidated plan
    6R item 14) and is deliberately left untouched.

    Degrades to a loud SKIP when the live TTS server on :5123 is down (module
    convention) or when the configured backend is not mlx (the case pins the
    engine-writer path, which is MLX-specific).
    """

    # Leading/trailing whitespace on purpose: the writer stores the STRIPPED
    # transcript, so the .txt content check proves the raw value reached the
    # writer and was coerced there (not pre-stripped away from it).
    TRANSCRIPT = "  Headless create flows through the engine writer.  "
    FIXTURE_RATE = 24000

    def setUp(self) -> None:
        if not self._server_reachable():
            self.skipTest(
                "live TTS server on :5123 is down — start it (tts server start) "
                "to run the create-from-audio E2E case"
            )
        try:
            import numpy  # noqa: F401
            import soundfile  # noqa: F401
            from gradio_client import handle_file  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"gradio_client/numpy/soundfile unavailable: {exc}")

        from qwen3_tts.core.config import get_backend

        if get_backend() != "mlx":
            self.skipTest(
                "configured backend is not mlx — the E2E create case pins the "
                "MLX engine-writer path"
            )

        from qwen3_tts.core.config import VOICE_PROMPTS_DIR

        self.voice_prompts_dir = VOICE_PROMPTS_DIR
        self.base_name = f"ui_e2e_engine_writer_{uuid.uuid4().hex[:8]}"
        self.tmp_dir = tempfile.mkdtemp(prefix="ui_create_e2e_")
        # Registered BEFORE anything can create the artifacts, so a failure
        # mid-test can never strand the pair in the real prompts dir.
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)
        self.addCleanup(self._remove_created_prompt_pair)

        import numpy as np
        import soundfile as sf

        # 1.0 s mono 24 kHz sine: at the native floor, so the writer
        # byte-copies (no resample) — and the containment guard passes because
        # gradio stages the upload under the system tempdir, which is one of
        # the writer's two allowed roots.
        t = np.linspace(0.0, 1.0, self.FIXTURE_RATE, endpoint=False)
        samples = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
        self.wav_path = os.path.join(self.tmp_dir, "ref.wav")
        sf.write(self.wav_path, samples, self.FIXTURE_RATE, subtype="PCM_16")

    @staticmethod
    def _server_reachable() -> bool:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(  # nosec B310
                f"{SERVER_URL}/health", timeout=2
            ) as resp:
                return resp.status == 200
        except (urllib.error.URLError, ConnectionError, OSError):
            return False

    def _remove_created_prompt_pair(self) -> None:
        for suffix in (".wav", ".txt"):
            path = self.voice_prompts_dir / f"{self.base_name}{suffix}"
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _terminate_ui(proc) -> None:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    @staticmethod
    def _choices(dropdown_result) -> list:
        """Extract choice values from a returned Dropdown payload."""
        if isinstance(dropdown_result, dict):
            raw = dropdown_result.get("choices") or []
        elif isinstance(dropdown_result, (list, tuple)):
            raw = dropdown_result
        else:
            raw = []
        values = []
        for entry in raw:
            if isinstance(entry, (list, tuple)) and entry:
                values.append(str(entry[-1]))  # (label, value) form
            else:
                values.append(str(entry))
        return values

    def test_create_from_audio_routes_through_the_engine_writer(self) -> None:
        from gradio_client import Client, handle_file

        launch_code = (
            f"import sys; sys.path.insert(0, '{PROJECT_DIR}'); "
            f"from qwen3_tts.interface.ui import build_ui; "
            f"demo = build_ui(); "
            f"import os; "
            f"demo.launch(server_name='127.0.0.1', server_port={UI_PORT}, "
            f"share=False, show_error=True, "
            f"allowed_paths=[os.path.realpath(os.path.expanduser('~/Downloads/Qwen3-TTS Output')), '/tmp'], "
            f"prevent_thread_lock=False)"
        )
        ui_proc = subprocess.Popen(  # nosec B603
            [sys.executable, "-c", launch_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )
        self.addCleanup(self._terminate_ui, ui_proc)

        wait_for_server(UI_URL, timeout=30)
        client = Client(UI_URL, verbose=False)
        result = client.predict(
            handle_file(self.wav_path),
            self.TRANSCRIPT,
            self.base_name,
            False,
            api_name="/create_voice_prompt",
        )

        self.assertIsInstance(
            result, (list, tuple), f"unexpected result shape: {result!r}"
        )
        self.assertEqual(len(result), 3, f"expected 3 outputs, got: {result!r}")
        status = str(result[0])
        self.assertEqual(
            status,
            f"Created MLX voice prompt: {self.base_name}",
            f"create did not succeed; status: {status}",
        )

        # The engine writer's artifacts, under the dir the UI resolves.
        wav_path = self.voice_prompts_dir / f"{self.base_name}.wav"
        txt_path = self.voice_prompts_dir / f"{self.base_name}.txt"
        self.assertTrue(wav_path.exists(), f"engine-writer .wav missing: {wav_path}")
        self.assertTrue(txt_path.exists(), f"engine-writer .txt missing: {txt_path}")
        self.assertEqual(
            txt_path.read_text(),
            self.TRANSCRIPT.strip(),
            "stored transcript must be the writer's stripped coercion",
        )

        # Write-time rate guarantee: >=24 kHz, mono.
        import soundfile as sf

        info = sf.info(str(wav_path))
        self.assertGreaterEqual(info.samplerate, 24000)
        self.assertEqual(info.channels, 1)

        # The refreshed lists reflect the creation.
        expected_choice = f"{self.base_name}.wav"
        for label, dropdown_result in zip(
            ("Available Voices", "Voice Prompt"), result[1:], strict=False
        ):
            self.assertIn(
                expected_choice,
                self._choices(dropdown_result),
                f"{label} dropdown does not list the created prompt",
            )


if __name__ == "__main__":
    sys.exit(main())
