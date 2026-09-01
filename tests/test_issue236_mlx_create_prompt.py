#!/usr/bin/env python3
"""Issue #236 -- MLX-native /create-voice-prompt (server endpoint).

``/create-voice-prompt`` has never worked on the MLX backend: the handler
routes every backend through the engine's torch-shaped
``create_voice_prompt`` -> ``model.create_voice_clone_prompt(...)``, an API
that NO mlx-audio version implements (verified 0.4.8 / 0.5.1 / upstream
master in docs/reviews/mlx-audio-0.5.1-evaluation-2026-09-01.md). Upstream
mlx-audio does cloning at GENERATION time (``prepare_zeroprompt``) and
consumes the prompt as a ``.wav+.txt`` pair; the CLI (``--mlx-only``) and
the UI already create that pair without any model. Only the server
endpoint is broken.

Contract pinned by these tests:

  * ``handle_create_voice_prompt(state, req, backend=...)`` dispatches on
    backend. On MLX: NO clone-loaded gate, NO ``inference_lock`` (no GPU
    work happens), the reference audio is validated
    (``ensure_min_sample_rate``) and stored as a ``.wav+.txt`` pair via the
    engine writer ``save_voice_prompt_mlx`` -- never via the torch engine
    ``create_voice_prompt``, never via the tools-layer writer, and no
    ``.pt`` is written.
  * Blank transcript with ``no_transcript=False`` -> 400
    ``transcript_required``, checked with ``.strip()`` (a whitespace-only
    transcript must not slip through a truthiness check) and BEFORE any
    audio decoding. ``no_transcript=True`` stores an empty ``.txt``
    (identical to what MLX generation produces with the flag today).
  * ``name=".pt"`` (empty base after extension strip) -> 400, not a
    filesystem write of ``".wav"``.
  * Client-input problems map to 4xx on the MLX branch: undecodable bytes
    -> 400 ``invalid_audio``; an unrepairable sub-24 kHz reference (the
    8 kHz prompt measured 3/3 token-cap runaways) -> 400
    ``unsupported_reference_audio`` -- NOT 500 ``creation_failed``.
  * The torch branch keeps today's shape, including the 503 gate when
    clone is not loaded (pinned with the branch FORCED via ``backend=
    "torch"`` so the test runs on every leg instead of being skipped).
  * The response is typed: ``CreateVoicePromptResponse(status, name)``.

RED-note (Gate A): the ``backend`` kwarg and ``save_voice_prompt_mlx``
are part of the deliverable -- tests passing ``backend=`` TypeError at
RED (that IS the missing dispatch, same class as T5's missing timeout
constant), classified genuine-RED in the ledger; the no-kwarg variants
drive TODAY's handler so the 503-instead-of-400 behavior gap is visible
independently of the signature.

Run: pytest tests/test_issue236_mlx_create_prompt.py -v --tb=short
"""

import asyncio
import base64
import io
import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import numpy as np
    import soundfile as sf

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

_skip = unittest.skipUnless(_HAS_DEPS, "requires numpy + soundfile")

_ENGINE_VOICE_PROMPT = "qwen3_tts.core.engine.voice_prompt"
_APP_PROMPTS = "qwen3_tts.server.app_prompts"
_TOOLS_CREATE = "qwen3_tts.tools.create_voice"


def _wav_bytes(seconds=1.0, rate=24000, channels=1, freq=220.0):
    """Real mono WAV bytes >=24 kHz (the e2e _wav_bytes pattern). Below-24 kHz
    fixtures need librosa to repair, which .venv-310 lacks -- gate those."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    if channels == 2:
        audio = np.stack([audio, audio], axis=-1)
    else:
        audio = np.stack([audio], axis=-1)
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def _make_state():
    """Complete app.state stand-in. Clone deliberately None: the MLX branch
    must not need it (that is the point), and the torch 503 pin needs it."""
    state = SimpleNamespace()
    state.auth_token = "test_token"  # nosec B105
    state.models = {"clone": None, "design": None, "custom": None}
    state.inference_lock = asyncio.Lock()
    state.gen_cache = {}
    state.gen_cache_lock = threading.Lock()
    state.server_config = {
        "security": {"max_text_length": 50000, "max_batch_size": 20},
    }
    return state


def _req(name="test_voice", audio_b64=None, transcript="hello world", no_transcript=False):
    from qwen3_tts.server.validation import CreateVoicePromptRequest

    return CreateVoicePromptRequest(
        audio_base64=audio_b64 if audio_b64 is not None else _wav_bytes(),
        name=name,
        transcript=transcript,
        no_transcript=no_transcript,
    )


class _PromptsDirPatch:
    """Patch BOTH independent VOICE_PROMPTS_DIR module bindings to one tmpdir.

    The server handler reads qwen3_tts.server.app_prompts's binding; the
    engine writer reads qwen3_tts.core.engine.voice_prompt's. Patching only
    one writes real files into the user's live voice_prompts/ directory
    (the mock-patch-seams trap). Use as a context manager.
    """

    def __init__(self, tmpdir):
        self.tmpdir = tmpdir
        self._cms = [
            patch(f"{_APP_PROMPTS}.VOICE_PROMPTS_DIR", tmpdir),
            patch(f"{_ENGINE_VOICE_PROMPT}.VOICE_PROMPTS_DIR", tmpdir),
        ]

    def __enter__(self):
        for cm in self._cms:
            cm.__enter__()
        return self.tmpdir

    def __exit__(self, *exc):
        for cm in reversed(self._cms):
            cm.__exit__(*exc)
        return False


@_skip
class TestMlxBranchCreatesPair(unittest.TestCase):
    """The MLX branch: no model, no lock, no tools, no .pt -- just a
    validated .wav+.txt pair."""

    def _drive_mlx(self, tmpdir, req):
        """Drive the handler with backend='mlx' under the tmpdir seams.
        Returns (result, writer_calls, tools_calls)."""
        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()
        writer_calls = []
        tools_calls = []

        # create=True: the writer is part of the deliverable. At RED the
        # recorder simply stays empty (the handler cannot call what does
        # not exist yet); at GREEN the stub delegates to the REAL writer so
        # the on-disk assertions below stay meaningful.
        def _writer_stub(base, audio_path, transcript):
            import importlib

            real = importlib.import_module(
                "qwen3_tts.core.engine.voice_prompt"
            ).save_voice_prompt_mlx
            writer_calls.append((base, transcript))
            return real(base, audio_path, transcript)

        async def _drive():
            with _PromptsDirPatch(tmpdir):
                with patch(
                    f"{_ENGINE_VOICE_PROMPT}.save_voice_prompt_mlx",
                    side_effect=_writer_stub,
                    create=True,
                ):
                    real_tools = __import__(
                        "qwen3_tts.tools.create_voice",
                        fromlist=["create_and_save_voice_prompt"],
                    ).create_and_save_voice_prompt
                    with patch.object(
                        real_tools,
                        "create_and_save_voice_prompt",
                        side_effect=lambda *a, **k: tools_calls.append(a),
                    ):
                        with patch(
                            f"{_ENGINE_VOICE_PROMPT}.create_voice_prompt",
                            side_effect=AssertionError(
                                "torch engine create must not run on the MLX branch"
                            ),
                        ):
                            return await handle_create_voice_prompt(
                                state, req, backend="mlx"
                            )

        result = asyncio.run(_drive())
        return result, writer_calls, tools_calls

    def test_mlx_branch_writes_wav_txt_pair(self):
        """The core deliverable: created -> a real .wav+.txt pair on disk,
        transcript stored stripped, and the .wav is 24 kHz mono."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, writer_calls, tools_calls, _locks = self._drive_mlx(
                tmpdir, _req(transcript="  hello world  ")
            )

            self.assertEqual(
                result,
                {"status": "created", "name": "test_voice"},
                f"unexpected response: {result!r}",
            )
            wav_path = os.path.join(tmpdir, "test_voice.wav")
            txt_path = os.path.join(tmpdir, "test_voice.txt")
            self.assertTrue(os.path.exists(wav_path), "no .wav written")
            self.assertTrue(os.path.exists(txt_path), "no .txt written")
            with open(txt_path) as f:
                self.assertEqual(f.read(), "hello world", "transcript must be stored stripped")
            info = sf.info(wav_path)
            self.assertEqual(info.samplerate, 24000)
            self.assertEqual(info.channels, 1)
            self.assertEqual(
                writer_calls,
                [("test_voice", "hello world")],
                "the engine writer must be invoked once with the stripped transcript",
            )
            self.assertEqual(
                tools_calls,
                [],
                "the server must not route through the tools-layer writer",
            )
            self.assertFalse(
                os.path.exists(os.path.join(tmpdir, "test_voice.pt")),
                "no .pt may be written on the MLX branch",
            )

    def test_mlx_branch_needs_neither_model_nor_lock(self):
        """Clone is None and inference_lock is never acquired: the MLX branch
        is inference-free by design. A recording wrapper around the REAL
        lock observes acquires without faking concurrency safety (a
        MagicMock lock is the documented anti-pattern)."""
        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        class _RecordingAsyncLock:
            def __init__(self):
                self._lock = asyncio.Lock()
                self.acquire_calls = 0

            async def acquire(self):
                self.acquire_calls += 1
                return await self._lock.acquire()

            def release(self):
                self._lock.release()

            async def __aenter__(self):
                await self.acquire()
                return self

            async def __aexit__(self, *exc):
                self.release()
                return False

            def locked(self):
                return self._lock.locked()

        with tempfile.TemporaryDirectory() as tmpdir:
            state = _make_state()
            state.inference_lock = _RecordingAsyncLock()

            async def _drive():
                with _PromptsDirPatch(tmpdir):
                    return await handle_create_voice_prompt(
                        state, _req(), backend="mlx"
                    )

            result = asyncio.run(_drive())
            self.assertEqual(result.get("status"), "created")
            self.assertIsNone(
                state.models.get("clone"),
                "precondition: clone stays None -- the branch must not need it",
            )
            self.assertEqual(
                state.inference_lock.acquire_calls,
                0,
                "the MLX branch acquired inference_lock -- it runs no inference",
            )

    def test_torch_engine_create_never_runs_on_mlx_branch(self):
        """The re-route mutant killer: if the MLX branch fell back to the
        torch path, the engine create (patched to raise) would explode and
        the clone-loaded gate would 503 first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _w, _t, _locks = self._drive_mlx(tmpdir, _req())
            self.assertEqual(result.get("status"), "created")

    def test_no_pt_written_even_when_torch_prompt_exists(self):
        """A pre-existing .pt must not gain anything; the MLX branch writes
        only the pair (mutant M-f: also writing a .pt)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(tmpdir, exist_ok=True)
            with open(os.path.join(tmpdir, "test_voice.pt"), "wb") as f:
                f.write(b"stale-torch-prompt")
            self._drive_mlx(tmpdir, _req())
            with open(os.path.join(tmpdir, "test_voice.pt"), "rb") as f:
                self.assertEqual(
                    f.read(),
                    b"stale-torch-prompt",
                    "the MLX branch must not touch an existing .pt",
                )


@_skip
class TestTranscriptPolicy(unittest.TestCase):
    """Blank transcript + no_transcript=False -> 400 on BOTH backends
    (torch already fails this request with a 500; 400 is strictly better).
    no_transcript=True stores an empty .txt on MLX."""

    def test_blank_transcript_400_before_decode(self):
        """Strip-based (whitespace-only must NOT slip through) and fail-fast
        (before any base64 decoding of a near-100 MB body)."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()
        decode_calls = []

        async def _drive():
            with patch(
                f"{_APP_PROMPTS}._decode_audio",
                side_effect=lambda b64: decode_calls.append(1),
            ):
                return await handle_create_voice_prompt(
                    state,
                    _req(transcript="   ", audio_b64=_wav_bytes()),
                    backend="mlx",
                )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(_drive())

        self.assertEqual(ctx.exception.status_code, 400)
        detail = ctx.exception.detail
        self.assertIsInstance(detail, dict, f"expected structured detail: {detail!r}")
        self.assertEqual(detail.get("error"), "transcript_required")
        self.assertEqual(
            decode_calls,
            [],
            "the 400 must fire before any audio decoding (fail fast)",
        )

    def test_blank_transcript_400_on_torch_branch_too(self):
        """The policy is backend-independent: torch changes only for an
        already-failing request (upstream raised 500 today)."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()
        state.models["clone"] = MagicMock(name="clone-model")

        async def _drive():
            return await handle_create_voice_prompt(
                state, _req(transcript=""), backend="torch"
            )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(_drive())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail.get("error"), "transcript_required")

    def test_blank_transcript_400_without_backend_kwarg(self):
        """No-kwarg RED variant: TODAY's handler (no dispatch) answers 503
        (the clone gate) for this request -- the 400 policy is missing
        independent of the new signature. On the mlx dev env the ambient
        backend is mlx; the assertion below is about the STATUS, which is
        503 today and 400 after the fix."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()

        async def _drive():
            return await handle_create_voice_prompt(
                state, _req(transcript="   ", audio_b64=_wav_bytes())
            )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(_drive())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail.get("error"), "transcript_required")

    def test_no_transcript_true_stores_empty_txt(self):
        """no_transcript=True is allowed: the pair is written with an empty
        .txt -- byte-identical to what MLX generation produces under the
        flag today (ref_text='')."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _w, _t, _locks = self._drive_mlx(
                tmpdir, _req(transcript="", no_transcript=True)
            )
            self.assertEqual(result.get("status"), "created")
            txt_path = os.path.join(tmpdir, "test_voice.txt")
            self.assertTrue(os.path.exists(txt_path))
            with open(txt_path) as f:
                self.assertEqual(f.read(), "")


@_skip
class TestMlxBranchErrorMapping(unittest.TestCase):
    """Client-input problems are 4xx on the MLX branch -- never 500
    creation_failed/unknown_error."""

    def test_invalid_name_dot_pt_400(self):
        """name='.pt' strips to an empty base: 400 invalid_name, and
        definitely no file literally named '.wav'."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()

        async def _drive():
            with tempfile.TemporaryDirectory() as tmpdir:
                with _PromptsDirPatch(tmpdir):
                    return await handle_create_voice_prompt(
                        state, _req(name=".pt"), backend="mlx"
                    )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(_drive())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail.get("error"), "invalid_name")

    def test_undecodable_audio_400_invalid_audio(self):
        """Bytes that soundfile cannot decode (and the staged .wav suffix
        means no pydub fallback server-side) -> 400 invalid_audio, not
        500 unknown_error/bug."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()
        garbage = base64.b64encode(b"this is not audio data at all").decode()

        async def _drive():
            with tempfile.TemporaryDirectory() as tmpdir:
                with _PromptsDirPatch(tmpdir):
                    return await handle_create_voice_prompt(
                        state, _req(audio_b64=garbage), backend="mlx"
                    )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(_drive())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail.get("error"), "invalid_audio")

    def test_unrepairable_sample_rate_400_unsupported_reference_audio(self):
        """An 8 kHz reference (3/3 token-cap runaways when used) that cannot
        be repaired -> 400 unsupported_reference_audio, NOT 500
        creation_failed/retry. librosa is blocked via sys.modules so the
        ensure_min_sample_rate guarantee fails deterministically in every
        env (the .venv-310 CI proxy has no librosa; the mlx env does)."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()
        low_rate_b64 = _wav_bytes(seconds=0.5, rate=8000)

        async def _drive():
            with tempfile.TemporaryDirectory() as tmpdir:
                with _PromptsDirPatch(tmpdir):
                    with patch.dict(sys.modules, {"librosa": None}):
                        return await handle_create_voice_prompt(
                            state, _req(audio_b64=low_rate_b64), backend="mlx"
                        )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(_drive())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail.get("error"), "unsupported_reference_audio")
        self.assertEqual(ctx.exception.detail.get("recovery"), "config")

    def test_unsupported_rate_400_without_backend_kwarg(self):
        """No-kwarg RED variant: today's handler 503s at the clone gate for
        this request -- the 4xx mapping is missing independent of the
        signature."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()
        low_rate_b64 = _wav_bytes(seconds=0.5, rate=8000)

        async def _drive():
            with tempfile.TemporaryDirectory() as tmpdir:
                with _PromptsDirPatch(tmpdir):
                    with patch.dict(sys.modules, {"librosa": None}):
                        return await handle_create_voice_prompt(
                            state, _req(audio_b64=low_rate_b64)
                        )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(_drive())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail.get("error"), "unsupported_reference_audio")


@_skip
class TestCacheAndListing(unittest.TestCase):
    """Cache invalidation after create; /prompts intersection listing."""

    def test_cache_cleared_after_mlx_create(self):
        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()
        clears = []

        async def _drive():
            with tempfile.TemporaryDirectory() as tmpdir:
                with _PromptsDirPatch(tmpdir):
                    with patch(
                        "qwen3_tts.core.engine.clear_voice_prompt_cache",
                        side_effect=lambda: clears.append(1),
                    ):
                        return await handle_create_voice_prompt(
                            state, _req(), backend="mlx"
                        )

        result = asyncio.run(_drive())
        self.assertEqual(result.get("status"), "created")
        self.assertEqual(
            clears,
            [1],
            "the engine prompt caches must be invalidated after a create "
            "(delete/rename/create all follow this rule)",
        )

    def test_prompts_lists_new_mlx_prompt(self):
        """/prompts on MLX lists .wav bases that have a matching .txt
        (intersection semantics -- net-new unit coverage for the listing
        side of the create)."""
        from qwen3_tts.server.app_prompts import (
            handle_create_voice_prompt,
            handle_list_prompts,
        )

        state = _make_state()

        async def _drive():
            with tempfile.TemporaryDirectory() as tmpdir:
                with _PromptsDirPatch(tmpdir):
                    await handle_create_voice_prompt(state, _req(), backend="mlx")
                    return handle_list_prompts(state, "mlx", None)

        listing = asyncio.run(_drive())
        names = [p if isinstance(p, str) else p.get("name", "") for p in listing["prompts"]]
        self.assertIn(
            "test_voice.wav",
            names,
            f"the freshly created pair must be listed; got {names}",
        )


@_skip
class TestTorchBranchPins(unittest.TestCase):
    """GREEN-by-design pins: the torch path keeps today's shape. These are
    classification pins, not REDs (the plan's Gate A ledger says so)."""

    def test_torch_503_gate_stays_live_forced_backend(self):
        """Forced backend='torch' (a direct handler call, so the pin runs on
        every leg): clone not loaded -> 503, exactly today's gate."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()

        async def _drive():
            with tempfile.TemporaryDirectory() as tmpdir:
                with _PromptsDirPatch(tmpdir):
                    return await handle_create_voice_prompt(
                        state, _req(), backend="torch"
                    )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(_drive())
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("Clone model must be loaded", str(ctx.exception.detail))

    def test_route_source_dispatches_with_get_backend(self):
        """Route-shape guard: the endpoint must pass get_backend() into the
        handler (the dispatch arg cannot be dropped silently), and must
        still await it directly (the #192 pin)."""
        import inspect

        from qwen3_tts.server import app as app_module

        src = inspect.getsource(app_module.create_voice_prompt_endpoint)
        self.assertIn(
            "await handle_create_voice_prompt(",
            src,
            "the route must await the async handler directly (#192 pin)",
        )
        self.assertIn(
            "get_backend",
            src,
            "the route must pass get_backend() into the handler so the "
            "MLX/torch dispatch actually engages",
        )

    def test_mlx_load_path_stays_create_free(self):
        """The #214 guard must survive: load_voice_prompt_mlx never creates
        (the fix adds a save function beside it, never inside the load
        path)."""
        import inspect

        from qwen3_tts.core.engine import voice_prompt

        src = inspect.getsource(voice_prompt.load_voice_prompt_mlx)
        self.assertNotIn(
            "save_voice_prompt_mlx",
            src,
            "the MLX load path must stay create-free (#214 pin)",
        )
        self.assertNotIn("sf.write", src)


if __name__ == "__main__":
    unittest.main()
