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
    ``transcript_required`` on BOTH backends, checked with ``.strip()``
    (whitespace-only must not slip through a truthiness check) and BEFORE
    any audio decoding. ``no_transcript=True`` stores an empty ``.txt``
    plus the ``.wav`` (identical to what MLX generation produces with the
    flag today).
  * ``name=".pt"`` (empty base after extension strip) -> 400
    ``invalid_name`` on both branches.
  * Client-input problems map to 4xx on the MLX branch: undecodable bytes
    -> 400 ``invalid_audio``; an unrepairable sub-24 kHz reference (the
    8 kHz prompt measured 3/3 token-cap runaways) -> 400
    ``unsupported_reference_audio`` -- NOT 500 ``creation_failed``.
  * The torch branch keeps today's shape, including the 503 gate when
    clone is not loaded (pinned with the branch FORCED via ``backend=
    "torch"`` so the test runs on every leg instead of being skipped).
  * The route passes ``get_backend()`` INTO the handler call (AST-pinned:
    an argument, not a mention) and the dispatch behaves through the
    patched ``qwen3_tts.server.app.get_backend`` seam.
  * The typed response model (``CreateVoicePromptResponse``) is pinned at
    the wire level in tests/test_response_contracts.py (deliverable 4 of
    the plan), not here.

Classification ledger (Gate A): the ``backend=`` tests TypeError at RED
-- that IS the missing dispatch (same class as T5's missing timeout
constant). The two no-kwarg variants drive TODAY's handler so the
503-vs-400 behavior gap is visible independently of the signature. The
rewrite-property case is librosa-gated (``.venv-310`` has no librosa);
the librosa-BLOCKED 400 test is its deterministic-every-env complement
for the negative path.

Run: pytest tests/test_issue236_mlx_create_prompt.py -v --tb=short
"""

import ast
import asyncio
import base64
import inspect
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

try:
    import librosa  # noqa: F401

    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

_ENGINE_VOICE_PROMPT = "qwen3_tts.core.engine.voice_prompt"
_APP_PROMPTS = "qwen3_tts.server.app_prompts"


def _wav_bytes(seconds=1.0, rate=24000, channels=1, freq=220.0):
    """Real WAV bytes. The default (24 kHz mono) exercises the byte-copy
    path; sub-24 kHz/stereo fixtures exercise the ensure_min_sample_rate
    rewrite, which needs librosa -- gate those with _HAS_LIBROSA."""
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
    state.auth_token = "test_token"
    state.models = {"clone": None, "design": None, "custom": None}
    state.inference_lock = asyncio.Lock()
    state.gen_cache = {}
    state.gen_cache_lock = threading.Lock()
    state.last_activity = 0.0
    state.shutdown_timer = None
    state.server_config = {
        "security": {"max_text_length": 50000, "max_batch_size": 20},
        "auto_shutdown_minutes": 0,
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

    The server handler/listing read qwen3_tts.server.app_prompts's binding;
    the engine writer reads qwen3_tts.core.engine.voice_prompt's. Patching
    only one writes real files into the user's live voice_prompts/
    directory (the mock-patch-seams trap). Use as a context manager.
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


def _drive_mlx(tmpdir, req, state=None):
    """Drive the handler with backend='mlx' under the tmpdir seams.

    Module-level (used from several classes). Returns
    (result, writer_calls, tools_calls). The writer stub delegates to the
    REAL save_voice_prompt_mlx, captured BEFORE the patch scope (resolving
    it inside the ``with patch`` would recurse on the patched attribute --
    Gate A round 1); at RED the writer does not exist, the recorder stays
    empty, and any call raises instead of silently passing.
    """
    import qwen3_tts.core.engine as facade  # the package __init__ IS the facade
    from qwen3_tts.server.app_prompts import handle_create_voice_prompt
    from qwen3_tts.tools import create_voice as tools_create

    if state is None:
        state = _make_state()
    writer_calls = []
    tools_calls = []
    # Seam pinned with the implementation contract: the handler imports the
    # writer FUNCTION-LOCALLY FROM THE FACADE (house rule), so the facade
    # binding is the patch that intercepts; a submodule patch would be inert.
    real_writer = getattr(facade, "save_voice_prompt_mlx", None)

    def _writer_stub(base, audio_path, transcript):
        writer_calls.append((base, transcript))
        if real_writer is None:
            raise AssertionError("save_voice_prompt_mlx is not implemented yet")
        return real_writer(base, audio_path, transcript)

    async def _drive():
        with _PromptsDirPatch(tmpdir):
            with patch.object(
                facade, "save_voice_prompt_mlx", side_effect=_writer_stub, create=True
            ):
                with patch.object(
                    tools_create,
                    "create_and_save_voice_prompt",
                    side_effect=lambda *a, **k: tools_calls.append(a),
                ):
                    with patch(
                        # create_voice_prompt lives in inference.py; the handler
                        # resolves the FACADE attr at call time — submodule
                        # patches are inert here too (Gate A round 2, fastapi).
                        "qwen3_tts.core.engine.create_voice_prompt",
                        side_effect=AssertionError(
                            "torch engine create must not run on the MLX branch"
                        ),
                    ):
                        return await handle_create_voice_prompt(
                            state, req, backend="mlx"
                        )

    result = asyncio.run(_drive())
    return result, writer_calls, tools_calls


class TestMlxBranchCreatesPair(unittest.TestCase):
    """The MLX branch: no model, no lock, no tools, no .pt -- just a
    validated .wav+.txt pair."""

    def test_mlx_branch_writes_wav_txt_pair(self):
        """The core deliverable: created -> a real .wav+.txt pair on disk,
        transcript stored stripped, and the .wav is 24 kHz mono."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, writer_calls, tools_calls = _drive_mlx(
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
                "the engine writer must be invoked once with the stripped transcript "
                "(kills any reroute through the torch path or the tools writer)",
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

    def test_rewrite_property_sub24k_stereo_lands_24k_mono(self):
        """The REWRITE pin (not the no-op copy): a 16 kHz stereo reference
        must land on disk as 24 kHz mono -- a below-native-rate prompt made
        MLX clone generation run to the token cap 3/3 times. Needs librosa
        to repair the rate; skipped where it is absent (the librosa-BLOCKED
        400 test is the deterministic complement for the failure path)."""
        if not _HAS_LIBROSA:
            self.skipTest("librosa not installed -- rate repair unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _w, _t = _drive_mlx(
                tmpdir, _req(audio_b64=_wav_bytes(seconds=0.5, rate=16000, channels=2))
            )
            self.assertEqual(result.get("status"), "created")
            info = sf.info(os.path.join(tmpdir, "test_voice.wav"))
            self.assertEqual(info.samplerate, 24000, "sub-24 kHz must be rewritten up")
            self.assertEqual(info.channels, 1, "stereo must be downmixed")

    def test_native_rate_stereo_still_downmixed(self):
        """Gate B round 1 (agy CRITICAL, empirically confirmed): a NATIVE-
        rate stereo file must still land as mono. The helper folds its own
        downmix into was_modified, so pre-mixing outside it made the writer
        byte-copy the ORIGINAL STEREO file. Runs everywhere (no librosa)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _w, _t = _drive_mlx(
                tmpdir, _req(audio_b64=_wav_bytes(seconds=0.5, channels=2))
            )
            self.assertEqual(result.get("status"), "created")
            info = sf.info(os.path.join(tmpdir, "test_voice.wav"))
            self.assertEqual(info.channels, 1, "stereo must be downmixed")
            self.assertEqual(info.samplerate, 24000)

    def test_native_rate_mono_is_a_faithful_copy(self):
        """The copy path stays byte-exact for an already-24 kHz mono file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _w, _t = _drive_mlx(tmpdir, _req())
            self.assertEqual(result.get("status"), "created")
            src_wav = os.path.join(tmpdir, "test_voice.wav")
            audio_in, sr_in = sf.read(io.BytesIO(base64.b64decode(_req().audio_base64)))
            audio_out, sr_out = sf.read(src_wav)
            self.assertEqual(sr_out, sr_in)
            self.assertEqual(len(audio_out), len(audio_in))

    def test_mlx_branch_needs_neither_model_nor_lock(self):
        """Clone is None and inference_lock is never acquired: the MLX branch
        is inference-free by design. A recording wrapper around the REAL
        lock observes acquires without faking concurrency safety."""
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

    def test_no_pt_written_even_when_torch_prompt_exists(self):
        """A pre-existing .pt must not gain anything; the MLX branch writes
        only the pair (mutant M-f: also writing a .pt). Asserts the drive
        SUCCEEDED first -- otherwise the .pt-untouched assertion would be
        hollow (a 400/503 path also writes nothing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(tmpdir, exist_ok=True)
            pt_path = os.path.join(tmpdir, "test_voice.pt")
            with open(pt_path, "wb") as f:
                f.write(b"stale-torch-prompt")
            result, _w, _t = _drive_mlx(tmpdir, _req())
            self.assertEqual(
                result.get("status"),
                "created",
                "precondition failed: the create itself did not succeed",
            )
            with open(pt_path, "rb") as f:
                self.assertEqual(
                    f.read(),
                    b"stale-torch-prompt",
                    "the MLX branch must not touch an existing .pt",
                )


class TestTranscriptPolicy(unittest.TestCase):
    """Blank transcript + no_transcript=False -> 400 on BOTH backends
    (torch already fails this request with a 500; 400 is strictly better).
    no_transcript=True stores an empty .txt plus the .wav."""

    def test_blank_transcript_400_before_decode(self):
        """Strip-based (whitespace-only must NOT slip through) and fail-fast
        (before any base64 decoding of a near-100 MB body). The decode stub
        returns REAL bytes so a dropped blank-check dies on the 400
        assertion, not on a None-type crash inside staging (Gate A round 1,
        agy)."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()
        decode_calls = []
        real_bytes = base64.b64decode(_wav_bytes())

        async def _drive():
            with patch(
                f"{_APP_PROMPTS}._decode_audio",
                side_effect=lambda b64: decode_calls.append(1) or real_bytes,
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
        independent of the new signature."""
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

    def test_no_transcript_true_stores_empty_txt_and_wav(self):
        """no_transcript=True is allowed: the pair is written -- empty .txt
        AND the .wav audio alongside it (Gate A round 1, agy)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _w, _t = _drive_mlx(
                tmpdir, _req(transcript="", no_transcript=True)
            )
            self.assertEqual(result.get("status"), "created")
            txt_path = os.path.join(tmpdir, "test_voice.txt")
            wav_path = os.path.join(tmpdir, "test_voice.wav")
            self.assertTrue(os.path.exists(txt_path), "no .txt written")
            self.assertTrue(
                os.path.exists(wav_path), "no .wav written alongside the empty .txt"
            )
            with open(txt_path) as f:
                self.assertEqual(f.read(), "")


class TestMlxBranchErrorMapping(unittest.TestCase):
    """Client-input problems are 4xx on the MLX branch -- never 500
    creation_failed/unknown_error."""

    def test_invalid_name_dot_pt_400_mlx(self):
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

    def test_invalid_name_dot_pt_400_torch(self):
        """The empty-base rejection is branch-independent (it fires before
        any backend divergence): forced-torch variant (Gate A round 1,
        agy gap)."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()

        async def _drive():
            with tempfile.TemporaryDirectory() as tmpdir:
                with _PromptsDirPatch(tmpdir):
                    return await handle_create_voice_prompt(
                        state, _req(name=".pt"), backend="torch"
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
        env (its function-local ``import librosa`` hits the None entry)."""
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
                    with patch.dict(os.environ, {"TTS_BACKEND": "mlx"}):
                        # TTS_BACKEND pins the AMBIENT resolution (no kwarg)
                        # deterministically: get_backend() honors the env
                        # first, so this passes on torch-default CI too.
                        with patch.dict(sys.modules, {"librosa": None}):
                            return await handle_create_voice_prompt(
                                state, _req(audio_b64=low_rate_b64)
                            )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(_drive())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail.get("error"), "unsupported_reference_audio")


class TestCacheAndListing(unittest.TestCase):
    """Cache invalidation after create; /prompts intersection listing."""

    def test_cache_cleared_after_mlx_create(self):
        """Both seams are recorded: the handler may import the clear
        function from the engine facade OR the voice_prompt submodule --
        patching only one would fail a correct implementation (Gate A
        round 1, agy)."""
        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        state = _make_state()
        clears = []

        def _record():
            clears.append(1)

        async def _drive():
            with tempfile.TemporaryDirectory() as tmpdir:
                with _PromptsDirPatch(tmpdir):
                    with patch(
                        "qwen3_tts.core.engine.clear_voice_prompt_cache",
                        side_effect=_record,
                    ):
                        with patch(
                            "qwen3_tts.core.engine.voice_prompt.clear_voice_prompt_cache",
                            side_effect=_record,
                        ):
                            return await handle_create_voice_prompt(
                                state, _req(), backend="mlx"
                            )

        result = asyncio.run(_drive())
        self.assertEqual(result.get("status"), "created")
        self.assertEqual(
            len(clears),
            1,
            "the engine prompt caches must be invalidated exactly once after "
            "a create (delete/rename/create all follow this rule)",
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
        # handle_list_prompts calls query_params.get(...): a plain dict works.
        query_params = {"offset": "0", "limit": "0"}

        async def _drive():
            with tempfile.TemporaryDirectory() as tmpdir:
                with _PromptsDirPatch(tmpdir):
                    await handle_create_voice_prompt(state, _req(), backend="mlx")
                    # An orphan .wav (no .txt) must stay INVISIBLE on the MLX
                    # listing: proves intersection, not union.
                    with open(os.path.join(tmpdir, "orphan.wav"), "wb") as f:
                        f.write(b"RIFF-mock")
                    return handle_list_prompts(state, "mlx", query_params)

        listing = asyncio.run(_drive())
        names = [p if isinstance(p, str) else p.get("name", "") for p in listing["prompts"]]
        self.assertIn(
            "test_voice.wav",
            names,
            f"the freshly created pair must be listed; got {names}",
        )
        self.assertNotIn(
            "orphan.wav",
            names,
            "a .wav without a .txt must NOT be listed on MLX (intersection, not union)",
        )


class TestRouteDispatch(unittest.TestCase):
    """M-e killers: the route must pass get_backend() INTO the handler
    call (AST: an argument, not a mention), and the dispatch must behave
    through the qwen3_tts.server.app.get_backend seam (the module-scope
    from-import means facade/definition patches do not reach it)."""

    def test_route_source_passes_get_backend_as_an_argument(self):
        """AST guard: get_backend(...) must appear as an argument of the
        handle_create_voice_prompt call. A substring check survives
        unused-import/comment mutants (Gate A round 1, both reviewers)."""
        from qwen3_tts.server import app as app_module

        src = inspect.getsource(app_module.create_voice_prompt_endpoint)
        self.assertIn(
            "await handle_create_voice_prompt(",
            src,
            "the route must await the async handler directly (#192 pin)",
        )
        tree = ast.parse(src)
        dispatch_args = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and "handle_create_voice_prompt" in ast.unparse(
                node.func
            ):
                dispatch_args = [ast.unparse(a) for a in node.args] + [
                    ast.unparse(k.value) for k in node.keywords
                ]
        self.assertTrue(
            any("get_backend" in a for a in dispatch_args),
            f"get_backend() must be an ARGUMENT of the dispatch call; got {dispatch_args}",
        )

    def test_route_dispatch_behavioral_via_app_get_backend_seam(self):
        """Behavioral M-e killer: with qwen3_tts.server.app.get_backend
        patched to 'mlx' (the only patch that reaches the route's
        module-scope from-import), the endpoint must run the MLX branch --
        on a torch-default machine (CI) an ignored/ambient-resolved backend
        would 503 instead."""
        from starlette.datastructures import Address
        from starlette.requests import Request

        from qwen3_tts.server import app as app_module

        state = _make_state()
        scope = {
            "type": "http",
            "app": SimpleNamespace(state=state),
            "headers": [],
            "path": "/create-voice-prompt",
            "method": "POST",
            "client": Address("127.0.0.1", 51000),
            "query_string": b"",
        }
        request = Request(scope)

        async def _drive():
            with tempfile.TemporaryDirectory() as tmpdir:
                with _PromptsDirPatch(tmpdir):
                    with patch.object(app_module, "get_backend", return_value="mlx"):
                        with patch(
                            "qwen3_tts.core.engine.save_voice_prompt_mlx",
                            create=True,
                            side_effect=lambda *a, **k: os.path.join(
                                tmpdir, "test_voice.wav"
                            ),
                        ):
                            with patch(
                                # Facade seam: the torch branch resolves
                                # create_voice_prompt from the engine facade at
                                # call time; a submodule patch is inert here.
                                "qwen3_tts.core.engine.create_voice_prompt",
                                side_effect=AssertionError("torch path must not run"),
                            ):
                                return await app_module.create_voice_prompt_endpoint(
                                    request, _req(), None
                                )

        result = asyncio.run(_drive())
        self.assertEqual(
            result.get("status"),
            "created",
            "the route ignored the patched get_backend() -- the dispatch does "
            "not engage through the app.get_backend seam (mutant M-e)",
        )


class TestLoadPathStaysCreateFree(unittest.TestCase):
    """The #214 guard must survive: load_voice_prompt_mlx never creates."""

    def test_mlx_load_path_stays_create_free(self):
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
