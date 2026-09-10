"""#193 / Step 1C Task 2: unlocked ASR ensure-load before the lock queue.

``_trim_icl_echo`` probes ASR under ``inference_lock`` (it runs inside
``run_inference``). When ASR is not loaded, the probe cannot run -- and since
#214 item 2 it must never lazily rebuild the model in-lock. So on a fresh
server every clone generation silently shipped untrimmed.

The fix (ratified design, option (b) keep-loaded): the server layer
ensure-loads ASR UNLOCKED in the generate paths, BEFORE the request queues
for ``inference_lock``, gated on the same three conditions the probe itself
requires (``generation.trim_icl_echo`` true + ``mode == "clone"`` + a
resolvable transcript). It mirrors the ``/transcribe`` ensure-load pattern
(``app_models.py``): unlocked load via ``asyncio.to_thread``, then the
existing under-lock degradation in ``_trim_icl_echo`` covers the
unload-in-the-window race (``handle_unload_asr`` holds ``inference_lock``,
so an unload can only land in the unlocked preload window).

The pins below drive the REAL ``handle_generate`` / ``handle_generate_stream``
with mocked inference (harness pattern from
``tests/test_batch_generation_state_ownership.py``) and patch at definition
sites: ``qwen3_tts.core.engine.asr.*``, the prompt loader, the engine facade.

Run: cd ~/Qwen3-TTS_UserFiles && conda run -n qwen3-tts-mlx python -m pytest tests/test_echo_trim_asr_preload.py -v

No GPU, models, or running server required.
"""

import asyncio
import contextlib
import os
import struct
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

try:
    import numpy as np  # noqa: F401
    import soundfile  # noqa: F401

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False  # noqa: F811

_skip = unittest.skipUnless(_HAS_DEPS, "requires numpy + soundfile")

_APP_GENERATION = "qwen3_tts.server.app_generation"
_ENGINE = "qwen3_tts.core.engine"
_ASR = "qwen3_tts.core.engine.asr"

REFERENCE = "This is my reference recording for cloning, thanks for listening."
SR = 24000


def _make_state():
    """Minimal app.state for exercising the real generation handlers.

    Mirrors the harness in tests/test_batch_generation_state_ownership.py,
    plus the pending-queue pieces the streaming path touches.
    """
    state = SimpleNamespace()
    state.auth_token = "test_token"  # nosec B105
    state.models = {
        "clone": MagicMock(),
        "design": MagicMock(),
        "custom": MagicMock(),
    }
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    state.generation_state = {
        "active": False,
        "start_time": 0.0,
        "text_length": 0,
        "mode": "",
        "batch_index": 0,
        "batch_total": 0,
        "chunk_index": 0,
        "chunk_total": 0,
        "generation_id": None,
        "cancelled": False,
    }
    state.request_queue = set()
    state.request_queue_lock = threading.Lock()
    _glock = AsyncMock()
    _glock.__aenter__.return_value = None
    _glock.__aexit__.return_value = None
    state.generation_lock = _glock
    state.pending_requests = []
    state.pending_lock = asyncio.Lock()
    state.last_activity = 0
    state.models_loaded = threading.Event()
    state.models_loaded.set()
    state.gen_cache = {}
    state.gen_cache_lock = threading.Lock()
    state.inference_lock = asyncio.Lock()
    state.eta_cache = {"median_rate": None, "last_updated": 0}
    state.eta_cache_lock = threading.Lock()
    state.shutdown_timer = None
    state.server_config = {
        "security": {"max_text_length": 50000, "max_batch_size": 20},
        "auto_shutdown_minutes": 0,
        "vllm": {"enabled": False, "fallback_to_torch": True},
    }
    return state


def _make_request(state):
    request = MagicMock()
    request.app.state = state
    request.headers = {"accept": "application/json"}
    return request


class _RecordingLock:
    """Wrap an asyncio.Lock, appending entries/exits to a shared event list.

    Gives the ordering pins a lock-entry marker to compare against the
    preload marker: the ASR ensure-load must complete BEFORE the handler
    first acquires inference_lock.
    """

    def __init__(self, events):
        self._inner = asyncio.Lock()
        self._events = events

    async def __aenter__(self):
        self._events.append("lock-enter")
        return await self._inner.__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        self._events.append("lock-exit")
        return await self._inner.__aexit__(exc_type, exc, tb)


class _StubConfigProvider:
    """Minimal ConfigLoader stand-in returning a controlled config dict.

    Keeps the pins off the real config.json: the preload gate reads
    ``generation.trim_icl_echo`` from whatever this returns.
    """

    def __init__(self, generation=None):
        self._generation = (
            {"trim_icl_echo": True} if generation is None else generation
        )
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        return {"generation": dict(self._generation)}


class _Mocks(SimpleNamespace):
    """Shared mock set: ASR state, ASR load, prompt loader, inference."""


def _make_mocks(events, threads, wav, *, streaming=False, transcript=REFERENCE):
    """Build the mock set driving the real handlers without real models.

    ASR state is modeled faithfully: the load mock flips a shared flag that
    the is_asr_loaded mock reads, so a successful preload makes every later
    check (the next batch item's, the engine's in-lock probe) see a warm
    model. Tests override either mock to simulate colder scenarios.
    """
    asr_state = {"loaded": False}

    def _is_asr_loaded():
        return asr_state["loaded"]

    def _load_asr_model():
        asr_state["loaded"] = True
        events.append("preload")
        threads.append(threading.current_thread())
        return True

    if streaming:

        def _inference(**kwargs):
            events.append("inference")
            return iter([(wav, SR)])

    else:

        def _inference(model, text, **kwargs):
            events.append("inference")
            return wav, SR

    return _Mocks(
        events=events,
        threads=threads,
        wav=wav,
        asr_state=asr_state,
        is_asr_loaded=MagicMock(side_effect=_is_asr_loaded),
        load_asr_model=MagicMock(side_effect=_load_asr_model),
        prompt_loader=AsyncMock(
            return_value={"transcript": transcript}
            if transcript is not None
            else {"ref_audio": "only-audio-no-text"}
        ),
        run_inference=MagicMock(side_effect=_inference),
    )


@contextlib.contextmanager
def _driven(mocks, *, streaming=False, probe=None):
    """Apply the patch set for driving the real handlers.

    Seams: ASR helpers at their definition site (the preload helper does
    attribute access on the asr module at call time), the prompt loader at
    its definition site in server/prompt_loading (function-local import at
    call time), inference on the engine facade (function-local import at
    call time), and the handler's own module-global validation/memory names.
    """
    engine_name = "run_inference_streaming" if streaming else "run_inference"
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch(
                f"{_APP_GENERATION}._check_memory_available",
                return_value=(True, 4096),
            )
        )
        stack.enter_context(
            patch(f"{_APP_GENERATION}._validate_generation_request")
        )
        stack.enter_context(
            patch(
                "qwen3_tts.server.prompt_loading.load_voice_prompt_serialized",
                mocks.prompt_loader,
            )
        )
        stack.enter_context(
            patch(f"{_ENGINE}.{engine_name}", mocks.run_inference)
        )
        stack.enter_context(
            patch(f"{_ASR}.is_asr_loaded", mocks.is_asr_loaded)
        )
        stack.enter_context(
            patch(f"{_ASR}.load_asr_model", mocks.load_asr_model)
        )
        if probe is not None:
            stack.enter_context(
                patch(
                    "qwen3_tts.core.engine.inference._transcribe_probe",
                    probe,
                )
            )
        yield


def _cleanup_cache(state):
    for entry in state.gen_cache.values():
        path = entry.get("main_file") or entry.get("file")
        if path and os.path.exists(path):
            os.unlink(path)


@_skip
class TestBatchEchoTrimPreload(unittest.TestCase):
    """handle_generate must ensure-load ASR unlocked, before the lock queue."""

    def test_preload_fires_once_before_the_lock(self):
        """Clone + trim on + transcript + ASR unloaded: one load, pre-lock.

        Two items: the load fires for item 0 only -- the is_asr_loaded()
        short-circuit makes item 1's repeat free. The event list pins the
        order strictly: preload BEFORE the first lock-entry, inference on
        every item after it. The load must also run OFF the event loop
        (async-offload policy, cf. tests/test_server_async_offload.py).
        """
        events = []
        threads = []
        wav = np.zeros(500, dtype=np.float32)
        mocks = _make_mocks(events, threads, wav)

        async def run():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            state.inference_lock = _RecordingLock(events)
            req = GenerateRequest(
                texts=["first", "second"],
                mode="clone",
                prompt_file="voice.wav",
            )
            try:
                with _driven(mocks):
                    return await handle_generate(
                        request=_make_request(state),
                        state=state,
                        req=req,
                        security={"max_text_length": 50000, "max_batch_size": 20},
                        config_provider=_StubConfigProvider(),
                    )
            finally:
                _cleanup_cache(state)

        result = asyncio.run(run())

        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(
            mocks.load_asr_model.call_count,
            1,
            "the ASR ensure-load must fire exactly once per batch, not "
            "once per item",
        )
        self.assertEqual(
            events,
            ["preload", "lock-enter", "inference", "lock-exit",
             "lock-enter", "inference", "lock-exit"],
            "the preload must complete BEFORE the generation first queues "
            "for inference_lock",
        )
        self.assertLess(events.index("preload"), events.index("lock-enter"))
        for thread in threads:
            self.assertIsNot(
                thread,
                threading.current_thread(),
                "load_asr_model ran on the event-loop thread; it must be "
                "offloaded via asyncio.to_thread",
            )

    def test_design_mode_never_preloads(self):
        """mode != clone can never echo, so no ASR state is even consulted."""
        events = []
        threads = []
        wav = np.zeros(500, dtype=np.float32)
        mocks = _make_mocks(events, threads, wav)

        async def run():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            req = GenerateRequest(
                text="hello world", mode="design", voice_description="friendly"
            )
            try:
                with _driven(mocks):
                    return await handle_generate(
                        request=_make_request(state),
                        state=state,
                        req=req,
                        security={"max_text_length": 50000, "max_batch_size": 20},
                        config_provider=_StubConfigProvider(),
                    )
            finally:
                _cleanup_cache(state)

        result = asyncio.run(run())

        self.assertEqual(len(result["results"]), 1)
        mocks.load_asr_model.assert_not_called()
        mocks.is_asr_loaded.assert_not_called()

    def test_x_vector_only_mode_never_preloads(self):
        """x_vector_only carries no transcript: no echo possible, no load."""
        events = []
        threads = []
        wav = np.zeros(500, dtype=np.float32)
        mocks = _make_mocks(events, threads, wav)

        async def run():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            req = GenerateRequest(
                text="hello world",
                mode="clone",
                prompt_file="voice.wav",
                x_vector_only_mode=True,
            )
            try:
                with _driven(mocks):
                    return await handle_generate(
                        request=_make_request(state),
                        state=state,
                        req=req,
                        security={"max_text_length": 50000, "max_batch_size": 20},
                        config_provider=_StubConfigProvider(),
                    )
            finally:
                _cleanup_cache(state)

        result = asyncio.run(run())

        self.assertEqual(len(result["results"]), 1)
        mocks.load_asr_model.assert_not_called()
        mocks.is_asr_loaded.assert_not_called()

    def test_trim_disabled_by_config_never_preloads(self):
        """generation.trim_icl_echo=false: nothing may load for the trim."""
        events = []
        threads = []
        wav = np.zeros(500, dtype=np.float32)
        mocks = _make_mocks(events, threads, wav)

        async def run():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            req = GenerateRequest(
                text="hello world", mode="clone", prompt_file="voice.wav"
            )
            try:
                with _driven(mocks):
                    return await handle_generate(
                        request=_make_request(state),
                        state=state,
                        req=req,
                        security={"max_text_length": 50000, "max_batch_size": 20},
                        config_provider=_StubConfigProvider(
                            {"trim_icl_echo": False}
                        ),
                    )
            finally:
                _cleanup_cache(state)

        result = asyncio.run(run())

        self.assertEqual(len(result["results"]), 1)
        mocks.load_asr_model.assert_not_called()
        mocks.is_asr_loaded.assert_not_called()

    def test_trim_defaults_on_when_generation_key_missing(self):
        """No generation.trim_icl_echo key: the default is TRUE.

        Must match _trim_icl_echo's own read
        (``.get("generation", {}).get("trim_icl_echo", True)``) exactly --
        a stricter default here would silently disable the trim preload for
        stock configs.
        """
        events = []
        threads = []
        wav = np.zeros(500, dtype=np.float32)
        mocks = _make_mocks(events, threads, wav)

        async def run():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            state.inference_lock = _RecordingLock(events)
            req = GenerateRequest(
                text="hello world", mode="clone", prompt_file="voice.wav"
            )
            try:
                with _driven(mocks):
                    return await handle_generate(
                        request=_make_request(state),
                        state=state,
                        req=req,
                        security={"max_text_length": 50000, "max_batch_size": 20},
                        config_provider=_StubConfigProvider({}),
                    )
            finally:
                _cleanup_cache(state)

        result = asyncio.run(run())

        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(
            mocks.load_asr_model.call_count,
            1,
            "a config without generation.trim_icl_echo must preload (the "
            "engine default is true)",
        )
        self.assertLess(events.index("preload"), events.index("lock-enter"))

    def test_unresolvable_transcript_never_preloads(self):
        """A prompt without any transcript attribute cannot echo: no load.

        Exercises the real ``_reference_text_from_prompt`` (definition site,
        unpatched): a dict prompt with no text-carrying key resolves to None.
        """
        events = []
        threads = []
        wav = np.zeros(500, dtype=np.float32)
        mocks = _make_mocks(events, threads, wav, transcript=None)

        async def run():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            req = GenerateRequest(
                text="hello world", mode="clone", prompt_file="voice.wav"
            )
            try:
                with _driven(mocks):
                    return await handle_generate(
                        request=_make_request(state),
                        state=state,
                        req=req,
                        security={"max_text_length": 50000, "max_batch_size": 20},
                        config_provider=_StubConfigProvider(),
                    )
            finally:
                _cleanup_cache(state)

        result = asyncio.run(run())

        self.assertEqual(len(result["results"]), 1)
        mocks.load_asr_model.assert_not_called()
        mocks.is_asr_loaded.assert_not_called()

    def test_asr_already_loaded_skips_the_load(self):
        """Warm ASR: the preload must be a free is_asr_loaded() check."""
        events = []
        threads = []
        wav = np.zeros(500, dtype=np.float32)
        mocks = _make_mocks(events, threads, wav)
        mocks.is_asr_loaded = MagicMock(return_value=True)

        async def run():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            req = GenerateRequest(
                text="hello world", mode="clone", prompt_file="voice.wav"
            )
            try:
                with _driven(mocks):
                    return await handle_generate(
                        request=_make_request(state),
                        state=state,
                        req=req,
                        security={"max_text_length": 50000, "max_batch_size": 20},
                        config_provider=_StubConfigProvider(),
                    )
            finally:
                _cleanup_cache(state)

        result = asyncio.run(run())

        self.assertEqual(len(result["results"]), 1)
        mocks.load_asr_model.assert_not_called()

    def test_preload_failure_warns_but_generation_completes(self):
        """A failed ensure-load must never fail the generation.

        The trim is cosmetic: log one warning and proceed. The under-lock
        degradation in _trim_icl_echo then simply skips the trim.
        """
        events = []
        threads = []
        wav = np.zeros(500, dtype=np.float32)
        mocks = _make_mocks(events, threads, wav)
        mocks.load_asr_model = MagicMock(
            side_effect=RuntimeError("hf: network unreachable")
        )

        async def run():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            req = GenerateRequest(
                text="hello world", mode="clone", prompt_file="voice.wav"
            )
            try:
                with _driven(mocks):
                    return await handle_generate(
                        request=_make_request(state),
                        state=state,
                        req=req,
                        security={"max_text_length": 50000, "max_batch_size": 20},
                        config_provider=_StubConfigProvider(),
                    )
            finally:
                _cleanup_cache(state)

        with self.assertLogs("tts", level="WARNING") as captured:
            result = asyncio.run(run())

        self.assertEqual(
            len(result["results"]),
            1,
            "the generation must complete even when the ASR preload fails",
        )
        self.assertIn(
            "echo trim",
            " ".join(captured.output),
            "the preload failure must be logged (once) as a warning",
        )

    def test_unload_in_window_degrades_to_untrimmed(self):
        """Preload loads, an unload lands in the window, the trim is skipped.

        The ratified window-race: ``handle_unload_asr`` holds
        inference_lock, so the unload can only land between the unlocked
        preload and the in-lock probe. The probe then finds ASR absent and
        MUST skip the trim -- never 503, never an in-lock rebuild. The mock
        inference runs the REAL ``_trim_icl_echo`` (definition site) with
        the probe patchable, proving the degradation on the exact seam.
        """
        events = []
        threads = []
        wav = np.zeros(500, dtype=np.float32)
        mocks = _make_mocks(events, threads, wav)
        probe = MagicMock()

        trimmed_lengths = []

        def _inference_with_real_trim(model, text, **kwargs):
            from qwen3_tts.core.engine.inference import _trim_icl_echo

            # The preload loaded ASR (mock set the flag); NOW a concurrent
            # /unload-asr lands in the window, so the engine's in-lock check
            # must find the model gone.
            mocks.asr_state["loaded"] = False
            trimmed, _ = _trim_icl_echo(
                wav,
                SR,
                REFERENCE,
                "clone",
                False,
                config={"generation": {"trim_icl_echo": True}},
            )
            trimmed_lengths.append(len(trimmed))
            events.append("inference")
            return wav, SR

        mocks.run_inference = MagicMock(
            side_effect=_inference_with_real_trim
        )

        async def run():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            req = GenerateRequest(
                text="hello world", mode="clone", prompt_file="voice.wav"
            )
            try:
                with _driven(mocks, probe=probe):
                    return await handle_generate(
                        request=_make_request(state),
                        state=state,
                        req=req,
                        security={"max_text_length": 50000, "max_batch_size": 20},
                        config_provider=_StubConfigProvider(),
                    )
            finally:
                _cleanup_cache(state)

        result = asyncio.run(run())

        self.assertEqual(
            len(result["results"]),
            1,
            "an unload landing in the preload window must degrade to an "
            "untrimmed generation, never a 503",
        )
        self.assertEqual(
            mocks.load_asr_model.call_count,
            1,
            "exactly one load (the preload); the engine must NOT rebuild "
            "ASR in-lock (#214 item 2)",
        )
        probe.assert_not_called()
        self.assertEqual(
            trimmed_lengths,
            [len(wav)],
            "the audio must ship untrimmed when ASR went missing under "
            "the lock",
        )


@_skip
class TestStreamEchoTrimPreload(unittest.TestCase):
    """handle_generate_stream mirrors the batch preload, pre-lock."""

    def test_stream_preload_fires_before_the_lock(self):
        """Streaming happy path: one unlocked preload, then the lock."""
        events = []
        threads = []
        wav = np.zeros(500, dtype=np.float32)
        mocks = _make_mocks(events, threads, wav, streaming=True)

        async def run():
            from qwen3_tts.server.app_generation import handle_generate_stream
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            state.inference_lock = _RecordingLock(events)
            req = GenerateRequest(
                text="hello world", mode="clone", prompt_file="voice.wav"
            )
            with _driven(mocks, streaming=True):
                response = await handle_generate_stream(
                    request=_make_request(state),
                    state=state,
                    req=req,
                    security={"max_text_length": 50000},
                    config_provider=_StubConfigProvider(),
                )
                return [chunk async for chunk in response.body_iterator]

        chunks = asyncio.run(run())

        self.assertTrue(chunks, "the stream produced no audio chunks")
        sample_rate, _length = struct.unpack("<II", chunks[0][:8])
        self.assertNotEqual(
            sample_rate, 0, "first frame is the terminal error sentinel"
        )
        self.assertEqual(
            mocks.load_asr_model.call_count,
            1,
            "the streaming path must ensure-load ASR exactly once",
        )
        self.assertLess(
            events.index("preload"),
            events.index("lock-enter"),
            "the preload must complete BEFORE the stream queues for "
            "inference_lock",
        )
        for thread in threads:
            self.assertIsNot(thread, threading.current_thread())

    def test_stream_design_mode_never_preloads(self):
        """Streaming negative: design mode never touches ASR."""
        events = []
        threads = []
        wav = np.zeros(500, dtype=np.float32)
        mocks = _make_mocks(events, threads, wav, streaming=True)

        async def run():
            from qwen3_tts.server.app_generation import handle_generate_stream
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            state.inference_lock = _RecordingLock(events)
            req = GenerateRequest(
                text="hello world", mode="design", voice_description="friendly"
            )
            with _driven(mocks, streaming=True):
                response = await handle_generate_stream(
                    request=_make_request(state),
                    state=state,
                    req=req,
                    security={"max_text_length": 50000},
                    config_provider=_StubConfigProvider(),
                )
                return [chunk async for chunk in response.body_iterator]

        chunks = asyncio.run(run())

        self.assertTrue(chunks)
        mocks.load_asr_model.assert_not_called()
        mocks.is_asr_loaded.assert_not_called()


if __name__ == "__main__":
    unittest.main()
