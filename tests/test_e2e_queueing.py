#!/usr/bin/env python3
"""E2E: the inference_lock queuing tier (#214 item 4 + T5).

Proves at wire level, against a REAL server, the properties the unit suites
pin with TestClient/stand-ins:

  1. /transcribe and /create-voice-prompt genuinely queue behind a running
     /generate (they serialize on the same inference_lock).
  2. Two concurrent /load-model calls coalesce into one construction
     (Phase 2c), live.
  3. T5: /unload-model of a mode whose generation is QUEUED must not
     produce the lying 200 — the unload runs only after the queued
     generation completes (route leaf lock + in-lock generation_state
     reset + post-lock slot re-read).

SERVER REQUIREMENT (different from every other e2e module): this tier is
only meaningful against a server started with rate limiting DISABLED:

    tts server stop && TTS_DISABLE_RATE_LIMITING=1 tts server start

Rate limits are read at server import, so restarting under the default env
makes the module SKIP via its preflight (never fail). Note
`tests/run_full_suite.py --test-type e2e` picks this module up
automatically, so the whole e2e profile inherits this requirement — see
tests/README.md for the hazard note. Server-state footprint: test_01
loads ASR (unloaded again at module teardown, best-effort); test_03
precondition-unloads then loads `design`; test_04 leaves `design`
UNLOADED at the end. An unload wipes ALL of gen_cache and the design
reload's warm-up stalls other generations, so tests 3/4 should not run
against a shared production server. Never unloads `clone` — it is the
module's shared input.

Run: pytest tests/test_e2e_queueing.py -m e2e -v
"""

import base64
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid

import pytest

from tests.e2e_helpers import first_available_voice_prompt

# E2E tests require a live server and make real generation requests.
# Gated behind the `e2e` marker so plain `pytest tests/` skips them (no
# hang). Opt in with: pytest tests/ -m e2e
pytestmark = pytest.mark.e2e

SERVER_URL = "http://127.0.0.1:5123"
AUTH_TOKEN_PATHS = [
    "~/.config/qwen3-tts/.voice_server_token",
    "~/.voice_server_token",
]

# The server's own client contract for model ops is 900s; the queued T5
# unload waits out a whole generation on top.
MODEL_OP_TIMEOUT = 950
GEN_TIMEOUT = 300

POLL_INTERVAL_SEC = 0.7  # >=0.6s: 0.25s default trips the 120/min global ceiling


@pytest.fixture(scope="session", autouse=True)
def check_server():
    """Skip all tests if server is not running."""
    if not _is_server_running():
        pytest.skip("TTS server not running on port 5123. Start with: tts server start")

    token = _get_auth_token()
    if not token:
        pytest.skip("No auth token found. Token should be at ~/.config/qwen3-tts/.voice_server_token")


@pytest.fixture(scope="module", autouse=True)
def rate_limit_preflight(check_server):
    """PROVE rate limiting is disabled, else skip the whole module.

    Fires ~11 rapid tiny /generate probes: the default generate limit is
    10/minute, so an un-restarted server 429s within the window and this
    module degrades to one skip instead of partial failures. Any 429 is
    also converted to a skip anywhere else in the module (429 means the
    environment is misconfigured for this tier, not that a test failed).
    """
    token = _get_auth_token()
    prompt = first_available_voice_prompt(SERVER_URL, token)
    payload_mode = (
        {"text": "probe", "mode": "clone", "prompt_file": prompt}
        if prompt
        else {"text": "probe", "mode": "design"}
    )
    for i in range(11):
        status, _ = _make_request(
            "/generate",
            data={**payload_mode, "text": f"probe {i}"},
            method="POST",
            timeout=60,
        )
        if status == 429:
            pytest.skip(
                "Rate limiting is ENABLED on the server. Restart with: "
                "TTS_DISABLE_RATE_LIMITING=1 tts server start"
            )
    yield
    # Best-effort teardown: test_01 loads ASR (~1-1.6 GB resident); unload
    # it so the module leaves no extra residency behind for later modules.
    _make_request("/unload-asr", data={}, method="POST", timeout=60)


def _get_auth_token():
    for path in AUTH_TOKEN_PATHS:
        token_path = os.path.expanduser(path)
        try:
            with open(token_path) as f:
                token = f.read().strip()
                if token:
                    return token
        except FileNotFoundError:
            continue
    return ""


def _is_server_running():
    try:
        resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
        return resp.status == 200
    except Exception:
        return False


def _make_request(endpoint, data=None, method="GET", token=None, timeout=120):
    """HTTP request -> (status, body). status 0 = transport error."""
    if token is None:
        token = _get_auth_token()
    url = f"{SERVER_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        raw = resp.read()
        if resp.headers.get("Content-Type", "").startswith("application/json"):
            return resp.status, json.loads(raw.decode())
        return resp.status, raw  # binary (e.g. /generate-stream audio)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


def _status_or_skip(status, body, context):
    """429/503-environment guard: convert environment problems into skips,
    never into test failures. Returns (status, body) on success."""
    if status == 429:
        pytest.skip(f"Rate limited during {context} — restart with TTS_DISABLE_RATE_LIMITING=1")
    if status == 503 and isinstance(body, dict):
        code = (body.get("detail") or {}).get("error") or body.get("error")
        if code in (
            "insufficient_memory",
            "model_not_loaded",
            "asr_unloaded",
            "model_unloaded",  # T5's own retryable re-read code
        ):
            pytest.skip(f"Server environment unavailable for {context}: {code}")
    return status, body


def _stream_has_real_audio(body):
    """Walk the length-prefixed frames: True iff at least one audio frame
    (sample_rate != the WS2 error sentinel) is present. A terminal error
    frame is 200-with-bytes but NOT audio — the shared parser's shape."""
    import struct as _struct

    from qwen3_tts.core.stream_protocol import STREAM_ERROR_SENTINEL_SR

    offset = 0
    has_audio = False
    while offset + 8 <= len(body):
        sr, length = _struct.unpack_from("<II", body, offset)
        offset += 8 + length
        if sr != STREAM_ERROR_SENTINEL_SR:
            has_audio = True
    return has_audio


def _poll(endpoint, key, predicate, timeout, context):
    """Poll a public status endpoint until predicate(value) — >=0.6s
    interval; 429 -> skip; returns the last observed value or None."""
    deadline = time.monotonic() + timeout
    value = None
    while time.monotonic() < deadline:
        status, body = _make_request(endpoint, timeout=10)
        if status == 429:
            pytest.skip(f"Rate limited polling {context}")
        if status == 200 and isinstance(body, dict):
            value = body.get(key)
            if predicate(value):
                return value
        time.sleep(POLL_INTERVAL_SEC)
    return None


def _mono():
    return time.monotonic()


def _first_result(body):
    """/generate's results[0] with the cancelled/short-batch guard (the
    repo's _first_result contract: never index bare)."""
    results = body.get("results") or []
    if body.get("cancelled") or not results:
        return {}
    return results[0]


def _classify_holder_failure(results):
    """When the `active` poll times out, the worker's swallowed environment
    failure (503 model_not_loaded / insufficient_memory, 429) must surface
    as a skip, not a misleading 'generation never became active' failure."""
    if results:
        status, body, _ = results[0]
        _status_or_skip(status, body, "holder generate (poll timeout)")
    return False


def _wav_bytes(seconds=1.5, rate=24000, freq=220.0):
    """Synthesized mono WAV >=24 kHz — audio_base64 is a WAV container, not
    float32, and a <24 kHz reference raises ensure_min_sample_rate."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        pytest.skip("numpy + soundfile required for the audio fixture")
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def _run_generate_in_thread(results, text, mode="clone", prompt=None):
    def _worker():
        payload = {"text": text, "mode": mode}
        if prompt:
            payload["prompt_file"] = prompt
        status, body = _make_request("/generate", data=payload, method="POST", timeout=GEN_TIMEOUT)
        results.append((status, body, _mono()))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


class TestE2EQueueing:
    """Wire-level proofs of the queuing tier."""

    def test_01_transcribe_queues_behind_generate(self):
        """/transcribe must complete only after the in-flight /generate
        that holds inference_lock (both 200; ordering asserted)."""
        token = _get_auth_token()
        status, body = _status_or_skip(
            *_make_request("/load-asr", data={}, method="POST", timeout=MODEL_OP_TIMEOUT),
            "ASR pre-load",
        )
        if status != 200:
            pytest.skip(f"ASR unavailable (/load-asr -> {status}); cannot run transcribe tier")

        prompt = first_available_voice_prompt(SERVER_URL, token)
        if not prompt:
            pytest.skip("No voice prompt available for clone generation")
        text = f"queueing tier transcribe test {uuid.uuid4().hex[:8]}"
        results = []
        gen = _run_generate_in_thread(results, text, prompt=prompt)

        active = _poll(
            "/generation-status", "active", lambda v: v is True, 60, "generate start"
        )
        if active is not True:
            _classify_holder_failure(results)
        assert active is True, "generation never became active (generate thread may have failed)"
        t_transcribe_fired = _mono()

        t_status, t_body = _status_or_skip(
            *_make_request(
                "/transcribe",
                data={"audio_base64": _wav_bytes(), "language": "en"},
                method="POST",
                timeout=MODEL_OP_TIMEOUT,
            ),
            "queued transcribe",
        )
        # Capture BEFORE the join: after join returns the generate is
        # already done, so any later timestamp would make the ordering
        # assertion vacuous.
        t_done = _mono()
        assert t_status == 200, f"transcribe failed: {t_status} {t_body}"
        assert isinstance(t_body.get("transcript"), str)

        gen.join(timeout=GEN_TIMEOUT + 60)
        assert len(results) == 1, "generate thread never completed"
        g_status, g_body, g_done = results[0]
        assert g_status == 200, f"generate failed: {g_status} {g_body}"
        assert _first_result(g_body).get("chunks", 0) >= 1, "not a real generation (cache echo?)"

        assert t_done >= g_done, (
            "transcribe completed BEFORE the generate finished — the two "
            "paths are not serializing on inference_lock"
        )
        assert t_transcribe_fired < g_done, "test ordering: transcribe fired after the generate ended"

    def test_02_create_voice_prompt_queues_behind_generate(self):
        """/create-voice-prompt must queue behind the in-flight generate
        (prompt-ops limit; namespaced + deleted in finally so this module
        never leaks a prompt into first_available_voice_prompt())."""
        token = _get_auth_token()
        prompt = first_available_voice_prompt(SERVER_URL, token)
        if not prompt:
            pytest.skip("No voice prompt available for clone generation")
        text = f"queueing tier prompt-create test {uuid.uuid4().hex[:8]}"
        results = []
        gen = _run_generate_in_thread(results, text, prompt=prompt)

        active = _poll(
            "/generation-status", "active", lambda v: v is True, 60, "generate start"
        )
        if active is not True:
            _classify_holder_failure(results)
        assert active is True, "generation never became active"
        fired = _mono()

        name = f"e2e_queueing_{uuid.uuid4().hex[:8]}"
        created = False
        body_exc = None
        try:
            c_status, c_body = _status_or_skip(
                *_make_request(
                    "/create-voice-prompt",
                    data={
                        "audio_base64": _wav_bytes(seconds=8),
                        "name": name,
                        # #236: blank transcript without no_transcript is a
                        # 400 on both backends — the prompt needs its
                        # reference text anyway.
                        "transcript": "e2e queueing tier reference transcript",
                    },
                    method="POST",
                    timeout=MODEL_OP_TIMEOUT,
                ),
                "queued create-voice-prompt",
            )
            if c_status == 500 and isinstance(c_body, dict) and "create_voice_clone_prompt" in json.dumps(
                c_body
            ):
                # Stale-server canary: this branch is dead on a fixed server
                # (the MLX create path is now inference-free and never calls
                # that API on ANY mlx-audio version — see the #236 re-scope).
                # Kept so a partially-upgraded server degrades to a skip.
                pytest.skip(
                    "STALE SERVER (pre-#236): the create path still calls "
                    "model.create_voice_clone_prompt(), which no mlx-audio "
                    "version implements. The create-queueing property stays "
                    "unit-covered by "
                    "tests/test_issue192_create_prompt_serialization.py."
                )
            assert c_status == 200, f"create-voice-prompt failed: {c_status} {c_body}"
            created = True
            # Capture BEFORE the join: after join returns the generate is
            # already done, and the ordering assertion would be vacuous
            # (same shape as test_01).
            done = _mono()

            gen.join(timeout=GEN_TIMEOUT + 60)
            assert len(results) == 1, "generate thread never completed"
            g_status, g_body, g_done = results[0]
            assert g_status == 200, f"generate failed: {g_status} {g_body}"
            assert _first_result(g_body).get("chunks", 0) >= 1

            assert done >= g_done, "create-voice-prompt completed before the generate finished"
            assert fired < g_done, "test ordering: create fired after the generate ended"
        except Exception as exc:
            # Preserve the original failure: the finally's cleanup assert
            # must never mask it.
            body_exc = exc
            raise
        finally:
            if created:
                d_status, d_body = _make_request(
                    "/delete-prompt",
                    data={"name": name},
                    method="POST",
                    timeout=60,
                )
                if body_exc is None:
                    assert d_status == 200, f"cleanup failed: {d_status} {d_body}"

    def test_03_load_model_dedup_live(self):
        """Two concurrent /load-model design POSTs coalesce: both 200,
        EXACTLY ONE carries deduped=true (the owner omits the field under
        response_model_exclude_unset; only waiters carry it)."""
        status, models = _make_request("/models", timeout=30)
        assert status == 200, models
        if models["models"]["design"]["loaded"]:
            u_status, u_body = _status_or_skip(
                *_make_request(
                    "/unload-model",
                    data={"model_type": "design"},
                    method="POST",
                    timeout=MODEL_OP_TIMEOUT,
                ),
                "design precondition unload",
            )
            assert u_status == 200, f"precondition unload failed: {u_status} {u_body}"

        results = []

        def _load():
            results.append(
                _make_request(
                    "/load-model",
                    data={"model_type": "design"},
                    method="POST",
                    timeout=MODEL_OP_TIMEOUT,
                )
            )

        threads = [threading.Thread(target=_load, daemon=True) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=MODEL_OP_TIMEOUT + 60)
        assert len(results) == 2, "a load thread never completed"

        # Classify in-main-thread: pytest.skip inside a worker thread would
        # not propagate, so environment failures surface here.
        for r_status, r_body in results:
            _status_or_skip(r_status, r_body, "live design load")
        statuses = [r[0] for r in results]
        assert statuses == [200, 200], f"load statuses: {results}"
        deduped_count = sum(
            1 for _, body in results if isinstance(body, dict) and body.get("deduped") is True
        )
        assert deduped_count == 1, (
            f"expected exactly one deduped=true, got {deduped_count} in "
            f"{[r[1] for r in results]}"
        )

    def test_04_unload_of_queued_generation_mode_waits_out_the_generation(self):
        """T5 live: clone holds inference_lock (long generate); a design
        /generate-stream parks on the lock; /unload-model design must NOT
        return before the design generation completes. On an unfixed server
        the unload 200s while design is queued (the lying 200) and /models
        flips to not-loaded — the state-contract violation this asserts."""
        token = _get_auth_token()
        prompt = first_available_voice_prompt(SERVER_URL, token)
        if not prompt:
            pytest.skip("No voice prompt available for clone generation")

        # Environment preconditions: the holder needs clone, the stream
        # needs design (test_03 loads it, but pin it so this test holds
        # standalone too).
        m_status, models = _make_request("/models", timeout=30)
        assert m_status == 200
        if not models["models"]["clone"]["loaded"]:
            pytest.skip("clone model not loaded — environment for the holder is unavailable")
        if not models["models"]["design"]["loaded"]:
            l_status, l_body = _make_request(
                "/load-model",
                data={"model_type": "design"},
                method="POST",
                timeout=MODEL_OP_TIMEOUT,
            )
            if l_status != 200:
                pytest.skip(f"design load failed ({l_status}); environment unavailable")

        # (i) long clone generate as the lock-holder. ~900 chars = 2 chunks
        # at the 500-char default (~80-140s on M2 Pro; GEN_TIMEOUT=300 holds
        # ~2x that — on a much slower machine this test may need a longer
        # GEN_TIMEOUT, and the failure surfaces as holder status 0).
        filler = (
            "This sentence intentionally gives the server real work to do. "
        )
        holder_text = (f"queueing tier T5 holder {uuid.uuid4().hex[:6]}. ") + filler * 14
        results = []
        holder = _run_generate_in_thread(results, holder_text, prompt=prompt)

        active = _poll(
            "/generation-status", "active", lambda v: v is True, 60, "holder generate start"
        )
        if active is not True:
            _classify_holder_failure(results)
        assert active is True, "holder generation never became active"

        # (iii) design stream parks on the lock; pending_requests registers
        # it BEFORE the acquire, so /queue-status is the wire-observable.
        stream_result = {}

        def _stream_design():
            payload = {
                "text": f"queueing tier T5 queued stream {uuid.uuid4().hex[:6]}",
                "mode": "design",
            }
            started = _mono()
            # _make_request returns (status, raw bytes) for non-JSON bodies.
            status, body = _make_request(
                "/generate-stream", data=payload, method="POST", timeout=MODEL_OP_TIMEOUT
            )
            stream_result.update(
                {"status": status, "body": body if isinstance(body, bytes) else b"",
                 "started": started, "done": _mono()}
            )

        streamer = threading.Thread(target=_stream_design, daemon=True)
        streamer.start()

        queued = _poll(
            "/queue-status", "queue_length", lambda v: (v or 0) >= 1, 30, "design stream queueing"
        )
        assert queued is not None, "design stream never became visible in /queue-status"

        # (v) fire the unload IMMEDIATELY, while design is queued.
        t_unload_fired = _mono()
        u_status, u_body = _status_or_skip(
            *_make_request(
                "/unload-model",
                data={"model_type": "design"},
                method="POST",
                timeout=MODEL_OP_TIMEOUT,
            ),
            "T5 unload",
        )
        t_unload_done = _mono()

        streamer.join(timeout=MODEL_OP_TIMEOUT + 60)
        holder.join(timeout=GEN_TIMEOUT + 60)

        assert stream_result, "design stream thread never completed"
        assert t_unload_fired < stream_result["done"], (
            "test ordering: the unload fired after the design stream already "
            "finished — the non-vacuous trigger check"
        )
        assert u_status == 200, f"unload failed: {u_status} {u_body}"

        # THE STATE CONTRACT — two coherent outcomes and one defect:
        # (A) FIFO held: real audio arrived AND the unload completed only
        #     after the stream. (B) sanctioned overtake: the unload's
        #     acquire beat the streamer's (queue-status proves registration,
        #     not the acquire), so the streamer got the re-read's terminal
        #     error frame — a clean retry, never an orphan run. The DEFECT
        #     (unfixed server): real audio AND unload-done-first (the lying
        #     200, #233 re-open); or a TRUNCATED stream (no error frame).
        stream_body = stream_result.get("body")
        stream_ok = stream_result["status"] == 200 and isinstance(stream_body, bytes) and _stream_has_real_audio(stream_body)
        if stream_ok:
            assert t_unload_done >= stream_result["done"], (
                "UNLOAD COMPLETED BEFORE THE QUEUED GENERATION — the lying "
                f"200: unload done at {t_unload_done:.2f}, design stream "
                f"done at {stream_result['done']:.2f}"
            )
        else:
            # Outcome B: the failure must be the in-band terminal frame —
            # a bare truncate (transport error) is the pre-fix shape.
            assert isinstance(stream_body, bytes) and stream_body, (
                "design stream produced neither audio nor an error frame "
                f"(status {stream_result['status']}) — truncated, not the "
                "clean retryable abort"
            )
            assert t_unload_done >= stream_result["done"], (
                "overtake outcome: the unload must still complete only "
                "after the streamer's error frame closed the stream"
            )

        assert len(results) == 1, "holder generate thread never completed"
        h_status, h_body, _ = results[0]
        assert h_status == 200, f"holder generate failed: {h_status} {h_body}"
        assert _first_result(h_body).get("chunks", 0) >= 1

        # Final state: design unloaded (the unload DID run, after the stream).
        m_status, models = _make_request("/models", timeout=30)
        assert m_status == 200
        assert models["models"]["design"]["loaded"] is False, (
            "design should be unloaded at the end (the queued unload ran last)"
        )
