#!/usr/bin/env python3
"""Reproduction harness for issue #192 — custom-mode token-cap runaway.

Issue #192: a custom-mode generation ran to the 2048-token cap without emitting
EOS, taking 215.8 s for a 16-character input, where healthy runs take ~7 s.

The runaway is not deterministic, so the point of this harness is to hold one
condition fixed at a time and count how many runs hit the cap. It drives
``_run_inference_mlx`` directly rather than the server, so the sampling
parameters are exactly what production sends, with no HTTP, cache, or chunking
in between.

Wall-clock alone is NOT a usable signal here — that is what caused the original
misattribution to machine contention. What distinguishes the runaway is the
token count: at 12.5 Hz, 2048 tokens is ~164 s of audio for an input that
should produce ~1.5 s. Emitted tokens are inferred from output duration, which
is the only handle available once ``model.generate_*`` has returned.

Measured results (2026-08-18, M2 Pro, 1.7B 8-bit): 0/60 with varying seeds and
0/30 with all three models resident. See
``docs/reviews/issue-192-custom-mode-runaway-2026-08-18.md``.

Usage:
    conda run -n qwen3-tts-mlx python scripts/probe_issue192.py --runs 60
    conda run -n qwen3-tts-mlx python scripts/probe_issue192.py --load-all --runs 30
    conda run -n qwen3-tts-mlx python scripts/probe_issue192.py --text "Token test three" --out rows.json
    conda run -n qwen3-tts-mlx python scripts/probe_issue192.py --churn --out churn.json

Exit codes: 1 if any run hit the cap (or the server log shows a cap warning),
2 for an indeterminate run (setup abort or unreadable server log), 0 otherwise.
"""
import argparse
import json
import logging
import os
import sys
import threading
import time

# The engine's own default for MLX; a run reaching this emitted no EOS.
DEFAULT_MAX_TOKENS = 2048

# mlx-audio's codec frame rate. Tokens are not returned by the generate call,
# so they are inferred from audio duration at this rate.
CODEC_FRAME_RATE_HZ = 12.5

# Within 2% of the cap counts as capped: the trailing partial frame and any
# post-processing trim make an exact equality test fragile.
CAP_TOLERANCE = 0.98

# The e2e text whose two failures are recorded in #192. "Token test 3" reaches
# the engine as this, because _normalize_text expands the digit.
DEFAULT_TEXT = "Token test three"

# What the server sends for a request that specifies neither speaker nor
# language, via `speaker or "Ryan"` in _run_inference_mlx.
DEFAULT_SPEAKER = "Ryan"
DEFAULT_LANGUAGE = "auto"

# Production sampling defaults from _get_mlx_gen_params().
SAMPLING = {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
}

# --- live-server churn mode -------------------------------------------------
# Added 2026-08-19, after the full e2e pass on d5d9aec failed to reproduce the
# runaway (0 cap warnings; result on the issue). The surviving hypothesis is
# the unserialized race between model ops and generation: handle_load_model
# acquires no lock, and handle_unload_model's active-generation guard is a
# lockless, mode-scoped check-then-act that clears the slot and then runs the
# global MLX backend cleanup. This mode drives exactly that window — concurrent
# authenticated /generate traffic in custom mode while a churn thread cycles
# /unload-model + /load-model, including the design reload whose warm-up
# inference is itself unserialized GPU work.

SERVER_BASE_URL = "http://127.0.0.1:5123"

# A runaway generation takes ~216 s; match the client read-timeout floor so
# the probe always sees the full result instead of abandoning the request.
CHURN_REQUEST_TIMEOUT_S = 600

# Connection failures mean the server died mid-probe (the known load/unload
# fragility). Stop everything after this many in a row rather than hang.
CHURN_MAX_CONSECUTIVE_FAILURES = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--load-all",
        action="store_true",
        help="Load clone and design too, reproducing the e2e resident set",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Seed run i with seed-base+i; omit to let each run seed freely",
    )
    parser.add_argument(
        "--churn",
        action="store_true",
        help="Live-server race probe: concurrent /generate (custom) while "
        "cycling /unload-model + /load-model. Needs a running server; "
        "ignores --runs/--load-all",
    )
    parser.add_argument(
        "--churn-generators",
        type=int,
        default=3,
        help="Concurrent generator threads (churn mode)",
    )
    parser.add_argument(
        "--churn-requests",
        type=int,
        default=4,
        help="Sequential requests per generator thread (churn mode)",
    )
    parser.add_argument(
        "--churn-pause",
        type=float,
        default=1.0,
        help="Seconds between churn ops (churn mode)",
    )
    parser.add_argument(
        "--churn-model",
        choices=("design", "clone", "none"),
        default="design",
        help="Churn target. design loads run a warm-up inference (concurrent "
        "GPU work); clone loads do not; none runs no churn thread — "
        "the ablation arms for the race-vs-pressure question",
    )
    parser.add_argument("--out", help="Write the per-run rows here as JSON")
    return parser.parse_args(argv)


def load_models(load_all):
    from qwen3_tts.core.engine import load_model

    names = ["clone", "design", "custom"] if load_all else ["custom"]
    models = {}
    for name in names:
        start = time.time()
        models[name] = load_model(name)
        print(f"loaded {name} in {time.time() - start:.1f}s", flush=True)
    return models


def report_memory():
    """Print MLX active memory, so a pressure run records what it achieved."""
    try:
        import mlx.core as mx

        print(f"mlx active memory: {mx.get_active_memory() / 1e9:.2f} GB", flush=True)
    except Exception as exc:  # noqa: BLE001 — diagnostics only, never fatal
        print(f"(no mlx memory reading: {exc})", flush=True)


def run_once(model, args, seed):
    from qwen3_tts.core.engine.inference import _run_inference_mlx

    params = dict(SAMPLING, max_new_tokens=args.max_tokens)
    if seed is not None:
        params["seed"] = seed

    start = time.time()
    audio, sample_rate = _run_inference_mlx(
        model,
        args.text,
        "custom",
        params,
        language=args.language,
        speaker=args.speaker,
    )
    elapsed = time.time() - start

    samples = int(getattr(audio, "shape", [0])[0]) if audio is not None else 0
    seconds = samples / sample_rate if sample_rate else 0.0
    approx_tokens = round(seconds * CODEC_FRAME_RATE_HZ)
    return {
        "seed": seed,
        "elapsed_s": round(elapsed, 1),
        "audio_s": round(seconds, 1),
        "approx_tokens": approx_tokens,
        "CAPPED": approx_tokens >= args.max_tokens * CAP_TOLERANCE,
    }


def server_log_path():
    """Absolute server-log path from config — never CWD-relative.

    A CWD-relative literal here would silently degrade the log-based cap
    counter to 0 whenever the probe ran from any other directory.
    """
    from qwen3_tts.core.config import LOG_FILE

    return str(LOG_FILE)


def read_auth_token():
    """Return the server auth token, or exit with the reason we cannot."""
    from qwen3_tts.core.config import _LEGACY_TOKEN_FILE, TOKEN_FILE

    for path in (TOKEN_FILE, _LEGACY_TOKEN_FILE):
        if path.exists():
            token = path.read_text().strip()
            if token:
                return token
    print(
        "no auth token found — is the server running? (tts server start)",
        file=sys.stderr,
    )
    sys.exit(2)  # indeterminate — never readable as "cap detected" (exit 1)


def server_request(method, path, token, payload=None, timeout=30):
    """One HTTP call; returns (status, parsed_json_or_None).

    HTTPError is returned as its status (e.g. the 409 unload guard) so callers
    record it as data. Connection-level errors propagate to the caller.
    """
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        SERVER_BASE_URL + path,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except Exception:  # noqa: BLE001 — detail is best-effort
            return exc.code, None
    try:
        return resp.status, json.loads(body)
    except ValueError:
        return resp.status, None


def audio_seconds_from_b64(audio_b64):
    """Decode the /generate audio_base64 (a WAV container) to seconds.

    Returns None when the payload is not decodable — recorded distinctly from
    0.0 so a decode failure can never read as a suspiciously short clip.
    """
    import base64
    import binascii
    import io
    import wave

    try:
        raw = base64.b64decode(audio_b64)
        if raw[:4] != b"RIFF":  # audio_base64 is a WAV container, never raw f32
            return None
        with wave.open(io.BytesIO(raw)) as wav:
            return wav.getnframes() / wav.getframerate()
    except (binascii.Error, EOFError, ValueError, TypeError, wave.Error):
        # Bad base64 padding or a truncated RIFF blob: a malformed body is a
        # data point, never a reason to kill the generator thread carrying it.
        return None


def count_new_cap_warnings(log_path, start_offset):
    """Count 'token cap' WARNING lines appended to the server log since start.

    Returns None when the log is unreadable — recorded distinctly from 0 so a
    broken signal can never read as a clean run. Substring scope: this also
    matches the clone-path cap wording (voice_prompt.py); valid here because
    churn mode drives custom/design generations only.
    """
    try:
        with open(log_path, "rb") as handle:
            handle.seek(start_offset)
            return handle.read().count(b"token cap")
    except OSError:
        return None


class ChurnProbe:
    """Shared state for the generator threads and the churn thread."""

    def __init__(self, args, token, t0):
        self.args = args
        self.token = token
        self.t0 = t0
        self.rows = []
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._consecutive_failures = 0

    def now_s(self):
        return round(time.time() - self.t0, 1)

    def record(self, row):
        # Append under the same lock as the failure counter so a late row can
        # never race the summary snapshot below.
        with self._lock:
            self.rows.append(row)
        print(json.dumps(row), flush=True)

    def note_failure(self):
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= CHURN_MAX_CONSECUTIVE_FAILURES:
                self.stop.set()
                print(
                    "server unreachable "
                    f"{self._consecutive_failures}x in a row — stopping",
                    file=sys.stderr,
                    flush=True,
                )

    def note_success(self):
        # Any HTTP response counts as liveness — 4xx/5xx don't trip the stop
        # knob, only connection failures do (and requests are bounded anyway).
        with self._lock:
            self._consecutive_failures = 0

    def snapshot(self):
        """Copy rows under the lock so summary counts and JSON can't disagree."""
        with self._lock:
            return list(self.rows)


def generator_worker(probe, gen_idx):
    """Fire sequential custom-mode /generate requests, varying text per request.

    Text varies so every request bypasses the generation cache (the seed is
    deliberately NOT part of the cache key, so varying it would not).
    """
    for req_idx in range(probe.args.churn_requests):
        if probe.stop.is_set():
            return
        # PID keeps texts unique across invocations — repeated runs must not
        # age into generation-cache hits, which return instantly and exercise
        # nothing (the seed is not part of the cache key, so it can't do this).
        text = f"{probe.args.text} churn {os.getpid()} {gen_idx} {req_idx}"
        start = time.time()
        try:
            status, data = server_request(
                "POST",
                "/generate",
                probe.token,
                payload={
                    "text": text,
                    "mode": "custom",
                    "speaker": probe.args.speaker,
                },
                timeout=CHURN_REQUEST_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 — a dead server is a finding
            probe.note_failure()
            probe.record(
                {
                    "kind": "generate",
                    "generator": gen_idx,
                    "text": text,
                    "start_s": probe.now_s(),
                    "error": str(exc),
                }
            )
            continue
        probe.note_success()
        # Parse defensively: the race under test could itself produce a
        # malformed body, and an unhandled exception here would kill this
        # generator thread — silently dropping its remaining requests and any
        # CAPPED row it would have recorded (a false "clean" run).
        parse_error = None
        audio_s = None
        seed = None
        results = (data.get("results") or []) if isinstance(data, dict) else []
        try:
            if results:
                seed = results[0].get("seed")
                if results[0].get("audio_base64"):
                    audio_s = audio_seconds_from_b64(results[0]["audio_base64"])
            approx_tokens = (
                round(audio_s * CODEC_FRAME_RATE_HZ) if audio_s is not None else None
            )
            capped = approx_tokens is not None and approx_tokens >= int(
                probe.args.max_tokens * CAP_TOLERANCE
            )
        except Exception as exc:  # noqa: BLE001 — malformed body is a finding
            parse_error = str(exc)
            approx_tokens = None
            capped = False
        probe.record(
            {
                "kind": "generate",
                "generator": gen_idx,
                "text": text,
                "start_s": probe.now_s(),
                "elapsed_s": round(time.time() - start, 1),
                "status": status,
                "error": (
                    data.get("detail")
                    if isinstance(data, dict) and status != 200
                    else None
                ),
                "audio_s": round(audio_s, 1) if audio_s is not None else None,
                "approx_tokens": approx_tokens,
                "CAPPED": capped,
                "seed": seed,
                "parse_error": parse_error,
            }
        )


def churn_worker(probe):
    """Cycle /unload-model + /load-model while generations are in flight.

    Design is the primary target: unloading it during a custom generation is
    allowed by the mode-scoped guard yet still runs the global MLX cleanup,
    and reloading it runs a warm-up inference — the only concurrent GPU work
    reachable through the API (/generate calls serialize on inference_lock).
    Clone churn keeps the alloc/free + cleanup + pressure but drops the
    warm-up — the ablation arm. Custom is unloaded every other cycle to probe
    the TOCTOU guard (409 when it engages, success when it slips).
    """
    cycle = 0
    while not probe.stop.is_set():
        targets = [probe.args.churn_model, "custom"] if cycle % 2 == 0 else [
            probe.args.churn_model
        ]
        for model in targets:
            if probe.stop.is_set():
                return
            for op in ("unload", "load"):
                if probe.stop.is_set():
                    return
                start = time.time()
                try:
                    status, data = server_request(
                        "POST",
                        f"/{op}-model",
                        probe.token,
                        payload={"model_type": model},
                        timeout=CHURN_REQUEST_TIMEOUT_S,
                    )
                except Exception as exc:  # noqa: BLE001 — a dead server is a finding
                    probe.note_failure()
                    probe.record(
                        {
                            "kind": "churn",
                            "op": op,
                            "model": model,
                            "start_s": probe.now_s(),
                            "error": str(exc),
                        }
                    )
                    continue
                probe.note_success()
                probe.record(
                    {
                        "kind": "churn",
                        "op": op,
                        "model": model,
                        "start_s": probe.now_s(),
                        "elapsed_s": round(time.time() - start, 1),
                        "status": status,
                        "detail": (data or {}).get("status") if data else None,
                    }
                )
                time.sleep(probe.args.churn_pause)
        cycle += 1


def run_churn(args):
    """Drive concurrent generation + model-op churn against the live server."""
    token = read_auth_token()
    try:
        status, _ = server_request("GET", "/health", token, timeout=5)
    except Exception as exc:  # noqa: BLE001 — setup failure, message is the point
        print(f"server not reachable at {SERVER_BASE_URL}: {exc}", file=sys.stderr)
        sys.exit(2)
    if status != 200:
        print(f"server health returned {status} — not healthy, aborting", file=sys.stderr)
        sys.exit(2)

    # Custom must be resident or every /generate 503s (the e2e pass's teardown
    # leaves it unloaded). Idempotent: "already_loaded" is the fast path.
    status, data = server_request(
        "POST",
        "/load-model",
        token,
        payload={"model_type": "custom"},
        timeout=CHURN_REQUEST_TIMEOUT_S,
    )
    if status != 200 or (data or {}).get("status") not in ("loaded", "already_loaded"):
        print(f"could not load custom model (status {status}): {data}", file=sys.stderr)
        sys.exit(2)

    try:
        log_offset = os.path.getsize(server_log_path())
    except OSError:
        log_offset = 0

    t0 = time.time()
    probe = ChurnProbe(args, token, t0)
    print(
        f"churn probe: {args.churn_generators} generators x "
        f"{args.churn_requests} requests, log offset {log_offset}",
        flush=True,
    )

    churn_thread = None
    if args.churn_model != "none":
        churn_thread = threading.Thread(target=churn_worker, args=(probe,), daemon=True)
        churn_thread.start()
    generators = [
        threading.Thread(target=generator_worker, args=(probe, idx))
        for idx in range(args.churn_generators)
    ]
    for thread in generators:
        thread.start()
    for thread in generators:
        thread.join()
    probe.stop.set()
    if churn_thread is not None:
        churn_thread.join(timeout=CHURN_REQUEST_TIMEOUT_S)

    cap_warnings = count_new_cap_warnings(server_log_path(), log_offset)
    if cap_warnings is None:
        print(
            f"WARNING: could not read server log {server_log_path()} — "
            "log-based cap count unavailable for this run",
            file=sys.stderr,
            flush=True,
        )
    # Summarize from a snapshot: the churn thread can outlive its join timeout
    # and append late rows, which would make summary counts and the JSON
    # "rows" array disagree.
    rows_snapshot = probe.snapshot()
    gen_rows = [r for r in rows_snapshot if r.get("kind") == "generate"]
    capped = [r for r in gen_rows if r.get("CAPPED")]
    audio_secs = sorted(r["audio_s"] for r in gen_rows if r.get("audio_s") is not None)
    churn_rows = [r for r in rows_snapshot if r.get("kind") == "churn"]
    guard_blocks = sum(
        1 for r in churn_rows if r.get("op") == "unload" and r.get("status") == 409
    )
    guard_slips = sum(
        1
        for r in churn_rows
        if r.get("op") == "unload" and r.get("model") == "custom" and r.get("status") == 200
    )

    non_ok = sum(1 for r in gen_rows if r.get("status") != 200)
    cap_warn_display = (
        "unavailable (log unreadable)" if cap_warnings is None else cap_warnings
    )
    print(
        f"\n=== churn probe: {len(capped)}/{len(gen_rows)} generations hit the "
        f"cap (client-side); {cap_warn_display} 'token cap' warnings in the "
        f"server log; {non_ok} generation(s) returned non-200 ===\n"
        f"=== churn ops: {len(churn_rows)} "
        f"(unload-guard engaged {guard_blocks}x, custom-unload slipped "
        f"{guard_slips}x); audio_s min/median/max = "
        f"{audio_secs[0] if audio_secs else '-'}/"
        f"{audio_secs[len(audio_secs) // 2] if audio_secs else '-'}/"
        f"{audio_secs[-1] if audio_secs else '-'} ===",
        flush=True,
    )

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                {"summary": {
                    "args": {
                        "churn_model": args.churn_model,
                        "churn_generators": args.churn_generators,
                        "churn_requests": args.churn_requests,
                        "churn_pause": args.churn_pause,
                        "text": args.text,
                    },
                    "generations": len(gen_rows),
                    "generations_non_ok": non_ok,
                    "capped": len(capped),
                    "cap_warnings_in_log": cap_warnings,
                    "churn_ops": len(churn_rows),
                    "unload_guard_engaged": guard_blocks,
                    "custom_unload_slipped": guard_slips,
                }, "rows": rows_snapshot},
                handle,
                indent=1,
            )
        print(f"wrote {args.out}", flush=True)

    if capped or (cap_warnings or 0) > 0:
        # Server-side warnings count as caps even if the client-side decode of
        # that row failed — the log is the independent signal.
        return 1
    if cap_warnings is None:
        return 2  # indeterminate — log signal unavailable
    return 0


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.setdefault("TTS_BACKEND", "mlx")

    if args.churn:
        return run_churn(args)

    models = load_models(args.load_all)
    report_memory()

    rows = []
    for i in range(args.runs):
        seed = None if args.seed_base is None else args.seed_base + i
        rows.append(run_once(models["custom"], args, seed))
        print(json.dumps(rows[-1]), flush=True)

    capped = [r for r in rows if r["CAPPED"]]
    tokens = sorted(r["approx_tokens"] for r in rows)
    print(
        f"\n=== {len(capped)}/{len(rows)} hit the {args.max_tokens}-token cap; "
        f"tokens min/median/max = {tokens[0]}/{tokens[len(tokens) // 2]}/{tokens[-1]} ===",
        flush=True,
    )

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=1)
        print(f"wrote {args.out}", flush=True)

    return 1 if capped else 0


if __name__ == "__main__":
    sys.exit(main())
