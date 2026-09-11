# Issue #193 — ICL Echo Trim Default: Evidence Document

**Date:** 2026-09-10
**Branch:** `fix/issue193-icl-echo-trim-default`
**Commits in scope:** `34a10d08` (Task 1 — engine: first-chunk cap + degradation-path comment) ..
`4ba2a4de` (Task 1 test pins) .. `baddb8a8` (Task 2 test pins, RED) .. `5fd5715d` (Task 2 —
server: unlocked ASR ensure-load, GREEN)

This document is Part 3 of the Task 3 brief for Step 1C of the consolidated backlog plan
(`docs/plans/2026-09-06-consolidated-backlog-priority.plan.md`, Step 1C). It records the defect
issue #193 names, the ratified decision and its amendment, and the mechanism of the three shipped
pieces with real code quotes verified against the branch head (`5fd5715d`). The counted gate runs
and the live-server smoke are appended at gate time by the controller (they run after this
document's dispatch); their sections below are placeholders and contain no output yet.

---

## The defect this fix closes

Two defects shipped together, because they share one code path:

**1. The trim was inert by default (issue #193 proper).** `generation.trim_icl_echo` defaults to
`true` and CLAUDE.md documented the feature as on-by-default, but `_trim_icl_echo`
(`qwen3_tts/core/engine/inference.py`) gated itself on `asr.is_asr_loaded()` — "Only opportunistic:
pulling a heavy ASR model into a generation that never asked for one would cost more than the
artifact it removes" (the pre-`34a10d08` comment). ASR is not in `load_at_startup` defaults, so on
a fresh server — the common case — EVERY clone generation shipped with the re-spoken reference
transcript's tail still at the head of the audio. The documented default was false in practice: the
config key consented to a feature that could never fire unless something else (a `/transcribe`
call) had happened to load ASR first.

**2. The WS9.4 combined-cap over-trim.** The multi-chunk batch path trims the COMBINED audio once
(the echo is a generation-head artifact, so per-chunk application would re-probe without catching
more), but the 50% cut cap was computed against `len(audio)` — the combined total:

```python
        cut = _find_silence_boundary(audio, sample_rate, estimate)
        max_cut = int(len(audio) * _ICL_MAX_TRIM_FRACTION)
```

(pre-`34a10d08`). On a, say, five-chunk generation, that licenses cutting half of ALL the audio —
far past any plausible echo, since the echo can only sit at the head of the FIRST chunk (the chunk
that carried the reference). The cap's shape was right for the single-chunk case it was written in
and wrong for the combined case it was being applied to.

---

## The ratified decision, and the amendment that changed where the load happens

**DECISION (recorded in the plan 2026-09-09, controller ruling, user-approved — option (b),
keep-loaded variant):** when `trim_icl_echo=true`, `mode=clone`, and a transcript resolves but ASR
is not loaded, force-load ASR for the trim probe and **keep it loaded** (the warm load is
1.8–1.9s — banked research — and amortizes to once per server lifetime via the process-lifetime ASR
globals). `trim_icl_echo: true` is the documented default and option (b) makes that default TRUE.
The keep-loaded variant makes the unload-after toggle unnecessary — **no new config key**;
`trim_icl_echo` itself is the consent.

**AMENDMENT (2026-09-09, post-survey — implementation shape only; the ruling stands):** the
recorded cost model covered only the WARM load. A COLD ASR load performs unbounded HuggingFace
network I/O (MLX pulls `whisper-large-v3-turbo`), and the originally recorded shape ran that load
in-engine — i.e. INSIDE `inference_lock` — so a first-ever clone generation on a fresh machine
would stall every other generation for the length of a model download. The sentence accepting the
in-engine force-load as "a documented deviation from `/load-asr`'s outside-`inference_lock` split"
is superseded: that deviation was accepted on warm-load reasoning that does not hold cold.

The repository had already ratified the correct pattern for exactly this problem in `/transcribe`
(`app_models.py`): load UNLOCKED before the generation queues for the lock, then re-check under the
lock and never rebuild there (`#214` item 2 — re-loading in-lock "would trade a cheap 503 for the
starvation this lock exists to prevent"). Step 1C mirrors that pattern with one deliberate
divergence: where `/transcribe` answers a retryable 503 `asr_unloaded` on an under-lock miss, the
echo trim SKIPS — a missing echo-trim is cosmetic, so the correct degradation is an untrimmed
generation, never a failed one.

The controller's earlier suggestion (time-box the cold load inside the lock, plus a one-shot
failure latch) was REJECTED and stays rejected: time-boxing can abort a nearly-complete download
repeatedly, and a hard latch fights `huggingface_hub`'s resume-partial behavior, so successive
attempts would never converge.

---

## Mechanism piece 1 — the unlocked ensure-load at the server layer

**`_ensure_asr_for_echo_trim`** (`qwen3_tts/server/app_generation.py:188-253`, commit `5fd5715d`)
runs in the request path BEFORE the generation queues for `inference_lock`. It is gated on the
same three conditions `_trim_icl_echo` probes with, so nothing loads for a generation that would
not use it: `mode == "clone"` (echo is a clone-only artifact), `generation.trim_icl_echo` truthy
(same `.get(..., True)` default chain as the probe's own read), and a transcript resolvable off the
voice prompt via the REAL `_reference_text_from_prompt` (no transcript, no echo). The body after
the docstring (`app_generation.py:227-253`):

```python
    if mode != "clone" or x_vector_only_mode:
        return
    # Same read as _trim_icl_echo's gate: default true.
    if not config.get("generation", {}).get("trim_icl_echo", True):
        return

    # Lazy + module-style: attribute access on the modules keeps the mock
    # seams at the definition sites (core.engine.asr.*, engine.inference),
    # matching how _trim_icl_echo itself reaches them.
    from qwen3_tts.core.engine import asr
    from qwen3_tts.core.engine.inference import _reference_text_from_prompt

    if not _reference_text_from_prompt(voice_prompt):
        return
    if asr.is_asr_loaded():
        return
    try:
        # NEVER on the event loop (async-offload policy, cf.
        # tests/test_server_async_offload.py): weight construction is
        # seconds of blocking work, the cold path minutes of network I/O.
        await asyncio.to_thread(asr.load_asr_model)
    except Exception as e:
        logger.warning(
            "ASR ensure-load for the ICL echo trim failed (%s); generating "
            "without the trim probe",
            sanitize_log(e),
        )
```

Four properties are deliberate:

- **Unlocked, and off the event loop.** The cold load is minutes of HF network I/O; `asyncio.to_thread` keeps both the event loop and `inference_lock` free while it runs. This mirrors
  `/transcribe`'s ensure-load (`app_models.py`).
- **Keep-loaded.** `asr.load_asr_model` populates the process-lifetime ASR globals; nothing
  unloads afterward. Per-item repeats in a batch are free via the `is_asr_loaded()` short-circuit.
- **Failure is a log line, not an error.** ONE warning, then the generation proceeds — the
  under-lock degradation (piece 2) skips the trim. Deliberately NO failure latch and NO retry
  suppression: a latch would fight `huggingface_hub`'s resume-partial behavior.
- **The config is resolved the same way the engine resolves it** — callers pass
  `(config_provider or DefaultConfigLoader()).load()` — so the gate reads exactly what the probe
  will read.

**Both wired call sites sit pre-lock.** Batch `handle_generate` (`app_generation.py:547-552`):

```python
            await _ensure_asr_for_echo_trim(
                mode,
                x_vector_only_mode,
                (config_provider or DefaultConfigLoader()).load(),
                voice_prompt,
            )

            # Acquire inference_lock ONLY for GPU-bound work: ...
            async with state.inference_lock:
```

and streaming `handle_generate_stream` (`app_generation.py:935-940`) — the lock there is acquired
inside `audio_stream_generator` when the response body is iterated, so anything in the handler
body is safely pre-queue. Pure cache-hit items return before the call site, so a pure cache hit
never triggers a load.

**`/ws` is deliberately NOT wired.** Step 1C scopes the server-layer ensure-load to the two
`app_generation.py` handlers; a `/ws` clone generation on a fresh server therefore still ships
untrimmed — the same symptom as #193 on that one surface. This is registered, not forgotten: 6R
item 19 in the consolidated plan (companion to 6R item 17, the `/ws` gap family).

---

## Mechanism piece 2 — the under-lock re-check that degrades instead of rebuilding

`_trim_icl_echo` keeps its ASR-absence check as the DEGRADATION path (`qwen3_tts/core/engine/inference.py:1158-1166`, comment rewritten by `34a10d08`; the check itself predates this branch —
what changed is its meaning and its company):

```python
    from qwen3_tts.core.engine import asr

    # Degradation path (#193 / Step 1C): the server layer ensure-loads ASR
    # UNLOCKED before the generation queues for inference_lock, so a miss
    # here under the lock means an unload landed in that window. Skip the
    # trim — never rebuild in-lock. This check must sit before the probe
    # because _transcribe_mlx lazily re-loads an unloaded model (asr.py).
    if not asr.is_asr_loaded():
        return audio, sample_rate
```

The two locks' interaction, end to end:

1. Request arrives; `_ensure_asr_for_echo_trim` loads ASR UNLOCKED (piece 1).
2. The generation queues for `inference_lock`.
3. `/unload-asr` HOLDS `inference_lock` (the `#214` item 2 closure, `ARCHITECTURE.md`
   "Unload-ASR race closure"), so an unload can never land mid-probe — the only place it can land
   is the unlocked preload window in step 1→2.
4. If it did, the in-lock check at `:1165` finds ASR absent and returns the audio untouched.
   The check sits BEFORE the probe because `_transcribe_mlx` lazily re-loads an unloaded model —
   without the guard, the probe itself would be the in-lock rebuild the whole design forbids.

So: degradation is untrimmed output; a failed generation and an in-lock rebuild are both
structurally unreachable from this path. This is the deliberate divergence from `/transcribe`'s
503 — cosmetic vs. functional loss.

**The pin that fails without the degradation** —
`tests/test_icl_echo_trim.py::TestTrimIclEchoGuards::test_skips_when_asr_not_loaded` (`:216`):
ASR patched absent, probe and `load_asr_model` both patched spy — asserts the probe is never
called, `load_asr_model` is NEVER called (no in-lock rebuild), and the audio comes back untouched.

**The end-to-end window-race pin** —
`tests/test_echo_trim_asr_preload.py::TestBatchEchoTrimPreload::test_unload_in_window_degrades_to_untrimmed` (`:596`): drives the REAL `handle_generate`; the preload loads (mock sets the flag),
then a concurrent unload "lands in the window" before the mocked inference runs the REAL
`_trim_icl_echo`. Asserts the result still ships (1 result, never a 503), `load_asr_model` fired
exactly once (the preload — the engine did not rebuild), the probe never ran, and the audio
shipped at its original length.

---

## Mechanism piece 3 — the 50% cap scales to the FIRST chunk (WS9.4, batch-only)

New kwarg `trim_cap_samples: int | None = None` on `_trim_icl_echo`
(`qwen3_tts/core/engine/inference.py:1138`) and on `_postprocess_chunk` (`:1291`), threaded
through (`:1308-1316`). The cap computation (`:1188-1190`):

```python
        cap_basis = trim_cap_samples if trim_cap_samples is not None else len(audio)
        max_cut = int(cap_basis * _ICL_MAX_TRIM_FRACTION)
        cut = min(cut, max_cut)
```

Only the multi-chunk batch path passes a basis — the FIRST chunk's length
(`run_inference`, `:1471-1480`):

```python
    result, sample_rate = _postprocess_chunk(
        result,
        sample_rate,
        gen_params,
        mode,
        config,
        reference_text=reference_text or _reference_text_from_prompt(voice_prompt),
        x_vector_only_mode=x_vector_only_mode,
        trim_cap_samples=len(all_audio[0]),
    )
```

The other three `_postprocess_chunk` call sites pass nothing, correctly: the single-chunk batch
path (`:1392`) trims audio whose first chunk IS the whole signal (basis `len(audio)` is the same
thing), and both streaming sites (`:1674` MLX, `:1717` torch) scope the trim to their first
emitted chunk already (`reference_text=ref_text if emitted == 0 else None`) — there the trimmed
audio is the first chunk, so the whole-audio basis is the first-chunk basis.

**The pins** — `tests/test_icl_echo_trim.py`:

- `TestTrimCapFirstChunk::test_trim_cap_samples_bounds_the_cut` (`:599`): a ~10 s generation with
  a 1 s cap basis loses at most 0.5 s — the old whole-audio basis would have licensed up to 5 s.
- `TestTrimCapFirstChunk::test_default_cap_basis_is_the_whole_audio` (`:610`): without the kwarg
  the basis stays `len(audio)` — callers that pass nothing keep the old behavior (backward
  compatibility is explicit, not incidental).
- `TestTrimCapWiredToSurfaces` (`:622`): drives all three surfaces with `_postprocess_chunk`
  spied — multi-chunk batch captures `trim_cap_samples == SR` (the FIRST chunk's length, not the
  7 s stubbed combined total, `:703`); single-chunk batch captures NO cap kwarg (`:709`); MLX
  streaming captures NO cap kwarg (`:714`).

---

## The degradation contract, stated once

For every clone generation on every wired surface:

| Situation | Outcome |
|---|---|
| ASR loaded (preload or earlier) at probe time | Trim probes and clips normally |
| ASR absent under the lock (unload landed in the preload window) | Untrimmed output; generation succeeds |
| Ensure-load itself fails (network, disk) | One warning; generation proceeds; trim skipped |
| `trim_icl_echo=false`, non-clone mode, `x_vector_only_mode`, or no resolvable transcript | No ASR load is even attempted; trim never probes |

A missing echo-trim is cosmetic. Losing a generation, or stalling the GPU-serialization lock on a
model download, is not — that ordering is the whole design.

---

## Gates run at dispatch (real output)

Runs by this implementer on the branch head (`5fd5715d` + the docs commits of this task), via
`conda run -n qwen3-tts-mlx`:

### Touched/new test modules, MLX env

```
conda run -n qwen3-tts-mlx python -m pytest tests/test_icl_echo_trim.py tests/test_echo_trim_asr_preload.py -q
```

```
============================== 55 passed in 2.32s ==============================
```

### Touched/new test modules, torchless CI-proxy leg (`.venv-310`) — RAN, not skipped

```
./.venv-310/bin/python -m pytest tests/test_icl_echo_trim.py tests/test_echo_trim_asr_preload.py -q
```

```
======================== 55 passed, 1 warning in 3.16s =========================
```

55/55 with 0 skipped on Python 3.10 without torch — the torchless leg actually ran the modules.

### ruff

```
conda run -n qwen3-tts-mlx ruff check qwen3_tts tests
```

```
All checks passed!
```

### Config-docs drift check

```
conda run -n qwen3-tts-mlx make check-config-docs
```

```
OK: CONFIG.md defaults match get_default_config() (66 keys).
```

Config surface is UNCHANGED by Step 1C (option (b) keep-loaded dropped the unload-after toggle —
no new key). The controller's CONFIG.md prose sync updates the row's behavior description only;
`trim_icl_echo` is not a `get_default_config()` key, so the drift checker does not inspect it.

---

## Part 1 — counted gate output (full non-E2E suite)

**Appended at gate time by the controller** (2026-09-10). Every invocation used
`conda run -n qwen3-tts-mlx`.

### Full non-E2E suite, counted

```
conda run -n qwen3-tts-mlx python -m pytest tests/ -m "not e2e" --ignore=tests/evaluations -q
```

```
==== 3342 passed, 4 skipped, 92 deselected, 3 warnings in 109.62s (0:01:49) ====
```

Exit code: 0.

**Delta from the 3325 passed / 5 skipped / 0 failed baseline (`8e44dad9`, pre-1C):**

- **+17 passed.** `git diff --numstat 8e44dad9..HEAD -- 'tests/*.py'` confines every test-file
  change to exactly this branch's own surface: `tests/test_echo_trim_asr_preload.py` (new,
  765/0 — the 11 server pins), `tests/test_icl_echo_trim.py` (+163/−2 — the rewritten
  degradation pin and the WS9.4 first-chunk cap pins), and `tests/run_batches.py` (+1 — the new
  module's `BATCHES` registration). No unrelated test file moved; this accounts for the
  pass-count increase.
- **−1 skipped (4 vs 5).** The skip that cleared is the pre-existing server-liveness-conditional
  `tests/test_ui_headless.py` create-from-audio case: the baseline run predated the live
  server's start; this run had it up. Unrelated to this diff.
- **0 failed**, matching the baseline.

### Static gates

```
conda run -n qwen3-tts-mlx ruff check qwen3_tts tests
```
```
All checks passed!
```

```
conda run -n qwen3-tts-mlx mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface
```
```
Success: no issues found in 58 source files
```

```
conda run -n qwen3-tts-mlx bandit -r qwen3_tts -c pyproject.toml
```
```
    Medium: 0
    High: 0
```
Exit code: 0 — 0 HIGH, 0 Medium confirmed.

### Torchless leg — RAN, not skipped

```
./.venv-310/bin/python -m pytest tests/test_icl_echo_trim.py \
  tests/test_echo_trim_asr_preload.py tests/test_engine_streaming.py -q
```
```
======================== 67 passed, 1 warning in 2.17s =========================
```

67 = 44 (`test_icl_echo_trim.py`) + 11 (`test_echo_trim_asr_preload.py`) + 12
(`test_engine_streaming.py`), 0 skipped, on Python 3.10 without torch — the CI-proxy leg
actually ran the touched modules.

### Batch runner — owning batches 1, 3, 4

```
conda run -n qwen3-tts-mlx python tests/run_batches.py --batch {1,3,4}
```
```
Total: 1/1 batches passed   (batch 1)
Total: 1/1 batches passed   (batch 3)
Total: 1/1 batches passed   (batch 4)
```

All exit codes: 0.

---

## Part 2 — live smoke (freshly restarted server, ASR unloaded)

**Appended at gate time by the controller** (2026-09-10). Server restarted onto the branch by the
user (`tts server stop && tts server start`, PID 47494 started 16:49:30 EDT); prompt `LT_4`
(`.pt`+`.wav`+`.txt` — transcript resolvable). Authed via the standard bearer token. One honest
timeline note: the machine slept ~6.5 h between leg 1 and leg 2 (log timestamps are the server's
own clock and are reproduced verbatim).

### Leg 1 — fresh server, ASR unloaded (the issue #193 repro condition)

ASR state BEFORE, via `/models`: `{"asr_loaded":{"loaded":false,"backend":null,"model_name":null}}`.

```
POST /generate  {"texts": ["<264 chars>"], "mode": "clone", "prompt_file": "LT_4.wav"}
→ HTTP 200, {"n_results": 1, "cancelled": false}
```

Server log, the complete arc:

```
16:52:06 [tts.engine] INFO: Loaded MLX ASR model
16:52:53 [tts.engine] INFO: Inference complete: 264 chars, 46.7s, mode=clone [mlx]
16:52:53 [tts.engine] INFO: Transcribing (MLX): /var/folders/.../icl_probe_4p451pju.wav
16:52:57 [tts.engine] INFO: Transcription complete: 54 chars in 4.5s
```

Reading: the force-load fired on a fresh server BEFORE the generation (16:52:06), the generation
ran 46.7 s, then the echo-trim probe transcribed the output head under the lock (4.5 s — inside
the banked 3.2–7.3 s measurement). No 503, no error; the result shipped. This is the exact shape
the ratified design mandates: load unlocked → generate → probe.

**No `Trimmed … ICL reference echo` line appears** in either leg: the probe found no echo in
these samples' heads (echo presence is model/sample-dependent). The trim itself — detection,
silence-snap, cap — is pinned by the unit pins (`TestTrimIclEchoBehaviour`,
`TestTrimCapFirstChunk`); what the live leg proves is the production wiring that was inert before
this step: the force-load, the probe execution, and clean success.

### Leg 2 — the unload leg, including a disclosed first attempt

`/unload-asr` → `{"status":"unloaded"}`; `/models` confirms unloaded. The FIRST regeneration used
a byte-identical payload and was served from cache with NO load and NO inference — correct
by-design behavior (pre-lock cache hits return before the preload call site), but it did not
exercise the re-fire, so it is disclosed rather than counted:

```
23:24:15 [tts.engine] INFO: Unloaded MLX ASR model
23:24:16 [tts] INFO: ASR model unloaded.
23:24:16 [tts] INFO: Generation cache hit (pre-lock) for text 1/1
```

The cache-busted retry (fresh text) proves the keep-loaded re-fire:

```
23:28:25 [tts.engine] INFO: Loaded MLX ASR model
23:28:43 [tts.engine] INFO: Inference complete: 181 chars, 18.4s, mode=clone [mlx]
23:28:43 [tts.engine] INFO: Transcribing (MLX): /var/folders/.../icl_probe_5r_wrh88.wav
23:28:50 [tts.engine] INFO: Transcription complete: 69 chars in 6.5s
```

```
→ HTTP 200, {"n_results": 1, "cancelled": false}
```

Reading: after an explicit `/unload-asr`, the next clone generation force-loads ASR again
(23:28:25) and runs the probe — keep-loaded, self-healing, no client-visible failure.

### What the live leg proves — and what it does not

Proven in production: the server-layer ensure-load fires unlocked on fresh and post-unload
servers; the probe runs under the lock after generation; generations succeed with real results in
both the fresh-server and post-unload states; a pre-lock cache hit triggers no load (by design).

Not proven here (covered by pins instead): the microsecond unload-lands-in-the-window race and
the untrimmed degradation — the window is real but not deterministically hittable over HTTP; the
end-to-end pin `test_unload_in_window_degrades_to_untrimmed` drives the REAL `handle_generate`
and the REAL `_trim_icl_echo` through exactly that interleaving (`tests/test_echo_trim_asr_preload.py`).
