# Consolidated Backlog — Priority Execution Plan

**Generated:** 2026-09-06 · **Method:** Blueprint (research → design → draft → adversarial review → register)
**Baseline:** `main` @ `fc9bfec`, working tree clean except a pre-existing dirty `config.json` (untouched by policy).

## Why this file exists

Every open GitHub issue and every unchecked item across `docs/plans/*.md` was re-checked against
current source on 2026-09-06 — several were stale (see the corrections merged in PR #259, `main` @
`fc9bfec`). This file is the **single prioritized execution order** for what's left, and supersedes
the standalone `phase3-ux-behavior-bugs.plan.md` draft mentioned in prior session notes.

**Verification depth is not uniform — read this before trusting a step's context brief at face
value.** Fully re-verified against current source, with exact file:line citations, and
adversarially reviewed against that source (2026-09-06): issues #193, #237, #238; Lanes H/I; the
four P2-1 line counts; Phase-3 items 3d and 3f. **Carried forward from existing plan docs without
an independent re-check this pass**: Phase-3 items 3c, 3e, 3g (Steps 3C/3D/3E) and the E2E
timeout/crash claims in Wave 4 — each of those steps says so explicitly in its own text, and each
opens with "re-verify against current source" as its first task for exactly that reason. Treat the
unverified blocks as more likely to contain the kind of stale-since-written drift PR #259 already
found twice, not less.

**One duplicate found and consolidated:** GitHub issue **#238** ("`/update-model-config` leaves
stale `model_load_times`") and the master plan's Phase-3 item **3e/M8** ("`/update-model-config`
calls `unload_model_cleanup`") are the same fix, filed independently. Step 1B below closes both.

## Legend

- **Wave** — a batch of steps with no shared write-surface; steps inside a wave are parallel-safe.
  Waves are ordered — do not start wave *N+1* until wave *N*'s file-overlapping steps have landed
  (a step in wave *N+1* touching a file wave *N* also touched must rebase first).
- **Model tier** — `strongest` (a judgment call, high blast radius, or a documented past risk) vs
  `default` (well-specified, mechanical, low risk).
- **Rollback** — every step is its own feature branch + squash-merge PR; revert is `git revert
  <squash-commit>` on `main`, or delete the branch pre-merge. Never `--amend`, never direct-to-main
  (this repo's standing policy).
- Branch prefixes matter for CI: `fix/*`/`feat/*`/`feature/*` get the push matrix; `docs/*`/
  `chore/*` get none until a PR opens (still fine — PRs always get the matrix).

## 2026-09-06 addendum: four parallel cross-cutting reviews folded in

After the initial plan was registered, four independent analyses ran in parallel against current
source (`ecc:python-reviewer`, `ecc:security-reviewer`, a coverage-gap analysis, and a static E2E
health assessment) — dispatched per `superpowers:dispatching-parallel-agents` (independent domains,
no shared state, single round each). Their findings are folded in below as **Wave 0** (new,
highest-priority — correctness/security bugs found by two independent reviews converging on the
same defect in one case), corrections to **Wave 4** (the E2E agent falsified Wave 4's original
premise — see that section), a new **Wave 4B** (test-coverage gaps), and new steps appended to
**Wave 6** (code-quality findings). Dead-code cleanup (`Step 6·0`) was executed directly and is
recorded in Wave 6.

One more convergence worth naming: security's **MEDIUM-4** (the UI's MLX voice-prompt create path
bypasses the engine writer's security guards) and the E2E agent's **G1** (Voice Management tab has
zero E2E coverage at any level) are the same file (`interface/ui/voice_management.py`) — fixing the
security gap and adding its test coverage belong in the same effort; see Step 0G.

---

## Wave 0 — critical findings from the 2026-09-06 cross-cutting review (highest priority, ahead of Wave 1)

### Step 0A — Bound `max_chunk_chars`: found independently by both python-review and security-scan

- **Model tier:** default · **Branch:** `fix/max-chunk-chars-bounds` · **Parallel with:** 0B, 0D (different files); rebase against 0C/0E if those land first (shared files, see below)
- **Context:** `GenerateRequest.max_chunk_chars` (`server/validation.py:51`) is the only numeric
  request field with no `Field(...)` bounds — every sibling (`temperature`, `max_new_tokens`,
  `seed`) has them, and CLAUDE.md documents a `0`–`10000` config-side range that isn't enforced on
  the request path. A negative value does two things at once: (1) `_prepare_text_chunks`
  (`inference.py:1243`, gate is `max_chunk_chars > 0`) disables chunking entirely, so a 50,000-char
  request becomes one uninterruptible `generate()` call; (2) `_stream_thread_join_timeout`
  (`app_generation.py:105-108`) computes `min(negative, text_len)` → the 90s floor, so both
  `/generate-stream` and `/ws` release `inference_lock` 90s into a multi-hour generation **while the
  model is still on the GPU** — precisely the unsynchronized-concurrent-inference condition the
  entire #192/#214 serialization effort exists to prevent, and the function's own docstring says
  "never reintroduce a constant here." Verified live: `max_chunk_chars=-1, text_len=50000 →
  _stream_thread_join_timeout returns 90.0`.
- **Tasks:** write a failing test asserting a negative/out-of-range `max_chunk_chars` is rejected at
  the request-validation layer (400); RED; add `Field(default=None, ge=0, le=10000)` to
  `GenerateRequest`; also add a defensive clamp inside `_stream_thread_join_timeout` itself
  (`effective_chars = min(max_chunk_chars, text_len) if max_chunk_chars and max_chunk_chars > 0 else
  text_len`) so a direct engine caller (not just the HTTP layer) can't reopen this; GREEN.
- **Verify:** new test alongside `tests/test_streaming_thread_lifecycle.py`; `pytest -k
  max_chunk_chars`; `ruff`; `mypy`.
- **Exit criteria:** negative/out-of-bounds `max_chunk_chars` rejected at the API boundary; the join
  timeout can't collapse to the floor even if the schema guard is bypassed.

### Step 0B — Fix stale model reference after an unload-then-reload race

- **Model tier:** strongest (concurrency-correctness, same defect family as #192/#214) · **Branch:** `fix/require-model-under-lock-returns-model` · **Depends on:** rebase after Step 1B lands (both touch `model_loading.py`/`app_models.py` territory) and before Step 2 (Lane H reuses this exact mechanism — see its risk note)
- **Context:** `_require_model_under_lock` (`app_generation.py:127-158`) only asserts
  `state.models[mode] is not None` — it does not return the model, and no caller rebinds. Capture
  sites: `app_generation.py:223` (batch), `:742`/`:875` (stream), `websocket.py:306`/`:433` (ws). An
  unload **followed by a reload** inside the capture→acquire window passes the guard (slot is
  non-`None` again) while the request runs inference against the *old*, backend-cleaned-up model
  object — exactly the scenario the guard's own docstring warns about, just from the other
  direction. `/create-voice-prompt`'s torch path has the same gap with an even wider window and no
  post-lock recheck at all (`app_prompts.py:428-512` — see Step 0B's companion fix below).
- **Tasks:** change the signature to `_require_model_under_lock(state, mode) -> Any` returning
  `state.models[mode]`; rebind at all five call sites above (`model = _require_model_under_lock(...)`)
  before entering inference; apply the same fix to `/create-voice-prompt`'s capture at
  `app_prompts.py:428` (call it immediately inside `async with state.inference_lock`, use the
  returned model). Write a failing test that interleaves unload+reload inside the capture window and
  asserts the *new* model object is what actually runs; extend
  `tests/test_issue214_unload_queued_window.py`.
- **Verify:** `pytest tests/test_issue214_unload_queued_window.py tests/test_voice_server.py -v`; `ruff`; `mypy`.
- **Exit criteria:** an unload-then-reload interleaving is proven to use the fresh model object at every one of the six call sites, with a regression test per site (or one parametrized test covering all six).

### Step 0C — `generation_state` threading discipline: guarded by the wrong lock type, mutated unlocked in places

- **Model tier:** strongest · **Branch:** `fix/generation-state-thread-safety` · **Files:** `app_generation.py`, `websocket.py`, `app_models.py`, `app.py` — **overlaps Step 1A's write surface (`app_generation.py`, `app.py`) and may be the deeper mechanism behind #237** (a lost cancel from a race on this exact dict). Resolve this step and Step 1A together, or do this one first and re-check whether #237's symptom still reproduces afterward — don't fix the same race twice independently.
- **Context:** `state.generation_lock` is an `asyncio.Lock`, but `_chunk_progress`
  (`app_generation.py:454-460`, `:860-867`) mutates `generation_state` from inside
  `asyncio.to_thread(...)` / the streaming inference thread — an `asyncio.Lock` provides zero mutual
  exclusion against a non-event-loop thread. Separately, the `/generate-stream` path updates
  `generation_state` at `:849-858` and resets it at `:956-968` **without** taking `generation_lock` at
  all, while the batch path, `/ws`, and `/cancel-generation` all do. `handle_unload_model`
  (`app_models.py:226-229`) also reads `active`/`mode` with no lock. Consequence: a
  `/cancel-generation` landing between the streaming acquire and the `:856` `"cancelled": False`
  write is silently clobbered.
- **Tasks:** introduce a small wrapper (e.g. a `GenerationStateGuard` class) owning a
  `threading.Lock` (not `asyncio.Lock` — both threads and the event loop touch this state) with
  `begin()` / `update_progress()` / `reset_if_owner(generation_id)` / `snapshot()` methods; route
  every read and mutation of `generation_state` through it, including the two currently-unlocked
  streaming sites and `handle_unload_model`'s read. This also gives `/generation-status`
  (`app.py:601-608`) and `/queue-status` (`app.py:618-622`) one atomic `snapshot()` instead of a
  multi-key unlocked read. Write a failing test reproducing the streaming-path clobbered-cancel
  scenario; RED; implement; GREEN; re-run Step 1A's tests (if landed) to confirm no regression.
- **Verify:** full server test suite; `ruff`; `mypy`.
- **Exit criteria:** every `generation_state` read/write goes through the guard; the previously-unguarded streaming clobber is fixed with a regression test; #237's fix (Step 1A) and this step don't duplicate work — cross-reference confirmed in both PR bodies.

### Step 0D — WebSocket connection-slot leak on the pre-auth path

- **Model tier:** default · **Branch:** `fix/ws-slot-leak-preauth` · **Parallel with:** 0A, 0B
- **Context:** `_ws_try_acquire` (`websocket.py:90`) reserves a slot, but `_ws_release` is only
  reached on paths where `await websocket.close(...)`/`send_json(...)` succeed. `await
  websocket.accept()` (`:95`) is outside any `try`, so a client that vanishes mid-handshake leaks a
  slot permanently; the three auth-failure branches (`:111-113`, `:120-123`, `:135-136`) call
  `close()`/`send_json()` *before* `_ws_release`, so if the peer is already gone those raise and the
  release is skipped. Same defect class the `:101-105` comment says was already fixed once, via a
  different escape — repeated slot leaks exhaust `_WS_MAX_TOTAL`.
- **Tasks:** wrap everything after `_ws_try_acquire` in `try: ... finally: _ws_release(app_state,
  client_ip)`, deleting the four scattered release calls. Write a test that patches
  `websocket.close`/`send_json` to raise and asserts `app_state._ws_connections` returns to empty
  afterward.
- **Verify:** `pytest tests/test_websocket*.py -v`; `ruff`; `mypy`.
- **Exit criteria:** slot count returns to baseline after a simulated mid-handshake disconnect and after each auth-failure branch, even when the close/send call itself raises.

### Step 0E — Gradio UI can launch unauthenticated and publicly shared

- **Model tier:** strongest (security-sensitive, user-facing default behavior change) · **Branch:** `fix/gradio-share-requires-auth`
- **Context:** `share = bool(os.environ.get("TTS_UI_SHARE")) or IN_COLAB`
  (`generate_server.py:125-133`, `ui/_facade.py:574-581`, `ui/shared.py:872-887`,
  `colab_notebook.ipynb` cell 6) — every Colab run forces a public `*.gradio.live` URL, and `tts ui
  --share` does the same locally, with no `auth=` passed to `demo.launch()` anywhere. The UI process
  holds the server bearer token, so anyone with the link gets the full authenticated surface:
  generate, delete/rename prompts, `/update-model-config`, `/shutdown`, plus hard-delete of
  `~/Downloads/Qwen3-TTS Output` files and Gradio's file route over `allowed_paths` (currently all of
  `~/Downloads`, not just the app's subfolders). ARCHITECTURE.md's Security section doesn't cover the
  UI surface at all.
- **Tasks:** add `auth=` (or `auth_dependency`) to `get_gradio_launch_kwargs()` whenever `share` is
  truthy — generate a random user/password at launch and print it to the console/log; refuse to
  launch with `share=True` and no credentials configured. Mirror in the Colab notebook cell. Narrow
  `allowed_paths` to the Automated Output / Manual Downloads subfolders specifically, not all of
  `~/Downloads`. Add a Security section entry for the Gradio UI to ARCHITECTURE.md.
- **Verify:** manual launch with `TTS_UI_SHARE=1` (or Colab-simulated), confirm a shared link requires the printed credentials; `ruff`; `mypy`.
- **Exit criteria:** no code path launches a publicly-shared Gradio instance without authentication; `allowed_paths` narrowed; ARCHITECTURE.md updated.

### Step 0F — Auth-failure error handling gaps (audit-log bypass + streaming error disclosure)

- **Model tier:** default · **Branch:** `fix/auth-and-stream-error-sanitization` · **Bundled: two related but distinct error-handling gaps, both small, same theme.**
- **Task 1 — `verify_auth` 500s instead of 401ing on a non-ASCII bearer token, skipping the audit log:** `secrets.compare_digest` (`app.py:251-252`) raises `TypeError` on non-ASCII input — it fails closed, but the R-26 audit-log line (`app.py:256-262`) never fires, so probing produces no "Auth failure" log entries and a 500+traceback instead of a cheap 401. Fix: compare as bytes (`token.encode("utf-8", "replace")` vs. the stored token's bytes) or wrap in `try/except TypeError` falling through to the existing 401 branch. While there, replace the three `.replace("Bearer ", "")` call sites (`app.py:211,233,251`) with a proper case-insensitive scheme strip (`removeprefix`) — a global `.replace()` mangles a token that happens to contain the substring "Bearer ".
- **Task 2 — `/generate-stream`'s terminal error frame ships an unsanitized exception string:** `thread_error[0]` (raw `str(e)`, set at `app_generation.py:908`) is JSON-wrapped and sent as-is at `:982-983` — the `/ws` sibling correctly runs it through `_sanitize_error` (`websocket.py:603-608`) first. This is the one path shipping absolute filesystem paths / HF cache locations to the client, the same CWE-209 class already closed on `/health`. Fix: `yield encode_stream_error_frame(_sanitize_error(thread_error[0]))`.
- **Verify:** a test posting a non-ASCII `Authorization` header and asserting 401 + an audit-log entry; a streaming-error test asserting the terminal frame's message is sanitized; `ruff`; `mypy`.
- **Exit criteria:** both gaps closed with regression tests; no unsanitized exception text reaches any client-facing surface.

### Step 0G — MLX voice-prompt UI create bypasses the engine writer's security guards (closes security MEDIUM-4 + starts E2E gap G1)

- **Model tier:** strongest (security-relevant write path + new test surface) · **Branch:** `fix/ui-voice-create-use-engine-writer`
- **Context:** `interface/ui/voice_management.py:123-168` re-implements `save_voice_prompt_mlx`
  inline (`sf.read`/`sf.write`/`shutil.copy` directly) instead of calling it — so it has none of the
  writer's protections: no realpath/home-or-tempdir containment on the source path, no zero-sample
  check, no `.wav`-removal rollback if the `.txt` write fails, no transcript strip. It's the only
  voice-prompt write path in the codebase without the source-containment guard, and (per Step 0E) is
  reachable from a publicly-shared UI. The writer's own comment at `voice_prompt.py:468-470` claims a
  peer guard exists in `tabs_generation.py` — verified false; the only guard there is on the
  history-replay path, not create.
- **Tasks:** delete the inline MLX branch in `voice_management.py`; call `save_voice_prompt_mlx(base_name, audio_path, transcript)` instead, mapping `UnsupportedReferenceAudioError`/`sf.LibsndfileError` to `gr.Error` the way `app_prompts.py:461-491` already does; correct the stale comment at `voice_prompt.py:468-470`. This is also the natural place to add the E2E coverage the separate analysis flagged as the single highest-severity coverage gap (create/delete/rename/preview on the Voice Management tab currently has zero E2E coverage at any level) — at minimum add a create-from-audio E2E case exercising this exact fixed path, deferring the rest of that gap (delete/rename/preview) to Wave 4's E2E work.
- **Verify:** unit test asserting the UI path now goes through the same containment/rollback guards as the server path; one new E2E test for create-from-audio via this UI flow; `ruff`; `mypy`.
- **Exit criteria:** UI create path provably shares the engine writer's guards (same containment, same rollback behavior); stale comment corrected; at least one E2E test exists for this flow where none did before.

---

## Wave 1 — independent bug fixes (parallel-safe, 5 lanes)

Step 1D/1E are test-infrastructure hygiene, not a correctness bug — placed in Wave 1 rather than
alongside Wave 6's other mechanical cleanup because gate integrity has to be trustworthy *before*
the correctness work in this and later waves relies on it. Fixing it last (Wave 6's stated rationale
for everything else) would mean every wave in between ran against a batch gate known to be silently
incomplete.

### Step 1A — Fix #237: cancel landing in a batch item's pre-lock window is erased

- **Model tier:** default · **Branch:** `fix/issue237-batch-cancel-race` · **Parallel with:** 1B, 1C, 1D — but see the corrected write-surface note below; touches `app.py` in addition to `app_generation.py`, unlike any other Wave-1 step, so it stays parallel-safe only because nothing else in this wave touches `app.py`.
- **Context (corrected during adversarial review — the original draft misdescribed both the window and the fix's write surface):**
  The race is real but bigger than one file. `/cancel-generation` (`app.py:862`) sets
  `state.generation_state["cancelled"] = True` and nothing else — there's no per-cancel target id,
  and `generation_state`'s init in `app_lifespan.py` carries `generation_id: None` with no
  `cancel_target_id` field. The batch loop (`app_generation.py`) re-clears `cancelled: False` on
  every item's in-lock state update at `:450` (comment: this exists to stop a *stale* flag from a
  **different concurrent request** truncating this batch).
  **The actual erasable window is `:343` (the per-item cancel check) → `:450` (the in-lock reset)**
  — not "the pre-lock cache check," which is earlier (`:290-326`) and is already caught correctly
  at `:343`.
  **A second, separate drop path exists and must be fixed in the same step:** `/cancel-generation`
  (`app.py:860`) returns `{"status": "no_active_generation"}` without setting the cancel flag at all
  when `state.generation_state["active"]` is `False` — and `active` is only set `True` at
  `app_generation.py:438-440`, **inside** `inference_lock`. A cancel for batch item 0, arriving
  before the lock is first acquired, is rejected outright by this check and never reaches the flag.
  Fixing only the `:343→:450` window (as the original draft proposed) leaves this path unfixed.
- **Files:** `qwen3_tts/server/app.py` (`/cancel-generation` — needs a way to attribute a cancel to
  the in-flight `generation_id` rather than a bare global boolean, and must not bounce a cancel that
  arrives before `active` flips true), `qwen3_tts/server/app_lifespan.py` (`generation_state`'s
  initial schema — add whatever field the attribution scheme needs), `qwen3_tts/server/app_generation.py`
  (the batch loop's cancel-check-and-reset).
- **Tasks:**
  1. Write a failing test that drives a batch generation, injects a cancel timed to land in the
     `:343→:450` window, and asserts the batch actually stops.
  2. Write a second failing test that injects a cancel **before** `active` is first set to `True`
     (i.e. before item 0 acquires `inference_lock`) and asserts it is not silently dropped as
     `no_active_generation` — the batch must still stop.
  3. Confirm both fail today (RED).
  4. Design and implement an attribution scheme (e.g. record the target `generation_id` alongside
     `cancelled` when `/cancel-generation` is called, and have the batch loop only honor a cancel
     addressed to *its own* `generation_id` — this is what actually prevents the stale-flag
     cross-request scenario the existing comment describes, more precisely than an unconditional
     `False` reset does) that closes both windows without reopening the cross-request stale-flag bug.
  5. GREEN on both new tests, then confirm the original stale-flag-from-another-request scenario
     still passes.
- **Verify:** `conda run -n qwen3-tts-mlx python -m pytest tests/test_voice_server.py tests/test_e2e_queueing.py -v`; register new tests in `tests/run_batches.py` if they land in a new module; `ruff check qwen3_tts tests`; `mypy qwen3_tts/{core,server,interface}`.
- **Exit criteria:** both new tests prove their respective races fixed (fail pre-fix, pass post-fix); full non-E2E suite green; issue #237 closed in the PR body, explicitly noting both fixed windows.

### Step 1B — Fix #238 and Phase-3 3e/M8: `/update-model-config` resource-cleanup gap

- **Model tier:** default · **Branch:** `fix/issue238-update-model-config-cleanup` · **Parallel with:** 1A, 1C, 1D
- **Context:** `handle_update_model_config` (`qwen3_tts/server/app_models.py:280`) nulls all three
  model slots directly (`state.models[name] = None`, `:329-330`) but never calls
  `unload_model_cleanup()` and never pops `state.model_load_times`. Compare `handle_unload_model`
  (`app_models.py:259-274`), which does both. Result: `/models` keeps reporting a stale
  `load_time_sec` for a model that was just invalidated by a config change, until the next load
  overwrites it.
  **These are two distinct defects sharing one root cause, not one bug** — the master plan's 3e/M8
  wanted the missing `unload_model_cleanup()` call (memory reclaim); GitHub issue #238 wanted the
  stale `model_load_times` fixed (reporting accuracy). Both tasks below are required; do not treat
  either as optional because the other is "the real fix."
- **Tasks:**
  1. Write a failing test: call `/update-model-config` after a model has a recorded `load_time_sec`,
     then hit `/models` and assert the entry is gone (not stale). This covers #238.
  2. Write a second failing test asserting `unload_model_cleanup()` is actually invoked when
     `/update-model-config` changes settings (e.g. spy/mock it and assert called) — do not let this
     ship untested just because the `model_load_times` test above happens to pass without it. This
     covers 3e/M8.
  3. RED on both — today neither happens.
  4. Fix: in `handle_update_model_config`, after nulling the three model slots, call
     `unload_model_cleanup()` **once** (it is a no-arg module-level function —
     `core/engine/asr.py:328`, `gc.collect()` + a backend-wide `empty_cache()` — not a per-model
     call; calling it three times would be redundant, not "more correct"), matching
     `handle_unload_model`'s single call. Then pop `state.model_load_times` for each of the three
     names. Keep the existing `model_config_epoch` bump logic unchanged — it's correct and unrelated
     to this gap.
  5. GREEN on both tests.
  6. Update `~/.claude/plans/review-entire-repo-for-ancient-possum.md`'s Phase-3 section to mark
     3e/M8 done, citing this PR — this is the step that closes that loop; Step 3D is explicitly told
     to skip M8, so if this step doesn't record the closure nothing else will.
- **Verify:** `conda run -n qwen3-tts-mlx python -m pytest tests/test_voice_server.py -v -k update_model_config`; `ruff`; `mypy`.
- **Exit criteria:** both new regression tests pass; issue #238 closed in the PR body; master plan's 3e/M8 marked done in the same PR (task 6 above) — do not re-do M8 in Step 3D.

### Step 1C — Decide, then fix, #193: clone output includes the re-spoken reference echo by default

- **Model tier:** strongest (genuine design tradeoff, not mechanical) · **Branch:** `fix/issue193-icl-echo-trim-default` · **Parallel with:** 1A, 1B, 1D
- **Context:** `_trim_icl_echo` (`qwen3_tts/core/engine/inference.py:1131`) only runs when
  `asr.is_asr_loaded()` is already true — it deliberately never force-loads ASR (comment: "pulling a
  heavy ASR model into a generation that never asked for one would cost more than the artifact it
  removes"). On a fresh server (ASR not loaded, which is the common case — ASR isn't in
  `load_at_startup` defaults) the trim is silently inert and cloned output includes the re-spoken
  reference tail, even though `generation.trim_icl_echo` defaults to `true` and CLAUDE.md documents
  the feature as on-by-default.
- **Decision needed first (this is why the step is `strongest` tier) — the choice is binary, not
  three-way (a third option was eliminated during adversarial review — see below):**
  (a) leave as-is and fix the **documentation** instead (trim is opportunistic-only, never
  guaranteed) — lowest cost, lowest value;
  (b) force-load ASR transiently for the trim probe (first 4s only) when `trim_icl_echo=true` and no
  ASR is already loaded, then leave ASR loaded or unload it afterward per a new config toggle —
  highest value, adds real latency/memory cost to every clone generation on a fresh server.
  **Ruled out: threading `reference_text` from the voice-prompt's stored transcript as a
  substitute for loading ASR.** This is already implemented (`inference.py:1385` and `:1639` both
  pass `reference_text or _reference_text_from_prompt(voice_prompt)`) and does **not** touch the
  actual gate. `_trim_icl_echo` uses ASR to transcribe the *generated output's head*
  (`inference.py:1163`, `head_text = _transcribe_probe(...)`) — `reference_text` is only the
  comparison target at `:1164`, not a substitute for that transcription. The gate that makes the
  feature inert is `:1155` (`if not asr.is_asr_loaded(): return audio, sample_rate`), which a stored
  transcript does nothing to change. An implementer who picks this option ships no fix at all.
  **Record the decision** (with rationale) in this plan file before writing code, the same way
  FOLLOWUP-1 was recorded in `consolidated-roadmap.md`.
- **Tasks (after the decision is recorded):** write the failing test for the chosen behavior, RED,
  implement, GREEN, update `CLAUDE.md`'s `trim_icl_echo` row if the on-by-default claim changes.
- **Verify:** `conda run -n qwen3-tts-mlx python -m pytest tests/test_icl_echo_trim.py -v`; live clone smoke on a **freshly started server with ASR unloaded** (the exact repro condition) — `curl` `/generate` with a clone prompt and confirm the reference tail is absent (or confirm the doc now accurately says it can be present, if option (a) is chosen).
- **Exit criteria:** decision recorded; either behavior fixed with a passing regression test, or CLAUDE.md corrected to stop overpromising — issue #193 closed either way, with the disposition stated in the closing comment.

### Step 1D — Lane I: de-hollow `tests/test_server_peaks.py`, and add a static guard against the whole class of bug

- **Model tier:** default · **Branch:** `fix/test-server-peaks-unittest` · **Parallel with:** 1A, 1B, 1C
- **Context:** confirmed 2026-09-06 — `tests/test_server_peaks.py`'s classes
  (`TestGenerateResultPeaksField`, `TestCalculateWaveformPeaksExistence`) are plain pytest-style
  classes, not `unittest.TestCase`. `tests/run_batches.py` executes every batched module via
  `python -m unittest` (`:384-390`), which collects only `TestCase` subclasses — this module has
  been passing hollow in every batch gate since it was written, the same false-green family issue
  #181 already fixed elsewhere.
  **Corrected during adversarial review, both the sabotage and the harness prescription in the
  original draft were wrong for this file:** the module contains no `/generate` call and no app —
  its four tests only assert (a) that `GenerateResult` (from `server/validation.py`) accepts a
  `peaks` field, and (b) that `calculate_waveform_peaks` returns the right length/range for known
  inputs. "Sabotage `/generate`'s response" doesn't touch either of those and would leave the module
  green under pytest too — an unfalsifiable proof. There's also no app/client to build, so the
  `test_peaks_caching.py`-style `TestClient` + `_init_app_state` harness this step originally
  prescribed is unnecessary machinery for four pure unit assertions.
  **Also discovered during review: this is not a one-module problem.** At least two other *batched*
  modules are entirely pytest-style and therefore fully hollow in the batch gate today —
  `tests/test_validation.py` (batch 5, 7 plain classes) and `tests/test_audio_pipeline.py` (batch 1,
  3 plain classes) — plus plain `class Test…:` declarations were spotted in `test_create_voice_functions`,
  `test_server_vllm_integration`, `test_fastapi_app_ext`, `test_solid_analyzer`, `test_voice_helpers`,
  `test_error_handling`, `test_ocp_strategy`, `test_ai_regression`, and `tests/security/*`. Converting
  only `test_server_peaks.py` and calling gate-integrity solved would be wrong. This step covers the
  one module plus the guard that prevents recurrence; **Step 1E (new, inserted after review) covers
  sweeping the rest.**
- **Tasks:**
  1. **Prove the hollowness with a sabotage that can actually fail:** break
     `calculate_waveform_peaks` (e.g. make it always return an empty list), confirm this makes the
     module's own assertions fail under plain pytest (`pytest tests/test_server_peaks.py -v`), then
     confirm `python -m unittest tests.test_server_peaks -v` still reports OK despite the same
     sabotage — *that's* the real, falsifiable proof of hollowness. Revert the sabotage before
     continuing.
  2. Rewrite the module's two classes as `unittest.TestCase` subclasses. No app/TestClient/fixture
     is needed — these are plain function-call assertions; a bare `class TestGenerateResultPeaksField(unittest.TestCase):` conversion (methods renamed to `test_*` if not already, assertions changed from `assert` to `self.assertX` as needed) is sufficient. Keep every existing assertion.
  3. Re-run the Step 1 sabotage against the converted module: it must now go RED under
     `python -m unittest tests.test_server_peaks`, and GREEN after reverting.
  4. Add a static guard extending `tests/test_async_test_hygiene.py`'s AST-based approach (or a
     sibling check) that fails if any batched module (cross-reference `tests/run_batches.py`'s
     `BATCHES` list) defines a top-level test class that does not inherit `unittest.TestCase` —
     this is what actually prevents recurrence; without it, the next new test module can reintroduce
     the exact same hollow-gate bug.
- **Verify:** `conda run -n qwen3-tts-mlx python -m unittest tests.test_server_peaks -v` (confirm real test count) and `pytest tests/test_server_peaks.py -v`; confirm batch 3 (`run_batches.py:114,167` — the module's actual batch assignment, not its docstring, which doesn't mention batching) now runs it for real; run the new static guard against the current `BATCHES` list and confirm it flags the *other* hollow modules discovered this pass (expected — that's Step 1E's job, not this one's, but the guard should already be catching them).
- **Exit criteria:** module runs identically (same pass/fail) under both runners; sabotage proof documented in the PR body; static guard merged and correctly flagging the remaining hollow modules (which Step 1E will then fix).

### Step 1E — Sweep the remaining hollow batched test modules (inserted after adversarial review)

- **Model tier:** default · **Branch:** `fix/hollow-batched-test-modules` · **Depends on:** Step 1D's static guard landing first (this step's exit criterion is that the guard goes quiet)
- **Context:** the guard added in Step 1D will flag every batched module with a non-`TestCase`
  top-level test class. Known-hollow as of 2026-09-06: `tests/test_validation.py` (batch 5, 7
  classes) and `tests/test_audio_pipeline.py` (batch 1, 3 classes) — both confirmed fully pytest-style,
  fully hollow under the batch runner today. Additionally spotted but not individually confirmed
  hollow (verify each before assuming): `test_create_voice_functions`, `test_server_vllm_integration`,
  `test_fastapi_app_ext`, `test_solid_analyzer`, `test_voice_helpers`, `test_error_handling`,
  `test_ocp_strategy`, `test_ai_regression`, `tests/security/*`.
- **Tasks:** for each module the Step 1D guard flags: confirm real hollowness with a sabotage proof
  (same discipline as Step 1D task 1 — do not assume it's hollow just because the class isn't
  `TestCase`; some pytest-style classes may already coincidentally get collected some other way),
  convert to `unittest.TestCase`, re-verify the sabotage now goes RED. This can be split across
  several small PRs (one per module or small module-group) rather than one giant PR — low risk,
  high count, good candidate for parallel sub-agents once the guard's flagged list is in hand.
- **Verify:** the Step 1D guard reports zero flagged modules; `python -m unittest discover tests` and `pytest tests/` produce the same pass/fail set for every converted module.
- **Exit criteria:** static guard passes clean; every batched module's tests actually execute under the batch runner.

---

## Wave 2 — serial after Step 1A (same file: `app_generation.py`)

### Step 2 — Lane H: load design/custom models on demand in `/generate` (RUNBOOK contract)

- **Model tier:** default · **Branch:** `fix/on-demand-model-load-generate` · **Depends on:** 1A must land and be rebased onto first (same file, `/generate`'s model-lookup path vs. the batch-cancel path — different functions, but the same module; rebase to avoid a stale diff).
  **Hard gate added during adversarial review — do not merge until Wave 4 (Step 4B) has either
  ruled out or fixed the server's repeated-load/unload memory-pressure crash.** This step turns
  on-demand model loading from a manual, deliberate UI click into something any unauthenticated-path
  `/generate` request can trigger routinely — it multiplies how often the exact failure mode Wave 4
  investigates gets exercised, before anyone knows why it happens. If Wave 4 finds a real leak,
  shipping this step first makes the leak fire far more often in production.
- **Additional risk flagged during review:** the original draft said to keep the load "outside
  `inference_lock`, matching the existing voice-prompt-load precedent" — that precedent doesn't
  transfer. A voice-prompt load is disk I/O + tensor deserialize; a *model* load is the exact class
  of GPU-adjacent work that issues #192/#214 spent multiple PRs proving must be serialized (warm-up
  itself had to move under `inference_lock` after shipping unlocked once). Do not design this as a
  bespoke unlocked `await load_model(...)` call — route it through the same per-load-record/claim
  mechanism `model_loading.py` already provides for `/load-model` (`claim_model_load`), which exists
  precisely to make concurrent/racing loads safe. Reusing it, rather than reinventing locking here,
  is both less risky and less code.
- **Context:** confirmed 2026-09-06 — `/generate` and `/generate-stream` still return `model_not_loaded` (`app_generation.py:232`, `:748`) when `design`/`custom` aren't loaded, even though `design.load_at_startup`/`custom.load_at_startup` are `false` by design (on-demand load is documented — RUNBOOK / CLAUDE.md — as "one click away" with a live ETA badge). The handler never actually attempts the on-demand load itself; only the UI's manual "Load Model" button does.
- **Tasks:**
  1. Write the failing test: POST `/generate` with `mode="design"` (or `"custom"`) while unloaded, mock `load_model`/`run_inference`, assert 200 and that `load_model` was called once (not a 503).
  2. RED — today this 503s with `model_not_loaded`.
  3. Implement on-demand load in the handler's model-resolution step using `model_loading.py`'s
     existing `claim_model_load` per-load-record mechanism (see risk note above) rather than a new
     unlocked `await load_model(...)` — this gets the double-load-prevention and the loading-state
     badge infra for free, since `/load-model` already relies on it. On load failure, return the
     same sanitized 503 shape `/load-model` produces (via `_recover_from_failed_load`, PRF-5) —
     never a bare 500.
  4. Concurrency: two simultaneous first-requests for the same unloaded model must not double-load
     — this should fall out of task 3's reuse of `claim_model_load`, but test it explicitly with two
     concurrent requests to confirm.
  5. Update the error text for the now-rarer genuinely-cannot-load path to point at `POST /load-model` / the Manage Models tab instead of "restart."
  6. Update CLAUDE.md's Key Settings row for `load_at_startup` and the Server API table's `/generate`
     description to reflect the new on-demand behavior — both currently describe manual-load-only.
- **Verify:** `conda run -n qwen3-tts-mlx python -m pytest tests/test_voice_server.py -v -k generate`; full non-E2E suite; `ruff`; `mypy`.
- **Exit criteria:** Wave 4's gate (above) satisfied; `/generate` on an unloaded design/custom model loads it instead of erroring; concurrent-first-request test passes without double-load; CLAUDE.md updated; RUNBOOK's documented behavior is now actually true.

---

## Wave 3 — Phase-3 UX/behavior sweep remainder

*(Sub-item detail for 3c/3e/3g is inlined below — the master plan's version at
`~/.claude/plans/review-entire-repo-for-ancient-possum.md` is no longer needed to evaluate these
steps, per the adversarial review's finding that external, unversioned references made them
unfalsifiable. 3e's M8 sub-item is dropped here — it's Step 1B above.)*

### Step 3A — 3d: make `seed_lock_chunks` real

- **Model tier:** default · **Branch:** `fix/phase3-seed-lock-chunks` · **Depends on:** rebase after 1C (same file, `inference.py`) **and after Step 2 lands** (write surface now includes `app_generation.py`, which Step 2 also touches — see correction below).
- **Context (corrected during adversarial review — "all three runners" was false):** confirmed 2026-09-06 — `seed_lock_chunks` appears in `inference.py` in exactly **two** places, both inside `run_inference` alone: the signature (`:1330`) and `seed = gen_params.get("seed") if seed_lock_chunks else None` (`:1402`). When `seed_lock_chunks=False`, seed is `None` for every chunk (no seeding at all), **not** the prescribed per-chunk derived seed (`base_seed + chunk_index`).
  **`run_inference_streaming` has no `seed_lock_chunks` parameter at all** (`inference.py:1586-1600`)
  — its two call sites, `app_generation.py:874` and `websocket.py:432`, pass none. There is exactly
  one runner today, not three; making the flag "real" for streaming means **adding** a new parameter
  to `run_inference_streaming` and threading it through both call sites, in two modules this step
  did not originally list.
- **Files:** `qwen3_tts/core/engine/inference.py` (both `run_inference` and `run_inference_streaming`), `qwen3_tts/server/app_generation.py` (streaming call site), `qwen3_tts/server/websocket.py` (streaming call site). `tests/test_seed_lock_chunks` exists and was registered in Batch 4 in a prior session, but only pins the *current* (still-unimplemented) inner seeding — it will need updating alongside the fix, not just unmocking.
- **Tasks:** write the failing test asserting per-chunk seeds differ deterministically (`base_seed`, `base_seed+1`, ...) when `seed_lock_chunks=False` and a seed was supplied, for both `run_inference` and `run_inference_streaming`; RED; implement in `run_inference` (fix the existing logic) and add the equivalent parameter + logic to `run_inference_streaming` plus its two call sites; GREEN; update `tests/test_seed_lock_chunks` to pin the new inner seeding for both runners.
- **Verify:** `conda run -n qwen3-tts-mlx python -m pytest tests/test_seed_lock_chunks.py tests/test_engine_streaming.py -v`; `ruff`; `mypy`.
- **Exit criteria:** per-chunk derived seeding verified for both the batch and streaming runners; regression tests pinned for both.

### Step 3B — 3f: confirm-and-close `models.<type>.revision` MLX wiring

- **Model tier:** default · **Branch:** none (read-only confirmation) or `docs/phase3-revision-confirm` if a doc note is needed · **Parallel with:** everything in Wave 3
- **Context:** confirmed 2026-09-06 — `revision=revision` is already threaded through multiple call sites in `qwen3_tts/core/engine/model_loader.py` (lines ~298, 352, 400, 415), sourced from `get_model_revision(model_type)`. This looks **already resolved**, contradicting the master plan's listing of it as open.
- **Tasks:** trace the exact MLX load call path end-to-end (not just grep hits) to confirm `revision` actually reaches the `mlx_audio`/`huggingface_hub` load call for all three model types; if confirmed, mark 3f done in the master plan with the file:line evidence — no code change. If a gap is found (e.g. one model type's MLX path bypasses this), scope a small follow-up fix instead.
- **Verify:** manual trace + one live load with a non-`"main"` revision configured for a test model type, confirming the loader actually requests that revision (check server log or a mocked `huggingface_hub` call).
- **Exit criteria:** either closed with evidence (fastest outcome), or a scoped follow-up filed with the exact gap.

### Step 3C — 3c: timeouts & MLX prompt naming bundle

- **Model tier:** default · **Branch:** `fix/phase3-timeouts-prompt-naming` · **Parallel with:** 3D, 3E (different files)
- **Context/tasks (inlined from the master plan during adversarial review, so this step is
  evaluable without that external file — not individually re-verified against current source this
  pass; re-check each sub-item first):**
  1. New `ASR_LOAD_TIMEOUT_SEC=900` in `http_client.py`, drift-guarded against the server-side
     timeout the same way `LOAD_MODEL_TIMEOUT_SEC`/`UNLOAD_MODEL_TIMEOUT_SEC` already are, for the
     UI's `/load-asr` call (currently likely on a shorter default timeout — confirm before fixing).
  2. Voice preview in the UI uses `_generation_timeout` (the char-length-scaled timeout helper)
     instead of a flat/short timeout.
  3. Drop the unconditional `.pt` suffix assumption in MLX voice-prompt naming (MLX prompts are
     `.wav`+`.txt` pairs, not `.pt` — confirm the exact site with a grep for `.pt` string literals
     near prompt-naming code before fixing).
  4. REPL's `/prompt` command becomes backend-aware (currently assumes one backend's naming/format).
- **Verify:** relevant UI/CLI test modules + `ruff` + `mypy`.
- **Exit criteria:** all 4 sub-items above verified fixed with a regression test each; `~/.claude/plans/review-entire-repo-for-ancient-possum.md`'s Phase-3 section updated to mark 3c done, citing this PR.

### Step 3D — 3e remainder: server hygiene bundle (minus M8, done in Step 1B)

- **Model tier:** default · **Branch:** `fix/phase3-server-hygiene` · **Parallel with:** 3C, 3E
- **Context/tasks (inlined during adversarial review; excluding M8, closed in Step 1B; not
  individually re-verified against current source this pass — re-check each first):**
  1. `/stats` and `/models` handlers wrapped in `asyncio.to_thread` (every sibling endpoint already
     does this — confirm which of these two don't yet).
  2. `pending_requests` removed via a `finally` block (confirm the current site leaks it on an
     exception path).
  3. Base64 encode/decode and tempfile creation moved off the event loop (`asyncio.to_thread`).
  4. `backend="vllm"` with vLLM disabled produces a clear config-validation error instead of a
     confusing failure downstream.
  5. Streaming `generation_state` updates happen under `generation_lock` (confirm a race exists
     today by checking whether streaming's state writes are unguarded, unlike the batch path's).
  6. `max_chunk_chars` gets Pydantic `Field` bounds (currently likely unbounded int).
  7. `FileNotFoundError` detail messages get path-sanitized before reaching the client (CWE-209
     pattern, matching the `/health` redaction already done elsewhere).
- **Verify:** server test suite + `ruff` + `mypy` + bandit (touches error-detail sanitization).
- **Exit criteria:** all 7 sub-items above verified fixed with regression tests; master plan updated to mark 3e done (minus M8, already closed in Step 1B), citing this PR.

### Step 3E — 3g: CLI/tools bundle

- **Model tier:** default · **Branch:** `fix/phase3-cli-tools-bundle` · **Parallel with:** 3C, 3D
- **Context/tasks (inlined during adversarial review; not individually re-verified against
  current source this pass — re-check each first):**
  1. `tts voice create` returns a non-zero exit code on failure (confirm it currently returns 0
     regardless).
  2. `tts uninstall config` writes atomically via the existing `save_config` helper instead of a
     direct write.
  3. `tts config`'s subprocess call handles `TimeoutExpired` and a missing config-wizard binary
     gracefully instead of an unhandled exception.
  4. Healthcheck's `size_str` formatting handles values ≥1 TB correctly (confirm it currently
     mis-formats or truncates at GB).
  5. Cache/uninstall CLI help-text fixes (cosmetic — confirm the exact wording issue first).
  6. `create_voice`'s file-open path handles a timeout instead of hanging indefinitely.
  7. `ui/__init__._SUBMODULES` and `components` — confirm the exact issue (likely a missing/stale
     entry causing an import or lazy-load gap).
  8. REPL prompt path text — confirm the exact display bug before fixing.
- **Verify:** CLI test modules + `ruff` + `mypy`.
- **Exit criteria:** all 8 sub-items above verified fixed with regression tests where applicable; master plan updated to mark 3g done, citing this PR; **after 3B–3E all land, the whole Phase 3 section of the master plan is complete** — no separate `phase3-ux-behavior-bugs.plan.md` draft is needed (superseded by this file).

---

## Wave 4 — E2E gate integrity, then the reliability investigation it was blocking

**Corrected 2026-09-06 by a static E2E health analysis — this wave's original premise was stale, not
just unverified.** `test_08_load_model`/`test_10_load_unload_cycle` do **not** fail via a 15s
`wait_for_function` timeout as the plan (and CLAUDE.md) claimed — that failure mode is *suppressed*:
the waits are wrapped in `except Exception: pass` (`test_e2e_playwright.py:921-922, 995-1000,
1014-1019`), the table check is demoted to a `print("[warn]...")` best-effort
(`wait_for_table_row_refreshed`, `:523`), and `_wait_for_model_state()`'s return value — the
"authoritative check" per its own comment — is discarded at every call site (`:925, 955, 964, 1001,
1020`). These tests are **known-hollow, not known-red**: they pass even if the design model never
loads. Step 4A (new) fixes the test gate itself before Step 4B (the original investigation,
re-scoped) tries to diagnose anything through it.

### Step 4A — Un-hollow `test_08_load_model`/`test_09`/`test_10` (prerequisite for 4B)

- **Model tier:** default · **Branch:** `fix/e2e-model-load-assertions`
- **Tasks:** assert the discarded `_wait_for_model_state()` return at all five call sites
  (`test_e2e_playwright.py:925, 955, 964, 1001, 1020`); remove or narrow the blanket `except
  Exception: pass` wrapping the waits so a genuine failure surfaces instead of passing silently;
  restore `wait_for_table_row_refreshed`'s check to a hard assertion rather than a warn-and-continue.
  Run these three tests against a live server **before** touching anything else — expect them to
  newly fail if the underlying load/unload behavior is actually broken, which is itself information
  Step 4B needs.
- **Verify:** `pytest tests/ -m e2e -k "load_model or load_unload_cycle"` on a freshly restarted, idle server (check `/generation-status` first — a queued generation masquerades as a timeout, a known trap from a prior session).
- **Exit criteria:** all three tests make real, non-swallowed assertions; their current pass/fail status against a live server is now trustworthy (whichever way it comes out).

### Step 4B — Diagnose repeated-load/unload server death (re-scoped after 4A)

- **Model tier:** strongest (open-ended root-cause investigation) · **Branch:** `fix/e2e-load-cycle-investigation` (or a docs-only finding if no code fix is warranted) · **Depends on:** Step 4A landing first
- **Context:** the server has been observed dying mid-load after several load/unload cycles
  (`.voice_server.log` ends mid-load), suspected memory pressure on the M2 Pro — never root-caused,
  and confirmed by the E2E analysis to have **no test that cycles ≥5 times** (`test_10` does exactly
  one cycle); memory-trend tests exist (`test_e2e_performance_{batch,stress}.py`) but live in modules
  the default batch runner doesn't execute and never pair sampling with load/unload cycling.
- **Tasks:** with 4A's now-trustworthy tests as ground truth, cycle load/unload ≥5 times on a
  quiesced server, capturing `.voice_server.log` + `/stats` memory figures each cycle; confirm or
  rule out memory pressure. If it's a genuine leak, file it as its own tracked issue with evidence
  rather than folding a blind fix in here.
- **Verify:** memory trend across ≥5 load/unload cycles on a freshly restarted, idle server.
- **Exit criteria:** root-caused with a fix + regression test, or root-caused and filed as a new tracked issue with evidence — "uninvestigated" is not an acceptable end state.

### Step 4C — Extract the duplicated Playwright harness before adding new E2E coverage

- **Model tier:** default · **Branch:** `refactor/e2e-shared-page-object`
- **Context:** `GradioPage` (the project's real Page Object Model, `test_e2e_playwright.py:192`) is
  used only in that one file — `test_e2e_history_clear_copy.py`, `test_e2e_tab_navigation.py`, and
  `test_e2e_wavesurfer_live.py` each re-implement locators *and* their own `subprocess.Popen(...)` UI
  launch on their own port. Four independently-drifting copies of the riskiest code in the suite.
  Every new E2E test in Step 4D is cheaper to write once this is fixed.
- **Tasks:** extract `GradioPage` plus a shared UI-launch/port helper into `tests/e2e_ui.py`; migrate all four modules onto it; keep behavior identical (this is a pure extraction, not a rewrite).
- **Verify:** all four E2E modules still pass unchanged against a live server.
- **Exit criteria:** one shared harness, zero duplicated `Popen`/locator blocks.

### Step 4D — Close the highest-severity new E2E gaps (in priority order)

- **Model tier:** default · **Branch:** `fix/e2e-coverage-gaps` (may split into several PRs — this is "batch small same-shape work" territory per gap, not one PR)
- **Context (gaps found by the 2026-09-06 static analysis, none previously tracked):**
  1. **Voice Management tab** (create/delete/rename/preview) — zero E2E coverage at any level; `/rename-prompt`, `/preview-prompt`, `/prompt-details` have zero hits. Highest severity — rename has rollback-on-failure logic, create/delete touch the filesystem, and Step 0G above just changed the create path. Start here (Step 0G already adds one create-from-audio case — this task covers delete/rename/preview).
  2. `/ws` WebSocket streaming — confirmed zero E2E coverage (use bounded receives with explicit timeouts; starlette's `TestClient.receive()` is unbounded and has hung CI before).
  3. Stream wire format end-to-end — the sentinel error frame + `X-Seed` header are unit-tested only; no E2E parses a real `/generate-stream` frame.
  4. `/cancel-generation` contract — `test_06` clicks Stop but tolerates every outcome; the `cancelled: true` + short-`results` contract CLAUDE.md warns about has no E2E guard.
  5. Lower priority, same pattern: `/update-model-config`/`/update-startup-config` (zero E2E — notable since Wave 0/Step 1B changes this exact handler), prosody presets, `x_vector_only_mode`, history Download button, a CLI E2E tier (currently none exists — batch/srt/dialogue/repl/watch are UI/HTTP-tested only, never via the CLI itself).
- **Also fix while touching this area (structural, not new coverage):** `make test-e2e`/batch 6 runs only `test_e2e_playwright.py` (13 of 90 E2E tests) — either expand it to cover all 12 E2E modules or rename the target so its 1-of-12 scope stops reading as "the E2E suite passed"; two of three core generation modes (design/custom) silently skip under bare `pytest -m e2e` depending on load order — make that skip loud instead of quiet; add Playwright tracing-on-failure + JUnit XML output (no failure artifacts exist today, which directly blocks diagnosing Step 4B's crash if it recurs); stop rewriting the tracked `.claude/.mcp.json` at `setUpModule` time (`e2e_helpers.py`); make `TTS_DISABLE_RATE_LIMITING=1` a loud module-level precondition for `test_e2e_security_validation.py` instead of 11 per-test 429 skips.
- **Verify:** each new/fixed test passes against a live, freshly restarted server.
- **Exit criteria:** gaps 1-4 closed with real (non-tolerant) assertions; the structural fixes landed; `make test-e2e`'s actual scope matches its name or its name says what it actually covers.

---

## Wave 4B — test-coverage gaps (from the 2026-09-06 coverage analysis)

*(Numbered 4B to slot between Wave 4 and Wave 5 by priority — closing coverage gaps outranks the
blocked/low-severity waves, and has no dependency on Wave 4; the waves are adjacent by priority,
not data flow. Its steps are **4B.1–4B.4** — do not confuse them with Wave 4's **Step 4B**, the
load/unload investigation.)*

**Where the numbers come from:** a `--cov` run in the `qwen3-tts-mlx` env (`pytest tests/ -m
"not e2e" --ignore=tests/evaluations --cov=qwen3_tts --cov-report=term-missing`): **88%
aggregate** (10,873 statements, 1,332 missed; 3,132 passed / 4 skipped / 92 deselected, 55 s).
The documented 80% target **is met and already CI-gated** (`--cov-fail-under=80`,
`.github/workflows/test.yml:173`) — no headline emergency. The finding is that **the gate is
aggregate-only**: 13 modules sit below 80% individually while the total stays green, and every
P0 gap below is invisible to it. This analysis also confirms `/ws` has zero e2e coverage
(matches Wave 4's finding), and that `/generate-stream` is touched only incidentally by
`test_e2e_queueing.py:534` as a lock-contention vehicle — nothing asserts the stream protocol.

**Env caveat (drives Step 4B.3):** the run above was in `qwen3-tts-mlx`, where
torch/mlx/gradio import but `qwen_tts` does not. CI has neither torch nor mlx — so CI's 80%+ and
this 88% cover partly disjoint line sets. A companion run in the `qwen3-tts` (torch) env
disambiguates the `model_loader.py` gaps (item 12 below).

### Step 4B.1 — P0: server/concurrency failure arms

- **Model tier:** default · **Branch:** `test/p0-server-failure-arm-coverage` (may split
  per-module) · **Source:** coverage analysis items 1–5, verified against current source by the
  originating agent 2026-09-06; re-check line numbers at implementation time as usual.
- **Context:** every line below is a failure arm — the code that runs when something goes wrong,
  which is exactly what the next concurrency change needs exercised before it can trust the
  suite:
  1. `server/websocket.py` (91%, 24 missed) — `_stream_generation` 487-499 (`_HTTPException` →
     `send_json` → return; the nested-`detail` unwrap shape that bit #214 Phase 2b),
     `inference_thread` 457-462 catch-all, `_cancel_watcher` 224-228 disconnect handling,
     191-194 over-length text rejection. Integration tests (starlette TestClient websocket) +
     one e2e. **Bounded receives only** — starlette's unbounded `receive()` has hung CI before.
  2. `server/app_lifespan.py::_background_load` 648-665 (module 89%, 40 missed) — the two #214
     arms where startup waits on an HTTP-owned load record (timeout expiry; FAILED-record
     branch), both writing `model_load_errors`, which `/health` surfaces. Unit test with a
     fabricated `state.model_loads` record.
  3. `server/app_lifespan.py::_run_warmup_under_inference_lock` 568-580 — the
     `concurrent.futures.TimeoutError` → `future.cancel()` arm from #192/#211, untested. Unit
     test with a patched future.
  4. `core/engine/asr.py` (62%, 62 missed — the lowest-coverage risk-bearing module) —
     `unload_asr_model` 260-281 real body (both MLX/torch branches + the Phase-2b gc-ordering
     fix) never runs — only the no-op path is covered; `_ensure_asr_torch_loaded` 26-51
     entirely uncovered; `_ensure_mlx_whisper_processor` 103-139 HF-download-shim error
     branches uncovered. Unit tests with module-global stubbing — register `addCleanup`
     restorations **before** mutating module globals (standing project discipline).
  5. `core/engine/voice_prompt.py` (75%, 52 missed) — `save_voice_prompt_mlx` 517-535
     orphan-`.wav` rollback (a `.txt`-write failure must `os.remove` the `.wav` and re-raise so
     a 400 `invalid_audio` can't swallow a server fault — #236 code whose dedicated test file
     never reaches this path); `_load_pt_safe` 119-151 corrupted-`.pt` fallback **including the
     path-traversal ValueError guard at 138-141** (zero coverage — a genuine gap, not an env
     artifact: torch imports fine in the coverage env); `migrate_orphan_mlx_prompts` 302-329.
- **Tasks:** per item, write the failure-arm tests described; any new test module gets
  registered in `tests/run_batches.py`'s `BATCHES` (enforced by `tests/test_batches_coverage.py`).
  A test that cannot pass against current code is a bug report, not a test problem — file it in
  the PR as a finding.
- **Verify:** targeted `pytest` with `--cov=qwen3_tts.server --cov=qwen3_tts.core.engine
  --cov-report=term-missing` on the touched modules; full non-E2E suite; `ruff`; `mypy`.
- **Exit criteria:** each listed line range is exercised by a passing test (or has a filed bug
  with the failing test as its evidence); all five modules move above 80% — the floor the
  aggregate gate already claims to enforce.

### Step 4B.2 — P1: engine correctness invariants + client contracts

- **Model tier:** default · **Branch:** `test/p1-invariant-coverage` · **Source:** coverage
  analysis items 6–10.
- **Context:**
  6. `inference.py::run_inference_streaming` torch fallback 1673-1713 (module 85%) — the entire
     chunked torch streaming loop never executes in tests, including the
     echo-trim-first-chunk-only rule and the `_postprocess_chunk` streaming/batch-parity
     invariant CLAUDE.md states as a design guarantee.
  7. `inference.py::_transcribe_probe` 1084-1108 — ICL echo-trim WAV staging + the tempfile
     cleanup `finally`.
  8. `server/client/generator.py` (83%, 31 missed) — `generate_streaming` alias-resolution
     `ValueError` branch, malformed `X-Seed`/JSON-decode failures, preset merge,
     `generate_dialogue`.
  9. `app_generation.py` 485-521 (module 91%) — the vLLM circuit-breaker decision block (CLOSED
     / open / fallback-disabled-raise paths).
  10. `interface/ui/voice_management.py::create_voice_prompt` (79%, 47 missed, all in one
      function) — the torch branch requiring a running server + base64 upload to
      `/create-voice-prompt`. **Overlaps Step 0G** (which fixes the MLX branch and adds one E2E
      case for this tab): this item is the *unit-level* torch-branch coverage — coordinate the
      two PRs, don't duplicate.
- **Tasks/Verify:** same discipline as 4B.1, scoped to these items.
- **Exit criteria:** items 6–10 exercised; the streaming/batch `_postprocess_chunk` parity
  invariant (item 6) pinned by a test that fails if the two paths drift.

### Step 4B.3 — P2: CLI/support + remaining sub-80% modules + the torch-env companion run

- **Model tier:** default · **Branch:** `test/p2-remaining-coverage`
- **Context:**
  11. `cli_server.py` (66%, 95 missed — lowest % in the repo) — prioritize `_kill_server_process`
      225-241 (SIGTERM→SIGKILL escalation, interacts with PM2 autorestart) over the lower-risk
      `status`/`log` display paths.
  12. `core/engine/model_loader.py` (62%, 91 missed) — `_do_load`, `_load_model_torch`,
      `_patch_deepcopy_for_bnb`, `_resolve_load_kwargs`. `qwen_tts` is absent in both the
      coverage env and CI — scope as stub-based unit tests, or accept/document as env-gated
      after the companion run below disambiguates.
  13. Remaining sub-80% modules (descending risk): `audio_processing.py` 77%, `vllm_client.py`
      79%, `config/models.py` 76%, `tools/check_config_docs.py` 76%,
      `interface/ui/tabs_generation.py` 73%, `server/client/config_fetcher.py` 73%,
      `core/config/auth.py` 72% (small, but it's the auth-header path), `__main__.py` 0%
      (trivial, 2 statements).
  Plus: **companion coverage run in the `qwen3-tts` (torch) env** — same command, different env
  — to separate real gaps from env artifacts (see the env caveat above). Its output decides
  item 12's disposition.
- **Exit criteria:** items 11/13 covered or explicitly dispositioned in the PR; item 12 either
  stub-tested or documented env-gated with the torch-env run as evidence.

### Step 4B.4 — Per-module coverage floor (ratcheted allowlist) — close the aggregate-only blind spot

- **Model tier:** default · **Branch:** `test/per-module-coverage-floor` · **Depends on:**
  4B.1–4B.3 landing first (ratchet at then-current levels, which by then sit closer to the
  floor).
- **Context:** the CI gate is aggregate-only — every P0 gap in 4B.1 was invisible to it. An
  aggregate gate lets coverage silently migrate: new well-covered code raises the total while a
  critical module rots underneath it.
- **Tasks:** introduce a per-module floor via a ratcheted allowlist (a checked-in
  `coverage-floors.json` + a small CI script asserting each module's measured % against its
  floor is the simplest shape); seed floors at measured then-current values; a PR may only raise
  or meet its floor, never lower it; modules exit the allowlist as they cross 80.
- **Verify:** CI green with the gate on; deliberately regress one module's coverage locally and
  confirm the gate goes red even while the aggregate stays above 80%.
- **Exit criteria:** a module dropping below its floor fails CI independently of the aggregate —
  the exact blind spot that hid 4B.1's items is closed.

---

## Wave 5 — blocked backlog (infrastructure-gated, low priority)

### Step 5 — HIGH-1/MED-2/HIGH-2: vLLM Docker validation + event-loop decoupling

- **Model tier:** default · **Branch:** `fix/vllm-docker-validation` (when unblocked)
- **Context:** confirmed 2026-09-06 still open — no Docker+GPU validation of vLLM params, and the event loop is not yet proven non-blocked during vLLM generation. vLLM is this project's optional third backend (mlx/torch are default); this has been carried as blocked on unavailable hardware across multiple prior sessions.
- **Tasks:** blocked until a Docker environment with GPU access for vLLM is available. When unblocked: validate vLLM params against a real Docker deployment (HIGH-1/MED-2); prove the event loop isn't blocked during vLLM generation across the full request path (`tests/test_vllm_async_nonblocking.py` already does this at the client-call level with a real detector — confirmed 2026-09-06, it drives the real `AsyncVLLMClient.generate()` with a heartbeat/elapsed-time bound, not a mock; extend the same pattern up through the FastAPI route handler if that layer isn't already covered) (HIGH-2).
- **Verify:** N/A until unblocked — no code changes are made under this step while blocked.
- **Exit criteria (this is a non-goal, not a completion condition — it never turns "done" on its own):** re-evaluate the moment a Docker+GPU environment becomes available for this project, or quarterly alongside the upstream-watch cadence (#112) in case vLLM's role in the project changes, whichever comes first. Do not attempt implementation without the environment.

---

## Wave 6 — code-quality cleanup (low severity, mechanical, sequence last)

*(2026-09-06 addendum: Steps 6F–6Q below are new from the four cross-cutting reviews — small,
low-risk fixes that may run before or alongside the 6A–6E splits; the splits keep their own
ordering among themselves, and Step 6E's targets were expanded in place by the python-review
analysis rather than duplicated as a new step.)*

### Step 6·0 — Dead-code cleanup (EXECUTED 2026-09-06, via /ecc:refactor-clean)

- **Status: DONE.** Full report below; this step needed no branch/PR — both deletions were made
  directly and verified, consistent with how low-risk this category of change is.
- **Method:** `vulture qwen3_tts --min-confidence 60` → 97 raw findings. Per standing project
  memory ([[project_dead_code_false_positives]]: vulture has historically produced only false
  positives here — framework dispatch, decorators, `Protocol` contracts, and re-exports it can't
  see), every finding was independently verified against real source (decorator presence, grep for
  callers, dynamic-dispatch patterns, test references) rather than trusted at face value.
- **Result: 92 of 97 were confirmed false positives** — 23 Pydantic `BaseModel` schema fields
  misread as "unused variables" (`server/validation.py`), 28 FastAPI/Click decorator-dispatched
  functions, 7 dunder/protocol-required signatures, 7 documented re-exports/test-DI hooks, 2
  framework-object attribute writes (`torch.backends.cudnn.benchmark`, `app.state.limiter`), 15
  confirmed-live via real internal/test callers, 3+5 already-documented-live per this exact memory
  (`engine_vllm.py`, `interface/ui/components.py`), 2 incidental (partial tuple-unpacking, nothing
  cleanly deletable).
- **2 SAFE-tier genuine dead code found and deleted:**
  1. `qwen3_tts/interface/generate_interactive.py:208` — `self._rich_task_id = None`, a
     write-once/read-never instance attribute (the method that would populate it,
     `_run_rich`, uses a local `task_id` instead and never assigns back to `self`).
  2. `qwen3_tts/interface/ui/generation.py:178-189` — `_cancel_if_confirmed()`, a wrapper function
     with zero callers repo-wide (superseded by the `ConfirmButton`/`confirm_step` pattern now
     standard elsewhere in this UI package).
- **2 CAUTION-tier items found and deliberately left alone** (public client-library API, zero
  internal callers, but no way to rule out an external consumer — "skip if uncertain" applies):
  `server/client/_base.py:239 TTSClient.reload_config()`, `server/client/config_fetcher.py:39
  ConfigFetcherMixin.get_health()`. Revisit only if the client library's external-consumer surface
  is ever formally scoped down.
- **Verification:** baseline `pytest tests/ -m "not e2e" --ignore=tests/evaluations` (3132 passed,
  4 skipped) before either deletion; targeted `-k "interactive or progress"` re-run after deletion
  1 (118 passed); full re-run after both deletions (3132 passed, 4 skipped — identical to
  baseline) plus `ruff check` (clean, no new unused-import fallout) and `mypy` (clean, 57 files).
  `tests/evaluations/` is excluded from all these runs — its collection error is the known
  pre-existing torchcodec native-lib issue the held `fix/torchcodec-collection-guard` branch
  addresses, unrelated to this work.
- **Net effect on Wave 6's remaining scope:** negligible line-count impact (2 lines total) — does
  not change the split targets in Steps 6B–6E below.

### Step 6A — P2-2: split four oversized functions

- **Model tier:** default · **Branch:** `refactor/p2-2-function-split`
- **Context:** repo audit (2026-07-31) found 4 functions over the 50-line SRP guideline via `python -m qwen3_tts.tools.solid_analyzer qwen3_tts`: `edit` (109 lines), `rebuild` (106), `stop` (95), `_generation_options` (67). **Re-run the analyzer first** — these figures are 5+ weeks stale.
- **Tasks:** re-run `solid_analyzer`; for each still-over-limit function, extract cohesive blocks (audit's own note: `edit`/`rebuild` are CLI command bodies and split cleanly into validate/apply/report phases). Characterize existing behavior with tests before extracting if coverage is thin.
- **Verify:** `python -m qwen3_tts.tools.solid_analyzer qwen3_tts` (confirm 0 or fewer violations); full non-E2E suite; `ruff`; `mypy`.
- **Exit criteria:** each function at or under 50 lines (or a documented reason it can't cleanly split further); no behavior change (existing tests pass unmodified).

### Step 6B — P2-1: split `app.py` (1064 lines)

- **Model tier:** strongest (structural risk — read [[mock_patch_seams_block_file_splits]] and [[codeql_dismissals_dont_follow_moved_code]] equivalents first: `grep` every `@patch("qwen3_tts.server.app.*")` across `tests/` before moving anything, and expect CodeQL to re-raise any previously-dismissed alert on relocated code) · **Branch:** `refactor/p2-1-split-app-py`
- **Context:** grown from 821 (2026-07-31 audit) to 1064 lines. Route-thin-wrapper pattern already established (handlers live in `app_generation.py`/`app_models.py`/`app_prompts.py`); `app.py` itself likely still carries setup/middleware/route-registration bulk that can split along existing seams.
- **Tasks:** identify a natural split (e.g. middleware/CORS/rate-limit setup vs. route registration) that doesn't cross the response-contract test boundary (`tests/test_response_contracts.py`); move in one PR; re-run mock-patch-target greps after.
- **Verify:** full non-E2E suite; `ruff`; `mypy`; CodeQL (via a full PR — not just the local harness, per this repo's own documented CodeQL blind spots).
- **Exit criteria:** `app.py` under 800 lines; zero test collateral damage; no re-raised CodeQL alerts.

### Step 6C — P2-1: split `interface/generate.py` (902 lines)

- **Model tier:** strongest (same structural-risk caveats as 6B) · **Branch:** `refactor/p2-1-split-generate-py`
- **Context:** grown from 865 to 902 lines. Companion modules already exist (`generate_helpers.py`, `generate_interactive.py`, `generate_server.py`) — likely candidate for moving remaining CLI-argument-parsing or dispatch logic into one of them or a new sibling.
- **Verify/Exit criteria:** same pattern as 6B, scoped to this file.

### Step 6D — P2-1: split `interface/ui/shared.py` (887 lines)

- **Model tier:** strongest (same structural-risk caveats) · **Branch:** `refactor/p2-1-split-shared-py`
- **Context:** confirmed 2026-09-06 at 887 lines — over the 800-line guideline, not previously called out by exact figure in the standing structural-debt memory (memory only said "still breach" without a number). This is a UI module widely imported by other UI modules per CLAUDE.md's own note about module-style access patterns for mocking — expect the mock-patch-seam risk to be highest here of the four files.
- **Verify/Exit criteria:** same pattern as 6B; extra care re-running every `@patch("...shared...")` grep across `tests/` and `interface/ui/` before and after.

### Step 6E — P2-1: split `core/engine/inference.py` (1769 lines) — last, highest risk

- **Model tier:** strongest · **Branch:** `refactor/p2-1-split-inference-py`
- **Context:** the largest and fastest-growing of the four (1110 → 1769 lines since the 2026-07-31 audit — it absorbed the ICL echo-trim, seed-lock, and postprocess-chunk work landed in the interim). This is the project's core dispatch/postprocessing engine — split last, after 6B–6D establish a working rhythm for this repo's split-and-verify process, and after Steps 1C/3A (which also touch this file) have both landed and been rebased past.
  **Concrete extraction targets (added by the 2026-09-06 python-review analysis — replaces "identify a natural split" guesswork):** the DSP chunk-combination block (`_snap_to_zero_crossing` … `_crossfade_chunks`, ~175 lines, numpy-only) → new `core/engine/chunk_combine.py`; the PRF-8 ICL echo-trim block (~190 lines) → new `core/engine/icl_echo.py`. Those two extractions alone take the file to ~1400 lines, leaving dispatch + backend adapters — start there before considering anything bigger. Same mock-patch-seam caveat as 6B applies (`grep` every `@patch("...inference...")` across `tests/` before moving anything).
- **Verify/Exit criteria:** same pattern as 6B, with extra scrutiny — this file backs both backends' generation paths; run the full E2E suite (not just non-E2E) before merging, live-smoke both `mlx` and `torch` backends post-split if feasible.

### Step 6F — Bound both streaming queues (backpressure)

- **Model tier:** default · **Branch:** `fix/bounded-stream-queues` · **Source:** python-review M1
- **Context:** both streaming producers build **unbounded** `asyncio.Queue`s —
  `app_generation.py:798` and `websocket.py:413` (both verified 2026-09-06: `asyncio.Queue()`).
  With no backpressure, a slow consumer (bad connection, paused tab) makes the producer thread
  accumulate an entire generation in RAM before the consumer drains any of it.
- **Tasks:** give both queues `maxsize≈32` (named constant); move the producer's put onto a
  thread-safe bounded put via `asyncio.run_coroutine_threadsafe` (the producer runs in a worker
  thread, so neither a bare `await put` nor a naive `put_nowait` loop is correct);
  **preserve the `None` sentinel + `done` Event contract** both consumers rely on; test with an
  artificially slow consumer asserting queue depth never exceeds the bound.
- **Verify:** `pytest tests/test_engine_streaming.py tests/test_websocket*.py -v`; full non-E2E
  suite; `ruff`; `mypy`.
- **Exit criteria:** both queues bounded with the slow-consumer test proving it; the
  sentinel/done contract unchanged (all existing streaming tests pass unmodified).

### Step 6G — `_error_response` → `NoReturn`, delete the fall-through guards

- **Model tier:** default · **Branch:** `refactor/error-response-noreturn` · **Source:**
  python-review M2
- **Context:** `_error_response` (`validation.py:487-504`) is typed `-> None` although it always
  raises — so mypy can't prove unreachability after a call, forcing ~10 hand-written
  fall-through guard blocks across 4 files to exist purely to satisfy the checker.
- **Tasks:** annotate `typing.NoReturn`; delete the 10 guards; if any deletion produces a mypy
  error, that call site had a real fall-through path — investigate it before forcing anything.
- **Verify:** `mypy qwen3_tts/{core,server,interface}`; full server test suite; `ruff`.
- **Exit criteria:** guards deleted, mypy green, no new `# type: ignore` added anywhere.

### Step 6H — Fix the `reset_activity_timer` lost-cancel race

- **Model tier:** default (latent today) · **Branch:** `fix/activity-timer-race` · **Source:**
  python-review M3
- **Context:** `reset_activity_timer` (`app_lifespan.py:278-295`, verified) does an unlocked
  read-modify-write on `app_state.shutdown_timer`: two concurrent requests can both cancel, then
  both start a timer; the loser's handle is overwritten and never cancelled → premature
  auto-shutdown. It also spawns a `threading.Timer` per request. Latent only because
  `auto_shutdown_minutes` defaults 0 — fix **before** anyone enables the feature, not after.
- **Tasks:** replace with a single long-lived timer thread comparing a `last_activity`
  timestamp periodically (one thread, no per-request spawn, no lost-cancel window); failing test
  that interleaves two reset calls and asserts exactly one timer is live.
- **Verify:** unit tests around `reset_activity_timer`; full server suite; `ruff`; `mypy`.
- **Exit criteria:** the interleaving test proves no orphan timer survives; per-request thread
  spawn gone.

### Step 6I — `register_backend`: replace 10 positional args with a request dataclass

- **Model tier:** strongest (public extension-point signature change) · **Branch:**
  `refactor/inference-request-dataclass` · **Source:** python-review M4
- **Context:** `register_backend` (`inference.py:47`, verified — a documented public OCP
  extension point) dispatches via 10 positional args; `voice_description`/`speaker` are adjacent
  same-typed params a backend author can silently transpose.
- **Tasks:** introduce a frozen `InferenceRequest` dataclass; narrow the `Protocol` to accept
  it; migrate the internal strategies; keep a positional-compat shim only if an external
  backend registration actually exists (none does in-repo — confirm by grep before deciding);
  update the OCP documentation describing the signature.
- **Verify:** full engine test suite; `ruff`; `mypy`.
- **Exit criteria:** dispatch is dataclass/keyword-based; the transposition hazard is
  structurally gone; docs updated.

### Step 6J — Finish threading `config_provider` (DI is half-wired)

- **Model tier:** default · **Branch:** `fix/config-provider-full-threading` · **Source:**
  python-review M5
- **Context:** `config_provider` DI is half-wired: `run_inference`/`run_inference_streaming`
  thread it into `_postprocess_chunk`, but ~8 other engine call sites (`_build_torch_params`,
  `_get_max_chunk_chars`, etc.) call module-level `load_config()` directly — a silent
  split-brain config for any non-default-provider caller (tests, embedders).
- **Tasks:** grep every direct `load_config()` call under `core/engine/`; thread
  `config_provider` through each (parameter defaulting to the module-level call); test with a
  non-default provider asserting the threaded value wins at every site.
- **Verify:** engine test modules; `ruff`; `mypy`.
- **Exit criteria:** no direct `load_config()` in the engine's request path — one config source
  per request.

### Step 6K — `handle_generate` needs a catch-all exception envelope

- **Model tier:** default · **Branch:** `fix/handle-generate-catchall` · **Source:** python-review M7
- **Context:** `handle_generate` (`app_generation.py:659-666`) catches a narrow exception tuple;
  `AttributeError`/`KeyError` (library API drift) escapes to a bare 500 with no
  `_error_response` envelope — contradicting the project's own documented lesson elsewhere that
  narrow tuples miss library API drift (CLAUDE.md, `_background_load`).
- **Tasks:** add a trailing `except Exception` mapping to a 500 `unknown_error` envelope
  (sanitized via `_sanitize_error`, matching sibling paths); test that an `AttributeError` from
  a mocked engine produces the envelope, not a bare 500.
- **Verify:** server test suite; `ruff`; `mypy`.
- **Exit criteria:** no exception path in `handle_generate` bypasses `_error_response`.

### Step 6L — Extract `GenerationCache` (server-handler oversized functions)

- **Model tier:** strongest (structural — mock-patch-seam caveats apply) · **Branch:**
  `refactor/generation-cache-extraction` · **Source:** python-review M8
- **Context:** a *new* oversized-function finding, distinct from Step 6A's four (different
  files): `handle_generate` 539 lines, `_stream_generation` 338, `handle_generate_stream` 296,
  `websocket_tts_handler` 213, `load_model_deduped` 213, `handle_create_voice_prompt` 194 — all
  over the <50-line guideline. Highest-value split first: `handle_generate`'s cache
  lookup/eviction/write (~120 lines, 3 near-duplicate read blocks) → a new
  `server/generation_cache.py` `GenerationCache` class — this removes duplication rather than
  just relocating it.
- **Tasks:** grep `@patch("qwen3_tts.server.app_generation.*")` across `tests/` first
  (standing mock-patch-seam discipline); extract `GenerationCache` and migrate
  `handle_generate` onto it; remaining oversized handlers follow in the same pattern, one PR
  each.
- **Verify:** full server test suite; `ruff`; `mypy`; CodeQL via the full PR.
- **Exit criteria:** `GenerationCache` extracted with the three near-duplicate reads collapsed
  to one; each touched handler meaningfully smaller; zero test collateral damage.

### Step 6M — `ServerState` Protocol + incremental handler typing

- **Model tier:** default · **Branch:** `refactor/server-state-typing` · **Source:** python-review M10
- **Context:** server handlers are almost entirely unannotated and `app.state` is structurally
  untyped — mypy runs without `disallow_untyped_defs`, so it checks essentially nothing in
  these bodies. This is *why* Step 0B's stale-model bug and Step 6J's split-brain config were
  invisible to the type gate.
- **Tasks:** add a `ServerState` `Protocol` in a new `server/state_types.py`; annotate
  module-by-module, enabling `disallow_untyped_defs` per-module via `[[tool.mypy.overrides]]`
  as each lands; start with `model_loading.py` + `validation.py` (the two modules behind the
  most Wave 0/1 steps).
- **Verify:** `mypy` with the new per-module strictness; full server suite.
- **Exit criteria:** the two starting modules fully annotated and strictly checked; the
  override mechanism in place for the rest to opt in incrementally.

### Step 6N — vLLM: untyped-but-lock-holding path needs typing or explicit gating

- **Model tier:** default · **Branch:** `fix/vllm-path-gating` · **Source:** python-review M11
- **Context:** `vllm_client.py`/`engine_vllm.py` (995 lines) are mypy-excluded, yet
  `app_generation.py:506` awaits `vllm_adapter.generate(...)` **while holding `inference_lock`**,
  and `app_lifespan._maybe_start_vllm_adapter` starts a `subprocess.Popen` directly in the
  lifespan startup path (up to 300 s). The type gate sees none of it.
- **Tasks:** either annotate the vLLM modules and drop the exclusion (preferred — its own PR),
  or gate the whole path behind an explicit experimental flag and document it as unchecked;
  either way, add the test proving the startup-path `Popen` no-ops cleanly when vLLM isn't
  configured (it should already — pin it).
- **Verify:** `mypy` post-annotation (or the flag test); startup no-op test; full suite.
- **Exit criteria:** the vLLM path is either type-checked like everything else or explicitly
  flagged and documented experimental; the startup no-op is proven by test.

### Step 6O — ReDoS guard on `_EMAIL_RE` (unauthenticated event-loop stall)

- **Model tier:** default · **Branch:** `fix/email-regex-redos` · **Source:** security MEDIUM-2
- **Context:** `_EMAIL_RE` (`text_processing.py:88`, verified) is quadratic on adversarial
  input — measured 3.57 s for one 50,000-char string (= `max_text_length`, a single allowed
  request). `_normalize_text` runs in `asyncio.to_thread`, but CPython's `re` doesn't release
  the GIL — so this stalls the **whole event loop**, including unauthenticated
  `/health`//`/ready`, for the duration.
- **Tasks:** rewrite so the `@`-split halves can't backtrack into each other (Python `re` has
  no possessive quantifiers — unroll the ambiguity), or gate behind a cheap `"@" in text`
  pre-check + a per-segment length cap; add a benchmark test asserting a 50,000-char
  adversarial input normalizes in <100 ms.
- **Verify:** the benchmark test; full text-processing suite; `ruff`; `mypy`.
- **Exit criteria:** adversarial input at `max_text_length` no longer stalls the loop by orders
  of magnitude; normalization output unchanged for legitimate emails (existing tests prove it).

### Step 6P — Bound the remaining `GenerateRequest` string fields

- **Model tier:** default · **Branch:** `fix/generate-request-string-bounds` · **Source:**
  security MEDIUM-5 · **Same file and fix-shape as Step 0A** (`validation.py` `Field` bounds) —
  land after 0A or bundle into its PR if timing aligns.
- **Context:** only `text`/`texts` are length-checked on `GenerateRequest`;
  `voice_description`, `instruct`, `speaker`, `language`, `prompt_file` are unbounded — a ~99 MB
  `instruct` is accepted and fed to the model **while holding `inference_lock`**.
- **Tasks:** `Field(max_length=...)` on each (suggested 4000/4000/64/16/255 — confirm against
  real usage before pinning); tests asserting oversize values are rejected at the boundary.
- **Verify:** `pytest tests/test_validation.py -v`; full suite; `ruff`; `mypy`.
- **Exit criteria:** every request string field bounded; oversize rejected with 422, not 500.

### Step 6Q — Batch of small same-shape fixes (python-review L2–L8 + security LOW-1–LOW-4)

- **Model tier:** default · **Branch:** `chore/small-fixes-batch` (may split into 2–3 PRs by
  theme; each item lands with its own test where testable) — composed as ONE batch step per the
  small-same-shape-work convention. **Re-verify each item at implementation time; all found
  2026-09-06.**
- **Items:**
  1. `cleanup_resources` (`app_lifespan.py:766`): the `del model` is a no-op (release comes
     from the next line's `models[name] = None`), and unlike every other unload path it never
     calls `unload_model_cleanup()` — add the call, drop the dead `del`. *(L2)*
  2. Missing `raise ... from` in exception translation: `app_models.py:536`,
     `app_prompts.py:441`, `app_prompts.py:206-208`, `validation.py:391-394`. Consider adding
     ruff's `B` selection (would also catch B008/B023, e.g. the per-iteration closure at
     `websocket.py:206`). *(L3)*
  3. f-string logging → lazy `%s` at `app_lifespan.py:497,501`, `app_generation.py:493,516,518`;
     consider ruff's `G` selection to enforce mechanically. *(L4)*
  4. `_gen_cache_key` (`validation.py:484`) truncates SHA-256 to 16 hex chars (64 bits) — safe
     at current scale; widen to 32 anyway. *(L5)*
  5. `_crossfade_chunks` (`inference.py:870-871`) returns the caller's array uncopied in the
     single-chunk case, asymmetric with the multi-chunk branch's `.copy()` — make both copy
     (trap for the next editor; line numbers will move after Step 6E's split). *(L6)*
  6. Redundant `except (FileNotFoundError, OSError)` (`app_lifespan.py:502,798`, `app.py:973`) —
     `FileNotFoundError` is an `OSError` subclass; collapse to `OSError`. *(L7)*
  7. `run_server` (`app.py:1008-1009`) adds logging handlers unconditionally — a second
     in-process call (tests, embedded use) duplicates every log line; guard with a
     `_logging_configured` flag or a `logger.handlers` check. *(L8)*
  8. *(security LOW-1)* Runtime extras (`huggingface_hub`, `pyrubberband`, `pydub`,
     `accelerate`, `bitsandbytes`, `vllm-omni`, `qwen-tts`) carry no version floors and sit
     outside `requirements.lock` (which covers test+ui+dev only). Add floors; regenerate the
     lock; extend lock/OSV coverage to them. `pydub` (last release 2021, shells out to ffmpeg)
     runs on user-supplied audio — floor it deliberately.
  9. *(security LOW-2)* Sidecar path derivation via global `.replace(".wav", ".json")` /
     `.replace(".json", ".wav")` (`interface/ui/shared.py:697,762,798` — all three verified
     2026-09-06) → `os.path.splitext`; any `.wav`/`.json` substring in a parent directory gets
     rewritten today. Correctness bug on a delete/write sink, contained by directory scoping
     (hence LOW, not HIGH).
  10. *(security LOW-3)* `api_key_env` (`interface/ui/shared.py:160-163,205`) is config-driven
      with no allowlist — any env-var name in `config.json` gets its value sent to Anthropic.
      Local-user-writable only, so defense-in-depth: allowlist to `ANTHROPIC_API_KEY` / a
      `*_API_KEY` pattern.
  11. *(security LOW-4, docs)* ARCHITECTURE.md:65 says the auth token is "auto-cleaned on
      shutdown"; `app_lifespan.py:495-501` deliberately preserves it (0600 perms). Correct the
      doc — the residual-credential posture should be stated accurately. (The other half of
      that finding — the Gradio UI missing from the security docs — is Step 0E.)
- **Verify:** per-item targeted tests where applicable; `ruff` (with any newly enabled rule
  selections); `mypy`; confirm `make check-config-docs` unaffected (item 11 is ARCHITECTURE.md,
  not CONFIG.md).
- **Exit criteria:** all 11 items landed (or individually dispositioned in the PR with a
  reason); no behavior change beyond the fixes themselves.

---

## Independent tracks (not gated by the waves above — different domain or decision-gated)

### Track T1 — Prosody preset builder v3 (feature, not backlog cleanup)

- **Status:** fully speced, two 4-reviewer rounds already SHIP-WITH-EDITS. Plan: `~/.claude/plans/include-this-as-well-elegant-reef.md`.
- **Gate:** **your explicit go-ahead** — this is new user-facing scope, not an autonomous bug fix. Confirmed 2026-09-06 still unimplemented (`core/config/presets.py` still 118 lines, no save/delete functions).
- **When to run:** anytime after you approve it; does not depend on or block any wave above (touches `core/config/presets.py`, `interface/voice_helpers.py`, `interface/ui/tabs_generation.py`, `interface/ui/_facade.py` — no overlap with the waves).

### Track T2 — mlx-env FastAPI repair (0.135.1 → 0.141.1)

- **Status:** deliberately held as its own isolated restart window since the 2026-09-05/06 dependabot session (two transport-library bumps in one week would be unattributable if something regressed). Confirmed 2026-09-06 still on 0.135.1.
- **When to run:** anytime, isolated from the waves — pick a quiet window, bump, smoke-test `/generate` + `/generate-stream` + `/ws` per the #223 uvicorn-bump protocol precedent, restart, verify.

### Track T3 — Disk-space reclamation Phases 1–2 (non-code, direct mode)

- **Status:** fully audited, nothing executed. Plan: `~/.claude/plans/goal-reduce-storage-usage-ancient-adleman.md`. Free-space figures are explicitly volatile (iCloud eviction) — re-measure with `df`/`du` immediately before acting, not from the plan's cached figures.
- **When to run:** anytime, entirely outside the git repo/branch/PR workflow — this is macOS housekeeping, not a code change.

### Standing watch — not an execution step

- **Issue #112** (Upstream Watch) is a passive monthly-refreshed dashboard, not actionable work. No step needed; re-check only if a ⚡ blocker clears in its next auto-comment.

---

## Summary

**44 pending execution steps across 8 waves** — Wave 0: 0A–0G (7) · Wave 1: 1A–1E (5) · Wave 2: 2
(1) · Wave 3: 3A–3E (5) · Wave 4: 4A–4D (4) · Wave 4B: 4B.1–4B.4 (4) · Wave 5: 5 (1) · Wave 6:
6A–6Q (17) — **plus 3 independent tracks** (feature, dependency, and non-code housekeeping, none
gated by the waves) **+ 1 passive watch** (no action). Step 6·0 (dead-code cleanup) was already
executed directly on 2026-09-06 and is recorded in Wave 6; it is not counted among the pending
steps. Ordered by: critical correctness/security findings from the 2026-09-06 cross-cutting
reviews first (Wave 0), then independent bug fixes and gate-integrity work (Wave 1), the
on-demand-load feature gated on the reliability investigation it would otherwise multiply
(Wave 2 — hard-gated on Wave 4's Step 4B; read "Wave 2 before Wave 4" as branch-creation order,
not merge order), the Phase-3 sweep closing out a whole prior plan (Wave 3), that open
reliability question itself plus the E2E coverage it needs (Wave 4), coverage-gap closing
(Wave 4B), infrastructure-blocked work (Wave 5), then mechanical low-severity cleanup last
(Wave 6 — sequenced last because the code it touches is still being modified by the earlier
waves; within Wave 6, the 6F–6Q small fixes may run before or alongside the 6A–6E splits, whose
structural risk is what actually wants the last position).

**Provenance of the additions:** the original 19-step plan (waves 1–6) was Blueprint-drafted and
adversarially reviewed against source (verdict below). The 2026-09-06 addendum then folded in
four parallel cross-cutting reviews (`ecc:python-reviewer`, `ecc:security-reviewer`, a
coverage-gap analysis, a static E2E health assessment): **Wave 0** — the HIGH-severity findings,
0A–0G, where 0A is convergent (python-review and security-scan independently found the same
`max_chunk_chars` defect) and 0G bundles a security fix with the E2E gap on the same file;
**Wave 4's in-place correction** — the E2E agent falsified the wave's original premise (the
failures were suppressed by swallowed exceptions and discarded assertions, not timeout-shaped),
so 4A un-hollows the tests before 4B investigates through them; **Wave 4B** — 13 coverage gaps
from an 88%-aggregate / 13-sub-80%-module analysis, grouped into 4B.1–4B.4 plus the
per-module-floor gate; **Wave 6's 6F–6Q** — 9 python-review MEDIUM steps, 2 security MEDIUM
steps, and 1 batched step carrying 7 python-review LOWs + 4 security LOWs, with two findings
already folded into existing steps (python M6 → 0B's tasks, python L1 → 0F task 1) and M9
becoming an in-place expansion of Step 6E's extraction targets rather than a duplicate step.

**Adversarial review (2026-09-06, Opus-tier, against source directly, not just this file's prose):
verdict SHIP-WITH-EDITS.** Five HIGH findings and all MEDIUM/LOW findings are folded into the steps
above — most substantively: Step 1C's original three-way decision had a non-viable option that was
already shipped and would have closed #193 with no actual fix (removed); Step 1A's original fix
covered only one of two ways the cancel gets dropped and named the wrong file (rescoped to three
files, both windows); Step 1D's original sabotage plan could not fail and its harness prescription
didn't match the module (both corrected, plus Step 1E inserted to sweep 2+ other confirmed-hollow
batched modules the review surfaced); Step 3A's "all three runners" claim was false — streaming
needs a new parameter added, not just unmocking (corrected, write surface expanded). Full findings:
`/private/tmp/claude-502/-Users-ericepstein-Qwen3-TTS-UserFiles/88bcce41-a83c-4f4c-9841-d27835b0ff60/tasks/a1204c7ae9cff963b.output`
(session-local; not durable — the corrections themselves are what's durable, captured above).
