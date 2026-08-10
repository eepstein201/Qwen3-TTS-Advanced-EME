# Qwen3-TTS Review Remediation Plan

> **For agentic workers:** Implement workstream-by-workstream in isolated worktrees (`superpowers:using-git-worktrees`). Each workstream = one feature branch + PR (project mandates feature branches; never commit to main). Steps use `- [ ]` checkboxes. Full per-step TDD code is elaborated at execution time per workstream; this is the master sequencing + task-level plan.

**Source:** `~/.claude/session-data/python-review-2026-08-10.md` (0 CRITICAL / 9 HIGH / 33 MEDIUM / 16 LOW)
**Goal:** Remediate the review's HIGH-impact defects and unify the divergent streaming pipeline, in independently-shippable workstreams ordered by risk-to-data-integrity.
**Architecture:** No change to the dispatch/lazy-import model. The one structural change is collapsing the batch (`/generate`) and streaming (`/generate-stream`, `/ws`) generation paths onto a shared post-processing + state-lifecycle core. All other workstreams are localized, surgical fixes.
**Tech Stack:** Python 3.11, FastAPI + Starlette + WebSocket, Gradio 6.x, MLX/torch backends, pytest/unittest, conda envs `qwen3-tts` (torch) / `qwen3-tts-mlx` (mlx).

## Global Constraints (apply to every task)
- Lazy imports mandatory: torch/mlx/transformers/mlx_audio/qwen_tts only inside functions.
- Feature branches only: `git checkout -b fix/<workstream>`; Claude commits to the branch, user merges to main.
- No AI authorship attribution anywhere (commits, comments, PRs).
- Validation gate per task: `conda run -n qwen3-tts-mlx ruff check qwen3_tts tests && conda run -n qwen3-tts-mlx python -m pytest <touched test> -v -m "not e2e"`.
- Full pre-push gate: ruff + mypy + bandit + the batch owning touched files + `pytest -m "not e2e"` (not just batches).
- Restart server to pick up changes: `tts server stop && tts server start` (server uses editable install).
- Tests are AAA-pattern, descriptive names; new async tests use `IsolatedAsyncioTestCase` (never bare `async def` on `TestCase`); register new test modules in `tests/run_batches.py` BATCHES.

## Requirements Restatement
Eliminate silent data loss / wrong output (4 HIGHs), the batch/streaming output divergence (1 HIGH), and crash-on-edge-case paths (2 HIGHs), plus the two remaining HIGHs (vLLM pool leak, history clobber). Then progress the MEDIUMs by theme. Each fix ships with a regression test that fails before / passes after.

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| Streaming unification (WS2) changes audio output for existing stream users | MEDIUM | Add output-equivalence tests (batch vs stream) on fixed seeds before refactoring; gate behind the same post-processing calls |
| Concurrency fixes (WS4) unobserved without a concurrent-load test harness | MEDIUM | Add a pytest test that fires interleaved batch + cancel; don't rely on single-threaded unit tests |
| torch-backend fix (H3) can't be exercised on MLX-only CI | HIGH | Unit-test the guard with a stub returning `[]`; mark a torch-only smoke as e2e |
| `handle_generate` (cc 41) decomposition risks behavior drift | MEDIUM | Characterize current behavior with parametrized tests FIRST, then refactor |
| python-multipart bump (WS7) could break FastAPI file-upload contract | LOW | It's a patch/minor bump within the 0.0.x line; run `/create-rompt` e2e after |

---

## WS1 — Silent-data-loss & wrong-output HIGHs (ship first; low risk, high value) — ✅ DONE (PR #154, merged 2026-08-10)
**Branch:** `fix/silent-data-loss` · **Findings:** H1, H4, H5, H6 · **Complexity:** Small–Medium

### Task 1.1 — H1: re-clear stale `cancelled` per batch item — ✅ done
- **Files:** `qwen3_tts/server/app_generation.py:294` (modify); `tests/test_server_generation.py` (add)
- **Action:** In the per-item `generation_state.update(...)` inside `inference_lock`, add `"cancelled": False` (mirror the streaming path at l.658). This clears any flag a concurrent `/cancel-generation` set between items.
- **Test (AAA):** `test_concurrent_cancel_does_not_truncate_other_batch` — mock two interleaved requests where B is cancelled; assert A returns all submitted items. Use a fake `inference_lock` that releases between items.
- **Validate:** `pytest tests/test_server_generation.py -k cancel -v -m "not e2e"`

### Task 1.2 — H4: mark watch files processed only after success — ✅ done
- **Files:** `qwen3_tts/interface/generate_interactive.py:713` (modify); test
- **Action:** Move `processed_files.add(event.src_path)` from l.713 to after the successful `sf.write` (~l.742). On failure the file is NOT added → retried next event.
- **Test:** `test_watch_mode_retries_file_after_generation_failure` — patch generator to raise once then succeed; assert the file is processed on the second cycle.
- **Validate:** `pytest tests/test_interactive.py -k watch -v`

### Task 1.3 — H5: thread `gen_params` into the REPL — ✅ done
- **Files:** `qwen3_tts/interface/generate.py:638`, `qwen3_tts/interface/generate_interactive.py:471` (modify)
- **Action:** Add `gen_params` parameter to `run_repl`; pass it from `generate.py:638` (`run_repl(config, use_server, gen_params)`); delete the internal reconstruction at l.500–501.
- **Test:** `test_repl_honors_cli_sampling_flags` — invoke `run_repl` with `gen_params={"temperature":0.5,"seed":42}`; assert the params reach the generation call (mock the generator, capture kwargs).
- **Validate:** `pytest tests/test_interactive.py -k repl -v`

### Task 1.4 — H6: per-entry error recovery in SRT (and batch local path) — ✅ done
- **Files:** `qwen3_tts/interface/cli/srt.py:88–119`, `qwen3_tts/interface/generate.py:131–146` (modify)
- **Action:** Wrap each entry's generation in `try/except`; log the entry + error; continue. Build the combined output from successful entries. Mirror `dialogue.py:162–203`.
- **Test:** `test_srt_continues_after_entry_error` — 3 entries, middle one raises; assert 2 audio files + combined output produced, failure logged.
- **Validate:** `pytest tests/test_cli_srt.py -v`

---

## WS2 — Unify batch + streaming pipeline (centerpiece; highest leverage, highest risk)
**Branch:** `refactor/unify-streaming-pipeline` · **Findings:** H2 + Theme A (3 MEDIUMs) · **Complexity:** Large
**Goal:** One post-processing + state-lifecycle core called by `/generate`, `/generate-stream`, `/ws`. Retires H2 + 3 MEDIUMs in one refactor.
**Shape (research-confirmed):** the unifying abstraction is a **shared async generator that yields post-processed chunks** — batch path = `async for` collector, streaming path = `StreamingResponse(gen)`, WS = relay to socket. Per-chunk steps (time-stretch, ASR echo-trim on chunk-1, audio validation) live in `_postprocess_chunk`; **LUFS stays batch-only** (EBU R128 integrated loudness is inherently whole-signal — the relative gate is a global statistic — so it *cannot* be per-chunk; architectural, not a bug). Blocking MLX/torch inference inside the generator must be offloaded via `asyncio.run_in_executor` or the event loop stalls.

### Task 2.1 — Characterize current behavior (do NOT change behavior yet)
- **Files:** `tests/test_engine_streaming.py` (add)
- **Action:** Add output-equivalence + divergence-characterization tests on fixed seeds: (a) batch applies clone_speed/LUFS/echo-trim; (b) stream currently does NOT; (c) stream ignores `max_chunk_chars`. These tests document the CURRENT divergence so the refactor is provably behavior-preserving where intended.
- **Validate:** tests pass against current code (they assert today's behavior as a baseline).

### Task 2.2 — Extract `_postprocess_chunk(chunk, cfg, ctx)` helper
- **Files:** `qwen3_tts/core/engine/inference.py` (modify); the batch path's `_trim_icl_echo → _maybe_apply_speed → _maybe_apply_lufs` sequence (l.1168–1177, 1244–1255) → a single helper.
- **Action:** Extract the per-chunk-feasible steps (`_maybe_apply_speed`, `_validate_audio`, ASR echo-trim on chunk-1) into `_postprocess_chunk`. **LUFS is NOT per-chunk** — EBU R128 integrated loudness's relative gate is a global statistic over all blocks, so `pyloudnorm.integrated_loudness()` cannot run incrementally (research-confirmed). LUFS stays batch-only (run after collecting all chunks in the batch adapter); streaming/WS either skip LUFS or use a documented per-chunk peak/RMS approximation (non-R128). State this in the streaming docstring.
- **Interfaces:** `_postprocess_chunk(chunk: np.ndarray, cfg: Mapping, ctx) -> np.ndarray` — consumed by both batch and streaming.
- **Test:** unit-test `_postprocess_chunk` directly (speed applied, audio validated, returns new array).

### Task 2.3 — Wire `_postprocess_chunk` into the streaming paths
- **Files:** `core/engine/inference.py:1376–1476` (`run_inference_streaming`), `server/app_generation.py:676`, `server/websocket.py:396`
- **Action:** Call `_postprocess_chunk` on each yielded chunk. Pass `max_chunk_chars=req.max_chunk_chars` and `config_provider` to both streaming call sites (currently omitted). For WS, also add `progress_callback` if chunk monitoring is desired.
- **Test:** update the Task 2.1 divergence tests → now assert stream output == batch output for the per-chunk steps (speed/validation equivalence).

### Task 2.4 — Shared `generation_state` lifecycle for the WebSocket path
- **Files:** `server/websocket.py:433–472` (modify); `server/app_generation.py:651–660` (reference pattern)
- **Action:** Inside `_stream_generation`'s `inference_lock` block, set `generation_state.update({"active":True,"start_time":time.time(),"text_length":len(text),"mode":mode,"generation_id":str(uuid.uuid4())[:8]})`; reset in `finally` guarded by generation_id ownership (same pattern as HTTP paths). This makes WS generations visible to `/generation-status`, `/queue-status`, and `detect_degraded_generation`.
- **Test:** `test_websocket_generation_visible_in_generation_status` — start a WS gen (mocked slow); assert `/generation-status` reports `active:true`.

### Task 2.5 — Signal mid-stream errors via an in-band terminal error frame (research-corrected)
- **Files:** `server/app_generation.py:773` + streaming wire format; client parser `interface/generate_server.py:315–344`
- **Action:** Do **NOT** rely on raising to truncate the connection — research: once HTTP 200 headers commit, a raise makes the client unable to distinguish a server error from a network drop and loses all error context. Instead, define an **in-band terminal error frame** for the length-prefixed streaming wire format (sentinel length/flag + JSON `{error, code}`), send it, then close cleanly. The WebSocket path already sends an application error message — add close code `1011` (or `4xxx`) per RFC 6455 §7.4. Update the client parser (`generate_server.py`) to surface the error instead of treating it as a parse failure.
- **Test:** `test_http_streaming_emits_terminal_error_frame_on_mid_stream_failure` + `test_client_surfaces_stream_error_frame`.

---

## WS3 — Edge-case crash hardening
**Branch:** `fix/crash-hardening` · **Findings:** H3, H7 + Theme C (validate_config non-dict, parse_ssml, show_history JSONL, generate_via_server KeyError, 6× generic raise Exception) · **Complexity:** Medium
- **H3** `inference.py:255`: add `if not wavs: raise RuntimeError("torch generation returned no audio segments")` (mirror MLX l.417). Unit-test with stub returning `[]`.
- **H7** `generate_server.py:315–344`: wrap parse loop in `try/except (struct.error, ValueError)`; bound `audio_len` against `MAX_CHUNK_SIZE` (define constant, e.g. 200 MB).
- **validate_config** `config/io.py:92`: type-guard `if not isinstance(config, dict): raise ValueError(...)`; extend `load_config` except to `(TypeError, ValueError)`.
- **parse_ssml** `generate_helpers.py:338`: replace `"s" in time_str` with `endswith` + try/except `float()`.
- **show_history** `generate_helpers.py:271`: per-line `try/except json.JSONDecodeError` (skip corrupt line).
- **generate_via_server** `generate_server.py:245`: `resp.json().get("results")` + clear error if None.
- **generic raise Exception** `generate_server.py:189,217,219,243,299,358`: define `TTSGenericError(RuntimeError)`; use `ConnectionError` for the network case (l.358); update callers.

## WS4 — Concurrency hardening
**Branch:** `fix/concurrency` · **Findings:** Theme B + H8 · **Complexity:** Medium
- **ASR unload** `asr.py:234–252`: acquire `_asr_lock` around read-null-cleanup (capture old ref under lock; `gc.collect`/`empty_cache` outside).
- **Cache TOCTOU** `app_generation.py:219–224`: read the cache file inside the lock OR catch `FileNotFoundError` from `_read_cache_file_b64` and treat as cache miss (fall through to generation).
- **update_model_config** `app_models.py:378–380`: add active-generation guard mirroring `handle_unload_model` (409 if `generation_state["active"]`).
- **H8 vLLM stop()** `engine_vllm.py:415–420`: make `stop()` close the client explicitly (await if loop available; sync close otherwise) — don't discard `_client` without closing.
- **inference seed/dtype** `inference.py:1191,207`: research — under the current single-Lock serialization, global `manual_seed`/`mx.random.seed` is *functionally safe*, but adopt a **per-call `torch.Generator` defensively** (decoupled from global state, intent-revealing, survives the day the Lock is removed). **FIRST confirm generation is truly serialized** through one lock; if two generations can ever run concurrently on the same model, the global reseed is *already a silent reproducibility bug* (draws interleave, last reseed wins) and `torch.Generator` becomes mandatory. *Verify the server's lock topology before choosing.*

## WS5 — Resource leaks & atomicity
**Branch:** `fix/resource-leaks` · **Findings:** Theme D · **Complexity:** Small
- **vLLM clone temp .wav** `engine_vllm.py:469,548,604`: `try/finally` `os.unlink(request["input"]["prompt_audio"])` for clone mode in `generate()` and `generate_stream()`.
- **create_voice temp** `tools/create_voice.py:64–67`: `try/finally` cleanup of `wav_path`; use `tempfile.NamedTemporaryFile(suffix=".wav", dir=USER_FILES_DIR, delete=False)` for a unique name.
- **uninstall_config atomicity** `tools/uninstall.py:193`: replace raw `open("w")` with `save_config(default_config)` (gets atomic write + cache invalidation).

## WS6 — Async/event-loop blocking
**Branch:** `fix/async-blocking` · **Findings:** Theme E · **Complexity:** Small
- **/stats** `app.py:498`: `return await asyncio.to_thread(handle_stats, state, state.server_config)` (every sibling endpoint already does this).
- **apply_model_settings** `ui/shared.py:197–223`: reduce per-model poll + surface progress (Gradio generator-yield) or raise the per-model timeout; the 30s×3 frozen-UI window is the bug.

## WS7 — Dependency CVEs
**Branch:** `chore/dep-cves` · **Findings:** §6 · **Complexity:** Small
- **python-multipart 0.0.22 → 0.0.31** (5 DoS CVEs; FastAPI multipart parser; reached by `/create-voice-prompt`). Bump in `pyproject.toml`; run `/create-voice-prompt` e2e.
- Also bump in mlx env: `pillow`, `click`, `idna`, `msgpack`, `pygments`, `pytest` (dev), `pip`.
- **gradio 6.8.0 (torch env)** — leave tracked as known structural debt (transformers<5 knot); do NOT attempt in this plan.

## WS8 — Quality & complexity (lower priority)
**Branch:** `refactor/quality` · **Findings:** LOW cluster + cc decomp · **Complexity:** Medium
- Decompose `handle_generate` (cc 41), `_handle_generation` (35), `run_repl` (31), `websocket_tts_handler`/`_stream_generation` (24/22). **Characterize behavior with parametrized tests first.**
- Extract `_ensure_path_under_home()` from the 4× duplicated path-containment check in `shared.py` (brings it under the 800-line ceiling).
- Remove dead code: `ProgressIndicator` no-op construction (`model_management.py:123,182`), `prosody_preset` dead branch (`generation.py:204`).

## WS9 — i18n & correctness edge cases
**Branch:** `fix/correctness-edge-cases` · **Findings:** Theme F + G i18n · **Complexity:** Medium
- `encoding="utf-8"` on the 3 transcript `open()` calls (`voice_prompt.py:66,245,316`).
- `_TIME_RE` range validation (`text_processing.py:105`): return original substring when h/m/s out of range.
- `max_chunk_chars=0` truly disables chunking (`inference.py:1042–1073`) OR fix the doc — *decide after confirming the 765-char single-chunk anomaly from profiling*.
- Multi-chunk `_trim_icl_echo` cap (`inference.py:1244`): apply per-first-chunk or scale the cap.
- Metal retry keyword tightening (`inference.py:1308`): drop generic `"kernel"`, require Metal-specific phrases.

---

## Validation
```bash
# Per-task (fast):
conda run -n qwen3-tts-mlx ruff check qwen3_tts tests
conda run -n qwen3-tts-mlx python -m pytest <touched_test> -v -m "not e2e"

# Pre-push gate (full):
conda run -n qwen3-tts-mlx ruff check qwen3_tts tests
conda run -n qwen3-tts-mlx mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface
conda run -n qwen3-tts-mlx bandit -r qwen3_tts -c pyproject.toml
conda run -n qwen3-tts-mlx python -m pytest -m "not e2e"          # NOT just run_batches.py
# E2E (opt-in, needs server + TTS_DISABLE_RATE_LIMITING=1):
TTS_DISABLE_RATE_LIMITING=1 tts server start
conda run -n qwen3-tts-mlx python -m pytest tests/test_e2e_security_auth.py -m e2e -v
```

## Progress
- **WS1 — merged** as PR #154 (`fix/silent-data-loss`), 2026-08-10. All four HIGHs (H1, H4, H5, H6) shipped with regression tests, CI green.
- **Separate from this plan:** PR #153 (`fix(server): FastAPI hardening`) also merged 2026-08-10 — six ad-hoc server-hardening fixes (CWE-209 on `/health`, pre-auth rate-limit middleware, WS Origin validation, streaming-safe body-size middleware) from a standalone FastAPI review, not sourced from `python-review-2026-08-10.md`. Overlaps WS4 Task 2.4's WS `generation_state` goal, so that item is already satisfied when WS2 reaches it.
- **WS2–WS9 — not started.**

## Acceptance
- [x] WS1 HIGHs each have a regression test (fails before / passes after)
- [ ] WS2–WS4 remaining HIGHs each have a regression test (fails before / passes after)
- [ ] WS2 streaming output matches batch for per-chunk processing (equivalence tests green)
- [ ] Every workstream passes ruff + mypy + bandit + `pytest -m "not e2e"`
- [ ] No new `# type: ignore`, no bare `except:`, no mutable defaults introduced
- [ ] Feature branches only; no commits to main

## Deep-research validation — RESOLVED (focused web-research pass, 2026-08-10)
1. **Streaming unification pattern — CONFIRMED + sharpened:** use a shared async generator both paths consume (batch = `async for` collector; streaming = `StreamingResponse(gen)`), not a bare "function + adapters". Offload blocking inference via `run_in_executor`. Folded into WS2 Goal/Shape. *(FastAPI Stream Data docs; SO on offloading in async generators.)*
2. **Mid-stream error signaling — CORRECTION:** "raise → truncate" is wrong (client can't distinguish server error from network drop, loses context). Use an **in-band terminal error frame** + clean close; WS adds close code `1011`. Task 2.5 rewritten. *(FastAPI #10138; RFC 6455 §7.4.)*
3. **LUFS in streaming — CONFIRMED:** EBU R128 integrated loudness is whole-signal (relative gate is global); cannot be per-chunk. LUFS is batch-only by design. Task 2.2 sharpened. *(pyloudnorm paper; EBU R128 spec.)*
4. **torch RNG under serialization — CONFIRMED + defensive:** global reseed safe under single-Lock, but use per-call `torch.Generator` defensively; mandatory if concurrency is ever allowed. WS4 updated. *(PyTorch reproducibility notes; HF diffusers.)*
