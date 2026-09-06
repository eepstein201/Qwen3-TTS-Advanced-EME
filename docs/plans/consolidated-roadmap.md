# Qwen3-TTS Consolidated Development Roadmap

> **Status:** All P0–P2 items complete. Priority 2 enhancements (R-23/27/29/43) complete. GEN-1 (inference_lock release) shipped (#59, 2026-07-21). **2026-08-03 Santa audit (24 defects) resolved at HIGH severity**: H7 (#125), H1–H6 (#126), and the deferred Important I1 — WS client-disconnect misclassified as cancel (#137, `2e42e05`) — all merged to `main`; 13 MEDIUM + 4 LOW remain (see `audit_2026_08_03_ws_concurrency_cluster` memory). **P2-1 `config.py` split shipped** (`5a22d58` → `core/config/` package, largest submodule 464 lines); **P3 housekeeping shipped** (`3979881`). **PQA batch (2026-07-30 research): PRF-1..8 all shipped** (PRs #143–#149, #129/#164); **PRF-9 measured NO-GO and closed** (#187, 2026-08-15 — see row); PRF-10 unbuilt. **GEN-2 response contracts shipped** (#185, 2026-08-15). Open work is: vLLM-backend performance (HIGH-1/2, MED-2), PRF-10, P2-1 residuals (`inference.py`/`generate.py`/`app.py`/`shared.py`), P2-2, the remaining PRF-9 follow-up ticket (runaway guard — see the measurement doc; **MLX `max_tokens`/`lang_code` forwarding is now shipped** — all 6 MLX call sites map the kwargs and clamp to 4096, `tests/test_mlx_generate_kwargs.py`), and upstream-blocked research.
> **Last Updated:** 2026-08-15 — Lane F reconciliation after PR #187 (PRF-9 measurement doc): PRF-9 closed as measured-NO-GO; also swept the stale open-work list (GEN-2 shipped #185; PRF-7 fully complete via #164). 2026-08-14 — reconciled the PRF table against the 2026-08-09 merges (#143–#149): PRF-1 (#143 `d449325`), PRF-2 (#146 `7b6c336`, intentional design divergence — see row), PRF-3 (#145 `c4172e8`), PRF-4 (#144 `ecba434`), PRF-5 (#148 `97124c1`), PRF-6 (#147 `71533c1`), PRF-8 (#149 `bc7f3ec`); PRF-7 `mlx-audio>=0.4.6` merged via dependabot #129 (`15d9a7d`), 0.4.7 floor via #164 (`eafd0a2`). PRF-4 is no longer open — SDPA is the default (`advanced.attn_implementation="auto"`), FA2 opt-in. Earlier: 2026-08-07 pass recorded the 08-03 audit HIGH-resolution + I1 (#137), the `config.py` split (`5a22d58`), P3 closure (`3979881`).

---

## Verification policy

Every ✅ below was confirmed against current source on 2026-07-21 (`file:line` cited). Items marked **OPEN** have no matching implementation in tree. **TDD convention:** each OPEN item lists acceptance criteria; write the failing test first (red), implement (green), refactor. Source findings live in `docs/reviews/e2e-review-2026-07-01.md`.

---

## ✅ Completed

| ID | Task | Completed |
|----|------|-----------|
| P0-1 | Remove duplicate Pydantic models (`validation.py`) | ✅ |
| P0-2 | Consolidate print helpers (`tools/_shared.py`) | ✅ |
| CRITICAL-1 | Deconstruct `test_voice.py` (31k tokens → 8 domain files) | ✅ |
| CRITICAL-2 | Fix Docker IPC (`--ipc=host`, HF cache volume) | ✅ |
| CRITICAL-3 | Add `_require_server` decorator (client package) | ✅ |
| P1-1 | `_get_config_value` helper in `config.py` | ✅ |
| P1-2 | `_prepare_text_chunks` helper in `inference.py` | ✅ |
| P1-3 | Audio processing constants (`LUFS_TARGET`, `DEFAULT_SAMPLE_RATE`) | ✅ |
| P1-4 | `_extract_error_message` helper in `server/client/_base.py` | ✅ |
| P2-1 | CLAUDE.md trimmed to <300 lines | ✅ |
| P2-2 | Split `ui.py` into `interface/ui/` package | ✅ |
| P2-3 | Split `generate.py` into `interface/cli/` package | ✅ |
| P2-4 | Split `TTSClient` into `server/client/` package | ✅ |
| HIGH-3 | WebSocket bidirectional audio streaming (`websocket.py`) | ✅ |
| R-13 | Rate limiting (per-IP / per-token / hybrid) | ✅ |
| R-23 | LUFS normalization toggle — `_maybe_apply_lufs` reads `generation.lufs_normalize` + `lufs_target`; wired into single- and multi-chunk paths (`inference.py:642-653,779,844`); ships OFF by default | ✅ 2026-07-21 |
| R-27 | Configurable silence gap — `generation.silence_gap_seconds` switches silence-gap concat vs 50 ms crossfade (`inference.py:836-842,943-946`); default `0.0` | ✅ 2026-07-21 |
| R-29 | `TranscribeRequest.language` pattern — `Field(default="en", pattern=r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")` (`validation.py:100`) | ✅ 2026-07-21 |
| R-30 | ASGI body-size 413 middleware — `limit_request_body_size`; `MAX_REQUEST_BODY_BYTES = 2 * MAX_AUDIO_BASE64_BYTES` (`app.py:258-296`) | ✅ 2026-07-06 |
| R-43 | `create_voice.main(argv=None)` accepts args directly — `parser.parse_args(argv)` (`tools/create_voice.py:269,309`) | ✅ 2026-07-21 |
| R-50 | Thread-safe concurrent generation (`_history_lock`) | ✅ |
| R-51 | History panel chunk count — `/generate` returns `"chunks"`; client stores `last_chunk_count` | ✅ |
| QC-1..10 | Type hints, `click.echo`, specific exceptions, extracted helpers, token-ownership guard, `TTS_LOG_LEVEL`, quant/transformers pins, xdist fixture | ✅ |
| E2E-1 | Token-write fail-fast — `_write_auth_token` raises `RuntimeError` on `OSError` (`app_lifespan.py:142-143`, called `:324`) | ✅ 2026-07-21 |
| E2E-2 | `/cancel-generation` streaming bridge — `_should_stop_streaming` checks `stop_event` **and** `generation_state["cancelled"]` (`app_generation.py:47-54,641-645`); test `tests/test_streaming_cancel.py` | ✅ 2026-07-21 |
| E2E-3 | Silent-error logging — `logger.warning` added at `app_prompts.py:118,214`, `app_models.py:364`, `ui/shared.py:160,198`, `config.py:448`, `client/voices.py:37` | ✅ 2026-07-21 |
| E2E-4 | `trusted_proxies` allowlist — `TTS_TRUSTED_PROXIES` env, loopback default; `X-Forwarded-For` honored only for trusted hosts (`app.py:139-149,155-169`) | ✅ 2026-07-21 |
| E2E-5 | `/generation-status` no longer leaks — public body omits `eta_sec`/`batch_total`/`chunk_total` (`app.py:421-429`) | ✅ 2026-07-21 |
| E2E-6 | vLLM clone temp-file cleanup — `tmp_path` `os.unlink`'d in `finally` (`vllm_client.py:239,294-304`) | ✅ 2026-07-21 |
| GEN-1 | Release `inference_lock` before WAV-encode + peaks — lock narrowed to inference + `chunk_count` capture (`app_generation.py:262,389`); encode/cache/peaks run lock-free; AST test `tests/test_generation_lock_scope.py` (PR #59, full test matrix green) | ✅ 2026-07-21 |

---

## Priority 1: Performance & Infrastructure (Open)

| ID | Task | Impact | Effort | Files |
|----|------|--------|--------|-------|
| **GEN-2** | Add `response_model=` Pydantic contracts to the 22/24 routes currently returning raw `dict` | Medium | Medium | `server/app.py` (+ models in `validation.py`) |
| **HIGH-1** | vLLM multimodal params: `--limit-mm-per-prompt audio=1`, `--enable-chunked-prefill`, `bfloat16` **(vLLM backend only)** | High | Medium | Docker configs, vLLM init |
| **HIGH-2** | Decouple FastAPI from vLLM inference via `httpx.AsyncClient` **(vLLM backend only)** | High | High | `app.py`, `engine_vllm.py` |
| ✅ **MED-1** (#182, `208cc00`, 2026-08-15) | ~~Wavesurfer peaks: confirm caching~~ **Resolved with corrected premise:** generation-cache hits didn't recompute peaks — they **omitted them entirely** (both hit paths returned no `peaks` field). Peaks now stored on the `gen_cache` entry and echoed on hits; computed exactly once per audio asset (`tests/test_peaks_caching.py`). | Medium | Low-Med | `audio_processing.py`, `app_generation.py` |
| **MED-2** | Optimize `engine_vllm.py` parameters (`max_model_len`, `tensor_parallel_size`, …) **(vLLM backend only)** | High | Medium | `engine_vllm.py` |

**Why GEN-2:** 22 of 24 routes return untyped `dict` — no response contract for clients and thin generated OpenAPI schema; structural debt flagged in the 2026-07 e2e review.
**Scope note (HIGH-1/HIGH-2/MED-2):** these apply only to the optional **vLLM backend** (Linux/datacenter). Default deployments (MLX on Apple Silicon, torch elsewhere) are unaffected — prioritize only if vLLM is in use.

**Acceptance criteria (test-first):**
- **GEN-2:** each newly-typed route has a test asserting the JSON response validates against its Pydantic model; `app.openapi()` generates with no warnings.
- **HIGH-1 / MED-2:** a Docker-vLLM smoke test asserting the server starts with the new params and `/health` returns 200.
- **HIGH-2:** a test asserting the request handler does not block the event loop during a mocked vLLM call.
- **MED-1:** a test asserting peak computation runs at most once per audio asset (cache hit on second request).

---

## Performance / Quality / Accuracy — 2026-07-30 research (PRF-1..8 ✅ shipped; PRF-9 ✅ measured-NO-GO, closed #187; PRF-10 open)

Adopted from [`perf-research-2026-07-30.md`](perf-research-2026-07-30.md) (five-track sweep; ~40 primary sources). That file is the **baseline the quarterly upstream sweep diffs against** — keep the `PRF-*` IDs stable for cross-reference. Cites below reconciled against `main` @ `e94b806` (paths drifted from the research doc's indicative values; corrected here). Context on the underlying upstream defects lives in the research doc's "Upstream quality/accuracy notes."

**Upstream is frozen:** all six Qwen3-TTS models unchanged since 2026-01-29, `qwen_tts` 0.1.1 with zero merged PRs since mid-March. Every remediation below is **local or third-party** — there is no upgrade target.

**Status (2026-08-15):** PRF-1..8 all shipped. PRF-9 measured 2026-08-15 and **closed NO-GO** (#187, `docs/reviews/prf9-max-new-tokens-measurement-2026-08-15.md`); PRF-10 unbuilt.

| ID | Task | Axis | Impact | Effort | Files (reconciled) |
|----|------|------|--------|--------|--------------------|
| ✅ **PRF-1** (#143, `d449325`, 2026-08-09) | **Fix Chinese number normalization** — `num2words(…, lang='zh')` raises `NotImplementedError`, swallowed by `_safe_transform()`; cardinal/ordinal/date/currency normalization **silently no-ops for all Chinese input**. Added a `zh` branch (digits→汉字; borrowed Coqui `chinese_mandarin_cleaners`). | Accuracy | **High** (primary language) | **Trivial** | `core/engine/text_processing.py:108` (`_safe_transform`), `:150-185,250-283` (norm steps) |
| ✅ **PRF-2** (#146, `7b6c336`, 2026-08-09) — **shipped with an intentional design divergence** | **Phase-aligned chunk splices.** Roadmap design prescribed a zero-crossing snap + RMS level-match *before* the existing raised-cosine crossfade. Measurement showed the zero-crossing snap **worsened** alignment (0.87→0.71 of reference RMS; on a crossfade it ignores crossing direction and can land anti-phase). **Shipped instead:** normalized cross-correlation phase alignment over a 10 ms lag window + ±3 dB RMS level-match *before* the raised-cosine crossfade (0.71→~0.97 of reference RMS); the zero-crossing snap is retained **only** for the no-overlap hard-splice path. The divergence is deliberate — do not "fix" it back to the prescribed design. | Quality | High (dominant chunk artifact) | Low | `core/engine/inference.py:563-613` (`_crossfade_chunks`), `:834-837` (concat) |
| ✅ **PRF-3** (#145, `c4172e8`, 2026-08-09) | **Normalize HH:MM:SS time strings** (regex + num2words) — proven failure (upstream #328: `15:16:36` seconds garbled). | Accuracy | Med | Low | `core/engine/text_processing.py` |
| ✅ **PRF-4** (#144, `ecba434`, 2026-08-09) | **Torch default FA2 → SDPA** — `_apply_cuda_optimizations` used to auto-select `flash_attention_2` on Ampere+ whenever `flash_attn` was installed (`model_loader.py:110-122`); upstream #333 reports NaN logits with `flash_attention_2` on exactly these GPUs (L4/A100). **Shipped:** SDPA is the default (`advanced.attn_implementation="auto"` → SDPA), FA2 opt-in behind the explicit config flag. Reverses the 2026-03-23 conclusion. | Correctness | **High** | Low | `core/engine/model_loader.py:110-122`, `docs/00-Foundations/ARCHITECTURE.md:160-161` |
| ✅ **PRF-5** (#148, `97124c1`, 2026-08-09) | **Recover from failed model loads** — mlx-audio #827: Base cloning goes ~2.4× slower after a failed swap. **Shipped:** every `/load-model` failure path runs `_recover_from_failed_load()` (clears the model slot, drops stale `load_time`, runs the unload cleanup); recovery is non-fatal so it never masks the load error. | Robustness | High | Low | `server/app_models.py` (`_recover_from_failed_load`), `server/app_lifespan.py` |
| ✅ **PRF-6** (#147, `71533c1`, 2026-08-09) | **Clone rate control via post-hoc pyrubberband time-stretch** (not via `instruct`) — upstream #290: model rate-control broken in clone. **Shipped:** `generation.clone_speed` (0.5–2.0, unset by default); `gen_params["speed"]` overrides; failures return unstretched audio. Design/custom are never stretched. | Robustness | Med | Low | `core/engine/audio_processing.py:163-195`, `core/engine/inference.py` |
| ✅ **PRF-7** (0.4.6 via dependabot #129, `15d9a7d`; 0.4.7 floor via #164, `eafd0a2`, 2026-08-15) | **Bump mlx-audio 0.4.5 → 0.4.6 → 0.4.7+** — ICL cache (clone TTFT ~−300 ms), streaming leak fix (#852/v0.4.2), continuous batching (v0.4.3), ~13% RTF (v0.4.6). **Validated 2026-08-15 post-merge:** pip resolved to **0.4.8** (floor `>=0.4.7`), `pip check` clean, transformers 5.0.0rc3 → 5.15.0 absorbed cleanly, full non-E2E suite 2766 passed, live clone-mode smoke generation 11 s / valid audio. | Speed | High | Low (dep bump) | `pyproject.toml:29` |
| ✅ **PRF-8** (#149, `bc7f3ec`, 2026-08-09) | **ASR-trim the ICL echo-tail** (upstream #341) — reuse existing ASR to detect/clip any reference-tail echo at the head of cloned output. **Shipped** as `generation.trim_icl_echo` (default `true`): ASR-transcribes the first 4 s, matches the longest reference-tail suffix (≥3 words), clips at the following silence, capped at 50% of output. Only fires when ASR is already loaded and a reference transcript is resolvable; `x_vector_only_mode` is never probed. Inert until the server plumbs `reference_text` through (follow-on). | Quality | Med | Low–Med | `core/engine/inference.py:117-128`, `core/engine/voice_prompt.py`, `core/engine/asr.py` |
| ✅ **PRF-9** (#187, `583eaea`, 2026-08-15) — **measured NO-GO, closed** | **Investigate raising MLX `max_new_tokens=2048` cap** — measured on M2 Pro/16 GB with chunking disabled. **The premise was false on MLX:** `max_new_tokens` (and `language=`) never reach the model — mlx-audio's params are `max_tokens` (default **4096**) and `lang_code`, and our kwargs are swallowed by `**kwargs` — so the *effective* shipped cap is 4,096 tokens @ **12.5 Hz** (~327.7 s), not 2048 @ 12 Hz. Raising it fails anyway: long single-call generations suffer **non-deterministic EOS-failure runaway loops** (the cap is the only guard — proven by per-20 s RMS fingerprints), and memory exhausts (13.5 GB MLX-active on the clean 8,192-token run; 16.5 GB over-commit at 16,384). **Chunked generation stays the long-form architecture.** Full data: `docs/reviews/prf9-max-new-tokens-measurement-2026-08-15.md`; follow-up tickets (MLX `max_tokens`/`lang_code` forwarding bug, runaway guard) listed there. | Quality | High (structural) | Low to test / n-a to ship | ~~`core/engine/inference.py:385-393,514` (swallowed kwargs)~~ **forwarding fixed 2026-08-16** (`_split_mlx_params`/`_mlx_lang_code`, all 6 MLX sites, clamped to 4096), `docs/reviews/prf9-max-new-tokens-measurement-2026-08-15.md` |
| **PRF-10** | *(Optional)* **Task-Vector emotion control** (arXiv:2606.05367) — training-free inference-time interpolation between neutral↔emotional **x-vectors** (emotional prosody lives in the speaker embedding). Exploits `x_vector_only_mode`; torch path. **Not built.** | Quality (new capability) | Med | Med | `core/engine/voice_prompt.py`, `core/engine/inference.py` |

**Execution order (low-risk quick wins first):** PRF-1 → PRF-4 → PRF-3 → PRF-2 → PRF-5 → PRF-6 → PRF-7 → PRF-8 → PRF-9 → PRF-10.

**Acceptance criteria (test-first — red → green → refactor):**
- **PRF-1:** a test asserting Chinese cardinals/ordinals/dates/currency normalize to汉字 (not a silent no-op); regression test that `_safe_transform` no longer swallows the `zh` path.
- **PRF-2:** a test asserting spliced chunk boundaries snap to a zero crossing and RMS-match within tolerance; existing crossfade tests still green.
- **PRF-4:** a test asserting `_apply_cuda_optimizations` returns `sdpa` by default on Ampere+ (FA2 only when the opt-in flag is set), even with `flash_attn` installed; ARCHITECTURE.md hardware table updated to match.
- **PRF-3:** a test asserting `15:16:36` and similar HH:MM:SS strings expand to spoken time, not garbled digits.
- **PRF-5:** a test asserting a failed model swap triggers recovery (restart/reset) rather than a persistent slowdown; load/unload handlers re-emit consistent state.
- **PRF-6:** a test asserting clone output honors a requested rate via post-hoc time-stretch (duration changes with the rate factor).
- **PRF-7:** MLX-env install resolves with `mlx-audio>=0.4.6` and the `transformers`/`gradio`/`hub` pins intact (`pip check` clean); a smoke generation passes. **Do not** touch `requirements.lock` (mlx excluded by policy).
- **PRF-8:** a test asserting a reference-tail echo at the head of cloned output is detected and clipped; `x_vector_only_mode` path unaffected.
- **PRF-9:** *validation gate, not a ship gate* — a documented long-form stability + peak-memory measurement on M2 Pro before any cap change; ship only if both pass.
- **PRF-10:** a test asserting neutral↔emotional x-vector interpolation produces a bounded, monotonic prosody shift; torch-only, gated behind `x_vector_only_mode`.

**Deferred (do not act — watched by the upstream GHA + crons):** speculative decoding (see R-28 below), vLLM prefix-caching the voice-prompt prefix (deployment decision), SageAttention (monkey-patch only), vLLM mainline TTS (absent), FlashAttention-4 (not a native HF value), Qwen3-ASR-1.7B as Whisper replacement (MLX availability unverified), new Qwen3-TTS models (upstream frozen). **Dropped:** EAGLE-3 for TTS, G2P/phoneme frontend, FA3/FA4/SageAttention on T4, full NeMo Text Processing (Pynini won't pip-install on macOS). Rationale in the research doc's KEEP-MONITORING / DROP sections.

---

## Priority 2: Enhancements (Resolved)

| ID | Task | Impact | Effort | Files |
|----|------|--------|--------|-------|
| **FOLLOWUP-1** | ✅ **Decided 2026-08-15: keep `design.load_at_startup`/`custom.load_at_startup` at `false`/`false`** (only `clone` loads at startup). Rationale: ~5 GB of additional startup memory buys availability for models most sessions never open; on-demand load is one click in Manage Models and now shows a live ETA badge (PR #175). No `config.json` change needed — the existing default-pinning tests already cover keep-false. | Low | Low | `config.json` (unchanged) |

**Acceptance:** decision recorded here with rationale; if flipped to `true`, update `config.json`, CLAUDE.md "Key Settings", and add a test asserting startup loads the expected models. (Keep-false: done — decision recorded 2026-08-15; no flip, no new test required.)

---

## Priority 3: Future / Upstream-Dependent

| ID | Task | Blocker | Effort |
|----|------|---------|--------|
| **R-28** | Speculative decoding (1.5-3x speedup; 0.6B as draft for 1.7B). See `2026-03-23-speculative-decoding-research.md`. Phase 1 = monitor upstream. **2026-07-30 update:** PCG (arXiv:2511.13732) is now ICASSP-2026–accepted but ships **zero code, zero adopters**; closest analogue SSD (arXiv:2505.15380) is 1.4×/lossy/code-less. Blocker moved "no theory → theory exists, nothing reusable." Re-check ~2027-01. | Upstream library support | High |
| **FUTURE-1** | Entropy-based hallucination monitoring | vLLM forward-pass modification | High |
| **FUTURE-2** | GFlowNet distribution alignment | Research integration | High |
| **FUTURE-3** | Adaptive attention head deactivation | Per-model profiling | High |

**Trigger-gated watches (2026-07-30 sweep — no action until the trigger fires):**
- **vLLM prefix-caching the voice-prompt prefix** — lossless, exact, lowest-effort speed win, but only if the talker is served through vLLM (it isn't). Deployment decision, not a research dependency.
- **SageAttention** — monkey-patch only, no TTS benchmarks. Trigger: becomes a native HF `attn_implementation`.
- **vLLM mainline TTS** — absent; vLLM-omni separate/unmerged. Trigger: TTS lands in mainline vLLM.
- **FlashAttention-4** — active beta (Hopper/Blackwell only), not a native HF value. Trigger: native HF wiring + stability.
- **Qwen3-ASR-1.7B** as Whisper replacement — beats Whisper-large-v3 (zh especially), MLX availability unverified. Trigger: confirm `mlx-community/Qwen3-ASR*`.
- **New Qwen3-TTS models** — upstream frozen since 2026-01-29. Trigger: new model ID under `Qwen/`.

---

## CI & Quality Debt (from 2026-07-01 e2e review — Open)

Long tail tracked in `docs/reviews/e2e-review-2026-07-01.md`. Highest-value items:

- ~~**CI gates:** only batches 1–3 are gated; add batches 4 + 5 + `tests/security/*` + `tests/evaluations/*`. Add `ruff` / `mypy` / `bandit` to CI (currently local-only). Add `--cov-fail-under=80`.~~ ✅ **Resolved** — `.github/workflows/test.yml` now runs batches 1–5, the `lint` job runs `ruff` + `mypy qwen3_tts/{core,server,interface}` + `bandit -r qwen3_tts --severity-level high`, and the `coverage` job enforces `--cov-fail-under=80`.
- **Docker:** `docker-compose.yml` vLLM build `context: ..` → `.`; reconcile `Dockerfile.vllm` vs `docker/vllm.Dockerfile`.
- **MLX model-config matrix:** run 1.7B + 0.6B across bf16 / 8bit / 4bit with smoke generation + `/health` `model_size`/`mlx_quantization` verification.
- **Manage-Models table refresh latency:** load/unload handlers must re-emit table update (test_09/10 issue).
- **Structural debt** (re-measured `main` @ `2e42e05`): ~~`config.py` 1432~~ ✅ split → `core/config/` package (`5a22d58`); ~~`_facade.py` 1293~~ ✅ split to 525 (`#92`); `inference.py` **1129**, `generate.py` **902**, `app.py` **831**, `shared.py` **803** (newly breached); `handle_generate` 395 lines; broad `except Exception` at `inference.py:627,637`; f-string logging; duplicate logger `engine_vllm.py:34/76`; unguarded file handle `engine_vllm.py:268`.
- **LOW cleanups:** ~~dead `# nosec`~~ ✅ (#180, 2026-08-15: 3 of 5 were dead and removed; the 2 live B104 Colab binds kept with justifications); dead `return` after `_error_response()`; ~~audit-log WS auth failures~~ ✅ (#180: all three auth-failure branches log WARNING with sanitized client IP); `_sanitize_error` regex over-strip; name-length mismatch 128 vs 255 (`config.py:592` vs `validation.py:19`). **New (2026-08-15):** `/generate` returns `model_not_loaded` (recovery: restart) for design/custom instead of loading on demand — RUNBOOK.md claims models load "on demand when a request needs them"; either auto-load in the handler or fix the doc. `tests/test_server_peaks.py` uses a pytest fixture the batch runner's `python -m unittest` invocation can't see — its tests run hollow in batch gates (same family as the #181 false-green fixes).

---

## Repo Review 2026-07-23 (comprehensive sweep — 7/8 done; HYG-1 open)

Four-agent review across dead code, tests, static gates, E2E, structure, and UI/UX +
user-facing text. Repo is healthy (CI green, ruff clean, 2163+ tests, CLAUDE.md 287/300).
New actionable findings below; two UI bugs reproduced directly.

| ID | Task | Sev | Effort | Files |
|----|------|-----|--------|-------|
| ✅ **UI-1** (#83) | "Unload" button crashes first click — `on_unload_click` imports `get_model_table_data` from `.shared`, but it lives in `.model_management` → `ImportError`. Handler wired live at `:1004`. | HIGH | Low | `interface/ui/_facade.py:951` |
| ✅ **UI-2** (#83) | Dead startup-reload warning — checks `startup == "default"` but table only emits `"Yes"`/`"No"`; warning never fires. Also fabricated "~3-5 seconds" reload estimate. | MED | Low | `interface/ui/_facade.py:972,981` |
| ✅ **DOC-1** (#83) | Typo in coverage command `--cov=qwen3_tss` → `qwen3_tts`. | LOW | Trivial | `docs/CONTRIBUTING.md` |
| ✅ **DEDUP-1** (#85) | `process_batch` duplicated (both live + independently tested); dedup needs lazy import to avoid circular import with `cli/batch.py`. | MED | Med | `interface/generate.py:90`, `interface/cli/batch.py:25` |
| ✅ **UI-3** (#84) | UI slider defaults + Model-Settings/Tips prose hardcoded instead of read from config constants (`VALID_MODEL_SIZES`, `generation` defaults) — silent drift + user config ignored. | MED | Med | `interface/ui/generation.py:300-303,396-399`, `_facade.py:1097,1103,1218` |
| ✅ **UI-4** (#84) | CLI model-size `choices` hardcoded instead of `VALID_MODEL_SIZES`. | LOW | Low | `interface/generate.py:246`, `cli.py:221` |
| ✅ **A11Y-1** (#86) | Per-action status Textboxes unlabeled, no `aria-live`; "Stop"/"Confirm Cancel?" vocab drift. | LOW-MED | Med | `interface/ui/generation.py`, `_facade.py` |
| **HYG-1** | Working-tree clutter (untracked): `.voice_server.log.old`, `.tts_server*.log`, stray `test/` dir, loose media, empty scratch `.md`. Confirm per category before deleting. | LOW | Low | repo root |

**Structural debt** (extends the CI & Quality Debt list above; re-measured `main` @ `2e42e05`):
`config.py` ✅ split → `core/config/` package (`5a22d58`); `_facade.py` ✅ split to 525 (`#92`);
`inference.py` **1129**, `generate.py` **902**, `app.py` **831**, `shared.py` **803** exceed the
800-line limit.
✅ **JS-EXTRACT** (#88): the 543-line embedded JS in `get_streaming_player_js` moved to
`interface/static/streaming_player.js` via a cached loader; `wavesurfer_js.py` 814 → 309 lines
(now under the limit). Behavior-preserving — verified by 74 existing wavesurfer tests + 3 new guards.

**Execution:** Phases 1–2 complete and merged 2026-07-23 (#83 UI-1/UI-2/DOC-1, #84 UI-3/UI-4,
#85 DEDUP-1, #86 A11Y-1); JS-EXTRACT merged 2026-07-23 (#88). **Remaining:** HYG-1 (working-tree
clutter cleanup — confirm per category before deleting) + the structural-debt file splits noted
above (`config.py`, `_facade.py`, `inference.py`, `generate.py`, `app.py`). Full plan:
`~/.claude/plans/run-comprehensive-review-of-merry-gadget.md`.

---

## Recommended Execution Order

### Sprint 1 — Default-backend quick wins (low risk, no vLLM needed)
1. **GEN-2:** `response_model` contracts (incremental; improves OpenAPI + client contracts)
2. ~~**FOLLOWUP-1:** decide `load_at_startup` defaults (config-only)~~ ✅ decided 2026-08-15 — keep `false`/`false` (see Priority 2)
3. **MED-1:** verify/cache wavesurfer peaks

### Sprint 2 — Concurrency refinement ✅ (2026-07-21)
4. ~~**GEN-1:** release `inference_lock` during encode/peaks~~ — shipped (#59); AST lock-scope test added

### Sprint 3 — vLLM backend (only if vLLM is deployed)
5. **HIGH-1 + MED-2:** vLLM params
6. **HIGH-2:** decouple FastAPI from vLLM

### Sprint 4 — PQA adoptions (2026-07-30 research; default backends) — items 7–14 ✅ shipped 2026-08-09
7. ~~**PRF-1:** Chinese number normalization~~ ✅ #143
8. ~~**PRF-4:** FA2→SDPA default on Ampere+~~ ✅ #144
9. ~~**PRF-3:** HH:MM:SS time normalization~~ ✅ #145
10. ~~**PRF-2:** phase-aligned chunk splices~~ ✅ #146 (intentional divergence from the prescribed design — see table row)
11. ~~**PRF-5:** defensive restart on model-swap OOM~~ ✅ #148
12. ~~**PRF-6:** clone rate control via post-hoc time-stretch~~ ✅ #147
13. **PRF-7:** bump mlx-audio → 0.4.6 ◐ (0.4.6 merged via #129; 0.4.7 open as PR #164)
14. ~~**PRF-8:** ASR-trim the ICL echo-tail~~ ✅ #149
15. **PRF-9:** validate raising the MLX `max_new_tokens` cap (measure before shipping)
16. **PRF-10:** *(optional)* task-vector emotion control

---

## Success Criteria

- [x] R-29/R-30 validation gaps closed (2026-07-21)
- [x] Full test suite passes — ~2555 tests (2026-08-07 maintained baseline)
- [x] **GEN-1:** concurrent generations interleave during encode — shipped #59, 2026-07-21 (`tests/test_generation_lock_scope.py`)
- [x] **GEN-2:** ≥80% of routes carry `response_model` — shipped #185, 2026-08-15; re-verified 2026-09-06 (19/23 routes in `app.py`, the 4 gaps are the documented deliberate exceptions on binary/streaming routes). Checkbox had been left stale despite the Notes section already recording it as shipped.
- [ ] **HIGH-1/MED-2:** vLLM params validated in a Docker environment
- [ ] **HIGH-2:** FastAPI decoupled from vLLM — event loop not blocked during generation

---

## Notes

- **E2E Testing:** plan at `docs/plans/e2e-testing-implementation-plan.md`. Phases 1–2 complete (50 tests). Phases 3–4 (UI, cross-browser, a11y) are considered covered by `tests/test_e2e_playwright.py`.
- **AI Regression Prevention:** all implementations follow TDD (red-green-refactor).
- **Git Workflow:** feature branches; never push directly to main without approval.
- **Verification log:** reconcile pass confirmed ✅s against `main` on 2026-07-21; GEN-1 shipped via #59 (merged `752e265`). See `docs/reviews/e2e-review-2026-07-01.md` for source findings.
