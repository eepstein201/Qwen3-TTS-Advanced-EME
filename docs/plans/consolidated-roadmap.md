# Qwen3-TTS Consolidated Development Roadmap

> **Status:** All P0–P2 items complete. Priority 2 enhancements (R-23/27/29/43) complete. Open work is: one generation-path lock refinement (GEN-1), response-model coverage (GEN-2), the `load_at_startup` default decision (FOLLOWUP-1), vLLM-backend performance (HIGH-1/2, MED-2), and upstream-blocked research.
> **Last Updated:** 2026-07-21 — reconciled against code at `main` @ `6e8ec9d`

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

---

## Priority 1: Performance & Infrastructure (Open)

| ID | Task | Impact | Effort | Files |
|----|------|--------|--------|-------|
| **GEN-1** | Release `inference_lock` before WAV-encode + peak computation | Medium | Medium | `app_generation.py:216,278-280,360,385,392,419-421` |
| **GEN-2** | Add `response_model=` Pydantic contracts to the 22/24 routes currently returning raw `dict` | Medium | Medium | `server/app.py` (+ models in `validation.py`) |
| **HIGH-1** | vLLM multimodal params: `--limit-mm-per-prompt audio=1`, `--enable-chunked-prefill`, `bfloat16` **(vLLM backend only)** | High | Medium | Docker configs, vLLM init |
| **HIGH-2** | Decouple FastAPI from vLLM inference via `httpx.AsyncClient` **(vLLM backend only)** | High | High | `app.py`, `engine_vllm.py` |
| **MED-1** | Wavesurfer peaks: confirm caching — peaks now computed server-side via `asyncio.to_thread`, verify results are cached rather than recomputed per playback | Medium | Low-Med | `audio_processing.py`, `app_generation.py` |
| **MED-2** | Optimize `engine_vllm.py` parameters (`max_model_len`, `tensor_parallel_size`, …) **(vLLM backend only)** | High | Medium | `engine_vllm.py` |

**Why GEN-1:** blocking work is already off the event loop (`asyncio.to_thread`), but `inference_lock` is still held across WAV encode + peak computation, so concurrent generations serialize on non-inference work.
**Why GEN-2:** 22 of 24 routes return untyped `dict` — no response contract for clients and thin generated OpenAPI schema; structural debt flagged in the 2026-07 e2e review.
**Scope note (HIGH-1/HIGH-2/MED-2):** these apply only to the optional **vLLM backend** (Linux/datacenter). Default deployments (MLX on Apple Silicon, torch elsewhere) are unaffected — prioritize only if vLLM is in use.

**Acceptance criteria (test-first):**
- **GEN-1:** a test asserting two concurrent `/generate-stream` requests for different voices interleave (lock released during encode), rather than fully serializing.
- **GEN-2:** each newly-typed route has a test asserting the JSON response validates against its Pydantic model; `app.openapi()` generates with no warnings.
- **HIGH-1 / MED-2:** a Docker-vLLM smoke test asserting the server starts with the new params and `/health` returns 200.
- **HIGH-2:** a test asserting the request handler does not block the event loop during a mocked vLLM call.
- **MED-1:** a test asserting peak computation runs at most once per audio asset (cache hit on second request).

---

## Priority 2: Enhancements (Open)

| ID | Task | Impact | Effort | Files |
|----|------|--------|--------|-------|
| **FOLLOWUP-1** | Decide `config.json` defaults for `design.load_at_startup` and `custom.load_at_startup` (currently `false`/`false`; only `clone` is `true`). Trade-off: ~2.5 GB → ~7.5 GB startup memory vs always-available Design/Custom voices. | Low | Low | `config.json` |

**Acceptance:** decision recorded here with rationale; if flipped to `true`, update `config.json`, CLAUDE.md "Key Settings", and add a test asserting startup loads the expected models.

---

## Priority 3: Future / Upstream-Dependent

| ID | Task | Blocker | Effort |
|----|------|---------|--------|
| **R-28** | Speculative decoding (1.5-3x speedup; 0.6B as draft for 1.7B). See `2026-03-23-speculative-decoding-research.md`. Phase 1 = monitor upstream. | Upstream library support | High |
| **FUTURE-1** | Entropy-based hallucination monitoring | vLLM forward-pass modification | High |
| **FUTURE-2** | GFlowNet distribution alignment | Research integration | High |
| **FUTURE-3** | Adaptive attention head deactivation | Per-model profiling | High |

---

## CI & Quality Debt (from 2026-07-01 e2e review — Open)

Long tail tracked in `docs/reviews/e2e-review-2026-07-01.md`. Highest-value items:

- **CI gates:** only batches 1–3 are gated; add batches 4 + 5 + `tests/security/*` + `tests/evaluations/*`. Add `ruff` / `mypy` / `bandit` to CI (currently local-only). Add `--cov-fail-under=80`.
- **Docker:** `docker-compose.yml` vLLM build `context: ..` → `.`; reconcile `Dockerfile.vllm` vs `docker/vllm.Dockerfile`.
- **MLX model-config matrix:** run 1.7B + 0.6B across bf16 / 8bit / 4bit with smoke generation + `/health` `model_size`/`mlx_quantization` verification.
- **Manage-Models table refresh latency:** load/unload handlers must re-emit table update (test_09/10 issue).
- **Structural debt:** `config.py` 1432 lines, `_facade.py` 1293, `inference.py` 1097, `generate.py` 864; `handle_generate` 395 lines; broad `except Exception` at `inference.py:627,637`; f-string logging; duplicate logger `engine_vllm.py:34/76`; unguarded file handle `engine_vllm.py:268`.
- **LOW cleanups:** dead `# nosec` (`generate.py:538`, `generate_interactive.py:346`, `generate_server.py:336`, `_facade.py:114`, `shared.py:556-559`); dead `return` after `_error_response()`; audit-log WS auth failures (`websocket.py:51`); `_sanitize_error` regex over-strip; name-length mismatch 128 vs 255 (`config.py:592` vs `validation.py:19`).

---

## Recommended Execution Order

### Sprint 1 — Default-backend quick wins (low risk, no vLLM needed)
1. **GEN-2:** `response_model` contracts (incremental; improves OpenAPI + client contracts)
2. **FOLLOWUP-1:** decide `load_at_startup` defaults (config-only)
3. **MED-1:** verify/cache wavesurfer peaks

### Sprint 2 — Concurrency refinement
4. **GEN-1:** release `inference_lock` during encode/peaks (pair with a concurrency test)

### Sprint 3 — vLLM backend (only if vLLM is deployed)
5. **HIGH-1 + MED-2:** vLLM params
6. **HIGH-2:** decouple FastAPI from vLLM

---

## Success Criteria

- [x] R-29/R-30 validation gaps closed (2026-07-21)
- [x] Full test suite passes — 2258 tests (2026-07-01 e2e review; maintained baseline)
- [ ] **GEN-1:** concurrent generations interleave during encode (no longer serialized on `inference_lock`)
- [ ] **GEN-2:** ≥80% of routes carry `response_model`
- [ ] **HIGH-1/MED-2:** vLLM params validated in a Docker environment
- [ ] **HIGH-2:** FastAPI decoupled from vLLM — event loop not blocked during generation

---

## Notes

- **E2E Testing:** plan at `docs/plans/e2e-testing-implementation-plan.md`. Phases 1–2 complete (50 tests). Phases 3–4 (UI, cross-browser, a11y) are considered covered by `tests/test_e2e_playwright.py`.
- **AI Regression Prevention:** all implementations follow TDD (red-green-refactor).
- **Git Workflow:** feature branches; never push directly to main without approval.
- **Verification log:** this reconcile pass confirmed every ✅ against `main` @ `6e8ec9d` on 2026-07-21; see `docs/reviews/e2e-review-2026-07-01.md` for source findings.
