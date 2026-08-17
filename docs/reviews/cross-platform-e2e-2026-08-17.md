# Cross-Platform User-Story Verification — 2026-08-17

Branch: `fix/cross-platform-user-story-reconciliation` (base `main` @ `eb8fa8e`)
Plan: `docs/superpowers/plans/2026-08-17-cross-platform-user-story-verification.md`

**Status: IN PROGRESS.** Tasks 0–7 complete and committed. Task 8 (macOS) and
Task 9a (Linux container) largely complete with open items below. Task 9b/9c/9d,
Task S and Task 10 not started.

---

## Environment

| Property | Value |
|---|---|
| Host | macOS 27, Apple M2 Pro |
| Env | conda `qwen3-tts-mlx`, Python 3.11.14, gradio 6.20.0 |
| Backend | MLX, 1.7B, 8-bit |
| Docker | engine 29.7.2, Compose v5.4.0, VM aarch64, 12 CPU, 6.2 GB RAM |
| Container base | `python:3.11-slim-bookworm`, non-root uid 1000 |

A Linux container shares the host kernel: it proves **Linux** behavior only. It
does **not** validate macOS, Windows, or CUDA.

---

## Baseline (Task 0) — established BEFORE any change

| Gate | Result |
|---|---|
| ruff | All checks passed |
| mypy | Success, 53 source files |
| bandit | 0 issues at every severity |
| check-config-docs | OK, 66 keys |
| Full non-E2E suite | **2846 passed, 4 skipped** |
| Container `test_install_script.py` | **4 failed** — `FileNotFoundError: /app/install.sh` (F6, pre-existing) |
| Container full suite | **7 failed, 2828 passed** |

### B1 — pre-existing, out of scope

`tests/evaluations/test_speaker_similarity.py` fails at **collection** on macOS:

```
OSError: Could not load this library: .../torchcodec/libtorchcodec_core4.dylib
  Reason: Library not loaded: @rpath/libavutil.56.dylib
```

torchcodec wants FFmpeg 4's `libavutil.56`; the system has a newer FFmpeg.
Without `--continue-on-collection-errors` this aborts the **entire** suite
(`Interrupted: 1 error during collection`, 0 tests run).

**Proved pre-existing**, not reasoned: all five modified files plus the untracked
test were `git stash push --include-untracked`ed to give a pristine `HEAD`, and
collection reproduced the identical error (`no tests collected, 1 error`). Stash
popped, tree restored. It is an env defect in the conda env, not a product or
test defect. It also makes `run_batches.py --batch 5` red; batch 5 minus that one
module is **73 tests, OK**.

---

## Findings fixed (Tasks 1–7)

| # | Sev | Finding | Commit |
|---|-----|---------|--------|
| F1 | HIGH | `install.sh` heredoc still wrote `"language": "English"`, reverting PR #190 on every fresh install; tracked `config.json` matched it | `b9a50af` |
| F1b | MED | Same heredoc seeded `"default_clone_prompt": "default_clone.pt"`, a file that ships with nothing | `b9a50af` |
| F2 | HIGH | Gradio Create Voice byte-copied uploads with no resampling — the primary GUI path still produced runaway-generation prompts | `7d5133c` |
| F3 | MED | Legacy low-rate prompts loaded silently; `tts voice rebuild` appears to fix a voice it does not fix | `175d4b5` |
| F7 | HIGH | `ensure_min_sample_rate` logged and returned `was_resampled=False` when librosa was unavailable, and the caller then wrote the poisonous file | `bc4545c` |
| F8 | MED | Mono reduction sat after the rate early-return, so 48 kHz stereo reached the model as two channels | `bc4545c` |
| F6 | MED | `Dockerfile.test` never copied `install.sh`; the module had been erroring in every container run | `87ff5fc` |
| F4 | LOW | MLX clamps `max_new_tokens` to 4096 despite the schema's `le=8192` — documented | `296cfb2` |
| F5 | LOW | Colab notebook never wrote `config['language']`, inheriting the stale `English` | `e105188` |

Each product fix landed test-first: RED reproducer committed separately, then the
fix. Every RED failed for the predicted reason before being made green.

---

## Defects found DURING execution (not in the plan)

These were found by running the plan's own artifacts, and are the substantive
result of the verification pass.

### D1 — `LibsndfileError` subclasses `RuntimeError` (product, fixed)

Task 3b makes `ensure_min_sample_rate` raise `RuntimeError` to refuse an
undeliverable guarantee. The plan's UI caller wrapped `sf.read()` and
`ensure_min_sample_rate()` in **one** try block, catching `RuntimeError` →
`gr.Error`. But `soundfile.LibsndfileError` **is a subclass of `RuntimeError`**,
so any undecodable upload became a hard refusal instead of the intended
warn-and-byte-copy. Caught by two pre-existing tests
(`test_ui_voice_mgmt.py::TestCreateVoicePromptUI`). Fixed by splitting the two
calls into separate handlers with a comment recording why they must never share
one. Commit `bc4545c`.

### D2 — new test leaked cache state into a pre-existing test (test, fixed)

`load_voice_prompt_mlx` caches by prompt **name** in a module-level LRU. The new
`TestLegacyLowRatePromptWarnsOnLoad` loaded a prompt named `legacy` and never
cleared it, so `test_voice_prompts.py::test_load_voice_prompt_mlx_pt_only_error`
later got a cache hit instead of its expected `FileNotFoundError` — passing alone,
failing in the full suite. Fixed with `addCleanup(vp.clear_voice_prompt_cache)`.
Commit `bc4545c`.

### D3 — container `gates` service was a hollow green (harness, fixed)

The plan's `gates` service ran `ruff` with `set -x` but **no `set -e`**. `ruff`
was not in the image at all (`Dockerfile.test` installed only `.[test]`; ruff,
mypy and bandit live in the `dev` extra), so the service printed
`ruff: command not found` and **exited 0** — reporting a passing static gate
having checked nothing. Fixed with `set -ex` plus the `dev` extra.

### D4 — tmpfs over `/home/testuser` masked the installed packages (harness, fixed)

The plan mounted `tmpfs` at `/home/testuser`, but the image pip-installs to
`/home/testuser/.local`. The tmpfs masked site-packages, so the suite service
died with `No module named pytest` from an image that demonstrably has pytest.

### D5 — `.ruff.toml` was never copied into the image (harness, fixed)

With ruff finally present, it ran on its **default** ruleset and reported
**780 errors** that do not reproduce on the host. A missing tool config does not
error, it silently changes the ruleset. Fixed by copying `.ruff.toml`, and guarded
by a new drift test so the whole class stays closed. `Dockerfile.test` and
`docker-compose.test.yml` are now copied too — `test_docker_config.py` opens
`Dockerfile.test`, which was the same F6 bug applied to its own image.

### D6 — `read_only: true` manufactured 42 false failures (harness, fixed)

Attribution, measured three ways rather than assumed:

| Image | Config | Result |
|---|---|---|
| `qwen3-tts:baseline` (pre-change) | plain `docker run` | 7 failed, 2828 passed |
| `qwen3-tts:test` (post-change) | plain `docker run` | 20 failed, 2832 passed |
| `qwen3-tts:test` | hardened compose | **62 failed**, 2788 passed |

So ~42 failures came from the hardening itself, not from any code change: the
CLI batch/SRT/dialogue paths and the Gradio facade legitimately write output
files, and ruff writes a cache. `read_only` was dropped. Its stated purpose —
"a run cannot mutate your checkout" — is already guaranteed by `Dockerfile.test`
COPYing the source in, since there is no bind mount. A hardening flag that
manufactures false failures makes the Linux signal less trustworthy, not more.
`network_mode: none`, `cap_drop: ALL`, `no-new-privileges`, `pids_limit` and the
non-root uid are all retained.

### D7 — `librosa` and `rubberband-cli` were missing from the test image (harness, fixed)

17 of the 20 unhardened failures were `ModuleNotFoundError: No module named
'librosa'`. Without it the reference-audio rate guarantee — the entire reason
this image exists for Task 9a Step 4 — went unverified on Linux. Adding the
`audio` extra then surfaced 3 more: `pyrubberband` shells out to the
`rubberband-cli` **binary**, which was not apt-installed. Both fixed.

### D8 — `test_voice_features` poisons librosa for every later module (test, OPEN)

**Container batch 2 is red because of this. Not yet fixed.**

`tests/test_voice_features.py` uses `patch.dict('sys.modules', {'pyrubberband':
None})` to force the librosa fallback. `patch.dict` snapshots `sys.modules` on
entry and **restores** it on exit — and librosa is imported for the first time
*inside* that context, so every librosa and numba module it pulled in is
**evicted** on exit. The next `import librosa` in the process then re-registers a
native numba extension and dies:

```
ImportError: cannot load module more than once per process
RuntimeError: Failed to resample 16000 Hz reference audio to 24000 Hz:
  cannot load module more than once per process. Refusing to write a
  below-native-rate reference, which would make clone generation fail to stop.
```

This is the numba/LLVM duplicate-registration hazard already known to this repo
(see the PR #100 investigation). `test_voice_features` guards *itself* with a
`skipTest` (lines 50, 70) but the poison leaks into every module that runs after
it in the same process.

Bisected precisely:

| Preceding module + `test_create_voice_functions` | Result |
|---|---|
| `test_voice_engine` | OK |
| `test_voice_prompts` | OK |
| `test_voice_generation` | OK |
| **`test_voice_features`** | **FAILED (errors=9)** |

`test_create_voice_functions` alone: **35 tests, OK**. Full `pytest -m "not e2e"`
in the container: **2857 passed, 0 failed** — pytest's import order never leaves
librosa evicted. Only the unittest batch runner hits it.

Not a regression from this branch: the old code's `except ImportError` wrapped
only `import librosa`, not `librosa.resample`, so the same `ImportError` would
have propagated. It became *reachable* only because librosa is now installed in
the image (D7). The new error message is doing its job — it names the failure
loudly instead of silently writing a bad reference.

**Proposed fix (not yet applied):** import librosa in `test_voice_features`
*before* entering `patch.dict`, so it is present in the pre-patch snapshot and
survives restoration. This keeps the fallback under test while removing the
eviction.

---

## Results

### macOS / M2 Pro (Task 8)

| Step | Result |
|---|---|
| ruff / mypy / bandit / check-config-docs | **green** |
| Full non-E2E suite | **2862 passed, 4 skipped** (baseline 2846 + exactly the 16 tests added) + B1 |
| Batches 1–4 | **pass** |
| Batch 5 | red **only** via B1; 73 tests OK without that module |
| Server freshness | started 12:57:39 after all edits; `/health` ok, clone loaded in 6.0 s; `openapi.json` shows `max_new_tokens` `maximum: 8192` |
| Legacy-prompt warning on the real reproducer | **passes** — exactly one warning naming `lt1`, `8000`, `24000`; silent for `lt1_24k` |
| Full E2E (`pytest -m e2e`) | **78 passed, 4 failed, 4 skipped** in 25m33s |

#### E2E failures — all latency, machine NOT quiesced

| Test | Failure |
|---|---|
| `test_ai_regression::test_all_backends_return_same_response_shape[clone]` | `/generate` timed out |
| `test_e2e_history_clear_copy::test_03_clear_all_two_step_with_visible_status` | Playwright 10 s `wait_for_function` timeout |
| `test_e2e_performance_batch::test_01_concurrent_generations_performance` | Request 0 took **232.3 s** (budget 120 s) |
| `test_e2e_performance_stress::test_01_server_handles_high_concurrent_load` | Request 1 timed out |

These ran while Docker was concurrently building images and running full suites
on the same 12-core machine — exactly the contention the plan warns about
("Task 8 needs a quiesced machine"). Per-inference timings in `.voice_server.log`
were healthy throughout (7–13 s for 15–33 char inputs), so this is contention,
not the degraded-server pathology. **Must be re-run on an idle machine before
these can be called pass or fail.** Not yet done.

### Linux arm64 container (Task 9a)

| Step | Result |
|---|---|
| Compose model | valid |
| Static gates | **pass** (`ruff` clean, config-docs OK) — genuinely, after D3/D5 |
| Full non-E2E suite | **2857 passed, 19 skipped, 0 failed** |
| Batch runner | **RED — batch 2**, via D8 above |
| F6 fixed in container | `test_install_script.py` now runs instead of erroring |

---

## Outstanding

1. **D8** — fix `test_voice_features` sys.modules eviction; re-run container batches.
2. **Task 8 E2E re-run** on a quiesced machine (4 latency failures).
3. **Task 8** remaining steps: CLI sweep, artifact assertions, batch/SRT/dialogue,
   voice lifecycle + F2/F3 regression proof, Gradio UI browser smoke, server lifecycle.
4. **Task 9a** Steps 3–6: config + sample-rate assertions inside Linux, isolation
   contract check (adjust for dropped `read_only`), checkout-not-mutated check.
5. **Task 9b** amd64 arch-parity (optional, skippable).
6. **Task 9c** push + GitHub Actions; **9d** hand `tests/validate_colab.ipynb` to the
   user for a real CUDA runtime. CUDA row stays **UNVERIFIED** until returned.
7. **Task S** cross-family adversarial gate (`agy`), then **Task 10** ship.
