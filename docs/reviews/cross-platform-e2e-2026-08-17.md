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

### D8 — `test_voice_features` poisons librosa for every later module (test, fixed)

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

**Fix applied** (`a76fc9b`): a `_preload_librosa()` helper imports librosa and
touches the lazy attributes the fallback calls, from `setUp` — i.e. *before*
`patch.dict` is entered — so librosa is in the pre-patch snapshot and survives
restoration. The fallback stays genuinely under test. After the fix the poisoned
pair runs 71 tests OK, the container batch runner is **6/6 green**, and macOS
gained a test (2862 → 2863) because the `skipTest` for the numba bug no longer
fires.

---

## Product findings NOT chartered by this plan (reported, not fixed)

### D9 — custom-mode generation runs to the token cap (HIGH, open)

Surfaced by the new `_warn_if_cap_reached()` this branch ships. Twice during the
E2E run:

```
WARNING: Generation hit the 2048-token cap without emitting EOS (mode=custom)
         — the audio is truncated.
INFO: Inference complete: 16 chars, 215.8s, mode=custom [mlx]
```

**A 16-character input took 215.8 s and still never emitted EOS.** This is the
same EOS-failure runaway class the plan addresses for clone references — but in
**custom** mode, which uses no reference audio at all, so the sample-rate fix
cannot be the cause.

This also **re-attributes an E2E failure**: `test_e2e_performance_batch`'s
"Request 0 took 232.3 s (expected < 120 s)" is not (only) machine contention, it
is this runaway. Recorded here rather than fixed: diagnosing why custom mode
fails to emit EOS is a separate investigation, and this plan's charter is the
reference-rate path. The warning that exposes it ships in `d4f34c9`.

### D10 — clone output length tracks the reference transcript, not the request (MED, open)

`tts "Local clone path check." -m clone -p lt1_24k` (23 characters) produced
**42.48 s** of audio; the same text via the server produced 16.56 s. The
reference `lt1_24k` is 41.34 s of audio with a 610-character transcript — so the
output duration matches the *reference*, not the request.

That is consistent with the known ICL echo behaviour (PRF-8 / upstream #341):
the model re-speaks the reference transcript before the requested text.
`trim_icl_echo` defaults on but **only fires when ASR is already loaded**, and
ASR was not loaded here, so nothing trimmed it. Consistent with, but not proven
by, transcription — I did not ASR-verify the content, so this is circumstantial
(duration match + documented mechanism), and is reported at that confidence.

Not a regression from this branch, and `lt1_24k` is at the correct 24 kHz, so it
is independent of the rate fix.

### Fixture and CLI notes (not defects in the product)

- The plan's `tts batch` fixture was wrong: `load_batch_file` returns the raw
  JSON array and expects **plain text strings**, with mode/speaker supplied as
  CLI flags — not per-item objects. The plan predicted this class ("that would
  be a fixture bug, not a product bug").
- The plan's `tts dialogue` fixture was likewise wrong: a `speakers` map
  declaring each speaker's mode is required, otherwise every line falls back to
  `mode=clone` and the default clone prompt.
- `tts generate --help` advertises `--max-new-tokens`, `--top-k`, `--top-p` etc.,
  but the bare `tts <text>` and `tts generate` paths both dispatch to an
  **argparse** parser that rejects `--max-new-tokens` as an unrecognised
  argument. The Click help and the parser that actually runs disagree. Minor,
  pre-existing, out of scope — recorded because it cost time here.
- `-o` rejects absolute paths by design (path-traversal guard); output is always
  resolved under `config.output_directory`.

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
| CLI read-only sweep (12 commands) | **all pass**; `tts doctor` → "All checks passed" |
| CLI generation, all modes + both routing paths | **5/5 produce real audio** (clone local, clone via server, design, custom, seeded) |
| Artifact assertions | every file mono, 24000 Hz, non-silent (peak 0.42–0.83), duration > 0.5 s |
| batch / SRT / dialogue | **all 3 pass** after correcting the plan's fixtures; 5/5 artifacts assert real audio |
| Voice lifecycle | create / info / rename / delete all work |
| **F2+F7 regression proof** | an 8 kHz source through `tts voice create` landed on disk at **24000 Hz, mono, 12.0 s** — duration preserved. CLI printed "Reference audio upsampled 8000Hz -> 24000Hz" |
| **F3 proof in a real generation** | `tts … -p lt1` emitted the warning naming both rates and stating `rebuild` will not fix it |
| Server lifecycle | `restart` → new PID, clone reloaded; `status`, `log`, `stop` all work |
| Gradio UI browser smoke | **NOT RUN** — see outstanding |

#### E2E failures — re-run on a quiesced machine, then attributed

The first E2E pass ran while Docker was concurrently building images and running
full suites on the same 12-core machine — exactly the contention the plan warns
about. All four were re-run with nothing else running:

| Test | First pass | Quiesced re-run | Verdict |
|---|---|---|---|
| `test_e2e_performance_batch::test_01` | FAIL (232.3 s vs 120 s) | **PASS** | contention (compounded by D9) |
| `test_e2e_performance_stress::test_01` | FAIL (timeout) | **PASS** | contention |
| `test_ai_regression::…[clone]` | FAIL | **FAIL** | **pre-existing** |
| `test_e2e_history_clear_copy::test_03` | FAIL | **FAIL** | **pre-existing** |

The two that still fail were then **proved pre-existing rather than reasoned
about**: `HEAD` was detached to `origin/main` (`eb8fa8e`), the server restarted on
that pristine code, and both reproduced identically —
`Failed: Request to /generate failed: timed out` and
`playwright TimeoutError: Page.wait_for_function: Timeout 10000ms exceeded`.
The branch was then restored. Neither is caused by this work, and neither is
fixed by it.

Original first-pass detail:

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
| Batch runner | **6/6 batches pass**, `SCRIPT_EXIT=0` (after D8) |
| F6 fixed in container | `test_install_script.py` **7 passed** — previously 4 failed with `FileNotFoundError` |
| Config reconciliation on Linux | `get_default_config()`, `config.json` and `install.sh` all agree on `auto`; `default_clone_prompt is None` |
| Sample-rate guarantee on Linux | 8000 → 24000 Hz with duration preserved; 48 kHz passed through untouched; stereo reduced to mono |
| Isolation contract (verified, not assumed) | uid **1000**, network **isolated** (`pypi.org` unresolvable), `CapEff=0000000000000000`, `NoNewPrivs=1`, voice_prompts tmpfs writable |
| Checkout not mutated | confirmed — the only tracked diffs were my own uncommitted files plus `.claude/.mcp.json`, a known E2E side-effect, since restored |

---

## Task S — cross-family adversarial gate (`agy`)

Reviewers: **B** `gemini-3.1-pro-high`, **C** `gpt-oss-120b-medium`. Same
byte-identical payload (the code diff, docs/plan excluded — 96–104 KB, under the
200 KB split threshold), two independent invocations, fresh conversation each
round. Claude-family reviewers deliberately not used.

| Round | B | C | Outcome |
|---|---|---|---|
| 0 | FAIL | FAIL | **2 real defects fixed** |
| 1 | FAIL | PASS (3 suggestions) | **2 real defects fixed**, 2 declined |
| 2 | FAIL | PASS (0 issues) | 0 new real defects |

### Round 0 — both reviewers, independently

- **Unreadable upload was copied unchecked.** Both flagged that the `sf.read`
  failure path logged a warning and byte-copied the file, shipping exactly the
  unverified prompt the code exists to prevent. **Fixed** (`d3a82bd`): now a
  `gr.Error`, and nothing is written.
- **Stereo never reached disk as mono** (B only, and it was right).
  `ensure_min_sample_rate` downmixed in memory but returned
  `was_resampled=False`; both callers use that flag to choose between writing
  the array and byte-copying the original, so a 48 kHz stereo upload landed on
  disk **still stereo**. My own unit test asserted the returned array, not the
  file — a hollow green. **Verified by reproducing it** (`ON DISK: 2 channels`),
  then **fixed**: the third return value now means `was_modified` (resampled OR
  downmixed), with an artifact-level test asserting the file's channel count.

### Round 1 — B only

- **`test_copies_every_repo_root_file_a_test_opens` was hollow.** It used a bare
  substring match, and the comment above the `COPY` names the same files — so it
  stayed green with `install.sh` deleted from the `COPY`, the exact regression it
  guards. **Verified empirically**, then **fixed** (`92d0d7e`) with an anchored
  `(?m)^COPY…` regex; confirmed the new assertion returns False on the broken file.
- **Misleading message**: a stereo downmix printed "upsampled 48000Hz -> 48000Hz".
  **Fixed** — the resample and downmix cases now print distinctly.

### Consciously declined, with reasons

- **"`load_voice_prompt_mlx` should raise, not warn"** (B, rounds 1 and 2).
  Declined: the plan chose advisory deliberately, and
  `test_load_still_returns_the_prompt` pins it. Raising would hard-fail every
  existing legacy prompt on load with no recourse but re-creation, which is a
  user-facing breaking change well beyond this plan's charter. C raised the same
  point as a *suggestion*, not a critical issue. **This is a genuine design
  question and is escalated to the repo owner rather than decided here.**
- **"`_warn_if_cap_reached` should raise"** (B round 1). Same class; truncated
  audio is still usable output and discarding it is a behaviour change.
- **"The static `install.sh` / Dockerfile tests are hollow proxies; assert the
  built container instead"** (B round 2). Partly addressed rather than declined:
  the container assertion already happens — Task 9a Step 2 runs
  `test_install_script.py` **inside the image** (7 passed). The module is
  documented as text-only so it stays runnable on any platform.
- **"The torch path bypasses the fix"** (B round 2). **Verified false**: the
  torch branch posts to `/create-voice-prompt`, whose handler writes only the
  `.pt` via `torch.save` after `load_audio_for_cloning()` (which resamples); its
  temp `.wav` is a `NamedTemporaryFile` outside `voice_prompts/`. There is no
  `.wav` for MLX to pick up later.

### Gate verdict

**Not unanimous.** C converged to a clean PASS; B holds one consistent design
objection plus one false positive. Per the plan's own rule — stop after three
rounds and escalate rather than loop — the residue is escalated, not silently
accepted. Four real defects were found and fixed by this gate, two of which
(the stereo-to-disk bug and the hollow Docker test) were genuine holes in my own
work that the macOS and Linux suites both passed straight through.

## Outstanding

1. **Task 8 Step 12** — Gradio UI browser smoke on macOS.
2. **Task 9b** amd64 arch-parity (optional, explicitly skippable per the plan).
3. **Task 9c** push + GitHub Actions; **9d** hand `tests/validate_colab.ipynb` to the
   user for a real CUDA runtime. CUDA row stays **UNVERIFIED** until returned.
4. **Task S** cross-family adversarial gate (`agy`), then **Task 10** ship.

Reported but deliberately **not** fixed here (out of charter, see above):
**D9** custom-mode token-cap runaway (HIGH), **D10** clone output tracking the
reference transcript, the `--max-new-tokens` CLI/parser mismatch, and the two
pre-existing E2E failures.

## Verification matrix

| Platform | Method | Status |
|---|---|---|
| macOS M2 Pro (MLX) | full suite + live E2E + CLI sweep + real audio | **verified**, minus the UI browser smoke |
| Linux arm64 (CPU) | hardened Compose: gates, suite, 6/6 batches, isolation contract | **verified** |
| Linux amd64 | buildx/QEMU import + collect-only | not run (optional) |
| Linux CI matrix | GitHub Actions on a `fix/*` branch | not run yet |
| Linux CUDA / Colab | `tests/validate_colab.ipynb`, user-run | **UNVERIFIED — pending user run** |
| Cross-model review | `agy` dual adversarial gate | not run yet |
