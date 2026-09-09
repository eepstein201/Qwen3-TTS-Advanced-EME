# TDD Evidence: Step 0G — UI voice-create routes through the engine writer

**Source plan:** docs/plans/2026-09-06-consolidated-backlog-priority.plan.md,
Step 0G (Wave 0).

**Branch:** `fix/ui-voice-create-use-engine-writer` (cut from main @ `649de48`,
the #281 0F squash; a docs-only 1C-decision commit `f81c351` was interleaved
onto the branch mid-task — no runtime effect, RED verified against base
behavior empirically).

## Problem and threat model

`interface/ui/voice_management.py` re-implemented the MLX voice-prompt store
inline (`sf.read`/`sf.write`/`shutil.copy`, the old `:96-170`) instead of
calling `save_voice_prompt_mlx` — making the UI create path the ONLY
voice-prompt write surface in the codebase with none of the engine writer's
guards: no realpath home-or-tempdir containment on the reference source, no
zero-sample refusal, no `.wav`-removal rollback when the `.txt` write fails.
Step 0E made the Gradio UI authenticated-but-shareable (`--share` /
`TTS_UI_SHARE` / Colab force a public `*.gradio.live` URL), so this unguarded
write path is reachable from a shareable UI: anyone handed the link could
point a create request at any readable file path on the host and have it
copied into the prompts dir (and `.txt`-failures orphaned half-written
prompts the MLX loader would then list).

## Survey-confirmed guard gaps (and one REFUTED claim)

| Guard | Engine writer (`voice_prompt.py`) | Old UI branch | Verdict |
|---|---|---|---|
| Source containment | realpath-normalized startswith(home, tempdir), guard inline on the sink variable | raw `audio_path` used directly | missing |
| Zero-sample refusal | raises `UnsupportedReferenceAudioError` | none — wrote an empty prompt | missing |
| >=24 kHz write-time rate | `ensure_min_sample_rate` (mono, never downsample, raise if undeliverable) | present (this half existed at the UI too) | present |
| Transcript strip | `(transcript or "").strip()` | `transcript.strip()` at call site | **plan claim REFUTED** — strip WAS present |
| None-coercion | `(transcript or "")` | `None.strip()` → AttributeError, swallowed by the broad `except` | missing |
| Rollback | `os.remove(wav)` on any store-phase failure | only the missing-transcript branch cleaned up | missing |

The stale comment at `voice_prompt.py:469-478` claimed the UI create path was
guarded in `tabs_generation.py` — verified false (that file's only guard is
the history-replay path); corrected in the GREEN commit.

## Task report

| Task | Scope | RED | GREEN |
|---|---|---|---|
| 1 | delete inline MLX branch; call the writer via the `app_prompts.py` gr.Error mapping shape; cache-clear preserved; 3 inline-pinning tests rewritten; comment/cross-ref collateral; 4 new pins | **4 failed / 25 passed** at base (all four right-reasons below) | **29/29** mlx AND torchless (RUN, not skip) |
| 2 | E2E create-from-audio case (batch 4); gates; this doc; plan + CLAUDE.md paperwork | n/a (integration case over the already-pinned fix) | full non-e2e **3300 passed / 4 skipped** (+1 exactly) |

## The RED run (Task 1, verbatim)

`conda run -n qwen3-tts-mlx python -m pytest tests/test_ui_voice_mgmt.py -v --tb=short`
against the pre-fix tree → **4 failed, 25 passed** (capture:
`.superpowers/sdd/step-0g-execution-quiet-harbor/task-1-red-output.txt`):

```
FAILED tests/test_ui_voice_mgmt.py::TestMlxCreateEngineWriterPins::test_mlx_create_clears_the_engine_prompt_cache - AssertionError: Expected 'clear_voice_prompt_cache' to be called once. Called 0 times.
FAILED tests/test_ui_voice_mgmt.py::TestMlxCreateEngineWriterPins::test_none_transcript_reaches_writer_and_is_coerced - AssertionError: Expected 'save_voice_prompt_mlx' to have been called once. Called 0 times.
FAILED tests/test_ui_voice_mgmt.py::TestMlxCreateEngineWriterPins::test_outside_home_reference_is_refused_and_writes_nothing - AssertionError: Error not raised
FAILED tests/test_ui_voice_mgmt.py::TestMlxCreateEngineWriterPins::test_transcript_write_failure_rolls_back_the_wav - AssertionError: True is not false : no orphan .wav may survive a failed transcript write
========================= 4 failed, 25 passed in 1.85s =========================
```

Right-reasons: (1) containment drove a REAL decodable 24 kHz wav placed
outside both allowed roots (home role re-pointed via `os.path.expanduser`,
tempdir role via `tempfile.gettempdir` — same shape as batch 3's
`TestReferenceSourceContainment`) and the inline branch stored it happily;
(2) rollback planted the `.txt` target as a DIRECTORY (filesystem
truth-injector — a `builtins.open` patch leaked into librosa/numba cache I/O
and passed hollowly) and the orphan `.wav` survived, the swallowed error
visible in the captured log; (3) the None pin failed with "called 0 times" —
see the pin-3 nuance below; (4) the cache pin failed because the inline
branch never cleared the engine cache (only the server path did).

**Pin-3 short-circuit nuance (disclosed):** the brief predicted base would
AttributeError-swell on `transcript=None`; in fact base's `not transcript`
guard fired FIRST, so base raised the friendly gr.Error and the writer was
never reached — the RED reason is writer-never-routed, not AttributeError.
The pin is still correctly shaped: it kills the naive GREEN rewrite
(`transcript.strip()` at the call site), where the AttributeError risk
actually lives.

## GREEN (Task 1, `d17c284`)

- The inline branch is deleted; the MLX path makes a function-local
  `from qwen3_tts.core.engine import UnsupportedReferenceAudioError,
  clear_voice_prompt_cache, save_voice_prompt_mlx` (mirroring
  `app_prompts.py`'s import shape) and calls
  `save_voice_prompt_mlx(base_name, audio_path, None if no_transcript else
  transcript)` — the RAW transcript reaches the writer, which strips; the
  box-wins semantics of `no_transcript=True` are preserved by the None
  conversion.
- gr.Error mapping clause-for-clause with `app_prompts.py`:
  `UnsupportedReferenceAudioError -> gr.Error(str(e))`,
  `ValueError -> gr.Error(str(e))`,
  `sf.LibsndfileError -> gr.Error("Audio could not be decoded …")`,
  `RuntimeError -> gr.Error(f"Failed to create prompt: {e}")`.
- `clear_voice_prompt_cache()` after a successful store — cache parity with
  the server path. This is the ONE deliberate behavior delta: the old inline
  branch never cleared the engine cache, so a create followed by a
  same-process generation could serve a stale prompt list.
- Collateral: dead imports removed; `app_prompts.py:454` cross-reference
  updated; `voice_prompt.py` stale comment corrected; CLAUDE.md app_prompts
  row gains the routing note. Torch branch and tab wiring untouched.
- Test rewrites (not deletes): `test_mlx_creates_wav_and_txt` now pins the
  exact writer call `save_voice_prompt_mlx("new_voice", "/tmp/audio.wav",
  "Hello world")`; `test_mlx_no_transcript_mode` pins the None pass-through;
  `test_mlx_no_transcript_no_flag_raises` pins the friendly error with
  `mock_writer.assert_not_called()`. The 4 new pins live in
  `TestMlxCreateEngineWriterPins` (audio-deps-guarded, RUN in both
  interpreters). Module total 29/29.

## Mutation verification (Task 1)

| Mutation | Caught by |
|---|---|
| conditional raw write keyed on the raw path's tmpdir-ness | SURVIVED — a dead mutant: the pin's own `tempfile.gettempdir` re-point makes the mutant's bypass condition unreachable, so it fell through to the real writer. Informative, not a pin weakness. |
| unconditional raw-path `shutil.copy` + txt write bypassing the writer | **killed by ALL FOUR pins** — containment `Error not raised`, rollback orphan-wav assertion, routing + cache `Called 0 times` |
| (reviewer mutant) cache-clear skipped | containment-adjacent cache pin RED |
| (reviewer mutant) writer's `os.remove` disabled | rollback pin RED |

Restores verified by `git checkout -- <file>` plus grep for the MUTANT marker
(zero hits).

## The E2E case (Task 2, `ee56e8c`)

`tests/test_ui_headless.py::TestUiCreateFromAudioEngineWriter::
test_create_from_audio_routes_through_the_engine_writer` (batch 4) launches
the real UI subprocess with the module's exact launch conventions, connects
gradio_client, and drives `/create_voice_prompt` (the auto-derived api name
of the Voice Management create button — confirmed against
`client.view_api()`) with:

- a 1.0 s mono 24 kHz sine `.wav` written by soundfile into a tmp dir — the
  client UPLOADS it, so the handler sees gradio's staging copy under the
  system tempdir, which is one of the writer's two allowed roots (the
  containment pass at E2E level rides the staging, exactly the surface the
  fix protects);
- a deliberately whitespace-padded transcript, so the `.txt` content
  assertion proves the writer's stripped coercion ran on the stored file.

Assertions: 3-tuple return with status `Created MLX voice prompt: <base>`;
the `.wav`+`.txt` pair exists under `VOICE_PROMPTS_DIR` as the UI resolves
it; `.txt` content == stripped transcript; stored audio is >=24 kHz mono;
both refreshed dropdown payloads list the new prompt. Cleanup is registered
BEFORE the create (addCleanup removes the pair even on failure; the UI
subprocess is terminated first via LIFO).

Degradation: `setUp` skips with a loud reason when the live server on :5123
is unreachable (module convention — verified: `skipped 'live TTS server on
:5123 is down — …'`, `OK (skipped=1)`), when the configured backend is not
mlx, or when gradio_client/numpy/soundfile are absent. It is a
`unittest.TestCase` because the batch runner drives modules via
`python -m unittest`, which does not collect plain functions. The script's
stale generation flow (6R item 14) is untouched — this class is independent
of it (create-from-audio, not generation).

Observed: `1 passed in 8.44s` live (server up, mlx env); inside batch 4 the
line reads `... ok`; nothing new appears in the real `voice_prompts/` after
the full suite + batch runs (37 files before, 37 after, 0 newer).

## Gates (COUNTED, re-run from `dec19d4` — the last commit that can affect them)

The E2E case landed as `ee56e8c`, then a 3-line follow-up (`dec19d4`) closed
the UI subprocess's PIPE streams in the test's cleanup (a ResourceWarning
under the unittest runner). Because that follow-up touched the test file
AFTER the first gate pass, every counted gate below was RE-RUN from
`dec19d4`. The only later commits are docs-only (`f63f336` sync, `a7d5676`
census fix), touching nothing the gates execute; the census row is measured at final HEAD.

| Gate | Command | Result |
|---|---|---|
| Full non-e2e | `conda run -n qwen3-tts-mlx python -m pytest tests/ -m "not e2e" -q --tb=short --ignore=tests/evaluations` | **3300 passed, 4 skipped, 92 deselected, 0 failed** in 118.39 s — exactly +1 over Task 1's 3299 (the E2E case, RUN with server up) |
| Batch 4 runner | `conda run -n qwen3-tts-mlx python tests/run_batches.py --batch 4` | **1/1 batches passed**; `Ran 786 tests … OK (skipped=1)` — the one skip is `test_ui_share_auth`'s pre-existing conftest-fixture-dependency test that skips by design under the unittest runner; the new test's line reads `... ok` |
| `test_ui_voice_mgmt` mlx | `conda run -n qwen3-tts-mlx python -m pytest tests/test_ui_voice_mgmt.py -q` | **29 passed** |
| `test_ui_voice_mgmt` torchless | `./.venv-310/bin/python -m pytest tests/test_ui_voice_mgmt.py -q` | **29 passed, 1 warning** — RUN, not skip |
| ruff | `conda run -n qwen3-tts-mlx ruff check qwen3_tts tests` | All checks passed (exit 0) |
| mypy (entry-point form) | `conda run -n qwen3-tts-mlx mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface` | Success: no issues found in 58 source files (the known pre-existing `annotation-unchecked` notes only) |
| `git diff -w` census | `git diff -w --stat d17c284..HEAD` | 4 files, +457/-6: `tests/test_ui_headless.py` +195/-1 (E2E case + pipe-close follow-up), evidence doc +244, plan +17/-4 (net), CLAUDE.md +1/-1; `config.json` never staged |
| Isolation stat | `find voice_prompts -type f` before/after the batch-4 + full runs | 37 → **37**, 0 files newer than the runs — nothing new in the real dir |

bandit was not re-run (not on this step's gate list): Task 2 touches only a
test module and docs.

## Commit list (in order)

- `6260ae2` — `test(ui): pin containment and rollback on the voice-create path` (Task 1 RED)
- `d17c284` — `fix(ui): route MLX voice-prompt creation through the engine writer` (Task 1 GREEN)
- `ee56e8c` — `test(ui): e2e create-from-audio through the engine-writer path` (Task 2)
- `f3bc1b1` — `docs(testing): UI voice-create engine-writer TDD evidence` — this file
- `570da67` — `docs(plans): record the 0F status-mark and the voice-prompt test landmine` — consolidated plan + CLAUDE.md clause fix
- `dec19d4` — `test(ui): close the e2e UI subprocess pipes on cleanup` — follow-up that
  triggered the from-HEAD gate re-run recorded above
- `f63f336` — `docs(testing): sync the engine-writer evidence with the final HEAD` — this
  file again (numbers/gates re-checked against final HEAD); docs-only, so the gate
  results above remain valid through final HEAD

## Accepted residuals (disclosed)

1. **Real-dir pollution incident + the SECOND landmine.** Task 1's FIRST full
   run (pre-fix test seam) wrote 8 stray artifacts into the REAL
   `voice_prompts/` (`low/low2/high/stereo_disk` `.wav+.txt`, mtime Sep 9
   13:05). The root cause was fixed in the same commit (the sample-rate class
   now patches `voice_prompt.VOICE_PROMPTS_DIR` too), but the reviewer found
   a SECOND, independent landmine that converted those orphans into user-
   visible corruption: `tests/test_engine.py:138-146` calls
   `migrate_orphan_mlx_prompts(clone_model=mock_model)` against the
   UNPATCHED real dir (its docstring assumes "no orphan .wav files in test
   env"), finds any orphan pair, "migrates" it by `torch.save`-ing a prompt
   built from the MagicMock — mid-pickle-corrupted ~668 B `.pt` stubs — and
   swallows the failure at `voice_prompt.py:328-329`. Result: 4 corrupt `.pt`
   stubs (Sep 9 13:09) that show up in `tts voice list` and are corrupt for
   the torch loader. Registered as **6R item 15** (fix = patch
   `VOICE_PROMPTS_DIR` in that test); cleanup of all 12 files is a manual
   user command (`rm voice_prompts/{low,low2,high,stereo_disk}.{wav,txt,pt}`).
   Task 2's own runs add nothing (isolation stat above).
2. **ValueError clause order — cosmetic LOW, deliberate split documented.**
   In `voice_management.py` the `except ValueError` clause (containment
   reject) surfaces the engine's raw message by design — the UI CAN reach
   containment (the server cannot; its uploads stage into TMPDIR), and the
   engine's "must live under the home directory or the system temp
   directory" text is the accurate, actionable message. Residual nuance: a
   NON-LibsndfileError `ValueError` raised during the decode phase would
   also surface raw engine text rather than the friendly invalid_audio
   string; `LibsndfileError` subclasses RuntimeError (not ValueError), so
   the clause order is not itself a correctness bug today.
3. **Pin-3's nuance** (repeated for the record): base never AttributeErrors
   on `transcript=None` — the `not transcript` guard short-circuits first —
   so the None pin's RED reason is writer-never-routed; the pin's value is
   killing the naive `.strip()`-at-call-site rewrite.
4. **The E2E case pins the happy path only.** The guards themselves
   (containment, zero-sample, rollback, cache) remain unit-pinned in
   `TestMlxCreateEngineWriterPins`; a create flow that satisfied the OLD
   inline branch would also satisfy the E2E, because the fixture is a legal
   tempdir upload. The E2E's job is integration health of the routed path
   plus the deliverable "at least one E2E test exists for this flow".
5. **Launch-string duplication.** The E2E class builds its own UI launch
   code instead of sharing `main()`'s, keeping `main()` byte-identical (its
   generation phase is the stale 6R item 14 surface, deliberately untouched
   here).
6. **Batch-4 ResourceWarnings are pre-existing.** The final batch output
   carries 48 ResourceWarning lines (16 actual unclosed-event-loop warnings;
   each prints ~3 matching lines) from modules byte-identical between
   `d17c284` and HEAD (`test_ui_facade` et al.); the new test's own result
   line is clean — the one warning it DID emit (an unclosed PIPE stream) was
   fixed in `dec19d4`, dropping the count from 60 to 48.
