# TDD Evidence — Phase 0a: hollow tests get real assertions

**Source plan:** `~/.claude/plans/review-entire-repo-for-ancient-possum.md`, Phase 0a
(findings T2 and T15, plus the non-empty `/queue-status` item).
**Branch:** `fix/hollow-tests-real-assertions` · **Base:** `main` @ `e1f3d6b`

## Why the RED gate looks unusual here

Phase 0a adds assertions to tests that already pass. The code under test is
already correct, so a conventional "write failing test → fix code" cycle does not
apply and manufacturing a failure would be dishonest. The plan therefore
specifies a **narrow inverted-assertion RED gate**: for each new assertion,
temporarily invert it (or simulate the defect it is meant to catch), observe the
failure, restore, observe the pass. That proves the assertion is *live* —
executed, reaching real state — which is exactly what the old bodies were not.

This mattered: the first probe found the assertions in the security module would
have been **dead code on an unreachable branch** (below).

## The defect that made these tests hollow

`tests/test_e2e_security_validation.py` gated its only checks behind
`if status in [200, 202]`. Probing the live server showed every payload returns:

```
XSS "<script>alert('xss')</script>" -> {"status": 400, "body": "{'detail': 'prompt_file required for clone mode'}"}
TEMPLATE '{{7*7}}'                  -> {"status": 400, "body": "{'detail': 'prompt_file required for clone mode'}"}
```

Clone mode requires `prompt_file`; without it `/generate` 400s at validation and
never reaches generation. **The 200 branch never executes.** T2 recorded "zero
assertions" — true, but the deeper defect is that the request never exercised
XSS/template handling at all. Scope grew from the planned 2 tests to **4**:
`test_01_sql_injection_prevented`, `test_02_xss_prevention_in_text`,
`test_05_command_injection_prevented`, `test_06_template_injection_prevented`.

Two assertion designs were rejected as unsound, both measured, not guessed:

| Rejected assertion | Why it fails |
|---|---|
| `"49" not in json.dumps(response)` (template evaluated?) | base64 audio contains "49" by chance — measured `True` on a 7.9 MB clone of `{{7*7}}`. Coin flip, not a check. |
| `"49" not in scrubbed_metadata` | the only numeric metadata is `seed`, a random 9-digit int → contains "49" ≈8% of runs. Flaky. |

The sound invariant is **non-reflection of the payload** (deterministic, and the
actual injection surface of a JSON API: a reflected payload in an error `detail`
is what a browser renders), plus **real audio on the accept path**.

## Task report

| Plan task | What was done | Validation command | Result |
|---|---|---|---|
| T2 (security e2e) | Non-reflection asserted on *every* status; `_scrubbed()` strips `audio_base64`/`peaks` before any substring search; 2 new accept-path tests (`test_02b`, `test_06b`) supply a real `prompt_file` so the 200 branch is finally exercised | `pytest tests/test_e2e_security_validation.py -m e2e` | **17 passed** in 28.86s |
| T15 (REPL) | 27 no-assert tests given behavioral assertions; helper now returns a frozen `ReplRun` (transcript + mocks); assertions grounded in *observed* REPL output, not guessed | `pytest tests/test_generate_interactive_ext.py` | **57 passed** |
| T15 (immutability) | `test_fastapi_app_ext3.py` — the two "Immutable update: … is NOT mutated" comments now have `copy.deepcopy` before/after assertions | `pytest tests/test_fastapi_app_ext3.py` | **31 passed** |
| 0a item 4 | `test_queue_status_reports_non_empty_queue` — pins the count (3 → 2 as the queue drains) and `active` independent of depth | `pytest tests/test_fastapi_server.py -k queue_status` | **2 passed** |

## RED evidence (inversions — all reverted)

**1. Non-reflection assertion is live** — inverted `payload not in body` → `payload in body`:

```
E   AssertionError: XSS <script>alert('xss')</script>: payload reflected verbatim in the response body —
    payload="<script>alert('xss')</script>", body={"detail": "prompt_file required for clone mode"}
4 failed
```
Proves the check runs and reaches the real response body on the branch that
actually executes.

**2. Real-audio assertion is live** — inverted `raw[:4] == b"RIFF"` → `b"XXXX"`:

```
E   AssertionError: XSS accept path: audio_base64 is not a WAV container (magic=b'RIFF')
2 failed
```
`magic=b'RIFF'` in the failure message proves real WAV bytes were decoded and
inspected — the accept path genuinely generated audio.

**3. REPL helper is live in every test** — pointed the shared banner assertion at
a string the REPL never prints:

```
E   AssertionError: '=== NEVER PRINTED ===' not found in '\n=== TTS REPL Mode =...
16 failed, 41 passed
```
16 failures = the helper executes in 16 tests.

**4. Immutability assertion detects real mutation** — simulated the handler
mutating the caller's dict:

```
E   AssertionError: {'models': {'clone': {'load_at_startup': True}}} != {'models': {}}
E   + {'models': {}} : /update-startup-config mutated the caller's config dict in place.
2 failed
```

**5. Queue count reads real state** — emptied `pending_requests`:

```
E   AssertionError: /queue-status miscounted the pending queue: {'queue_length': 0, 'active': True}
E   assert 0 == 3
```

**6. One unforced RED** (not an inversion): `mock_alias.assert_called_with("narrator")`
failed against the real signature `get_voice_alias('narrator', {})` — a genuine
wrong-expectation catch, corrected to the observed signature.

All inversions reverted; `grep` confirms zero `RED-GATE` markers remain on disk,
and each module re-verified GREEN afterwards.

## GREEN evidence

```
$ ruff check qwen3_tts tests
All checks passed!

$ mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface
Success: no issues found in 53 source files

$ pytest tests/ -m "not e2e" --ignore=tests/evaluations/test_speaker_similarity.py
2975 passed, 4 skipped, 88 deselected in 34.82s

$ pytest tests/test_batches_coverage.py
3 passed
```

bandit: 7 stale-`nosec` warnings only — byte-identical to the pre-change
baseline (tracked as a Phase 6a cleanup item). No new test modules were created,
so `BATCHES` needs no change; the coverage guard confirms it.

## Coverage and known gaps

- **Runtime cost:** the 2 new accept-path tests each perform a real clone
  generation (~30–80 s on MLX/M2 Pro). They are `-m e2e`, opt-in, and use one
  representative payload per attack class rather than all 4–5 — the code path is
  identical per payload, and 18 real generations would put the module at ~23 min.
  This is a deliberate bound and is stated here rather than left silent.
- **Not fixed here:** 5 no-assert tests remain in `tests/test_generate_helpers.py`
  (`test_colab_skips_playback`, `test_unsupported_platform_warns`,
  `test_missing_player_warns`, `test_colab_prints_path`,
  `test_xdg_open_not_found`). The plan scopes those as a "smaller cluster";
  deferred to keep this PR reviewable.
- **Pre-existing blocker, newly tracked:** `tests/evaluations/test_speaker_similarity.py`
  fails at *collection* (`Could not load libtorchcodec`), which makes
  `pytest tests/ -m "not e2e"` abort with `Interrupted` and run **zero** tests.
  Proven pre-existing by stashing all changes and re-running (byte-identical
  error). Suite evidence above therefore uses `--ignore` for that one module —
  a workaround, not a fix. Now tracked as **P1** in the plan's "Found during
  implementation" section, owned by Phase 0.
