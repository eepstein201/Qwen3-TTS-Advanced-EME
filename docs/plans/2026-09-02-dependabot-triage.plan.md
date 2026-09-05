# Plan: Resolve or Intentionally Defer All 11 Open Dependabot PRs

**Date:** 2026-09-02 · **Repo:** Qwen3-TTS_UserFiles · **Baseline:** `main` @ `d210aaa` → re-validated 2026-09-05 against `main` @ `e1a52e2`
**Master-plan hook:** Phase 8 "Dependabot triage" (`~/.claude/plans/review-entire-repo-for-ancient-possum.md:247-252`), re-verified 2026-09-02 against the current PR set.
**Review status:** REVISED after a 4-reviewer adversarial round (ecc:architect + ecc:python-reviewer + ecc:fastapi-reviewer, all FIX-FIRST; santa Reviewer A opus, FAIL). Every finding below is folded; santa round 2 (fresh dual reviewers) gates execution start. See "Review record".

## Re-validation 2026-09-05 (pre-execution drift check — every fact re-swept)

- **PR set unchanged** — same 11 numbers, no Dependabot re-cut. CI re-sweep matches the facts table: 8 PRs fully green, #242 coverage red (run `33638725848` still `completed/cancelled`), #241 + #244 docker-cpu-probe red and still `ResolutionImpossible` (failing-log re-read). All 11 `MERGEABLE` (7 CLEAN; #242/#241/#244 UNSTABLE = checks red but no conflicts). `main` advanced `d210aaa`→`e1a52e2` (4 commits: #248 PM2-aware CLI/UI + its docs, #249 onboarding, #250 CLAUDE.md condense) **without touching `pyproject.toml`, `requirements.lock`, or `.github/`**; every cited line anchor re-confirmed (`app.py:1040`, `ecosystem.config.cjs:12-13`, `test.yml:148/173/216-236`, pyproject lines 9/49/96, all PR "from" floors).
- **anthropic moved: 1.4.0 released 2026-09-04** — `>=1.2.0` now resolves to **1.4.0** (1.3.0 was current at plan time). 1.4.0 `requires_dist` verified 2026-09-05: same shape (httpx2<3,>=2.0.0; anyio<5,>=3.5.0 — env 4.12.1 satisfies; typing-extensions<5,>=4.14 — env 4.15.0 satisfies; Python >=3.10). Facts row + Tier B updated; the dry-run gate's expected install set is unchanged.
- **Server/PM2 DOWN at re-validation** — `pm2 jlist` returns `[]` (even the `tts-server-5123` registration is gone — reboot without resurrect, or `pm2 kill`) and nothing listens on 5123. Execution precondition, not a plan change: **step 0 added to the #223 protocol**.
- Still true, re-confirmed: branch protection absent (404); dependabot.yml ignore-free + weekly (next pip run ~2026-09-09 — execute the ignore-PR before it); torchcodec collection failure still live in the mlx env (collection abort re-produced); lock pins and both envs' installed versions identical to the facts table (incl. hub 1.5.0 vs 0.36.2, httptools 0.7.1, mlx 0.32.0, pytest 9.0.2); #223/#245 both touch `docker/requirements-vllm.txt` + `pyproject.toml`.

## Context

Eleven dependabot PRs are open (5 created in August, 6 re-cut 2026-09-02 13:55). Master-plan non-goal: "No gradio/transformers/vllm major bumps (Phase 8 rationale)". PR #100 root-caused a native crash to the transformers≥5/hub≥1 bump **in the torch env** (torch 2.13.0 + numba 0.64.0 + llvmlite 0.46.0 + hub 0.36.2 — the documented repro). `pyproject.toml:49` forbids raising the gradio floor past ~6.15 for the same env-knot reason.

**Verified facts (all checked 2026-09-02, corrected per review round):**

| Fact | Evidence |
|---|---|
| CI (re-swept 2026-09-05, unchanged): 8 PRs fully green (17 checks each — CodeQL + scan-scheduled skip on PR branches); #244 + #241 each fail only **docker-cpu-probe** (ResolutionImpossible at 1m28s / 2m5s vs 10-11m passes — genuine resolver failure, not flake; same-day `main` run is green); #242 fails only **coverage** (`conclusion: cancelled` at 30m23s == `timeout-minutes: 30`, `test.yml:148`; coverage runs `pytest -m "not e2e"` directly at `test.yml:173`, so the #196 faulthandler bound does NOT protect it — recurrence on re-run is plausible) | `gh pr checks` sweep + job logs |
| docker-cpu-probe builds production `.[torch,server,audio,ui]` via `Dockerfile.cpu-probe:41` with **unpinned ranges (no lock)** — outcome depends on PyPI state at run time; main can go red with zero repo change | `.github/workflows/test.yml:218-236` |
| CI installs only `.[test]` / `.[dev]` — `prompt-enhancer` (anthropic) has **zero CI surface** | `test.yml:66,134,169` |
| Lock pins: gradio==6.20.0, mypy==2.3.0, pytest-asyncio==1.4.0 (**already**), ruff==0.16.2, **`uvicorn[standard]==0.52.1`** (`requirements.lock:297` — uvicorn enters via the `test` extra, `pyproject.toml:96`). mlx and anthropic genuinely absent | lock grep (note: `^uvicorn==` misses the extras spelling — grep `^(uvicorn|uvicorn\[standard\])==`) |
| mlx env (PM2 server env): transformers 5.15.0, gradio 6.20.0, starlette 1.6.0 (repaired today), **uvicorn 0.41.0 and fastapi 0.135.1 — BOTH below pyproject floors** (0.52.1 / 0.141.1), numba 0.64.0 + llvmlite 0.46.0 + torch 2.10.0 installed, hub 1.5.0, anthropic 0.120.2, pytest-asyncio 1.3.0 | `pip list` + dist-info |
| torch env: transformers 4.57.3, gradio 6.8.0, starlette 0.52.1 (drifted), uvicorn 0.41.0, pytest-asyncio 1.3.0, numba 0.64.0, llvmlite 0.46.0 | `pip list` |
| The **real mlx-vs-torch differentiator** for the transformers≥5 crash: huggingface-hub **1.5.0 (mlx, works daily)** vs **0.36.2 (torch, repro env)** — NOT numba presence (both envs have numba 0.64.0) | dist-info both envs |
| uvicorn 0.41.0→0.52.4 is an **11-minor env jump**, not a patch bump: 0.49.0 raises httptools floor to ≥0.8.0 (env has 0.7.1 — `pip install -U uvicorn` alone would leave an undeclared pair); **0.50.0 switches `--ws auto` to `websockets-sansio`** (env currently serves `/ws` on legacy `websockets` impl; `app.py:1040` runs `uvicorn.run` with defaults) and **introduces exit code 3 on startup failure** (`STARTUP_FAILURE = 3`, uvicorn `config.py:80` @ 0.52.4) — interacting with the `_acquire_startup_lock` losing-racer abort × PM2 `autorestart: true, max_restarts: 3` (`ecosystem.config.cjs:12-13`; `start.cjs` propagates the child exit code) | uvicorn release notes 0.43–0.52.4 + installed `uvicorn/protocols/websockets/auto.py` + reviewer web-verification |
| **[RESEARCH-VERIFIED 3-0]** anthropic 1.x (1.0.0 on 2026-08-20 → **1.4.0** released 2026-09-04 — 1.3.0 was current at plan time, re-validated 2026-09-05 with requires_dist confirmed same shape — one minor past the PR's 1.2.0 floor — `>=1.2.0` resolves to 1.4.0 today) replaced `httpx` with `httpx2<3,>=2.0.0` — the single declared BREAKING change — pulling `httpcore2==2.12.0`, `idna>=3.18`, `truststore>=0.10`, `anyio>=4.10`; truststore replaces certifi; **Python floor raised 3.9→3.10** (env is 3.11 ✓). Our exact surface survives per MIGRATION.md (zero removal entries touch `max_tokens`/`system`/`messages.create`/`content[0].text`); removed elsewhere: sampling params `temperature`/`top_p`/`top_k` on message methods (→ TypeError; use `extra_body=` — **our sites pass none**), legacy Text Completions API, non-httpx2 `http_client=` | anthropic-sdk-python v1.0.0 release notes + MIGRATION.md + PyPI requires_dist (deep-research `wf_4fc98880`, 3-0 adversarial votes) |
| The only in-repo anthropic tests are mock/no-key paths (`tests/test_ui_shared_ext.py:69-84` injects a MagicMock via `patch.dict("sys.modules")`; `tests/evaluations/test_llm_judge.py:23-48` never reaches `_call_anthropic`) — they cannot detect a 1.x signature change | test reads |
| anthropic call sites (exact): `qwen3_tts/interface/ui/shared.py:179`, `tests/evaluations/llm_judge.py:81` — `Anthropic(api_key=)` + `messages.create(model=, max_tokens=, system=, messages=)` + `response.content[0].text` | repo grep (2 lazy import sites only; server process has zero) |
| `main` has **no branch protection** (API 404) — a red `coverage` check does not block merging #242, and nothing external stops a serial sequence if a merge turns main red | `gh api` branch-protection check |
| `.github/dependabot.yml` exists, two ecosystems (pip + github-actions), **no ignore rules**, weekly schedule (next pip run ~2026-09-09) | repo read |
| #223 and #245 both touch `docker/requirements-vllm.txt` (different lines) — mergeable, no conflict chain | PR diffs |
| `tests/evaluations/test_speaker_similarity.py` currently fails COLLECTION in the mlx env (torchcodec `libavutil.60.dylib` dlopen) → plain `pytest` aborts there; the held branch `fix/torchcodec-collection-guard` exists to stop exactly this | known state |

## Dispositions — 7 merge · 1 evaluate-then-merge · 3 defer

### Tier A — merge serially (all CI-green today)

Cadence per PR: I re-verify `gh pr checks` green → hand the exact `gh pr merge <N> --squash` command → **you run it** → I watch the merge commit's CI run to green → next. **Red-on-main rule (applies to every merge):** if the merge commit's run fails, the sequence STOPS; I cut a `revert/...` branch reverting the squash commit; you merge the revert; main must be green before anything re-merges; the offending PR is re-classified (Tier C or hold). One-at-a-time is safe: no two PRs share a pyproject line.

| Order | PR | Bump | Why safe | Local action before merge |
|---|---|---|---|---|
| 1 | **#165** | pytest-asyncio `>=0.23`→`>=1.4.0` | Lock already pins 1.4.0; CI has resolved 1.4.0+ for weeks; asyncio_mode is configured NOWHERE (strict mode is the default in both 0.23 and 1.4 — `tests/test_async_test_hygiene.py`'s premise survives unchanged). **[RESEARCH-VERIFIED 3-0]** 1.3.0→1.4.0 carries NO breaking change to strict-mode default, `asyncio_mode` semantics, or the `event_loop` fixture — its changes are one deprecation (overriding the `event_loop_policy` fixture → replaced by the `pytest_asyncio_loop_factories` hook) plus a raised pytest floor of >=8.4.0 (envs run pytest 9.x ✓). **Post-merge both conda envs sit at 1.3.0, formally below the new floor — explicitly accepted posture** (envs already sit below other floors); upgrade opportunistically at the next env maintenance, not in this plan | None |
| 2 | **#243** | ruff `>=0.16.1`→`>=0.16.5` | CI lint leg ran WITH 0.16.5 (PR branch pyproject) and is green | None |
| 3 | **#207** | mypy `>=2.3.0`→`>=2.3.1` | CI mypy leg green on 2.3.1 | None |
| 4 | **#223** | uvicorn `>=0.52.1`→`>=0.52.4` | CI green, and 0.52.4 is itself a /ws fix — but the LOCAL jump is 0.41.0→0.52.4 (see facts). Full protocol below | **Full protocol (below)** |
| 5 | **#240** | mlx `>=0.32.0`→`>=0.32.2` | Patch bump on the MLX platform | `conda run -n qwen3-tts-mlx pip install "mlx==0.32.2"` (pinned — bare `-U` would jump past it) → run `tests/test_voice_engine` batch + ONE live generation smoke; **on smoke failure: `conda run -n qwen3-tts-mlx pip install "mlx==0.32.0"`, no merge command handed** |
| 6 | **#219** | osv-scanner-action 2.3.8→2.5.1 (PR workflow) | CI-only; green | None |
| 7 | **#220** | osv-scanner-action 2.3.8→2.5.1 (reusable) | CI-only; green | None |

**#223 local protocol (executes BEFORE its merge command):**
0. **Start the server** (2026-09-05 re-validation found the PM2 list EMPTY and port 5123 silent): `pm2 start ecosystem.config.cjs` from the repo root (fresh registration — `pm2 resurrect` has nothing to restore), then `pm2 save`, and confirm `/health` 200 before touching any version.
1. **Pre-bump baseline** (attribution record): `/health` 200, one `/generate`, one `/generate-stream`, one `/ws` round-trip on the now-running PM2 server.
2. Upgrade with the extras: `conda run -n qwen3-tts-mlx pip install -U "uvicorn[standard]>=0.52.4"` — expect the **httptools 0.7.1→0.8.x co-upgrade**; verify with `pip check`.
3. `pm2 restart tts-server-5123` → `/health` 200 + clean log.
4. **Live `/ws` smoke**: connect, auth first message, one short generation, terminal frame received, clean close (TestClient ws tests do NOT exercise uvicorn's protocol layer — only this does; 0.50.0 switched the serving implementation).
5. **Live `/generate-stream` smoke** (httptools parses its length-prefixed binary frames).
6. **Startup-lock flap checks** (exit-code-3 × PM2 interaction, `ecosystem.config.cjs:12-13`): (a) *external loser* — with the server up, run a second `tts server start`: it must exit promptly, the winner stays healthy, and the managed process's PM2 restart counter is unchanged; (b) *managed loser* — hold the startup flock externally (a foreground holder of `.voice_server.lock`), then `pm2 restart tts-server-5123`: the managed process becomes the losing racer, exits 3, and PM2's restart counter advances (bounded by `max_restarts: 3`); release the lock and `pm2 restart` once more → healthy again. Both sub-checks must pass — a foreground loser alone never exercises PM2's autorestart logic.
7. If ANY step fails: revert env to 0.41.0 (`conda run -n qwen3-tts-mlx pip install "uvicorn[standard]==0.41.0"` — note the co-upgraded httptools stays at 0.8.x, which satisfies 0.41.0's `>=0.6.3` floor; re-run the /ws smoke after the revert), do not hand the merge command, re-classify #223.

**Lock-regen follow-up (mandatory, CLAUDE.md rule) — after #243 + #207 + #223 (uvicorn is lock-tracked):**
- Branch `chore/lock-regen-dep-bumps` → exactly `python -m piptools compile pyproject.toml --extra test --extra ui --extra dev --output-file requirements.lock` (the CLAUDE.md command, byte-for-byte; **never add `--upgrade`** — whole-tree re-resolution would churn gradio and every transitive).
- **Expected diff is exactly 3 pins:** ruff 0.16.2→0.16.5, mypy 2.3.0→2.3.1, `uvicorn[standard]` 0.52.1→0.52.4. **Any other moved pin = blocker** — investigate before merging. (The lock is consumed by no CI/Docker job — stale-lock is reproducibility drift, not breakage, but it is exactly the drift class the starlette incident taught us to prevent.)
- Local gates → PR → you merge. CI note: `chore/*` branches get ZERO push-triggered CI (push allowlist) — the PR itself still gets its pull_request run; verify with `gh run list --branch chore/lock-regen-dep-bumps` rather than assuming green.

### Tier B — evaluate, then merge: **#242 anthropic `>=0.120.2`→`>=1.2.0`**

Sequence AFTER #223's restart so any failure stays attributable (the install lands httpx2/httpcore2/truststore/idna in the same site-packages the server uses, effective at next restart):
1. Re-run the cancelled `coverage` job once (`gh run rerun 33638725848 --failed` — the actual run id) — it is the 30-minute workflow timeout, not a test failure. If it re-cancelles: proceed anyway is AVAILABLE (no branch protection — a red check does not block the merge), but prefer waiting for a green run; the deciding evidence for anthropic safety is local, since CI has zero prompt-enhancer surface either way.
2. **Dry-run gate:** `conda run -n qwen3-tts-mlx pip install --dry-run -U "anthropic>=1.2.0"` — the "Would install" set must be **anthropic + httpx2 + httpcore2 + truststore + idna** (web-verified as exactly this set: env anyio 4.12.1, typing_extensions 4.15.0, docstring_parser 0.18.0, pydantic, jiter, sniffio all already satisfy; re-checked against 1.4.0's requires_dist 2026-09-05 — same expected set). `anyio` appearing as a mover is tolerated ONLY as an upgrade of the satisfied 4.12.1; **any other shared-env mover = stop and reassess**.
3. Install, then **real-SDK signature check** (the repo's tests mock anthropic entirely, so they prove nothing): `python -c "import anthropic, inspect; client = anthropic.Anthropic(api_key='mock'); print(inspect.signature(client.messages.create))"` — the dummy-key constructor is local-only (no network call); **the unbound class form does NOT work** (`messages` is a `cached_property`, so `Anthropic.messages.create` raises AttributeError — locally verified against 0.120.2). Accept `model`/`max_tokens`/`system`/`messages` kwargs present; plus import smoke of both call sites' module paths. Project floor already compatible: `requires-python = ">=3.10"` (`pyproject.toml:9`) == the anthropic 1.x Python floor.
4. **Live enhancer smoke (mandatory when `ANTHROPIC_API_KEY` is set):** trigger the Design-mode description enhancement once and confirm a sane expanded description. If the key is unset: state explicitly in the merge note that the merge ships on an untested transport (httpx swap changes proxy/timeout/retry semantics).
5. Merge only if 3 (and 4 when possible) pass. **On a failed live smoke: `conda run -n qwen3-tts-mlx pip install "anthropic==0.120.2"` restores the env (httpx2/httpcore2/truststore remain installed and unused — harmless) — do not hand the merge command.** No lock impact, no CI impact.

### Tier C — defer intentionally, close with rationale (3)

**Ordering pinned (both round-2 reviewers): the ignore PR moves to the FRONT of the whole sequence — it executes before Tier A.** Sequence: (i) post the rationale comments on #241/#244/#245 while they are still open (the closure record is ours); (ii) merge `feat/dependabot-ignore-majors` — this closes the re-cut window for the ENTIRE sequence (next pip run ~2026-09-09), and dependabot may auto-close the now-ignored PRs; (iii) close any that remain open manually. This meets Reviewer B's race concern (comments posted before any auto-close) and Reviewer A's window concern (no unignored window at any point of the sequence).

**Repo-change PR 0 (executes before Tier A):** branch `feat/dependabot-ignore-majors` (`feat/*` so the push filter runs CI) → `.github/dependabot.yml`, **pip entry only**, **range-scoped so patch/minor still flow**:
```yaml
# under the existing `pip` package entry:
    ignore:
      - dependency-name: "gradio"
        versions: [">=6.15"]
      - dependency-name: "transformers"
        versions: [">=5"]
      - dependency-name: "vllm"
        versions: [">=0.9"]
```
Local gates → PR → you merge. Then santa-loop dual review gates the push (see Execution design).

| PR | Bump | Closure-comment rationale (final wording) |
|---|---|---|
| **#244** | transformers `>=4.57.3`→`>=5.16.1` (torch extra) | Violates the deliberate torch-env transformers<5 posture — PR #100 root-caused a native crash to the transformers≥5/hub≥1 bump in the torch env (torch 2.13.0 + numba 0.64.0 + llvmlite 0.46.0 + hub 0.36.2). CI agrees: docker-cpu-probe fails ResolutionImpossible on this PR while same-day main is green — the direct resolver conflict is upstream `qwen-tts`'s own pin `transformers==4.57.3` vs this PR's `>=5.16.1`, which independently blocks the bump on top of the posture reason. (No mechanism is asserted beyond the documented repro; the mlx env's working transformers 5.15.0 differs by hub 1.5.0-vs-0.36.2 and is not transferable proof for the torch stack.) Track on #112 upstream watch. |
| **#241** | gradio `!=6.14.*`→`!=6.14.0.dev,>=6.26.0` | (a) torch-env knot: gradio 6.18+ needs huggingface-hub ≥1.2 and 6.26 specifically needs **≥1.16** (PyPI; the #241 CI log shows `gradio 6.26.0 depends on huggingface-hub<2.0,>=1.16.0` vs transformers 4.57.3's `<1.0`) — forbidden in the torch resolve (`pyproject.toml:49`'s own comment says "~6.15", which is imprecise on the exact boundary but right on the conclusion) — docker-cpu-probe fails ResolutionImpossible on this PR while same-day main is green. (b) The rewrite drops the whole-series `!=6.14.*` ban (moot while the floor is ≥6.26, but the ban should be restored verbatim whenever the floor is ever revisited — 6.14.x recurses in the Dataframe frontend). |
| **#245** | vllm `>=0.8`→`>=0.28.0` | 0.8→0.28 is a massive jump against `vllm_client.py`/`engine_vllm.py` written for 0.8-era APIs, with unpinned companion `vllm-omni`, unverifiable on this hardware (no CUDA). Master-plan explicit non-goal. Track on #112. |

**Tier C follow-through notes (santa PR-0 gate, 2026-09-05):**
- **The ignore also mutes *security* PRs inside the ignored ranges** (GitHub's filter is not update-type-specific). Watch the Security tab for gradio/transformers/vllm alerts; remediating inside an ignored range means lifting the ignore + manual bump. Alerts on versions *below* each cutoff still produce security PRs normally.
- **Known churn (informational):** every transformers 4.x patch/minor will still open a PR and still red docker-cpu-probe — upstream `qwen-tts` pins `transformers==4.57.3` exactly, so nothing above it can ever resolve. The `>=5` cutoff deliberately chose majors-only; close-on-sight until #112 moves.
- **Pre-existing, follow-up:** the `dependencies`/`ci` labels referenced by dependabot.yml do not exist in the repo (Dependabot posts label-not-found comments). Tiny chore PR, unrelated to this diff.

### Side finding — owned env repairs (executed, not "optional")

1. **Torch env** (AFTER #223 merges, so the floor is 0.52.4): `conda run -n qwen3-tts pip install -U "starlette>=1.6.0,<2" "uvicorn[standard]>=0.52.4" "fastapi>=0.141.1"` → `pip check` → note-only (server does not run here, so a fastapi co-bump has no attribution concern in THIS env — unlike the mlx env). Owner: this plan.
2. **mlx env fastapi 0.135.1 < 0.141.1 floor** — surfaced by review: the running pair (fastapi 0.135.1 + starlette 1.6.0) is not the lock's validated pair (fastapi==0.141.1 + starlette==1.6.0). **Repair in a SEPARATE later window, never in the same restart as the uvicorn bump** (two transport-layer bumps in one week makes any breakage unattributable). Logged as known drift with an owner: this plan's follow-up list.

## Execution design — team-builder + santa-loop baked in (user-directed)

- **Plan validation (this round):** team-builder panel (architect + python-reviewer + fastapi-reviewer) + santa Reviewer A reviewed the draft; all findings folded. Santa round 2 on the revised plan: **Reviewer A (opus, web-verifying) PASS with zero criticals; Reviewer B (`agy`/gemini-3.1-pro-high) FAIL** → gate NAUGHTY → all round-2 findings folded (flap-check redesign, Tier-C ordering to the sequence front, anyio whitelist tolerance, Tier B rollback command, unbound signature form, #240 pinned install + revert path, PM2 cite fix, arithmetic/date fixes, requires-python verification). **Santa round 3 — fresh Reviewer A + fresh Reviewer B, identical rubric, no memory of prior rounds — must return both-PASS (NICE) before any merge command is handed over.** Either FAIL → fix → fresh reviewers, max 3 rounds total, then escalate to you.
- **Per repo-change PR gate (lock-regen, dependabot.yml):** santa-loop dual review on the branch diff (Reviewer A opus + Reviewer B external, read-only) — NICE before push; then pre-push local gates (ruff, owning batch, `pytest -m "not e2e"`), then you push/merge.
- **Tier B is a merge decision, not a code change** — gated by the dry-run + signature + live-smoke steps above, not by santa.
- **Plan artifact:** at execution start, copy this plan into the repo at `docs/plans/2026-09-02-dependabot-triage.plan.md` (plan-docs-tracked-in-repo) and update `docs/plans/consolidated-roadmap.md` pointers at completion.

## Verification

- Per merge: `gh pr checks <N>` — verify the full check LIST is green (never just a count), then the merge commit's `main` run watched to green (`gh run list --branch main --limit 3`).
- **Red-on-main rule** (above) on any failure.
- After Tier A: full local suite in a **named env** — `python3 -m pytest tests/ -m "not e2e"` (pyenv 3.11.5, full-suite capable). Do NOT run it in the mlx env without `--ignore=tests/evaluations/test_speaker_similarity.py` (torchcodec collection abort) or land `fix/torchcodec-collection-guard` first. Plus `tts server` health smoke (PM2).
- After Tier B: signature check + live enhancer smoke recorded in the merge note.
- **Open-PR trajectory: 11 → 4 after Tier A → 3 after Tier B → 0 after Tier C.** Confirm with `gh pr list --state open` at each boundary.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| A Tier-A merge turns main red (no branch protection — nothing external stops the sequence) | Medium | Red-on-main rule: stop, revert branch, you merge the revert, green before continuing, PR re-classified |
| #223 env jump (0.41.0→0.52.4, WS impl switch) breaks /ws or streaming | Medium | 7-step local protocol incl. live /ws + /generate-stream smokes and flap check; env revert path defined; risk validated BEFORE the merge command exists |
| Version targets move mid-sequence (dependabot re-cuts, as on Aug 29) | Medium | Dispositions are version-independent; re-map if superseded; ignore-PR-first ordering shrinks the window |
| coverage job re-cancelles (workflow timeout, unprotected by the #196 bound) | Medium | Re-run once; merge-not-blocked is available (no branch protection) but prefer green; deciding evidence is local regardless |
| anthropic 1.x transport swap (httpx→httpx2) misbehaves behind the user's proxy setup | Low-Medium | Dry-run gate + signature check + mandatory live smoke when key present; explicit disclaimer if not; sequenced after #223 for attribution |
| Lock regen churns unrelated pins | Low | No `--upgrade`; expected 3-pin diff; any other mover = blocker |
| Serial CI waits drag | High (certainty, not failure) | Wall-clock expectation **2–4 h**, dominated by full main runs (coverage ceiling 30 m) + sibling re-runs |

## Complexity: MEDIUM
2–4 h wall-clock. Zero source-code changes; repo changes are `requirements.lock` + `.github/dependabot.yml` + the plan doc itself, each via feature branches you merge, each santa-gated.

## Review record (2026-09-02)

| Reviewer | Verdict | Key catches folded |
|---|---|---|
| ecc:architect | FIX-FIRST (9 findings) | uvicorn IS lock-tracked → regen scope +3rd pin; main-red rollback path; required-check/branch-protection question; check-LIST discipline over counts; wall-clock 2–4 h; Tier-C ordering |
| ecc:python-reviewer | FIX-FIRST (9 findings) | `uvicorn[standard]==` grep shape (root cause of the lock claim); **numba claim false** → #244 rationale rewritten on the documented repro; no-`--upgrade` + 3-pin expected diff; Tier B tests are mock-paths → real-SDK checks; scoped ignore rules; env floor drift statements |
| ecc:fastapi-reviewer | FIX-FIRST (7 findings) | **mlx env itself drifted (uvicorn 0.41.0, fastapi 0.135.1)**; #223 = 11-minor jump across 0.50.0 WS switch + 0.49.0 httptools floor → live /ws+stream smokes, [standard] extra, flap check, pre-bump baseline; exit-code-3 × PM2 interaction; anthropic httpx→httpx2 dry-run gate; fastapi repair in separate window |
| santa Reviewer A (opus) | FAIL → round 2 pending | All of the above independently + trajectory fix (11→4→3→0), exact line cites (:179/:81), named verification env + torchcodec caveat, Tier-C-ordering pin, red-on-main critical |
| deep-research `wf_4fc98880` (105/108 agents) | clusters 1-2 verified 3-0; clusters 3-6 no surviving web claims (3 verify agents lost to API 429s) | anthropic + pytest-asyncio rows above are research-stamped; **clusters 3-6 (gradio↔hub conflict, transformers numba status, uvicorn release notes, toolchain existence) rest on repo CI evidence — the docker-cpu-probe ResolutionImpossible reds are direct empirical proof of the resolver conflicts — plus the fastapi-reviewer's direct release-notes reads; treat those as reviewer-verified, not independently web-verified** |
| santa round 2 — Reviewer A (opus, general-purpose, full web) | **PASS**, 0 criticals | Independently web-verified ~25 claims: anthropic 1.3.0 requires_dist + MIGRATION.md@v1.3.0, dry-run set is exactly 5 packages, gradio 6.26 needs **hub>=1.16** (#241 CI log shows the exact conflict), uvicorn 0.49 httptools + 0.50 sansio + `STARTUP_FAILURE=3` at source, all lock/env/line claims. Suggestions folded: PM2 cite → `ecosystem.config.cjs:12-13`; #240 pinned install + revert path; Tier B rollback command; ignore-PR before Tier A; unbound signature form; arithmetic/date fixes |
| santa round 2 — Reviewer B (`agy`/gemini-3.1-pro-high) | **FAIL** → gate NAUGHTY → fixes above folded | 4 criticals, validity-checked before folding: dry-run whitelist hardening (anyio — tolerated-mover wording; empirically it will not move, env 4.12.1 satisfies); **PM2 flap check redesigned** (foreground loser alone never exercises autorestart — added managed-loser sub-check); **ignore-merge auto-close race** (comments-first ordering + early merge); requires-python (verified `>=3.10` at `pyproject.toml:9` — already compatible, recorded); 7+6 arithmetic typo |
| santa round 3 — Reviewer B (`agy`/gemini-3.1-pro-high) | **FAIL** — 1 critical | Signature check: unbound class access crashes (`messages` is a `cached_property`) — **locally verified mid-round** (exact AttributeError reproduced against installed 0.120.2; dummy-key form returns all four kwargs) and fixed in Tier B step 3; its suggestion (conda env prefix on #240) folded |
| santa round 3 — Reviewer A (opus, general-purpose, full web) | **FAIL** — 2 criticals | Critical 1 = the same signature defect (independently confirmed at source: `cached_property` in v1.3.0 `_client.py:284`) — already fixed. Critical 2 = two primary-source precision fixes, **folded**: gradio hub boundary corrected (6.15/6.16 need hub>=0.33.5; >=1.2 starts at 6.18; 6.26 needs >=1.16 — `pyproject.toml:49`'s "~6.15" comment is imprecise but right on conclusion); #244's actual resolver conflict named (upstream `qwen-tts` pins `transformers==4.57.3`). Lows folded: torch-env fastapi 0.135.1 added to repair; rollback leftovers stated (httptools 0.8.x survives #223 revert → re-run /ws smoke; httpx2 set survives anthropic revert); coverage run id pinned (33638725848) |
| **SANTA FINAL (3/3 rounds exhausted)** | **ESCALATED TO USER per protocol** | Round 1 A FAIL, round 2 A PASS/B FAIL, round 3 A FAIL/B FAIL. Every finding from every round is folded and (where executable) locally or primary-source verified. Remaining issues: none known — the last round's criticals were both fixed before this gate. Per the santa cap, execution start now requires manual review: **user approval of this plan is that gate** |
| **PR 0 gate (2026-09-05)** — Reviewer A (opus, local-verify + web) + Reviewer B (agy/gemini-3.1-pro-high, web) | **NICE** (A PASS 1M/3L, B PASS 0C) | Security-PR muting note + gradio-comment wording folded into the branch; transformers 4.x churn + missing labels recorded as Tier C follow-through notes; both confirmed: ranges behave as intended (6.14.x flows, 6.15.0+ blocked), actions entry untouched, no wanted PR at risk |
