---
name: dependency-triage
description: Triage a batch of Dependabot PRs in this repo — classify routine vs engine-adjacent bumps, gate risky ones behind a two-version churn probe and an evaluation doc, regenerate requirements.lock, verify in the torchless CI proxy, and hand off merge commands. Use when Dependabot PRs have accumulated, when asked to review or merge dependency updates, or after a weekly Dependabot run.
---

# Dependency Triage

Dependabot runs weekly against two ecosystems here (`pip` and `github-actions`), each capped at 10
open PRs. This is the procedure for working that queue down.

**Values below are quoted with their source file. If a quoted value disagrees with the file, the file
wins — update this skill.** Nothing here is validated automatically, so a stale value is silent.

## Step 1 — Enumerate the queue

```bash
gh pr list --label dependencies --json number,title,headRefName
```

Returns `[]` when there is nothing to triage. That is a valid outcome, not an error — stop there.

Both ecosystems apply the `dependencies` label; `github-actions` updates additionally carry `ci`
(source: `.github/dependabot.yml`).

## Step 2 — Classify each PR

Dependabot titles carry the package name, so this is decidable without opening the diff.

**Routine** — a version bump with no engine impact. GitHub Actions bumps are almost always routine.

**Engine-adjacent** — anything touching `mlx-audio`, `transformers`, `torch`, `gradio`, or
`fastapi`/`starlette`. These have broken this repo before and do **not** merge on green CI alone.
They require the Step 3 gate.

## Step 3 — The engine-adjacent gate (two versions, or it proves nothing)

A single probe run proves nothing: it measures whatever the server already had loaded. You need a
control and a candidate, measured the same day on the same machine.

**The mechanism that makes this real:** `--churn` is a pure HTTP client against
`127.0.0.1:5123` — it never imports the bumped package itself. What the *server process* imports is
therefore the only thing that varies, and the server always inherits the interpreter that launched it
(`qwen3_tts/cli_server.py`: `cmd = [sys.executable, "-m", "qwen3_tts.server.app"]`). So the candidate
run must **start the server from the sandbox interpreter**. Installing into a venv and then running
`tts server start` measures the old version twice and looks like a clean A/B result.

### 3a. Build the sandbox (working conda env untouched)

```bash
conda run -n qwen3-tts-mlx python -m venv --system-site-packages /tmp/dep-sandbox
/tmp/dep-sandbox/bin/pip install <package>==<candidate-version>
```

Watch pip's output: if it installs anything beyond the target package, those transitive bumps are part
of what you are measuring too — record them in the evaluation doc, or re-run with `--no-deps` and
resolve deliberately.

Pin the `venv` call to the conda interpreter — `--system-site-packages` only shadows correctly against
the matching site-packages layout. Only the bumped package is shadowed; no GB-scale conda clone
(source: `docs/reviews/mlx-audio-0.5.1-evaluation-2026-09-01.md`, "Method").

### 3b. Prove the two interpreters actually differ — before measuring anything

```bash
conda run -n qwen3-tts-mlx python -c "import <package> as p; print(p.__version__, p.__file__)"
/tmp/dep-sandbox/bin/python      -c "import <package> as p; print(p.__version__, p.__file__)"
```

**If these print the same version, the comparison is void — stop and fix the sandbox.** This is the
check that separates a real A/B from two identical runs wearing different filenames.

### 3c. Stop PM2 first

```bash
pm2 stop tts-server-5123
```

PM2 has `autorestart: true`. Skip this and it respawns the conda-env server under you, or you get a
second daemon fighting for port 5123 — the orphan-server failure from the 2026-09 pass.

### 3d. Measure each version

Neither launch path blocks until the server is ready, so poll before probing (both sides — the race
is identical):

```bash
wait_ready() { until curl -sf -o /dev/null http://127.0.0.1:5123/health; do sleep 1; done; }
```

```bash
# CONTROL — current version, via the normal path
tts server stop && tts server start
wait_ready
conda run -n qwen3-tts-mlx python scripts/probe_issue192.py --churn --out control.json
CONTROL_EXIT=$?                      # capture immediately — see 3e
grep -c "token cap without emitting EOS" .voice_server.log    # scoped to this run
tts server stop

# CANDIDATE — sandbox interpreter, bypassing the tts CLI so the shadowed package is loaded
/tmp/dep-sandbox/bin/python -m qwen3_tts.server.app &
SRV=$!
wait_ready
conda run -n qwen3-tts-mlx python scripts/probe_issue192.py --churn --out candidate.json
CANDIDATE_EXIT=$?                    # capture immediately — see 3e
grep -c "token cap without emitting EOS" .voice_server.log
kill $SRV

echo "control=$CONTROL_EXIT candidate=$CANDIDATE_EXIT"   # both must be 0
```

Defaults are 3 generator threads × 4 requests (`--churn-generators`, `--churn-requests`); `--churn`
ignores `--runs`/`--load-all`. The `conda run` prefix on the probe is the script's own documented
invocation — a bare `python` may resolve to an interpreter without `qwen3_tts`.

**⚠ This drives a live server, cycling `/unload-model` + `/load-model` while generating.** It will
disrupt anyone else using that server. Do not point it at one someone else depends on.

**Restore when done:** `rm -rf /tmp/dep-sandbox && pm2 start tts-server-5123`

### 3e. Read the results properly — the process exiting is not a pass

**Capture `$?` on the line immediately after each probe, as 3d does.** A bare `echo $?` at the end of
the block reports whatever ran last (`kill`, `pm2 start`) — not the probe. Same for the cap-warning
grep: run it per version, not once over a log holding both runs. The probe already tracks its own log
offset internally, which is what backs its exit code; the manual grep is a second pair of eyes only.

Both `CONTROL_EXIT` and `CANDIDATE_EXIT` must be `0` (source: the script's docstring):

| Exit code | Meaning |
|---|---|
| `0` | clean |
| `1` | a run hit the token cap, or the server log shows a cap warning |
| `2` | indeterminate — setup abort or unreadable server log |

```bash
echo $?                                              # must be 0
grep -c "token cap without emitting EOS" .voice_server.log*   # must be 0
```

**Then write the evaluation doc** at `docs/reviews/<dependency>-<version>-evaluation-<YYYY-MM-DD>.md`,
following `docs/reviews/mlx-audio-0.5.1-evaluation-2026-09-01.md`: open with `**Question:**`, state a
bold `**Verdict: GO/NO-GO**` before any detail, list each exact command under `## Method`, and compare
the two runs in a table headed with the real versions, e.g. `| Probe | 0.4.8 (control) | 0.5.1 (sandbox) |`.

## Step 4 — Cross-check the deliberate ignore bands

`.github/dependabot.yml` deliberately blocks these version bands (currently, at lines 13-19):

| Dependency | Blocked band |
|---|---|
| `gradio` | `>=6.15` |
| `transformers` | `>=5` |
| `vllm` | `>=0.9` |

These encode a real dependency knot (see `docs/CODEMAPS/dependencies.md` and the
`transformers`/`huggingface-hub` explanation in `CLAUDE.md`). **A PR proposing a version inside a
blocked band is a config bug to investigate, not a merge candidate.**

## Step 5 — Regenerate the lock when dependencies changed

Command (source: `CLAUDE.md`, "Reproducible installs"; rationale in `docs/CONTRIBUTING.md`):

```bash
python -m piptools compile pyproject.toml --extra test --extra ui --extra dev --output-file requirements.lock
```

**Never install this lock into the platform conda envs** (`qwen3-tts` / `qwen3-tts-mlx`). It pins
`test+ui+dev` for standalone test/CI environments only; `mlx` and `torch` extras are deliberately
excluded because they conflict on `transformers`. In the torch env, transformers 4.57.3 requires
`huggingface-hub<1.0` while the lock's gradio needs `>=1.2.0`, so installing it there breaks
`pip check`.

## Step 6 — Verify in the CI proxy, not your conda env

CI has **no torch**. A pass in `qwen3-tts-mlx` is not evidence CI will be green.

A dependency bump touches no source file, so "the tests for this change" needs a rule:

```bash
# find tests that import the bumped package
grep -rl "import <package>\|from <package>" tests/
```

**A near-empty result does not mean "no tests needed."** Engine dependencies are consumed inside
`qwen3_tts/` modules, not imported by name in tests — only one test file imports `mlx_audio` directly,
yet an mlx-audio bump is exactly the case the prior-art evaluation ran the full suite for.

**Rule: if the package is engine-adjacent per Step 2 (`mlx-audio`, `transformers`, `torch`, `gradio`,
`fastapi`/`starlette`), run the full suite regardless of what the grep returned:**

```bash
.venv-310/bin/python -m pytest -m "not e2e" -q
```

`.venv-310` is the torchless Python 3.10 venv kept as the CI proxy. CI's `coverage` job discovers more
than the batch runner does, so a passing batch is not proof.

## Step 7 — Disposition: verify, hand off, watch

**Never merge or push yourself** (enforced by `.claude/hooks/no-direct-push-main.py`). The cadence,
one PR at a time (source: `docs/plans/2026-09-02-dependabot-triage.plan.md`):

1. `gh pr checks <N>` — verify the full check **list** is green, never just a count.
2. Hand over the exact command for the user to run: `gh pr merge <N> --squash`
3. Watch the resulting `main` run to green: `gh run list --branch main --limit 3`
4. Next PR.

**Red-on-main rule:** if the merge commit's run fails, the sequence **stops**. Cut a `revert/...`
branch reverting the squash commit, hand over the revert merge, and get `main` green before anything
else merges. Re-classify the offending PR as blocked or held.

For a blocked-band or deferred PR: post a comment stating the rationale, then close it — do not
leave it open to be re-triaged from scratch next week.

## What is not checked for you

**No CI job validates that `requirements.lock` is in sync with `pyproject.toml`** — verified by
grepping `.github/workflows/` for `requirements.lock`/`pip-compile`/`piptools`: zero matches. If you
skip Step 5, nothing will tell you.

Vulnerability scanning is covered, just not by `pip-audit`: `.github/workflows/osv-scanner.yml` runs
Google OSV-Scanner (v2.5.1) on pull requests, pushes, merge groups, and a weekly schedule. Don't add
a redundant audit gate without checking what it already covers.

**But nothing is enforced.** `main` has **no branch protection and no rulesets** (`gh api
repos/<owner>/<repo>/branches/main/protection` → 404; `.../rulesets` → `[]`). A red OSV, coverage, or
test check does **not** prevent a merge — that is why Step 7 says to read the full check list
yourself. Nothing external will stop you from merging a broken PR, and nothing stops a merge from
turning `main` red.

## Prior art

`docs/plans/2026-09-02-dependabot-triage.plan.md` is the worked example — the last full pass through
this queue, PR by PR, with the lock regenerations and an orphan-server incident write-up. Read it
when a situation here is ambiguous. (Treat its headline tally as approximate: its own summary and
`MEMORY.md` say "5 closures" where GitHub shows 6 PRs closed-not-merged.)
