# mlx-audio 0.5.1 → 0.5.3 evaluation (dependabot #302) — 2026-09-22

**Question:** does the 0.5.1 → 0.5.3 bump change behavior at our call sites or revive the
#192 concurrency/cap failure mode? Engine-adjacent per the dependency-triage skill, so it
gets the two-version churn gate, not green-CI-alone.

**Verdict: GO for #302** — behaviorally inert at our call sites (the qwen3 generation path
is byte-identical), churn-clean A/B, full suite green under the candidate.

## Method

Working tree: `af7065c6` (post PR A squash) for BOTH runs — code state cancels out; only
the server's interpreter differs. Server restored to PM2/conda (0.5.1) after the gate.

```bash
# 3a — sandbox (single package, no transitives)
conda run -n qwen3-tts-mlx python -m venv --system-site-packages /tmp/dep-sandbox
/tmp/dep-sandbox/bin/pip install "mlx-audio==0.5.3"   # -> Successfully installed mlx-audio-0.5.3 (only)

# 3b — divergence proof (must differ BEFORE measuring)
conda run -n qwen3-tts-mlx python -c "import mlx_audio, importlib.metadata as im; print(im.version('mlx-audio'), mlx_audio.__file__)"
# 0.5.1 /Users/ericepstein/miniforge3/envs/qwen3-tts-mlx/.../mlx_audio/__init__.py
/tmp/dep-sandbox/bin/python -c "... same ..."
# 0.5.3 /tmp/dep-sandbox/.../mlx_audio/__init__.py
# (mlx_audio exposes no __version__ — dist-info is the authoritative check)

# 3c — PM2 stopped for the whole gate (was already stopped on arrival; verified pid 0)
# 3d — CONTROL (conda 0.5.1 via the normal tts path), then CANDIDATE (sandbox interpreter):
/tmp/dep-sandbox/bin/python -m qwen3_tts.server.app    # port owner verified: pid = sandbox python
# probe for BOTH sides (absolute env python — see Notes on the nohup+conda trap):
/Users/ericepstein/miniforge3/envs/qwen3-tts-mlx/bin/python scripts/probe_issue192.py \
    --churn --out /tmp/dep-eval/{control,candidate}.json
# per-run scoped cap grep: tail -n +<line-at-run-start> .voice_server.log | grep -c "token cap without emitting EOS"

# Step 6 — full suites
/tmp/dep-sandbox/bin/python -m pytest -m "not e2e" -q     # candidate version
.venv-310/bin/python -m pytest -m "not e2e" -q            # torchless CI proxy
```

## Findings

### 1. Source diff: our generation path is untouched

0.5.1 → 0.5.3 changed files (non-pycache): `codec/`, `sts/`, `stt/` (other modalities),
`stt/models/{vibevoice,voxtral,granite_speech5}`, `tts/models/{breeze_tts,omnivoice}`,
`lm/load.py`, `lm/models/{gemma3_text,ssm}`, `convert.py`, `registry.py`, `version.py`,
`utils.py`, plus their tests. **Byte-identical: `tts/models/qwen3/qwen3.py` (the model we
run), `lm/generate` (the vendored generator), `tts/utils.py` (our `load_model` entry).**
The one adjacent change — root `utils.py` (5 lines) — routes model_type detection through
the new `registry.model_type_from_config` helper with the same fallback chain
(architecture → name heuristics); our Qwen3-TTS configs carry explicit `model_type`.

### 2. Churn A/B (same machine, same day, ~30 min apart)

| Probe (3 gen × 4 req, design unload/load cycling) | 0.5.1 (control) | 0.5.3 (sandbox) |
|---|---|---|
| probe exit code | 0 | 0 |
| generations 200 | 12/12 | 12/12 |
| CAPPED generations | **0** | **0** |
| `token cap without emitting EOS` in scoped log window | **0** | **0** |
| non-200s (same kind+count both sides) | 2× 503 `model_unloaded` (designed retryable in-lock guard) + 2× 429 (live 10/min generate limit) | identical |
| churn ops / unload guard | 5 / engaged 0, `custom_unload_slipped` 1 | identical |

### 3. Suites

| Suite | 0.5.1 (torchless `.venv-310`) | 0.5.3 (overlay sandbox) |
|---|---|---|
| full `pytest -m "not e2e"` | 3771 passed / 17 failed | **3812 passed / 0 failed** / 5 skipped |

The 17 `.venv-310` failures are all in `tests/test_voice_prompt_sample_rate.py` +
`tests/test_create_voice_functions.py` — `ModuleNotFoundError: No module named 'librosa'`
(a `server`-extra dep the torchless proxy has never installed; pre-existing local-env
class, documented since the 4B.3 pass). CI's coverage job installs librosa and is green on
byte-identical content (PR #332, all checks). The sandbox side runs the env-gated tests
too, hence the higher count.

## Recommendation

1. **Merge #302.**
2. Post-merge: upgrade the mlx env (`pip install "mlx-audio==0.5.3"`) and
   `tts server restart` so the production server picks it up; the churn harness stays the
   standing gate for every future mlx-audio bump.
3. The lock is unaffected (mlx extras are deliberately excluded from `requirements.lock`).

## Notes

- **nohup + `conda run` trap:** wrapping the probe in `nohup sh -c 'conda run ...'` lost
  the shell's conda initialization and resolved a *different* conda
  (`/opt/homebrew/Caskroom/miniconda`) → `EnvironmentLocationNotFound`, exit 1 that is a
  setup abort, not a measurement. Same class as the known never-`timeout`-a-`conda run`
  rule. Fix: invoke the env python by absolute path inside wrappers. First control run
  discarded on this basis.
- Exit codes were captured by the wrapper's `echo PROBE_EXIT=$?` on the line immediately
  after the probe (a `cmd | tail; echo $?` shape reports tail's exit — bitten once on the
  venv suite echo, which is why the suite numbers above come from re-runs reading pytest's
  own summary lines).
- Readiness polled on `/ready` (503→200), not `/health` — health reports ok while models
  load. Server port ownership re-verified via `lsof`/`ps` before trusting the candidate.
