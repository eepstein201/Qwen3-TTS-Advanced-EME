# mlx-audio 0.4.8 → 0.5.1 evaluation (dependabot #222) — 2026-09-01

**Question:** does the bump change behavior at our call sites, and does it fix #236 (MLX create-voice-prompt)? The upstream carried the #192 concurrency chain (ml-explore/mlx#3078, Blaizzy/mlx-audio#638/#733), so the risk half gets the churn-probe treatment.

**Verdict: GO for #222 (routine, behaviorally inert at our call sites) — but it does NOT fix #236.** #236's premise is corrected below; it needs its own fix, not this bump.

## Method

- **Sandbox:** overlay venv (`python -m venv --system-site-packages`) on a copy of the working env's interpreter, `pip install mlx-audio==0.5.1` (latest satisfying #222's `>=0.5.0`); only `mlx_audio` is shadowed — mlx stays 0.32.0, mlx-lm 0.32.0/0.31.3 visible. No GB-scale conda clone needed; the working env is untouched.
- **Source diff:** installed 0.4.8 vs 0.5.1 modules we touch.
- **Smoke:** engine-level `load_model("design")` + 3 seeded-schema generations per env, fixed text/params.
- **Suite:** full `pytest -m "not e2e"` in the sandbox.
- **Churn probe:** `scripts/probe_issue192.py --churn` (3 generators × 4 requests, unload/load cycling) against a live server per version, same day, same machine.

## Findings

### 1. The qwen3 model layer is behaviorally inert 0.4.8 → 0.5.1

The module we exercise (`mlx_audio/tts/models/qwen3/qwen3.py`) diffs to **exactly 10 lines — pure import surgery** (`from mlx_lm.generate import stream_generate` → `from mlx_audio.lm.generate import ...`, i.e. #880's vendoring). Inside the vendored LM package: the sampler's one change is a benign mask fix (`False` → `mx.zeros(..., dtype=mx.bool_)`), the qwen3 LM model is identical modulo a comment, and `generate.py` is a **trimmed vendored copy** (14 functions vs mlx-lm 0.31.3's 63 — a reduction, not a new vintage). `mlx_audio/tts/utils.py` (our `load_model` entry) and `utils.py` are byte-identical.

### 2. #236 is NOT a version gap — the bump cannot fix it

`model.create_voice_clone_prompt` exists in **no** mlx-audio version (verified 0.4.8 locally, 0.5.1 in sandbox, upstream master via the qwen3 source). Upstream does cloning at generation time: `Model.generate(ref_audio=..., ref_text=...)` → `prepare_zeroprompt(ref_audio, ref_text)`. Our `/create-voice-prompt` routes both backends through the torch-shaped `create_voice_prompt` (`inference.py:1752`), so the MLX path throws `AttributeError` on every version. The MLX-native shape is to **skip model inference entirely** — validate the reference audio and store the `.wav+.txt` pair that MLX generation already consumes as `ref_audio`/`ref_text` (with `ensure_min_sample_rate` at write time, exactly as `tools/create_voice.py` and the UI path already do). Correct the issue accordingly.

### 3. Empirical results

| Probe | 0.4.8 (control) | 0.5.1 (sandbox) |
|---|---|---|
| load_model (design, cached) | 7.5 s | 7.2 s identical path |
| generations (fixed text, n=3/env) | 6.56 / 6.40 / 6.96 s audio | 7.52 / 6.6/6.56 / 7.92 s audio |
| byte-identical cross-version pair | 157440 samples | 157440 samples |
| all-finite audio | 3/3 | 3/3 |
| peak memory | 6.86 GB | 6.6–6.86 GB |
| full non-e2e suite | 3051 passed / 17 known-local fails | **3083 passed / 0 failed** (venv even runs the torch-gated tests; the 17 known-local failures pass) |
| churn probe (3×4, unload/load cycling) | **0/12 capped, 0 log warnings, exit 0**; audio min/med/max 5.4/6.2/7.0 s | **0/12 capped, 0 log warnings, exit 0**; audio min/med/max 4.6/7.8/9.4 s |

Both probes also show the PR #235 `model_unloaded` re-read firing live under churn (queued generations cleanly retried instead of orphan-running) — the first same-day A/B of the new 503 against real churn on both dep versions.

## Recommendation

1. **Merge #222** (routine, behaviorally inert at our call sites; churn-clean; full suite green).
2. **Re-scope #236** (comment on the issue + close-as-wontfix or retitle): not a version gap; needs an MLX-native create path (validate + store .wav/.txt, no model inference — `ensure_min_sample_rate` at write time) or an explicit "torch-only" config error.
3. Keep the churn-probe harness as the standing upgrade gate for any future mlx-audio bump.

## Notes

- The sandbox technique (overlay venv, `--system-site-packages`) is the cheap way to A/B a runtime dep without cloning conda envs; only the shadowed package differs.
- Related env drift (out of scope here): the mlx env sits at starlette 1.3.1 vs the declared `>=1.6.0,<2` server floor (covered functionally by the retained body-size middleware; repair opportunistically).
