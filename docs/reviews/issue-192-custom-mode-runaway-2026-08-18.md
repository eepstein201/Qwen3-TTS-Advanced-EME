# Issue #192 — custom-mode token-cap runaway: first measurement pass

**Date:** 2026-08-18
**Status:** open — two hypotheses closed, no reproduction achieved
**Harness:** `scripts/probe_issue192.py`
**Raw rows:** `docs/reviews/data/issue-192-runs-2026-08-18.json`

## The defect

A custom-mode generation ran to the 2048-token cap without emitting EOS,
taking **215.8 s for a 16-character input**, twice, ~6 minutes apart. Healthy
custom-mode inferences on the same machine in the same session took **~7 s**.

At 12.5 Hz, 2048 tokens is ~164 s of audio for an input that should produce
~1.5 s. The user gets minutes of looped or garbage audio behind an HTTP 200.

Found by the `_warn_if_cap_reached()` warning that PR #191 added. Before that
it was entirely silent: mlx-audio's loop is a bare `for step in
range(max_tokens)` with no exhaustion signal, and `_validate_audio()` checks
only clipping and silence.

## The input, identified

The log's `16 chars` is **`"Token test 3"`** from
`tests/test_e2e_security_rate_limiting.py:280`. `_normalize_text()` expands the
digit, so 12 raw characters reach the engine as `"Token test three"` — 16.

Both logged occurrences were this same text, which is what made them look
correlated and motivated testing the text as a cause first.

## Closed: this is not a parameter-forwarding bug

The issue's own suggested step 2 proposed that `_split_mlx_params` might be
mis-setting conditioning for custom mode specifically, since PR #190 made
`max_tokens`/`lang_code` live and custom/design use a different parameter name
than clone. Checked against the installed mlx-audio rather than inferred:

**The kwarg names are correct per mode.**

| Entry point | Language parameter | Our call site |
|---|---|---|
| `generate(...)` (clone) | `lang_code` | `lang_code=_mlx_lang_code(...)` |
| `generate_custom_voice(...)` | `language` | `language=_mlx_lang_code(...)` |
| `generate_voice_design(...)` | `language` | `language=_mlx_lang_code(...)` |

**A misnamed kwarg could not be silent on this path anyway.** Neither
`generate_custom_voice` nor `generate_voice_design` declares `**kwargs`, so a
wrong name raises `TypeError` rather than being swallowed. The silent-swallow
failure mode from #190 is structurally impossible for custom mode.

**Sampling parameters cannot suppress EOS.** `_sample_token` snapshots the EOS
logit *before* top-k/top-p filtering and writes it back *after*:

```python
eos_logit = logits[:, eos_token_id : eos_token_id + 1]
if top_k > 0 and top_k < logits.shape[-1]:
    logits = apply_top_k(logits, top_k)
logits = _apply_probability_filters(logits, top_p, min_p)
if eos_logit is not None:
    logits = mx.put_along_axis(logits, eos_idx, eos_logit, axis=-1)
```

EOS is therefore never filtered out, and it competes against an already
narrowed candidate set — if anything this *raises* its relative probability.
Our `top_p=0.95` (mlx-audio defaults to `1.0`) is not implicated.

**`instruct=""` is not a degenerate prompt.** We pass `instruct or ""` where
mlx-audio defaults to `None`, which looked like a plausible custom-mode-only
malformed-prompt cause. It is not: `_prepare_generation_inputs` guards with
`if instruct:`, so empty string and `None` take the same branch.

## Measured: two conditions ruled out

Both conditions drive `_run_inference_mlx` directly — no HTTP, no generation
cache, no chunking — with the parameters the server actually sends for a
request that specifies neither speaker nor language:

```
temperature=0.9  top_k=50  top_p=0.95  repetition_penalty=1.05
max_new_tokens=2048  speaker="Ryan"  language="auto"
```

| Condition | Runs | Hit cap | Tokens min/median/max | Elapsed |
|---|---|---|---|---|
| 60 distinct seeds, exact text + speaker, clean single-model process | 60 | **0** | 12 / 19 / 97 | 0.5–4.1 s |
| All three models resident (clone + design + custom), 3.1 GB active | 30 | **0** | 13 / 19 / 51 | 0.5–2.1 s |

The failures emitted ~2048 tokens in ~216 s. Nothing observed came within
**20×** of that, in either token count or wall-clock.

Conclusion: the runaway is **not** determined by the text, **not** by the seed,
and **not** by the resident model set or the memory pressure it creates.

## What is left

The remaining distinguishing factor is the live-server path. The e2e suite that
produced both failures also drives `/load-model` and `/unload-model` and issues
concurrent requests; neither probe reproduces any of that.

A model object used after its slot was cleared, or a generation racing a load,
fits the profile — intermittent, resistant to repetition, and capable of
producing genuinely degenerate output rather than merely slow output. That is
the next hypothesis to test, and it is the expensive one.

**Watch the warning, not the clock.** Wall-clock alone is what caused the
original misattribution to machine contention, and
`test_01_concurrent_generations_performance` passes on a quiesced machine:

```bash
TTS_DISABLE_RATE_LIMITING=1 tts server start
python -m pytest tests/ -m e2e -q --tb=line --continue-on-collection-errors
grep -A1 "token cap" .voice_server.log
```

## Reproducing this pass

```bash
conda run -n qwen3-tts-mlx python scripts/probe_issue192.py \
    --runs 60 --seed-base 1000 --out seeds.json
conda run -n qwen3-tts-mlx python scripts/probe_issue192.py \
    --load-all --runs 30 --out pressure.json
```

The harness exits `1` if any run hits the cap, so a longer sweep can be gated
rather than eyeballed. Seeding is deterministic: `--seed-base 1000` reproduces
27 and 66 tokens for the first two runs.

## Notes

Not a regression from #191, and not made worse by it — #191 only made it
visible. Raising `max_new_tokens` is not a workaround: PRF-9 measured ≥8192 as
unstable on 16 GB (`docs/reviews/prf9-max-new-tokens-measurement-2026-08-15.md`).

Context: `docs/reviews/cross-platform-e2e-2026-08-17.md` (finding D9).
