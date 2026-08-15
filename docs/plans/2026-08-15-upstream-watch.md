# Monthly Upstream Watch — 2026-08-15

Baseline diffed against `perf-research-2026-07-30.md` + the R-28 entry in
`consolidated-roadmap.md`. No prior monthly note existed (first run). Focused
web check only, no full sweep.

## Verdict: no roadmap action

Nothing material changed for the default MLX/torch backends. No watch trigger
fired; no roadmap slot re-opens. Keep monitoring.

## Deltas recorded this check

1. **qwen_tts (R-28 blocker)** — still 0.1.1 on PyPI (2026-02-06); no
   `draft_model`/`speculative_config` hook. Adjacent third-party: NVIDIA
   TensorRT-Edge-LLM plans MTP-draft for Qwen3-TTS-0.6B in its 0.7.1
   (issues #86/#87); their #87 notes Qwen3-TTS's built-in CodePredictor is
   *not* MTP speculative decoding. **Still blocked.**
2. **SageAttention** — transformers issue #39618 still open; not a native
   `attn_implementation`. v5's modular `load_model` enables a class-swap
   workaround (not a trigger). Diffusers has it natively (irrelevant here).
   **Still blocked.**
3. **mlx-audio** — 0.4.7 released (Blaizzy PR #866: model-kind registry,
   arktts). No speculative decoding / prompt caching for Qwen3-TTS.
   **Still blocked.** (Confirms our PR #164 bump is content-free for us.)
4. **PCG (arXiv:2511.13732)** — still no official code (Apple). An
   OpenResearch auto-reproduction page exists; not adoptable. **Still
   blocked.**
5. **Qwen3-TTS models** — HF collection unchanged since 2026-01-29;
   "Qwen-TTS-Flash" is API-only, not open weights. **Frozen.**
6. **vLLM-omni** — still a separate repo; upstream-merge is a stated goal,
   not done (issue #152). **Published throughput numbers now exist** (missed
   by the 2026-07-30 baseline): vLLM blog 2026-06-23 — Qwen3-TTS voice
   cloning +61.5% audio throughput, 2×H20 @ c=64; Baseten ~$3–4/M chars.
   Informational only while vLLM items stay parked.

Next check: ~2026-09-15 (durable one-shot scheduled).
