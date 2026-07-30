# Qwen3-TTS Performance / Quality / Accuracy — Research & Roadmap-Fit

*Generated: 2026-07-30 · Sources: ~40 primary (arXiv, GitHub, HuggingFace, PyPI) · Confidence: High*
*Scope: delta vs. the 2026-03-23 speculative-decoding + attention research; new items for the consolidated roadmap*

> **Purpose:** capture findings so they can be merged into `consolidated-roadmap.md`.
> Candidate items use a `PRF-*` prefix (not yet adopted — renumber on merge). File paths are
> indicative, from the architecture map in `CLAUDE.md`; reconcile against source before work
> (repo verification policy). The quarterly upstream sweep diffs against this file as its baseline.

---

## Executive summary

Five parallel research tracks (Qwen3-TTS models/`qwen_tts`, MLX/Apple Silicon, speech
speculative decoding, CUDA kernels/vLLM, quality/post-processing/ASR) surface **10 actionable
items** and update the status of the upstream-blocked roadmap items.

The single sharpest finding: **Qwen3-TTS upstream is frozen** — all six models unchanged since
2026-01-29, `qwen_tts` library at 0.1.1 (2026-02-06) with **zero merged PRs since mid-March**
and 55 open issues. There is no upgrade target and no upstream help coming, so improvement is
**local mitigation or third-party**. The highest-ROI items are a confirmed bug in *this* project's
own code (silent Chinese number normalization), a correctness reversal on FlashAttention-2, and
cheap DSP/Accuracy fixes. Speculative decoding (R-28) remains blocked: PCG matured to ICASSP-2026
but ships no code; nobody has shipped spec-dec for Qwen3-TTS.

---

## What changed since the 2026-03-23 research

| Area | March 2026 position | July 2026 reality |
|------|---------------------|-------------------|
| **FlashAttention-2** | "Best option for T4/A100; keep it" | **Reversed** — open upstream #333 (updated 2026-07-29) reports **NaN logits with `flash_attention_2`** for Qwen3-TTS. SDPA is the safer torch default until characterized. |
| **Speculative decoding (R-28)** | "EAGLE-3 doesn't apply; 0.6B-as-draft + PCG is the path" | PCG is now **ICASSP-2026–accepted** but **ships zero code, zero library adoption**. Closest analogue SSD (8-layer draft, Qwen2.5-0.5B backbone) is **1.4×, lossy (WER 3.67→5.70), code-less**. Nobody has shipped it. Blocker moved "no theory → theory exists, nothing reusable." |
| **SageAttention** | "Monitor for native HF integration" | **Unchanged** — still monkey-patch-only, **no TTS benchmarks**. Monitor. |
| **vLLM** | "vLLM-omni supports Qwen3-TTS (PR #895)" | Mainline vLLM **still has no TTS**. vLLM-omni **separate, unmerged**, actively tuning Qwen3-TTS (v0.26.0rc1, 2026-07-28: cached decoder masks, compiled pre-transformer, codec-chunk ramp-up). No published throughput numbers. |
| **MLX / mlx-audio** | "MLX has its own attention; no action" | **Big delta is mlx-audio, not MLX core.** Qwen3-TTS support matured 0.3.0→**v0.4.6**: ICL cache, continuous batching, streaming-leak fix, ~13% RTF. No MLX 1.0 (core 0.32.0). |
| **Upstream maintenance** | (not assessed) | **Effectively unmaintained** — 0 merged PRs since mid-March; all remediation must be local. HuggingFace has formally requested benchmark verification (#347). |

---

## ADD — candidate roadmap items (actionable)

| ID | Task | Axis | Impact | Effort | Files (indicative) |
|----|------|------|--------|--------|--------------------|
| **PRF-1** | **Fix Chinese number normalization** — `num2words(…, lang='zh')` raises `NotImplementedError`, swallowed by `_safe_transform()`; cardinal/ordinal/date/currency normalization **silently no-ops for all Chinese input**. Add a `zh` branch (digits→汉字, borrow Coqui `chinese_mandarin_cleaners`). | Accuracy | **High** (primary language) | **Trivial** | `core/engine/text_processing.py` (~233-241, `_safe_transform`) |
| **PRF-2** | **Phase-aligned chunk splices** — add zero-crossing snap + RMS level-match *before* the existing raised-cosine crossfade (fade shape is already correct for correlated speech; phase/level is the missing piece). | Quality | High (dominant chunk artifact) | Low | `core/engine/inference.py` (`_crossfade_chunks` ~465-510; concat paths ~836-842, 943-946) |
| **PRF-3** | **Normalize HH:MM:SS time strings** (regex + num2words) — proven failure (upstream #328: `15:16:36` seconds garbled). | Accuracy | Med | Low | `core/engine/text_processing.py` |
| **PRF-4** | **Torch default FA2 → SDPA** until upstream #333 (NaN logits w/ `flash_attention_2`) is characterized; keep FA2 opt-in only. **Reverses the 2026-03-23 conclusion.** | Correctness | High | Low | `core/engine/model_loader.py` (~102, attn_implementation) |
| **PRF-5** | **Defensive server restart on model-swap OOM** — mlx-audio #827 (open): Qwen3-TTS Base cloning goes **~2.4× slower** after a failed swap. Likely maps to the known-red "server dies under repeated load/unload." | Robustness | High | Low | `server/app_lifespan.py`, `server/app_models.py` (load/unload) |
| **PRF-6** | **Clone rate control via post-hoc pyrubberband time-stretch** (not via `instruct`) — upstream #290 proves model rate-control is broken in clone (output always 41–48 s). Project already ships pyrubberband. | Robustness | Med | Low | `core/engine/inference.py` / `audio_processing.py` |
| **PRF-7** | **Bump mlx-audio → v0.4.6** — delivers ICL cache (clone TTFT ~−300 ms), streaming memory-leak fix (#852/v0.4.2), continuous batching (v0.4.3), ~13% RTF (v0.4.6). | Speed | High | Low (dep bump) | `pyproject.toml` (mlx extra), `requirements.lock` |
| **PRF-8** | **ASR-trim the ICL echo-tail** (upstream #341) — reuse the existing ASR feature to detect/clip any reference-tail echo at the head of cloned output. (`x_vector_only_mode` likely sidesteps #341 already.) | Quality | Med | Low–Med | `core/engine/inference.py` / `voice_prompt.py`; uses `core/engine/asr.py` |
| **PRF-9** | **Investigate raising the MLX `max_new_tokens=2048` cap** — the model natively supports **32,768 tokens (~40 min)**; chunking is a *backend* limit, not a model limit. Could halve/quarter chunk count → fewer seams. **Validate first:** 12 Hz long-form stability (paper: 25 Hz > 12 Hz for long-form) + M2 Pro memory. | Quality | High (structural) | Low to test / Med to ship | `core/engine/inference.py` (`_run_inference_mlx`), `CLAUDE.md`, `server/client/generator.py` (`_generation_timeout`) |
| **PRF-10** | *(Optional)* **Task-Vector emotion control** (arXiv:2606.05367) — training-free, inference-time interpolation between neutral↔emotional **x-vectors** (Qwen3-TTS emotional prosody lives in the speaker embedding, not the text/instruct path). Exploits existing `x_vector_only_mode`; torch path. | Quality (new capability) | Med | Med | `core/engine/voice_prompt.py` / `inference.py` |

**Suggested execution order (low-risk quick wins first):** PRF-1 → PRF-4 → PRF-5 → PRF-3 → PRF-2 → PRF-6 → PRF-7 → PRF-8 → PRF-9 → PRF-10.

---

## KEEP-MONITORING — upstream-blocked (do not act; watched by the GHA + crons)

- **R-28 Speculative decoding** — PCG (arXiv:2511.13732) ICASSP-2026–accepted but **no code/adopters**; SSD (arXiv:2505.15380) closest analogue, 1.4×/lossy/code-less; **zero shipped implementations** for Qwen3-TTS/CosyVoice/Step-Audio. Re-check in ~6 months.
- **Prefix-caching the voice-prompt prefix** (lossless, exact) is the *real* lowest-effort speed win — achievable today via **vLLM automatic prefix caching**, but only if the talker were served through vLLM (it currently isn't). A deployment decision, not a research dependency.
- **SageAttention** — still monkey-patch-only, no TTS benchmarks. Trigger: becomes a native HF `attn_implementation`.
- **vLLM mainline TTS support** — still absent; vLLM-omni separate/unmerged. Trigger: TTS lands in mainline vLLM.
- **FlashAttention-4** — real, active beta (CuTeDSL, beta24 2026-07-29), Hopper/Blackwell only, **not yet a native HF value**. Trigger: native HF wiring + stability.
- **Qwen3-ASR-1.7B** as a Whisper replacement — beats Whisper-large-v3 (esp. Chinese: WER 1.25 vs 2.08 zh / 4.51 vs 7.16 en), but **MLX availability unverified**. Trigger: confirm `mlx-community/Qwen3-ASR*`.
- **New Qwen3-TTS models** — upstream **frozen since 2026-01-29**; no successor. Trigger: new model ID under `Qwen/`.

---

## DROP / do not pursue

- **EAGLE-3 for TTS** — confirmed still text-only; exact-token verification is wrong for acoustically-fungible codec tokens; no draft heads exist for TTS.
- **G2P / phoneme frontend** — distribution mismatch (Qwen3-TTS reads raw text, not phoneme IDs; would degrade output). `gruut` (the one SSML-capable normalizer) was archived 2025-10.
- **FA3/FA4/SageAttention on Colab T4** — none target sm_75. Keep SDPA on T4.
- **NeMo Text Processing (full adoption)** — Pynini (OpenFst bindings) does not pip-install on macOS, blocking the MLX path. The targeted fixes (PRF-1, PRF-3) cover the real gaps; revisit only if normalization moves server-side.

---

## Upstream quality/accuracy notes (context for the items above)

Qwen3-TTS is genuinely SOTA on speaker similarity / naturalness vs. open (CosyVoice 2, F5-TTS) and commercial (MiniMax, ElevenLabs) rivals **per vendor numbers** — but those are self-reported with **no independent MOS/CMOS study, no "Limitations" section**, and HuggingFace has formally requested verification (#347, open). Treat as upper bounds. Confirmed community defects (all open, unfixed) that the ADD items mitigate: #341 (clone echoes reference tail), #290 (clone rate ignored), #239 (long-text rate drift), #328 (time strings), #333 (FA2 NaNs), #350 (concurrency unsupported), #318 (mixed-script Thai hang). Control is **natural-language `instruct`, not SSML** (no native SSML; no good standalone parser; `gruut` archived).

---

## Key sources (all accessed 2026-07-30)

- Models/library: HuggingFace `Qwen/Qwen3-TTS-12Hz-*` model cards & API; `QwenLM/Qwen3-TTS` repo + issues #333/#341/#290/#328/#347/#350/#318; PyPI `qwen-tts`; arXiv:2601.15621 (tech report, v1).
- MLX: `Blaizzy/mlx-audio` releases v0.3.0–v0.4.6, PR #685 (ICL cache), issues #720/#827/#851/#535; `ml-explore/mlx` releases v0.30.1–v0.32.0.
- Speculative decoding: arXiv:2511.13732 (PCG), 2505.15380 (SSD), 2410.13839 (multi-token+Viterbi), 2410.21951 (VADUSA), 2503.01840 (EAGLE-3), 2603.26364 (LLaDA-TTS); `SafeAILab/EAGLE`.
- CUDA/vLLM: `Dao-AILab/flash-attention` (README/setup.py/releases); HF Transformers attention-interface docs; `thu-ml/SageAttention`; `vllm-project/vllm` + `vllm-omni` releases v0.24.x–v0.26.0rc1; PyTorch 2.13.0.
- Quality/ASR: arXiv:2606.05367 (Task-Vector emotion); `coqui-ai/TTS` cleaners; `NVIDIA/NeMo-Text-Processing`; `snakers4/silero-vad`; `Rikorose/DeepFilterNet`; `SYSTRAN/faster-whisper`; `Qwen/Qwen3-ASR-1.7B`.

## Methodology

Five parallel research subagents (general-purpose, web-grounded), each instructed to cite primary
sources with dates and to flag unverifiable claims. One track dropped a fabricated source batch
after re-verification (SPAR-K; arXiv:2503.09215 ≠ speech). Vendor benchmark numbers are labeled
as such; the FA2 NaN finding and the `num2words('zh')` bug were independently confirmed.
