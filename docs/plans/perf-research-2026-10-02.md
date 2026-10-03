# Qwen3-TTS Performance / Quality / Accuracy — Quarterly Upstream Sweep

*Generated: 2026-10-02 · Five parallel research tracks · Baseline: `perf-research-2026-07-30.md`,
`2026-08-15-upstream-watch.md`, and `archive/2026-03-23-{speculative-decoding,attention-mechanisms}-research.md`*
*Pinned at sweep time: mlx 0.32.2 · mlx-audio 0.5.4 · qwen-tts 0.1.1 · torch 2.13.0 · transformers 4.57.3 (torch env) / 5.15.0 (mlx env)*

> **Purpose:** the delta since the 2026-07-30 full sweep, with revised roadmap-fit verdicts. New
> candidate items continue the `PRF-*` numbering (PRF-1..10 were the July batch). File paths are
> indicative. Claims marked *(verified locally)* were checked against this repo or the installed
> envs; everything else is web-sourced, and the speedup numbers are author-reported.

---

## Executive summary

1. **Upstream Qwen3-TTS is still frozen.** All five official checkpoints and the tokenizer are
   unchanged since 2026-01-23/29. `qwen-tts` is still 0.1.1, and `QwenLM/Qwen3-TTS` has had no
   commits since 2026-03-17. A stale bot closed the issues behind three of our workarounds
   (#333 FA2 NaN, #290 clone rate control, #328 time strings) as "not planned" between
   2026-08-15 and 2026-08-28. They were **not fixed**, so PRF-3/4/6 stay load-bearing.
2. **New project defect: there is no `0.6B-VoiceDesign` model.** Qwen publishes only five TTS
   models, and no 0.6B VoiceDesign exists under any author on HuggingFace, mlx-community
   included *(verified 2026-10-02 via the HF API)*. Yet `core/config/models.py:304,341` and
   `tools/model_cache.py:24,33,42` map `model_size="0.6B"` + design mode to those IDs, so that
   combination 404s at load time. Tracked as **PRF-11**.
3. **The Qwen3-ASR blocker was never real.** mlx-audio has shipped `stt/models/qwen3_asr` (plus
   `qwen3_forced_aligner`) since v0.3.1 (2026-01-29), and the **installed 0.5.4 has both**
   *(verified locally)*. `mlx-community/Qwen3-ASR-1.7B-{4,5,6,8bit,bf16}` exist (2026-01-29). The
   July KEEP-MONITORING trigger was already satisfied when it was written. Promoted to ADD as **PRF-12**.
4. **R-28 (speculative decoding) moved but did not clear.** arXiv:2609.37007 (Argmax, 2026-09-29)
   is a **lossless** speculative scheme for Qwen3-TTS's own CodePredictor (the 15-code RVQ inner
   loop), reporting 2.13× on an M3 Pro. NVIDIA TensorRT-Edge-LLM shipped "CodePredictor
   speculative decoding for Qwen3-TTS", but for TensorRT/Jetson only. Neither has MLX or torch
   code we can reuse. **Blocker updated:** "nothing reusable" → "a method aimed at our exact model
   exists; we would have to port it ourselves".
5. **mlx-audio 0.5.5→0.5.7 changes no Qwen3-TTS code** (compare `v0.5.4...v0.5.7`). The relevant
   EOS and repetition fixes (#897, #914) landed in 0.5.1 and are already in our pin. That has one
   consequence: the **PRF-9 NO-GO measurement (2026-08-15) predates those fixes**, so a re-measure
   is now cheap and justified (**PRF-13**).
6. **CUDA/vLLM: nothing actionable for our torch path.** FA4 gained a native HF
   `attn_implementation` in transformers 5.17, but it is still beta, Hopper/Blackwell only, and
   our torch env is pinned to transformers 4.57.3. vllm-omni reached 0.30.0 (Model Runner V2 is
   now the default on CUDA, still experimental). PyTorch 2.14 adds native GQA to the
   memory-efficient SDPA backend, the only item that could help T4. It is unbenchmarked and
   tied to the held torch bump (#279).

---

## Track 1 — Qwen3-TTS models + `qwen_tts`

| Finding | Source (date) | Maturity | MLX / torch / vLLM | Effort |
|---|---|---|---|---|
| No new checkpoints or revisions. Last-modified: 1.7B-Base 2026-01-23; 1.7B-{CustomVoice,VoiceDesign}, 0.6B-{Base,CustomVoice}, Tokenizer 2026-01-29 | https://huggingface.co/api/models?author=Qwen&search=Qwen3-TTS (2026-10-02) | — | — | — |
| **No `Qwen3-TTS-12Hz-0.6B-VoiceDesign` exists** (Qwen or mlx-community). Our config maps it anyway → PRF-11 | same; `?search=Qwen3-TTS-12Hz-0.6B-VoiceDesign` → `[]` (2026-10-02) | defect in this repo | all three backends | Low |
| `qwen-tts` still 0.1.1 (2026-02-06): no new `generate()` params, no speculative or cache hook | https://pypi.org/project/qwen-tts/ | — | torch | — |
| Upstream repo: last commit 2026-03-17; only merged PR ever is #15; no releases; Apache-2.0 unchanged | https://github.com/QwenLM/Qwen3-TTS | unmaintained | — | — |
| #333 / #290 / #328 closed **"not planned" by stale bot** (2026-08-28 / 08-15 / 08-21). No fix | https://github.com/QwenLM/Qwen3-TTS/issues/333 | — | torch (#333), all (#290/#328) | keep PRF-3/4/6 |
| #341 (ICL echo) still open. A 2026-08-20 user comment confirms that trimming the reference tail stops the echo, which corroborates PRF-8 | https://github.com/QwenLM/Qwen3-TTS/issues/341 | — | all | none |
| TensorRT-Edge-LLM v0.10.0 (2026-08-12) → v0.11.0 (2026-09-29): Qwen3-TTS VoiceDesign on-device, `/v1/audio/speech`, **CodePredictor speculative decoding** for Qwen3-TTS/Omni (CHANGELOG; exact version header unconfirmed) | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/CHANGELOG.md | released (TRT only) | none of ours | — |
| Community-only uploads (GGUF/ggml/ONNX VoiceDesign, 2026-09-29). Unofficial, not assessed | HF search | community | — | — |

## Track 2 — MLX + mlx-audio

| Finding | Source (date) | Maturity | Applicability | Effort |
|---|---|---|---|---|
| mlx-audio 0.5.5 (09-21), 0.5.6 (09-24), 0.5.7 (09-28): **no change under `tts/models/qwen3_tts/`**. 0.5.5's `qwen3.py` is a *different* model family; `tts/generate.py` passes `voice` only when not None; `max_tokens`/`lang_code`/`ref_audio`/`ref_text` are unchanged | https://github.com/Blaizzy/mlx-audio/compare/v0.5.4...v0.5.7 | released | MLX | Trivial (Dependabot #365) |
| Already in our 0.5.4 pin (via 0.5.1, 2026-08-31): #897 EOS no longer exempt from top-k/top-p; #914 windowed ICL repetition penalty (fixes speed-up on long text, #910); #895 Base rejects an unsupported `voice` | mlx-audio PRs #897/#914/#895 | released | MLX | none, but enables PRF-13 |
| #827 (clone ~2.4× slower after a failed swap) **still open**. PRF-5 recovery stays | https://github.com/Blaizzy/mlx-audio/issues/827 | — | MLX | — |
| #921 (open, 2026-08-28): ICL degenerates to no-EOS when `ref_text` does not cover all of `ref_audio`, reproduced on the official torch model too. Matches our transcript-mismatch → token-cap finding; there is no upstream fix | https://github.com/Blaizzy/mlx-audio/issues/921 | — | MLX + torch | input hygiene (see PRF-12 note) |
| #946 (ICL streaming cold-start distortion fix) **closed unmerged** 2026-09-07. The first streamed clone chunk may still distort | mlx-audio #946 | — | MLX streaming | monitor |
| MLX 0.32.3 (2026-09-29): GQA `sdpa_vector_2pass` tuning, quantized-matmul fixes, adaptive load I/O, `clear_streams` deadlock fix. No MLX 1.0. Gains unmeasured | https://github.com/ml-explore/mlx/releases/tag/v0.32.3 | released | MLX | Trivial |
| No speculative decoding, no new prompt caching, and no nvfp4/mxfp4/DWQ Qwen3-TTS conversions in mlx-audio or mlx-community. All TTS conversions dated 2026-01-22 | HF mlx-community | — | MLX | — |
| mlx-audio PR #966 (2026-09-19): per-frame KV cache for a depth decoder (Breeze TTS, RTF 1.63→0.62). The same pattern could apply to Qwen3-TTS's CodePredictor, which reportedly resets KV every frame (unverified) | https://github.com/Blaizzy/mlx-audio/pull/966 | PR, other model | MLX | monitor |

## Track 3 — Speech speculative decoding

| Finding | Source (date) | Maturity | Applicability | Effort |
|---|---|---|---|---|
| **arXiv:2609.37007 "RVQ Position Aware Speculative Decoding for On Device TTS"** (Argmax). Reuses the CodePredictor's frozen per-codebook heads as drafters plus 3 learned offset vectors (3,072 params), with tree verification under standard speculative sampling (lossless). 2.47 accepted/call; 2.13× on M3 Pro; WER/CER parity on 6 Fleurs languages. **Tested on 0.6B CustomVoice only**; speeds up the inner loop only | https://arxiv.org/abs/2609.37007 (2026-09-29) | paper; no public method code found (argmax-oss-swift v1.1.0, 2026-08-06, has a fused decoder only) | MLX/torch (port) | High |
| PCG (arXiv:2511.13732): still no official code. Apple page fetch failed this run (unverified) | — | paper | — | — |
| vllm-omni RFC #7837 (opened 2026-09-19): stage-local spec decoding for **Qwen3-Omni** only. No PRs | https://github.com/vllm-project/vllm-omni/issues/7837 | proposal | vLLM | — |
| Voice-prompt caching in the wild caches `create_voice_clone_prompt` output (embedding plus codec codes), which **we already do**. No measured TTFT exists for transformer-KV reuse of the ICL prefix on Qwen3-TTS | vllm-omni PR #2457; dingausmwald fork (self-reported ~0.7 s) | shipped (output cache) | all | none |
| EAGLE-3 for speech: no change. Still DROP | — | — | — | — |

## Track 4 — CUDA kernels + vLLM

| Finding | Source (date) | Maturity | Applicability | Effort |
|---|---|---|---|---|
| FA4 `flash-attn-4` 4.0.0b32 (2026-09-23): still beta, Hopper/Blackwell only | https://pypi.org/project/flash-attn-4/ | beta | not T4 | — |
| transformers **5.17** `from_pretrained` docs list `attn_implementation="flash_attention_4"` (the attention-backends page does not yet; docs inconsistent). Our torch env is pinned to transformers 4.57.3 by qwen-tts, so this is **unreachable** | https://huggingface.co/docs/transformers/en/main_classes/model | released (v5 only) | torch, blocked | — |
| #333 FA2 NaN: closed without root cause or fix. Keep SDPA default (PRF-4) | QwenLM #333 | — | torch | none |
| SageAttention: no native HF value. Only via the Hub kernel / `AttentionInterface.register` / monkey-patch. No TTS benchmarks | https://huggingface.co/docs/transformers/en/attention_interface | — | torch | — |
| **vllm-omni v0.30.0** (2026-09-25, on vLLM 0.30.0): Qwen3-TTS defaults to the experimental **Model Runner V2** on CUDA (`model_runner: v1` opts out); event-driven orchestration on by default; initial-silence codec tokens suppressed; Code2Wav fixes. No new published numbers (existing docs claim TTFA 733→64 ms at c=1). Some sources place these PRs under 0.28 | https://github.com/vllm-project/vllm-omni/releases/tag/v0.30.0 | experimental | vLLM (Step 5) | Med |
| Mainline vLLM: still no TTS (not directly re-checked) | — | — | vLLM | — |
| **PyTorch 2.14** (GA 2026-09-02): memory-efficient SDPA gains native GQA (the backend T4 can use); rank-3 SDPA input → fused kernels; `torch.while_loop` CUDA-graph capture. Nothing decode-specific benchmarked; sm_75 retention not fully confirmed | https://pytorch.org/blog/pytorch-2-14-release-blog/ | released | torch / T4 | tied to held #279 |

## Track 5 — Quality, post-processing, ASR

| Finding | Source (date) | Maturity | Applicability | Effort |
|---|---|---|---|---|
| **Qwen3-ASR on MLX**: supported by mlx-audio since v0.3.1 (2026-01-29); `stt/models/qwen3_asr` and `qwen3_forced_aligner` are present in the installed 0.5.4 *(verified locally)*. `mlx-community/Qwen3-ASR-1.7B-8bit` ≈ 2.46 GB. Open upstream ASR issues: #928 (short/no-speech audio mishandled), #927 (streaming timestamps vs `max_tokens`) | https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit ; https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/qwen3_asr/README.md | released | MLX | Low–Med (PRF-12) |
| WFST normalization: pynini 2.1.7 still ships manylinux-only wheels on PyPI; osx-arm64 only via conda-forge (WeTextProcessing works there). Conflicts with the "any Python env" test policy | https://pypi.org/project/pynini/#files | — | MLX blocked | keep DROP |
| **LACI "Taming Long-form TTS"** (arXiv:2609.16989, 2026-09-15): inference-only error detection plus rewind/regenerate with attention guardrails. On **Qwen3-TTS-0.6B**, worst-of-N WER on >1500-word prompts 35.2% → 3.4%. No code; needs attention internals | https://arxiv.org/abs/2609.16989 | paper | torch (MLX hard) | High |
| MagpieTTS-LF (2606.18485): cross-chunk state, NeMo-specific. Its finding that per-chunk gain normalization causes loudness jumps is already covered by our RMS matching | https://arxiv.org/abs/2606.18485 | paper | — | — |
| Prosody-boundary streaming (2603.06444): carry chunk N's tail as chunk N+1's prompt. Cheap analogue for clone mode; drift risk | https://arxiv.org/abs/2603.06444 | paper | MLX + torch | Med |
| Local quality metrics: UTMOSv2 (torch; MPS support/date unverified), WavLM-SV/ECAPA speaker similarity, VERSA (heavy). Reference denoising: DeepFilterNet / ClearerVoice-Studio, no new releases since July | https://github.com/sarulab-speech/UTMOSv2 ; https://github.com/modelscope/ClearerVoice-Studio | released | torch | Med |
| **EmoRES-TTS** (arXiv:2609.38157, 2026-09-29, Meta, code): splits an emotion steering vector into a shared "away-from-neutral" part plus an emotion residual. Tested on IndexTTS-2/CosyVoice2, not Qwen3-TTS. Would refine PRF-10's x-vector τ. Whether it cites 2606.05367 is unconfirmed | https://arxiv.org/abs/2609.38157 | paper + code | torch | Low on top of PRF-10 |
| Unverified search hits: "Qwen-Audio-3.0-ASR" (arXiv 2609.07549) / "Qwen-Audio-3.0-TTS" (2607.23938). Not opened; weights and MLX status unknown. Check next sweep | — | unverified | — | — |

---

## Roadmap-fit verdicts

### ADD — candidate roadmap items

| ID | Task | Axis | Impact | Effort | Files (indicative) |
|----|------|------|--------|--------|--------------------|
| **PRF-11** | **Fix the nonexistent `0.6B-VoiceDesign` mapping.** No such model exists upstream or on mlx-community. Either fall back to `1.7B-VoiceDesign` (with a logged warning) or reject `design` under `model_size="0.6B"` with a clear error; fix the cache-tool entries; add a test pinning every `MODEL_INFO`/`MLX_MODEL_INFO` ID to a known-published list | Correctness | Med (404 on a valid config combo) | Low | `core/config/models.py:297-312,334-348`, `tools/model_cache.py:24,33,42` |
| **PRF-12** | **Qwen3-ASR-1.7B as an optional MLX ASR model** for `/transcribe` and the PRF-8 echo-trim. It beats whisper-large-v3 on zh (WER 1.25 vs 2.08) and en (4.51 vs 7.16), and needs no new dependency (same `mlx_audio.stt` loader). **Validate first:** peak memory on 16 GB alongside a loaded TTS model (8bit ≈ 2.46 GB vs whisper-turbo), echo-trim word-match quality, and #928 short-clip behaviour. Keep Whisper as default until measured. Optional follow-on: use `Qwen3-ForcedAligner` to verify that a clone transcript covers the whole reference (the #921 / token-cap trap) | Accuracy | High (zh primary) | Low–Med | `core/engine/asr.py` (`_MLX_WHISPER_REPO`, `load_asr_model`), config key |
| **PRF-13** | **Re-measure the MLX `max_tokens` cap (PRF-9 follow-up).** The 2026-08-15 NO-GO predates mlx-audio 0.5.1's EOS-filter (#897) and windowed ICL repetition-penalty (#914) fixes. Re-run the PRF-9 protocol on 0.5.4+. *Validation gate only*: ship a cap change only if long-form stability and M2 Pro peak memory both pass | Quality (fewer seams) | Med | Low to measure | `docs/reviews/prf9-max-new-tokens-measurement-2026-08-15.md` protocol; `core/engine/inference.py` |

Housekeeping (not roadmap items): finish Dependabot **#365 mlx-audio ≥0.5.6** (content-free for
Qwen3-TTS per the compare, but it still needs the triage skill's churn gate), and bump MLX to 0.32.3
alongside it.

### KEEP-MONITORING

| Item | Status change this sweep | Trigger to act |
|---|---|---|
| **R-28 speculative decoding** | **Narrowed.** arXiv:2609.37007 gives a lossless, ~3K-param method aimed at Qwen3-TTS's CodePredictor (2.1× on M3 Pro); TRT-Edge-LLM shipped it for TensorRT only. No MLX/torch code | Public code for 2609.37007, or the method lands in mlx-audio, *or* a decision to port it ourselves (High effort; validate on 1.7B + Base/clone, which the paper did not test). Re-check 2027-01 |
| CodePredictor per-frame KV cache (mlx-audio #966 pattern) | New | A Qwen3-TTS PR in mlx-audio |
| LACI long-form rewind (2609.16989) | New; tested on Qwen3-TTS-0.6B | Code release |
| Chunk-tail carry-over as next ICL prompt (2603.06444) | New | Cheap experiment if seam complaints recur |
| EmoRES-TTS residual steering (2609.38157) | New; attaches to PRF-10 | Only if PRF-10 is built |
| Local quality-regression metrics (WER via our ASR + ECAPA/WavLM sim; UTMOSv2 optional) | New | Before any model/engine bump that changes output (e.g. a PRF-13 cap change) |
| Reference-audio denoising (DeepFilterNet / ClearerVoice) | Unchanged | User demand for noisy references |
| FlashAttention-4 | Native HF value in transformers 5.17, but beta + Hopper/Blackwell + our torch env stays on 4.57.3 | Stable FA4 *and* the torch env leaving transformers 4.x |
| SageAttention | No change | Native HF `attn_implementation` |
| vLLM / vllm-omni | 0.26 → 0.30.0, MRV2 default (experimental); mainline still no TTS | Step 5 (CUDA infra) |
| PyTorch 2.14 mem-efficient SDPA GQA (T4) | New | Lands with the #279 torch bump; benchmark on T4 then |
| mlx-audio #827 / #946 | Both unresolved (#946 closed unmerged) | Upstream fix → revisit PRF-5 / streaming first-chunk |
| New Qwen3-TTS models | Still frozen. Check the unverified "Qwen-Audio-3.0" papers next sweep | New model ID under `Qwen/` |
| ~~Qwen3-ASR on MLX~~ | **Removed: trigger already satisfied** → PRF-12 | — |

### DROP / do not pursue

- **EAGLE-3 for TTS:** unchanged. The *lossless RVQ-position* scheme above is a different method and is tracked under R-28.
- **G2P / phoneme frontend:** unchanged.
- **WFST normalization (NeMo / WeTextProcessing):** still no macOS PyPI wheels for pynini; conda-only.
- **FA3/FA4/SageAttention on Colab T4:** none target sm_75.
- **Self-converting nvfp4/mxfp4 Qwen3-TTS weights:** no community conversions, unverified TTS quality, and the 8-bit default is adequate on 16 GB.

---

## Methodology

Five parallel web-research subagents, one per track, each told to cite primary sources with dates
and to flag unverifiable claims. Tracks 3, 4 and 5 were re-dispatched once with tighter scopes
after stalls or permission-check timeouts. Local re-verification by the main session:
installed package versions in both conda envs; the `mlx_audio/stt/models/` listing (Qwen3-ASR
presence); the HF API for every Qwen TTS model ID and for `0.6B-VoiceDesign` (absent); and the
model maps in `core/config/models.py` / `tools/model_cache.py`. One subagent claim was corrected:
Dependabot #363 (vllm-omni ≥0.30) is **closed** (2026-10-01, folded into Step 5), not open.
