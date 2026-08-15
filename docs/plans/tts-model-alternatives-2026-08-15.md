# Qwen3-TTS Freeze & Alternative Voice Models: Research Report

*Generated: 2026-08-15 · Sources: ~40 (3 parallel research agents, commit APIs + primary model cards + news) · Confidence: Medium-High (activity verified via GitHub/HF APIs; performance claims largely single-source)*

**Reference architecture:** three-mode TTS (clone from 5–15 s audio / design from text description / 9 preset speakers), MLX-primary on Apple Silicon via `mlx-audio` 0.4.8 with a torch fallback, engine = dispatch layer (text_processing / audio_processing / voice_prompt / model_loader / inference) over the model package, FastAPI server, dual streaming paths, voice-prompt caching (.pt + .wav/.txt).

## Executive Summary

Qwen3-TTS's open weights are frozen with **no official explanation** — the promised 25Hz family never shipped while Alibaba iterates API-only successors (Qwen-Audio-3.0-TTS, Qwen3.5-Omni), matching its now-documented open-then-freeze pattern (Wan video, Omni line). Nothing in open source replicates Qwen's full three-mode shape; voice-design-from-description remains **unique** to Qwen3-TTS. But for the **clone mode** — the mode that matters most — there is a cheap, low-risk integration path: our MLX backend already sits on `mlx_audio`, and mlx-audio 0.4.8 ships ~10 actively-developed voice-cloning models behind the *same* `model.generate(text=, ref_audio=, ref_text=)` API Qwen uses. The strongest pilot candidate is **arktts/Audio8-TTS-Preview-0.6B** (Apache-2.0, transcript-ICL cloning identical in shape to Qwen Base, added to mlx-audio 0.4.7 two weeks ago). Recommendation: keep Qwen for Design/Custom modes (no substitute exists), pilot an mlx-audio-native alternative engine for clone mode behind the existing `advanced.backend`-style config dispatch.

## 1. Why the models are frozen

- **No official statement exists.** The repo README's only News entry remains the 2026-01-22 release; the standing promise "Other models … will be released in the near future" is ~7 months stale. A Qwen maintainer closed the 25Hz-tracking issue with "we will notify you when it's released" (2026-01-26); community re-asks through 2026-08-09 are unanswered; the release-request issue has been stale-bot-swept twice ([QwenLM/Qwen3-TTS#34](https://github.com/QwenLM/Qwen3-TTS/issues/34), [#294](https://github.com/QwenLM/Qwen3-TTS/issues/294); last repo commit 2026-03-17 per GitHub API).
- **An official 25Hz promise did exist** — @Alibaba_Qwen X post (2026-01-24): capability "coming in the upcoming open-source 25Hz model release." Nothing shipped against it ([X post](https://x.com/Alibaba_Qwen/status/2015073927564025899), cited in #294).
- **The API line shipped instead**: Qwen-Audio-3.0-TTS (2026-07-20, Flash ~300 ms / Plus tiers, hosted-only, "distinct from the open-weight Qwen3-TTS line") and Qwen3.5-Omni (API-only, includes voice cloning) ([marktechpost](https://www.marktechpost.com/2026/07/20/alibabas-tongyi-lab-releases-qwen-audio-3-0-tts-a-hosted-text-to-speech-model-in-flash-and-plus-tiers-across-16-languages/), [qwen.ai](https://qwen.ai/blog?id=qwen3.5-omni)).
- **It's a pattern**: Wan2.1/2.2 open → Wan2.5/2.6 API-only; Qwen2.5-Omni open → turbo APIs closed; The Information reads Qwen3.5-Omni's closure as "a potential shift away from a strategy that centers on open-source models" ([HF Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), [The Information](https://www.theinformation.com/briefings/alibabas-new-multimodal-ai-model-open-source)).
- **Strategic-withholding reading (community, plausible, unverified)**: the withheld 25Hz family is the single-codebook tier closest to Flash capability — 12Hz lacks emotion control on cloned voices ([arXiv 2601.15621](https://arxiv.org/html/2601.15621v1), [#34](https://github.com/QwenLM/Qwen3-TTS/issues/34)). Releasing it would narrow the open/API quality gap.
- **Verdict for planning:** assume the 12Hz weights are terminal. The monthly upstream watch (next: 2026-09-15) remains the trigger for any change, but nothing suggests movement.

## 2. The alternatives landscape (activity-verified)

Activity checked via GitHub commit APIs on 2026-08-15 ("active" = commits/release within ~6 months). Qwen3-TTS's last commit (2026-03-17) is *older than most competitors below*.

| Rank | Model | Last activity | License | Size | Clone approach | Fits our 3 modes? |
|---|---|---|---|---|---|---|
| 1 | **IndexTTS-2.5** (Bilibili) | commits 2026-08-13, release 2026-08-10 | ⚠️ Bilibili (not OSI; non-commercial) | 0.8B | transcript-free single clip + 8-dim emotion vector | clone only; no design/presets/streaming-doc |
| 2 | **Chatterbox v3/Turbo/Nano** (Resemble) | 2026-07-21 | MIT ✅ | 0.5B/350M/110M | transcript-free ~10 s clip; built-in watermarking | clone only; no streaming doc'd |
| 3 | **VibeVoice** (Microsoft) | 2026-07-24 | MIT (weights) ✅ — but TTS code pulled from repo 2025-09-05 | 1.5B + 0.5B realtime | ref-audio conditioned | clone + style voices; streaming ✅ |
| 4 | **CosyVoice 3** (FunAudioLLM) | 2026-05-25 | Apache-2.0 ✅ | 0.5B | **transcript-required ICL** (closest to Qwen Base) | clone + instruct (emotion/speed, not timbre); streaming 150 ms |
| 5 | **Fish/OpenAudio S2-Pro** | 2026-06-09 | ⚠️ Fish research license (weights) | 4B+400M dual-AR | 10–30 s ref, transcript-free; **15k+ text tags** (closest OSS analog to VoiceDesign) | clone + tag-control; streaming ~100 ms |
| 6 | **F5-TTS** | code active 2026-07-23, weights frozen 2025-03 | MIT code / CC-BY-NC weights | ~336M | flow-matching, transcript optional | clone only, no streaming |
| 7 | **Voxtral-4B-TTS** (Mistral) | released 2026-03-23 | ⚠️ CC-BY-NC 4.0 | 4B | transcript-free + named presets | clone + presets; streaming 70 ms |
| 8 | **GLM-TTS** (zai-org) | 2026-04-10 | Apache-2.0 ✅ | Llama-based | 3–10 s prompt audio | clone + speakers; streaming ✅ |

Inactive (failed the 6-month test): Orpheus (2025-12-05), Sesame CSM (2025-05-27), Step-Audio 2 (README-only drift), Mars5 (superseded), MegaTTS3 (maintenance-only). Kokoro has no cloning.

**The critical gap:** *no* competitor ships clone + design-from-description + preset speakers in one family. Qwen3-TTS VoiceDesign is still the only dedicated open design-a-voice-from-prose model. **Any migration keeps Qwen for Design mode** or drops that mode.

## 3. Apple-Silicon / MLX readiness — the integration seam

**mlx-audio 0.4.8 already ships ~10 cloning models behind the same API our engine calls** ([docs](https://blaizzy.github.io/mlx-audio/api-reference/tts/)): `load_model()` → `model.generate(text=, ref_audio=, ref_text=, …)` → `GenerationResult(audio, sample_rate, is_streaming_chunk)`. Our `model_loader`/`inference` dispatch is the natural multi-model seam — a new clone engine is a **new dispatch entry, not a rewrite**.

| Swap cost | Candidates | Notes |
|---|---|---|
| **Cheap — inside mlx-audio, same API** | **arktts/Audio8-0.6B** (v0.4.7, Apache-2.0, 11 langs, `ref_audio`+`ref_text` ICL exactly like Qwen Base, "inspired by Fish S2 Pro", [HF card](https://huggingface.co/mlx-community/Audio8-TTS-Preview-0.6b-bf16)) · Fish S2-Pro (bf16/8bit in mlx-community) · Chatterbox v3 + Turbo · Higgs v2/v3 · MOSS-TTS (8B→100M Nano) · Confucius4 | New `model_loader` entry + prompt-format check; verify per-chunk `_postprocess_chunk` assumptions (echo-trim, speed, LUFS) per model |
| **Medium — via the mlx-audio-plus fork** | CosyVoice 3 (identical API; `mlx-community/Fun-CosyVoice3-0.5B-2512-8bit` ~1.4 GB) | Risk = depending on a one-maintainer fork tracking upstream |
| **Expensive — separate stacks** | IndexTTS-2 (own repo, `reference_audio` API, **no streaming** → breaks `/generate-stream` + `/ws`, RTF ~1.3 on M2-class, non-commercial license) · F5-TTS-MLX (flow-matching paradigm, no voice-prompt caching concept) | Only worth it for specific capability needs |
| **Unavailable on MLX** | Step-Audio 2 (codec groundwork only), OpenAudio S1 | — |

**Benchmark reality check:** no rigorous head-to-head MLX benchmark exists (the only 2026 multi-model Mac table is torch-MPS/CPU, single-source — and notably measured Qwen3-TTS at RTF 2.08 **CPU-only** on Mac, reinforcing that our mlx-primary design is the right one; [LinkedIn benchmark](https://www.linkedin.com/posts/srinivas-karri-86b119_i-benchmarked-4-open-source-tts-models-on-activity-7445726908958461952-E2xb)). Any adoption decision needs our own RTF/quality measurement on the M2 Pro — the same harness lane F (PRF-9) would exercise.

## Key Takeaways

1. **Plan for Qwen3-TTS-12Hz being terminal.** The freeze is strategic (API monetization), not accidental; the 25Hz tier that would close the capability gap is exactly what's withheld. Keep the monthly watch, expect nothing.
2. **Don't migrate — add.** The architecture's dispatch layer + mlx-audio's unified API make a second clone engine a config-selectable addition. Keep Qwen for Design/Custom (irreplaceable), pilot an alternative for clone.
3. **Best pilot: arktts/Audio8-TTS-Preview-0.6B** — same package we already ship, Apache-2.0, 0.6B (our size tier), transcript-ICL cloning shaped like Qwen Base (our prompts already store `.wav`+`.txt` pairs), 11 languages. Caveats: it's a *preview* model, brand-new (2026-08-03), quality unmeasured — needs our own listening test + RTF measurement before any commitment.
4. **Runner-ups by constraint:** license-maximalist → Chatterbox v3 (MIT, transcript-free, watermarking); capability-closest to Qwen Base's ICL → CosyVoice 3 (but via a fork); richest text-control → Fish S2-Pro (but license-restricted); most active upstream → IndexTTS-2.5 (but non-commercial license + no streaming + heavy swap).
5. **Measure before choosing.** No trustworthy MLX benchmarks exist; run the same generation harness across Qwen/arktts/Chatterbox on the M2 Pro (RTF, peak memory, listening quality) — this doubles as the PRF-9 measurement infrastructure.

## Sources (primary)

1. [QwenLM/Qwen3-TTS README + issues #34, #294](https://github.com/QwenLM/Qwen3-TTS) — freeze evidence, 25Hz promise, unanswered re-asks (checked 2026-08-15)
2. [Qwen3-TTS technical report, arXiv 2601.15621](https://arxiv.org/html/2601.15621v1) — 12Hz/25Hz architecture split
3. [Qwen-Audio-3.0-TTS release coverage, marktechpost 2026-07-20](https://www.marktechpost.com/2026/07/20/alibabas-tongyi-lab-releases-qwen-audio-3-0-tts-a-hosted-text-to-speech-model-in-flash-and-plus-tiers-across-16-languages/) — API successor, hosted-only
4. [The Information — Alibaba multimodal open-source shift](https://www.theinformation.com/briefings/alibabas-new-multimodal-ai-model-open-source) — the freeze-as-strategy read
5. [GitHub commit APIs for all candidate repos](https://api.github.com) — activity verification, fetched 2026-08-15
6. [mlx-audio docs + releases](https://blaizzy.github.io/mlx-audio/) — supported-model catalog, unified generate API, v0.4.7/0.4.8 release notes
7. [mlx-community/Audio8-TTS-Preview-0.6b-bf16](https://huggingface.co/mlx-community/Audio8-TTS-Preview-0.6b-bf16) — arktts card
8. [mlx-audio-plus (CosyVoice 2/3 MLX fork)](https://pypi.org/project/mlx-audio-plus/) + [Fun-CosyVoice3-0.5B-2512-8bit](https://huggingface.co/mlx-community/Fun-CosyVoice3-0.5B-2512-8bit)
9. [solar2ain/mlx-indextts](https://github.com/solar2ain/mlx-indextts) — IndexTTS-2 MLX runner, RTF numbers, API shape
10. [lucasnewman/f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx) — F5 MLX port
11. [Mac TTS benchmark (torch-MPS/CPU), LinkedIn 2026-04-03](https://www.linkedin.com/posts/srinivas-karri-86b119_i-benchmarked-4-open-source-tts-models-on-activity-7445726908958461952-E2xb) — the only 2026 multi-model Mac table; single-source
12. Full per-agent findings briefs with all inline citations live in this session's research transcripts (3 agents, ~40 queries total).

## Methodology

Three parallel research agents (freeze cause / alternatives landscape / MLX readiness), 2–3 query variants per sub-question, commit-activity verified against GitHub APIs rather than snippets, model cards read at source. Single-source claims flagged inline. Performance figures are vendor/self-published on heterogeneous hardware — treat as directional only.
