# PRF-9 Measurement: MLX `max_new_tokens` — NO-GO (2026-08-15)

Lane F of `docs/plans/2026-08-15-parallel-tier1-tier2.md`. Runtime-only
measurement on the M2 Pro (16 GB); **no code or config changes ship with this
doc**. Verdict up front:

> **NO-GO — do not raise the MLX generation cap on this hardware.**
> At an effective 8,192-token cap: one of two long texts degenerated into a
> non-terminating repetition loop that filled the entire cap, and the clean
> run left only ~2.5 GB of the 16 GB unified memory free. At 16,384 tokens
> the process over-committed physical memory (16.5 GB active) and survived
> only because macOS compressed other processes. The binding constraint is
> not the cap — it is **EOS reliability and KV-cache memory** — so raising
> the cap makes things worse, not better.

## Environment

| Item | Value |
|------|-------|
| Machine | MacBook Pro (M2 Pro, 16 GB unified memory, Mac14,10) |
| Backend | `mlx`, model `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`, clone mode |
| Voice prompt | `lsmith` (.wav + .txt pair), seed 424242 |
| mlx / mlx-audio | 0.32.0 / 0.4.8 |
| Chunking | disabled (`max_chunk_chars: 0` request + `max_chunk_tokens: 100000` config) — one `generate()` call per text |
| Harness | stdlib-only `prf9_harness.py` (wall clock, WAV decode → duration/peak/RMS, `/stats` + `ps` polling at 5 s, log scrape) |

Texts (all deterministic, in the harness): probe 124 chars, A 2,278,
B 5,578, C 7,584, D 15,442 (= A+B+C).

## Headline findings

1. **The premise of PRF-9 is false on MLX.** `max_new_tokens` never reaches
   the model. mlx-audio's `generate()` names the cap `max_tokens`
   (`mlx_audio/tts/models/qwen3_tts/qwen3_tts.py:1153`, default **4096**) and
   swallows unknown kwargs — our engine passes `max_new_tokens=` (and
   `language=`; mlx-audio's param is `lang_code=`) into `**kwargs`, where both
   die silently (`qwen3_tts/core/engine/inference.py:385-393`, batch path;
   same shape on the streaming path at `:514`). **The effective shipped cap
   on MLX is 4,096 tokens, not 2,048** — `generation.max_new_tokens` and the
   `/generate` request field are dead knobs on this backend. The 2,048 cap
   documented in CLAUDE.md binds only the torch backend.
2. **The frame rate is 12.5 Hz, not 12.** mlx-audio computes streaming
   windows as `interval × 12.5`, and every cap in this measurement landed at
   exactly `duration × 12.5` tokens. 4,096 tokens = 327.68 s exactly;
   8,192 = 655.36; 16,384 = 1,310.72.
3. **Long single-call generations suffer non-deterministic runaway (EOS
   failure).** Same text, same seed, same effective cap terminated naturally
   in one server process and fell into a static repetition loop in another
   (sampling is reproducible within a process — identical RMS fingerprints —
   but not across restarts). The loop never emits EOS; **the cap is the only
   thing that stops it.** Raising the cap converts "truncated at 327 s" into
   "hundreds of extra seconds of garbage plus near-OOM".
4. **Memory scales with generated tokens and exhausts the machine well below
   16,384.** MLX active after clean load: ~3.0 GB. After A (1,442 tok):
   6.0 GB; B (3,245 tok): 9.0 GB; C (4,162 tok): 13.5 GB; B-runaway (8,191
   tok): 15.7 GB; D-runaway (16,384 tok): **16.5 GB — above physical**,
   absorbed by macOS memory compression. (`mlx_memory_peak_mb` is the
   allocator high-water including cache — 16.7–18.5 GB throughout — and
   overstates wired memory; `mlx_memory_active_mb` is the honest metric.
   Process RSS stays ~0.1–0.5 GB because Metal allocations are not counted.)
5. **Chunked generation (the shipped default) remains the right
   architecture** on this hardware: chunks of ≤500 chars stay far below any
   cap, keep KV cache small, and make truncation impossible. The silent
   truncation cliff only exists for single-call generation
   (`max_chunk_chars: 0`).

## Results

Tokens = duration × 12.5. "Cap" = effective `max_tokens` seen by the model
(see finding 1 — the request `max_new_tokens` was a no-op until the temp
patch described under Method).

| Effective cap | Text | Chars | Wall s | Audio s | Tokens | RTF | MLX active MB | Outcome |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096 (shipped) | probe | 124 | 12.4 | 7.1 | 89 | 1.74 | 4,303 | complete (EOS) |
| 4096 (shipped) | A | 2,278 | 72.3 | 115.4 | 1,442 | 0.63 | 6,013 | complete (EOS) |
| 4096 (shipped) | B | 5,578 | 190.7 | 259.6 | 3,245 | 0.73 | 9,011 | complete (EOS) |
| 4096 (shipped) | C | 7,584 | 194.0 | 327.68 | 4,096 | 0.59 | 11,062 | **truncated at exactly 4,096** |
| 8192 (patched) | probe | 124 | 7.2 | 6.0 | 75 | 1.20 | 4,554 | complete (EOS) |
| 8192 (patched) | B | 5,578 | 501.8 | 655.28 | 8,191 | 0.77 | 15,709 | **runaway loop to cap** |
| 8192 (patched, re-run at 4096) | B | 5,578 | 244.0 | 327.61 | 4,095 | 0.75 | 11,418 | **same loop, cut at 4,096** |
| 8192 (patched) | C | 7,584 | 246.5 | 332.97 | 4,162 | 0.74 | 13,505 | complete (EOS) |
| 16384 (patched) | probe | 124 | 8.0 | 6.3 | 79 | 1.27 | 5,376 | complete (EOS) |
| 16384 (patched) | D | 15,442 | 932.4 | 1,310.72 | 16,384 | 0.71 | 16,476 | **runaway to cap; > physical RAM** |

Natural speaking rate for this voice is ~21–23 chars/s (B natural: 21.5;
C: 22.8). D's 15,442 chars need ~9,000–10,000 tokens legitimately; it
generated 16,384 — roughly 40% of the output was degenerate continuation.

## Runaway analysis (per-20 s RMS energy)

Natural speech varies smoothly (B natural run: 0.032→0.038 across the
clip; A and C clean in every run). The degenerate B sample, by contrast:

- collapses at ~150 s (0.035 → 0.013), then
- locks at **exactly 0.0221 RMS for 160+ s** (a static loop), then
- at the 8,192 cap's extension, drifts to a second plateau of exactly
  0.0206 and holds it to the cap.

The 4,096-capped and 8,192-capped runs of that sample share a
bucket-for-bucket identical prefix — within one process the seeded sampling
reproduces exactly; across restarts it does not, which is why the same
text+seed terminated cleanly in the first server process and looped in the
second. D's runaway shows the same disease diffusely: ~100 s of static
energy mid-run, then an elevated-energy regime (0.046–0.052 vs the natural
0.028–0.035) and peak amplitude 0.59 (clean runs: 0.13–0.33) through to
the cap.

No NaN/Inf in any output; no `Low memory` warnings, Metal crash, or
half-retry lines in the server log at any tier. RTF stays 0.59–0.77 for
long texts (probe RTF 1.2–1.7 includes first-call warmup).

## Go/no-go against the plan's criteria

> GO iff at 8192: both texts stable AND `mlx_memory_peak_mb` leaves ≥ ~4 GB
> of 16 GB AND no OOM warnings/retries.

- **Stability: FAIL** — B degenerated at 8192 (655 s of looped audio where
  260 s was spoken).
- **Headroom: FAIL** — clean C left 13.5 GB active (2.5 GB free); the B
  runaway left 0.3 GB free.
- **16384 (informational): FAIL** — active 16.5 GB exceeds the 16 GB
  machine; the run survived on macOS compression, which is not a shipping
  posture.

## Recommended follow-ups (tickets, not in this lane)

1. **Bug: forward the cap on MLX** — map `max_new_tokens` → `max_tokens`
   in the clone call (`inference.py:385-393`) and the streaming generator
   (`:514`), or normalize the param name at the engine boundary. Until
   then, `generation.max_new_tokens` and the `/generate` field are silent
   no-ops on MLX and the Pydantic `le=8192` ceiling is unenforceable there.
2. **Bug: forward the language on MLX** — mlx-audio's param is `lang_code`;
   our `language=` is swallowed, so clone mode always runs `auto`.
3. **Runaway guard before any cap raise** — an EOS-skip detector
   (repetition check on generated codes, or a per-request token budget ≈
   chars × k) is a hard prerequisite; without it a raised cap only buys
   longer garbage.
4. **Close PRF-9 as measured-NO-GO** on 16 GB Apple Silicon for caps ≥
   8,192; keep chunked generation as the long-form architecture. Revisit
   only on hardware ≥ 32 GB or with a streaming runaway guard.
5. **Correct CLAUDE.md / ARCHITECTURE**: the "single MLX generate() capped
   at max_new_tokens=2048 (~170 s @ 12 Hz)" note is wrong on both numbers —
   effective cap 4,096 via mlx-audio default, frame rate 12.5 Hz (327.7 s
   ceiling). Fold into the fix PR for (1).
6. The measurement harness (stdlib-only, retained with this session's job
   artifacts) doubles as the RTF/peak-memory harness recommended by
   `docs/plans/tts-model-alternatives-2026-08-15.md` item 5 for benchmarking
   alternative clone engines.

## Method notes & deviations

- The tier plan's "config-only" mechanism does not work: `/generate`
  injects the request-schema default over `generation.max_new_tokens`
  (`app_generation.py:206`, ceiling at `server/validation.py:49`), so the
  cap must travel in the request body — and, per finding 1, still needs the
  engine kwarg mapping to reach MLX.
- Tier "2048" as planned was actually **4,096-effective** (the swallowed
  knob); it is reported as the shipped baseline. Tiers 8,192 and 16,384
  used a temporary uncommitted patch (cap forwarded as `max_tokens`;
  `le=16384` in `validation.py` for the top tier), both reverted after the
  run — the working tree was verified clean of all measurement mutations.
- One probe run crashed client-side after generation completed (harness
  thread bug, fixed); its server-side row (124 chars / 17.7 s) is
  consistent with the re-run and excluded from the table.
- The generation cache did not produce false hits (cap varies the key; no
  cache-hit lines in any measurement window).
- Server log cross-check: `Inference complete: N chars, X s` matched
  client-measured wall within 0.5 s on every row. Engine-side char counts
  differ slightly from sent text (2,318 vs 2,278 for A) — internal text
  normalization; audio-based metrics are unaffected.

— measured 2026-08-15, 13:05–14:00 EDT; rows and WAVs retained in the
session job directory (`prf9_results.jsonl`, `prf9_audio/*.wav`).
