#!/usr/bin/env python3
"""Reproduction harness for issue #192 — custom-mode token-cap runaway.

Issue #192: a custom-mode generation ran to the 2048-token cap without emitting
EOS, taking 215.8 s for a 16-character input, where healthy runs take ~7 s.

The runaway is not deterministic, so the point of this harness is to hold one
condition fixed at a time and count how many runs hit the cap. It drives
``_run_inference_mlx`` directly rather than the server, so the sampling
parameters are exactly what production sends, with no HTTP, cache, or chunking
in between.

Wall-clock alone is NOT a usable signal here — that is what caused the original
misattribution to machine contention. What distinguishes the runaway is the
token count: at 12.5 Hz, 2048 tokens is ~164 s of audio for an input that
should produce ~1.5 s. Emitted tokens are inferred from output duration, which
is the only handle available once ``model.generate_*`` has returned.

Measured results (2026-08-18, M2 Pro, 1.7B 8-bit): 0/60 with varying seeds and
0/30 with all three models resident. See
``docs/reviews/issue-192-custom-mode-runaway-2026-08-18.md``.

Usage:
    conda run -n qwen3-tts-mlx python scripts/probe_issue192.py --runs 60
    conda run -n qwen3-tts-mlx python scripts/probe_issue192.py --load-all --runs 30
    conda run -n qwen3-tts-mlx python scripts/probe_issue192.py --text "Token test three" --out rows.json

Exit code is 1 if any run hit the cap, so this can gate a longer sweep.
"""
import argparse
import json
import logging
import os
import sys
import time

# The engine's own default for MLX; a run reaching this emitted no EOS.
DEFAULT_MAX_TOKENS = 2048

# mlx-audio's codec frame rate. Tokens are not returned by the generate call,
# so they are inferred from audio duration at this rate.
CODEC_FRAME_RATE_HZ = 12.5

# Within 2% of the cap counts as capped: the trailing partial frame and any
# post-processing trim make an exact equality test fragile.
CAP_TOLERANCE = 0.98

# The e2e text whose two failures are recorded in #192. "Token test 3" reaches
# the engine as this, because _normalize_text expands the digit.
DEFAULT_TEXT = "Token test three"

# What the server sends for a request that specifies neither speaker nor
# language, via `speaker or "Ryan"` in _run_inference_mlx.
DEFAULT_SPEAKER = "Ryan"
DEFAULT_LANGUAGE = "auto"

# Production sampling defaults from _get_mlx_gen_params().
SAMPLING = {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--load-all",
        action="store_true",
        help="Load clone and design too, reproducing the e2e resident set",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Seed run i with seed-base+i; omit to let each run seed freely",
    )
    parser.add_argument("--out", help="Write the per-run rows here as JSON")
    return parser.parse_args(argv)


def load_models(load_all):
    from qwen3_tts.core.engine import load_model

    names = ["clone", "design", "custom"] if load_all else ["custom"]
    models = {}
    for name in names:
        start = time.time()
        models[name] = load_model(name)
        print(f"loaded {name} in {time.time() - start:.1f}s", flush=True)
    return models


def report_memory():
    """Print MLX active memory, so a pressure run records what it achieved."""
    try:
        import mlx.core as mx

        print(f"mlx active memory: {mx.get_active_memory() / 1e9:.2f} GB", flush=True)
    except Exception as exc:  # noqa: BLE001 — diagnostics only, never fatal
        print(f"(no mlx memory reading: {exc})", flush=True)


def run_once(model, args, seed):
    from qwen3_tts.core.engine.inference import _run_inference_mlx

    params = dict(SAMPLING, max_new_tokens=args.max_tokens)
    if seed is not None:
        params["seed"] = seed

    start = time.time()
    audio, sample_rate = _run_inference_mlx(
        model,
        args.text,
        "custom",
        params,
        language=args.language,
        speaker=args.speaker,
    )
    elapsed = time.time() - start

    samples = int(getattr(audio, "shape", [0])[0]) if audio is not None else 0
    seconds = samples / sample_rate if sample_rate else 0.0
    approx_tokens = round(seconds * CODEC_FRAME_RATE_HZ)
    return {
        "seed": seed,
        "elapsed_s": round(elapsed, 1),
        "audio_s": round(seconds, 1),
        "approx_tokens": approx_tokens,
        "CAPPED": approx_tokens >= args.max_tokens * CAP_TOLERANCE,
    }


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.setdefault("TTS_BACKEND", "mlx")

    models = load_models(args.load_all)
    report_memory()

    rows = []
    for i in range(args.runs):
        seed = None if args.seed_base is None else args.seed_base + i
        rows.append(run_once(models["custom"], args, seed))
        print(json.dumps(rows[-1]), flush=True)

    capped = [r for r in rows if r["CAPPED"]]
    tokens = sorted(r["approx_tokens"] for r in rows)
    print(
        f"\n=== {len(capped)}/{len(rows)} hit the {args.max_tokens}-token cap; "
        f"tokens min/median/max = {tokens[0]}/{tokens[len(tokens) // 2]}/{tokens[-1]} ===",
        flush=True,
    )

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=1)
        print(f"wrote {args.out}", flush=True)

    return 1 if capped else 0


if __name__ == "__main__":
    sys.exit(main())
