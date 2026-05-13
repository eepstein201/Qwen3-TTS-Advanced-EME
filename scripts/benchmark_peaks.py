#!/usr/bin/env python3
"""Benchmark waveform peaks calculation performance.

Measures how long it takes to calculate peaks for various audio durations.
Target: < 50ms for 10 seconds of audio.
"""

import time

import numpy as np

from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks


def benchmark_peaks_calculation():
    """Benchmark peak calculation performance."""
    print("Waveform Peaks Calculation Performance Benchmark")
    print("=" * 50)

    test_cases = [
        ("1 second", 24000),
        ("5 seconds", 120000),
        ("10 seconds", 240000),
        ("30 seconds", 720000),
        ("1 minute", 1440000),
    ]

    results = []

    for name, samples in test_cases:
        # Generate audio
        audio = np.random.randn(samples).astype(np.float32)

        # Time peak calculation
        start = time.time()
        peaks = calculate_waveform_peaks(audio, num_peaks=500)
        elapsed = time.time() - start

        ms_time = elapsed * 1000
        results.append((name, ms_time))

        # Calculate audio duration
        duration_sec = samples / 24000

        print(f"{name:12s} ({duration_sec:5.1f}s): {ms_time:6.2f}ms")

        # Verify performance
        if elapsed < 0.05:  # 50ms target
            print(f"  ✓ Target met (<50ms)")
        else:
            print(f"  ⚠ Above target (should be <50ms)")

    print()
    print("Summary:")
    print("-" * 50)
    print("Target: < 50ms for 10 seconds of audio")

    # Verify 10-second case
    ten_sec_result = next((r for r in results if r[0] == "10 seconds"), None)
    if ten_sec_result:
        _, ms_time = ten_sec_result
        if ms_time < 50.0:
            print(f"✓ 10s audio: {ms_time:.2f}ms (EXCELLENT - well under 50ms target)")
        elif ms_time < 100.0:
            print(f"✓ 10s audio: {ms_time:.2f}ms (GOOD - under 100ms)")
        else:
            print(f"⚠  10s audio: {ms_time:.2f}ms (NEEDS OPTIMIZATION - should be <50ms)")

    print()
    print("All benchmarks completed successfully!")


if __name__ == "__main__":
    benchmark_peaks_calculation()
