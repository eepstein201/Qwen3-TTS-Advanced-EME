#!/usr/bin/env python3
"""In-flight degradation detection for /health and /stats.

A wedged server answers /health with ``"status": "ok"`` and every model
reported loaded while taking minutes per character. That happened during the
2026-08-02 E2E work: the log showed ``Inference complete: 22 chars, 7314.2s``
(~332 s/char) and every generation-bearing test blew its timeout, reading as a
code regression rather than an environment problem.

Detection is deliberately on the IN-FLIGHT request, not completed history: that
generation ran for two hours before finishing, so a completed-samples design
would not have raised anything until long after every caller had timed out.

Run: pytest tests/test_health_degraded.py -v
"""

import types
import unittest

from qwen3_tts.server.app_lifespan import (
    _DEGRADED_MIN_ELAPSED_SEC,
    _DEGRADED_SEC_PER_CHAR,
    detect_degraded_generation,
)


def _state(active=True, elapsed=0.0, text_length=100):
    """Minimal app_state stand-in; now= is passed explicitly by callers."""
    return types.SimpleNamespace(
        generation_state={
            "active": active,
            "start_time": 0.0,
            "text_length": text_length,
        }
    )


class TestDetectDegradedGeneration(unittest.TestCase):
    def test_idle_server_is_not_degraded(self):
        result = detect_degraded_generation(_state(active=False), now=1e9)
        self.assertFalse(result["degraded"])
        self.assertIsNone(result["elapsed_sec"])

    def test_fast_generation_is_not_degraded(self):
        """A healthy generation: well under the threshold."""
        result = detect_degraded_generation(
            _state(text_length=100), now=_DEGRADED_MIN_ELAPSED_SEC + 100
        )
        self.assertFalse(result["degraded"])

    def test_short_elapsed_never_trips_however_few_chars(self):
        """s/char is meaningless early — 1 char at 10s must not fire."""
        result = detect_degraded_generation(
            _state(text_length=1), now=_DEGRADED_MIN_ELAPSED_SEC - 1
        )
        self.assertFalse(result["degraded"])
        self.assertIsNone(result["sec_per_char"])

    def test_the_observed_pathology_is_detected(self):
        """The real case: 22 chars, 7314s (~332 s/char)."""
        result = detect_degraded_generation(_state(text_length=22), now=7314.2)
        self.assertTrue(result["degraded"])
        self.assertGreater(result["sec_per_char"], _DEGRADED_SEC_PER_CHAR)

    def test_slow_but_working_long_generation_is_not_degraded(self):
        """Guards against false positives: a big request legitimately taking
        a long time has a LOW s/char and must stay healthy."""
        result = detect_degraded_generation(_state(text_length=9000), now=1800.0)
        self.assertFalse(result["degraded"])

    def test_unknown_text_length_falls_back_to_elapsed(self):
        """text_length 0 must not divide by zero nor declare health."""
        result = detect_degraded_generation(
            _state(text_length=0), now=_DEGRADED_MIN_ELAPSED_SEC + 1
        )
        self.assertTrue(result["degraded"])
        self.assertIsNone(result["sec_per_char"])


class TestHealthEndpointDoesNotLeakRequestSize(unittest.TestCase):
    """The public /health must expose the boolean and nothing more.

    /generation-status already strips eta_sec / batch_total / chunk_total from
    unauthenticated callers because they reveal the in-flight request's size.
    sec_per_char and text_length would reveal the same thing, so they belong on
    the authenticated /stats only.
    """

    def test_health_response_model_declares_degraded(self):
        """Without the field, FastAPI's response_model silently strips it."""
        from qwen3_tts.server.validation import HealthResponse

        self.assertIn("degraded", HealthResponse.model_fields)

    def test_health_response_model_omits_the_supporting_numbers(self):
        from qwen3_tts.server.validation import HealthResponse

        for leaky in ("sec_per_char", "text_length", "elapsed_sec"):
            self.assertNotIn(
                leaky,
                HealthResponse.model_fields,
                f"{leaky} reveals in-flight request size to unauthenticated "
                "callers; keep it on the authenticated /stats",
            )

    def test_detector_returns_numbers_for_authenticated_callers(self):
        """/stats consumes the whole dict, so the numbers must be present."""
        result = detect_degraded_generation(_state(text_length=22), now=7314.2)
        self.assertIn("sec_per_char", result)
        self.assertIn("threshold_sec_per_char", result)


if __name__ == "__main__":
    unittest.main()
