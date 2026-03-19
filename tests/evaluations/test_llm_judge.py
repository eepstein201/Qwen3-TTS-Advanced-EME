"""Tests for LLM-as-a-Judge evaluation prototype.

Tests the evaluation framework and rubric without requiring API keys.
Full evaluation tests are skipped when API keys are not available.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestLLMJudgeFramework(unittest.TestCase):
    """Test the LLM judge framework without API calls."""

    def test_rubric_exists(self):
        from tests.evaluations.llm_judge import RUBRIC

        self.assertIn("score", RUBRIC.lower())
        self.assertIn("1-5", RUBRIC)

    def test_evaluate_without_api_key(self):
        """Without API key, should return score=0 with explanation."""
        # Ensure no API key is set
        env_backup = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            from tests.evaluations.llm_judge import evaluate_prompt_adherence

            result = evaluate_prompt_adherence(
                original_prompt="Hello world",
                transcription="Hello world",
            )
            self.assertEqual(result["score"], 0)
            self.assertFalse(result["pass"])
            self.assertIn("API key", result["reasoning"])
        finally:
            if env_backup:
                os.environ["ANTHROPIC_API_KEY"] = env_backup

    def test_evaluate_returns_dict_structure(self):
        """Result should always have score, pass, and reasoning keys."""
        from tests.evaluations.llm_judge import evaluate_prompt_adherence

        result = evaluate_prompt_adherence("test", "test")
        self.assertIn("score", result)
        self.assertIn("pass", result)
        self.assertIn("reasoning", result)

    def test_unknown_provider_returns_error(self):
        """Unknown provider should return error result."""
        os.environ["TEST_KEY"] = "fake"
        try:
            from tests.evaluations.llm_judge import evaluate_prompt_adherence

            result = evaluate_prompt_adherence(
                original_prompt="test",
                transcription="test",
                llm_provider="unknown",
                api_key_env="TEST_KEY",
            )
            self.assertEqual(result["score"], 0)
            self.assertFalse(result["pass"])
        finally:
            del os.environ["TEST_KEY"]


if __name__ == "__main__":
    unittest.main()
