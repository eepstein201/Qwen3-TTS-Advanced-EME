"""LLM-as-a-Judge for TTS prompt adherence evaluation.

Evaluates whether generated audio transcription matches the original
text prompt in terms of content, tone, and completeness. Uses an LLM
with a strict rubric to score adherence.

This is a non-blocking evaluation — failures do not block CI/CD.
"""

import logging
import os

logger = logging.getLogger("tts.eval.llm_judge")

RUBRIC = """
You are evaluating TTS (Text-to-Speech) output quality. Given an original
text prompt and the transcription of the generated audio, score the output
on a scale of 1-5:

5 = Perfect: Transcription matches prompt exactly (minor punctuation differences OK)
4 = Good: All words present, minor word substitutions that don't change meaning
3 = Acceptable: Most content preserved, some words missing or substituted
2 = Poor: Significant content missing or many word errors
1 = Fail: Output is unintelligible or completely different from prompt

Respond with ONLY a JSON object:
{"score": <1-5>, "pass": <true if score >= 3>, "reasoning": "<brief explanation>"}
"""


def evaluate_prompt_adherence(
    original_prompt: str,
    transcription: str,
    llm_provider: str = "anthropic",
    api_key_env: str = "ANTHROPIC_API_KEY",
) -> dict:
    """Evaluate how well generated audio adheres to the original prompt.

    Args:
        original_prompt: The original text sent to TTS.
        transcription: Whisper transcription of the generated audio.
        llm_provider: LLM provider to use ("anthropic" or "openai").
        api_key_env: Environment variable name for API key.

    Returns:
        Dict with keys: score (1-5), pass (bool), reasoning (str).
        Returns {"score": 0, "pass": False, "reasoning": "..."} on error.
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return {
            "score": 0,
            "pass": False,
            "reasoning": f"No API key found in {api_key_env}",
        }

    user_message = (
        f"Original prompt:\n{original_prompt}\n\n"
        f"Audio transcription:\n{transcription}\n\n"
        "Score the transcription's adherence to the original prompt."
    )

    try:
        if llm_provider == "anthropic":
            return _call_anthropic(api_key, user_message)
        elif llm_provider == "openai":
            return _call_openai(api_key, user_message)
        else:
            return {"score": 0, "pass": False, "reasoning": f"Unknown provider: {llm_provider}"}
    except Exception as e:
        logger.error("LLM judge evaluation failed: %s", e)
        return {"score": 0, "pass": False, "reasoning": str(e)}


def _call_anthropic(api_key: str, user_message: str) -> dict:
    """Call Anthropic Claude API for evaluation."""
    import json

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=RUBRIC,
        messages=[{"role": "user", "content": user_message}],
    )

    result_text = response.content[0].text
    return json.loads(result_text)


def _call_openai(api_key: str, user_message: str) -> dict:
    """Call OpenAI API for evaluation."""
    import json

    import openai

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": user_message},
        ],
    )

    result_text = response.choices[0].message.content
    return json.loads(result_text)
