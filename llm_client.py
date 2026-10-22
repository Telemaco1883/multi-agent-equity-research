"""Thin client for the Hugging Face inference endpoint configured in config.py."""
from __future__ import annotations

import time

from huggingface_hub import InferenceClient

import config

_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    """Lazily build the inference client from backend configuration."""
    global _client
    if _client is None:
        config.validate_config()
        _client = InferenceClient(base_url=config.HF_INFERENCE_ENDPOINT, token=config.HF_TOKEN)
    return _client


def generate_response(
    system_prompt: str,
    user_content: str,
    max_tokens: int = 1500,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """Generate a chat completion, retrying on transient errors.

    Raises RuntimeError after all attempts fail.
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            result = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.4)
            content = result.choices[0].message.content
            if content and content.strip():
                return content.strip()
            last_error = RuntimeError("Empty response from model")
        except Exception as exc:  # network, rate limit, endpoint warming up, etc.
            last_error = exc
        if attempt < retries - 1:
            time.sleep(backoff * (2**attempt))
    raise RuntimeError(f"LLM request failed after {retries} attempts: {last_error}")
