"""Thin chat client supporting HF dedicated endpoints and HF Router."""
from __future__ import annotations

import time

from huggingface_hub import InferenceClient
from openai import OpenAI

import config

_hf_endpoint_client: InferenceClient | None = None
_hf_router_client: OpenAI | None = None


def _get_endpoint_client() -> InferenceClient:
    """Lazily build a dedicated HF Inference Endpoint client."""
    global _hf_endpoint_client
    if _hf_endpoint_client is None:
        config.validate_config()
        _hf_endpoint_client = InferenceClient(
            base_url=config.HF_INFERENCE_ENDPOINT,
            token=config.HF_TOKEN,
        )
    return _hf_endpoint_client


def _get_router_client() -> OpenAI:
    """Lazily build an OpenAI-compatible HF Router client."""
    global _hf_router_client
    if _hf_router_client is None:
        config.validate_config()
        _hf_router_client = OpenAI(
            base_url=config.HF_ROUTER_BASE_URL,
            api_key=config.HF_TOKEN,
        )
    return _hf_router_client


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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            if config.HF_INFERENCE_ENDPOINT:
                result = _get_endpoint_client().chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.4,
                )
                content = result.choices[0].message.content
            else:
                result = _get_router_client().chat.completions.create(
                    model=config.HF_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.4,
                )
                content = result.choices[0].message.content
            if content and content.strip():
                return content.strip()
            last_error = RuntimeError("Empty response from model")
        except Exception as exc:  # network, rate limit, endpoint warming up, etc.
            last_error = exc
        if attempt < retries - 1:
            time.sleep(backoff * (2**attempt))
    raise RuntimeError(f"LLM request failed after {retries} attempts: {last_error}")
