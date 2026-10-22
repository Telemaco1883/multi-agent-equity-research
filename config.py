"""Backend configuration loaded from environment variables (never exposed in the UI)."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st

        secret_value = st.secrets.get(name, "")
        if secret_value is None:
            return default
        return str(secret_value).strip() or default
    except Exception:
        return default


HF_INFERENCE_ENDPOINT: str = _secret("HF_INFERENCE_ENDPOINT")
HF_TOKEN: str = _secret("HF_TOKEN")
HF_ROUTER_BASE_URL: str = _secret("HF_ROUTER_BASE_URL", "https://router.huggingface.co/v1")
HF_MODEL: str = _secret("HF_MODEL")
FMP_API_KEY: str = _secret("FMP_API_KEY")


def missing_config() -> list[str]:
    """Return required env vars that are missing for the selected backend mode."""
    missing: list[str] = []
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    # If no dedicated endpoint is set, fall back to the HF Router (OpenAI-compatible API).
    if not HF_INFERENCE_ENDPOINT and not HF_MODEL:
        missing.append("HF_MODEL")
    if not FMP_API_KEY:
        missing.append("FMP_API_KEY")
    return missing


def validate_config() -> None:
    """Raise RuntimeError if backend credentials are missing."""
    missing = missing_config()
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Set them in the backend environment or a .env file."
        )
