"""Backend configuration loaded from environment variables (never exposed in the UI)."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

HF_INFERENCE_ENDPOINT: str = os.getenv("HF_INFERENCE_ENDPOINT", "").strip()
HF_TOKEN: str = os.getenv("HF_TOKEN", "").strip()
HF_ROUTER_BASE_URL: str = os.getenv("HF_ROUTER_BASE_URL", "https://router.huggingface.co/v1").strip()
HF_MODEL: str = os.getenv("HF_MODEL", "").strip()
FMP_API_KEY: str = os.getenv("FMP_API_KEY", "").strip()


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
