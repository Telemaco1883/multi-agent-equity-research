"""Backend configuration loaded from environment variables (never exposed in the UI)."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

HF_INFERENCE_ENDPOINT: str = os.getenv("HF_INFERENCE_ENDPOINT", "").strip()
HF_TOKEN: str = os.getenv("HF_TOKEN", "").strip()


def missing_config() -> list[str]:
    """Return the names of required environment variables that are not set."""
    required = {"HF_INFERENCE_ENDPOINT": HF_INFERENCE_ENDPOINT, "HF_TOKEN": HF_TOKEN}
    return [name for name, value in required.items() if not value]


def validate_config() -> None:
    """Raise RuntimeError if backend credentials are missing."""
    missing = missing_config()
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Set them in the backend environment or a .env file."
        )
