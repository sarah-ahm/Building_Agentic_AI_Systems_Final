"""Configuration and secret loading.

Secrets follow the exact course-lab pattern: prefer a local ``.env`` (via
``python-dotenv``), then fall back to Colab Secrets (``google.colab.userdata``).
Nothing is ever hard-coded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"

# Files we treat as source. Kept small on purpose (the tool is language-agnostic
# but the course target is TypeScript).
SOURCE_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".py"}

# Directories that never contain hand-authored source we want to document.
SKIP_DIRS = {"node_modules", "dist", "build", "coverage", ".git", "__pycache__"}


def get_api_key() -> str | None:
    """Return the Gemini API key from the environment or Colab Secrets, or None.

    Order: ``.env`` / real environment first (works locally and in CI), then
    ``google.colab.userdata`` (the only place a key lives inside Colab).
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key

    try:  # pragma: no cover - only runs inside Colab
        from google.colab import userdata

        return userdata.get("GEMINI_API_KEY")
    except Exception:
        return None


@dataclass
class Settings:
    """Runtime settings, mostly sourced from environment variables."""

    provider: str = "gemini"
    model: str = DEFAULT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    api_key: str | None = None
    max_file_bytes: int = 200_000
    concurrency: int = 4
    mock: bool = False

    @classmethod
    def from_env(cls, mock: bool = False) -> "Settings":
        """Build settings from environment variables (with sensible defaults)."""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "gemini"),
            model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            embed_model=os.getenv("GEMINI_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            api_key=None if mock else get_api_key(),
            mock=mock,
        )
