"""RepoScribe: an agentic, citation-grounded documentation generator."""

from .config import Settings
from .pipeline import run

__all__ = ["Settings", "run"]
__version__ = "0.1.0"
