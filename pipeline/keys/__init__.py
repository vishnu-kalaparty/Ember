"""
API keys for the 3 action-sentence labeling LLMs, read from the environment
(or from .env at the repo root; see .env.example).
"""

import os

from .. import config  # noqa: F401  (loads .env)

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

__all__ = ["CLAUDE_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY"]
