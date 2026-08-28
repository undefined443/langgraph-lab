"""Runtime configuration and LangSmith wiring.

This module centralises environment loading so every entry point (CLI, eval
script, LangGraph Studio) shares the same setup. Importing it has the side
effect of loading ``.env`` and, when a LangSmith key is present, enabling
tracing for the whole process.

The chat model is reached through an OpenAI-compatible endpoint. The defaults
point at OpenRouter, but any compatible gateway (OpenAI itself, DeepSeek,
a local vLLM, ...) works by overriding ``LLM_BASE_URL`` and ``LLM_API_KEY``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env once, as early as possible, before any LangChain module reads env.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable view of the knobs the demo exposes through the environment."""

    # Model id as the gateway expects it. OpenRouter uses "vendor/model" slugs;
    # pick one that supports function calling (this demo needs tool calls).
    model: str = os.getenv("LLM_MODEL", "openai/gpt-5.6-luna")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    # OpenAI-compatible endpoint. Default: OpenRouter.
    base_url: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

    # LangSmith project the traces land in. The LangSmith SDK reads the raw
    # env vars itself; this field is only for logging what we ended up with.
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "langgraph-lab")

    @property
    def api_key(self) -> str:
        """Gateway API key, trying the demo var then common fallbacks."""
        return (
            os.getenv("LLM_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )

    @property
    def tracing_enabled(self) -> bool:
        """True when LangSmith tracing is actually turned on for this run."""
        flag = os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}
        return flag and bool(os.getenv("LANGSMITH_API_KEY"))


def _mirror_langchain_env() -> None:
    """Copy ``LANGSMITH_*`` vars to the legacy ``LANGCHAIN_*`` names.

    Different releases of langchain-core look for one prefix or the other.
    Mirroring keeps tracing working regardless of which one wins.
    """
    pairs = {
        "LANGSMITH_TRACING": "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY": "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT": "LANGCHAIN_PROJECT",
        "LANGSMITH_ENDPOINT": "LANGCHAIN_ENDPOINT",
    }
    for src, dst in pairs.items():
        if os.getenv(src) and not os.getenv(dst):
            os.environ[dst] = os.environ[src]


_mirror_langchain_env()

settings = Settings()
