"""Chat model factory.

Kept in its own module so the graph, the evaluators, and any notebook all
construct the model the same way. Talks to an OpenAI-compatible endpoint
(OpenRouter by default) via ``ChatOpenAI``.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import settings

# Optional OpenRouter ranking headers. Harmless on other gateways.
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/langgraph-lab",
    "X-Title": "langgraph-lab",
}


def build_chat_model(**overrides) -> ChatOpenAI:
    """Return a configured ``ChatOpenAI`` instance.

    Args:
        **overrides: Keyword args that take precedence over ``Settings``,
            e.g. ``build_chat_model(temperature=0.7)``.
    """
    params = {
        "model": settings.model,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "timeout": 60,
        "base_url": settings.base_url,
        "api_key": settings.api_key,
    }
    if "openrouter.ai" in settings.base_url:
        params["default_headers"] = _OPENROUTER_HEADERS
    params.update(overrides)
    return ChatOpenAI(**params)
