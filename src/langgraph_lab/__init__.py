"""langgraph-lab: a minimal LangGraph + LangSmith demo agent."""

from .graph import build_graph, graph

__all__ = ["build_graph", "graph", "main"]


def main() -> None:
    """Console-script entry point (``uv run langgraph-lab``)."""
    from .cli import run

    run()
