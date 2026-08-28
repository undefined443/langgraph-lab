"""Interactive terminal chat for the agent.

Run with ``uv run main.py`` or ``uv run langgraph-lab``.

Demonstrates:
  * compiling the graph with a ``MemorySaver`` checkpointer
  * a ``thread_id`` that ties turns together into one conversation
  * streaming node-by-node updates instead of waiting for the final answer
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langsmith import uuid7
from rich.console import Console
from rich.panel import Panel

from .config import settings
from .graph import build_graph

console = Console()


def _print_banner() -> None:
    status = (
        f"[green]on[/green] -> project '{settings.langsmith_project}'"
        if settings.tracing_enabled
        else "[yellow]off[/yellow] (set LANGSMITH_TRACING=true + LANGSMITH_API_KEY)"
    )
    console.print(
        Panel(
            f"model     : {settings.model}\n"
            f"tracing   : {status}\n"
            f"commands  : /new  reset conversation   |   /exit  quit",
            title="langgraph-lab",
            border_style="cyan",
        )
    )


def _render_update(node: str, payload: dict) -> None:
    """Pretty-print one streamed node update."""
    for message in payload.get("messages", []):
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                console.print(
                    f"  [magenta]call[/magenta] {call['name']}({call['args']})"
                )
            if message.content:
                console.print(f"[bold green]assistant[/bold green] {message.content}")
        elif isinstance(message, ToolMessage):
            console.print(f"  [blue]result[/blue] {message.content}")


def run() -> None:
    """Start the read-eval-print loop."""
    graph = build_graph(checkpointer=MemorySaver())
    thread_id = str(uuid7())
    _print_banner()

    while True:
        try:
            user_input = console.input("\n[bold cyan]you[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            return

        if not user_input:
            continue
        if user_input == "/exit":
            console.print("bye")
            return
        if user_input == "/new":
            thread_id = str(uuid7())
            console.print(f"[dim]new thread {thread_id}[/dim]")
            continue

        config = {"configurable": {"thread_id": thread_id}}
        # In "updates" mode each streamed chunk is {node_name: state_delta}.
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates",
        ):
            for node, payload in chunk.items():
                _render_update(node, payload)
