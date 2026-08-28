"""The agent graph: a hand-built ReAct loop.

Topology::

    START -> agent -> (has tool calls?) --yes--> tools -> agent
                           |
                           no
                           v
                          END

``agent`` calls the LLM (with tools bound); ``tools`` executes whatever tool
calls came back; the conditional edge decides whether to loop again or stop.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .llm import build_chat_model
from .state import AgentState
from .tools import TOOLS

_SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a concise assistant demonstrating LangGraph. "
        "Prefer calling a tool over guessing: use `calculator` for arithmetic, "
        "`get_current_time` for the clock, and `search_knowledge_base` for "
        "questions about LangGraph or LangSmith. "
        "After using tools, answer in one short paragraph."
    )
)

_model_with_tools = None


def _get_model():
    """Build the tool-bound model on first use and cache it.

    Lazy so that merely importing this module (e.g. for graph inspection or in
    tests) does not require an API key.
    """
    global _model_with_tools
    if _model_with_tools is None:
        _model_with_tools = build_chat_model().bind_tools(TOOLS)
    return _model_with_tools


def _agent_node(state: AgentState) -> dict:
    """Call the model on the running history and return its reply."""
    response = _get_model().invoke([_SYSTEM_PROMPT, *state["messages"]])
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    """Route to the tool node when the model asked for a tool, else finish."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Assemble and compile the graph.

    Args:
        checkpointer: Optional persistence backend. Pass a ``MemorySaver`` (or
            a SQLite/Postgres saver) to get per-thread memory. Leave it ``None``
            when running under the LangGraph API, which supplies its own.
    """
    builder = StateGraph(AgentState)

    builder.add_node("agent", _agent_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", _should_continue, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


# Plain compiled graph for LangGraph Studio / `langgraph dev` (see langgraph.json).
graph = build_graph()
