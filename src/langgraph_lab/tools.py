"""Demo tools for the agent.

All tools are local and deterministic so the demo runs offline and produces
stable LangSmith traces. ``@tool`` turns a plain function into a structured
tool: the docstring becomes the description the model sees, and the type hints
become the argument schema.
"""

from __future__ import annotations

import ast
import operator
from datetime import UTC, datetime, timedelta

from langchain_core.tools import tool
from langsmith import traceable

# --- a tiny safe arithmetic evaluator ---------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a parsed arithmetic expression."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


# --- knowledge base fixture -----------------------------------------------

_KB = {
    "state": (
        "In LangGraph the state is a shared, typed object passed between nodes. "
        "Fields annotated with a reducer (e.g. add_messages) are merged instead "
        "of overwritten."
    ),
    "checkpointer": (
        "A checkpointer persists graph state per thread_id, which is what gives "
        "the agent multi-turn memory and enables time-travel / resume."
    ),
    "conditional edge": (
        "A conditional edge routes to the next node by running a function over "
        "the current state, e.g. 'go to tools if the last message has tool calls, "
        "otherwise finish'."
    ),
    "langsmith": (
        "LangSmith captures a trace tree for every run: each node, tool call and "
        "LLM call becomes a span with inputs, outputs, latency and token counts."
    ),
}


@traceable(run_type="retriever", name="kb_lookup")
def _kb_lookup(query: str) -> list[str]:
    """Naive keyword retrieval over the fixture KB.

    Decorated with ``@traceable`` so it shows up as its own span nested under
    the tool call in LangSmith.
    """
    q = query.lower()
    hits = [
        text
        for key, text in _KB.items()
        if key in q or any(w in q for w in key.split())
    ]
    return hits or ["No matching entry in the knowledge base."]


# --- the tools the agent can call ----------------------------------------


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression such as '3 * (4 + 5) / 2'.

    Supports + - * / // % ** and parentheses. Use this instead of doing mental
    math.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"error: {exc}"


@tool
def get_current_time(offset_hours: int = 0) -> str:
    """Return the current UTC time, optionally shifted by a whole-hour offset.

    Args:
        offset_hours: Hours to add to UTC, e.g. 8 for China Standard Time.
    """
    now = datetime.now(UTC) + timedelta(hours=offset_hours)
    label = "UTC" if offset_hours == 0 else f"UTC{offset_hours:+d}"
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} {label}"


@tool
def search_knowledge_base(query: str) -> str:
    """Look up LangGraph / LangSmith concepts in the local knowledge base."""
    return "\n".join(f"- {line}" for line in _kb_lookup(query))


TOOLS = [calculator, get_current_time, search_knowledge_base]
