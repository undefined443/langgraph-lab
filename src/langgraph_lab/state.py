"""Graph state definition.

The state is the single object that flows through every node. Here it only
carries the message history, but real apps add their own keys (retrieved docs,
a scratchpad, a step counter, ...).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the ReAct agent.

    ``messages`` uses the ``add_messages`` reducer: whatever a node returns
    under this key is appended to the running list (and message ids are
    de-duplicated) instead of replacing it.
    """

    messages: Annotated[list[AnyMessage], add_messages]
