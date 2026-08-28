"""LangSmith evaluation demo.

What it does:
  1. creates (once) a small dataset in LangSmith
  2. runs the agent graph over every example  -> this is the "target"
  3. scores each run with two evaluators:
       * correctness   -- an LLM-as-judge comparing answer vs. reference
       * used_expected_tool -- did the agent call the tool we expected?
  4. prints the experiment URL

Run: ``uv run scripts/evaluate.py``
Requires ``LANGSMITH_API_KEY`` and ``LLM_API_KEY`` in .env.
"""

from __future__ import annotations

import os
import sys

from langchain_core.messages import AIMessage, HumanMessage
from langsmith import Client, evaluate

# Ensure ``import langgraph_lab`` works when run as a plain script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langgraph_lab.config import settings
from langgraph_lab.graph import build_graph
from langgraph_lab.llm import build_chat_model

DATASET_NAME = "langgraph-lab-react"

# input question -> reference answer + which tool we expect the agent to use.
EXAMPLES = [
    {
        "inputs": {"question": "What is 17 * 23 + 4?"},
        "outputs": {"answer": "395", "expected_tool": "calculator"},
    },
    {
        "inputs": {"question": "In LangGraph, what is a checkpointer for?"},
        "outputs": {
            "answer": "It persists graph state per thread_id, giving the agent "
            "multi-turn memory and enabling resume / time-travel.",
            "expected_tool": "search_knowledge_base",
        },
    },
    {
        "inputs": {"question": "Roughly what is the current time in UTC+8?"},
        "outputs": {
            "answer": "The current UTC+8 wall-clock time.",
            "expected_tool": "get_current_time",
        },
    },
    {
        "inputs": {
            "question": "Explain conditional edges in LangGraph in one sentence."
        },
        "outputs": {
            "answer": "A conditional edge picks the next node by running a "
            "function over the current state.",
            "expected_tool": "search_knowledge_base",
        },
    },
]


def ensure_dataset(client: Client) -> None:
    """Create the dataset and examples the first time only."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        return
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME, description="ReAct agent smoke set"
    )
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[e["inputs"] for e in EXAMPLES],
        outputs=[e["outputs"] for e in EXAMPLES],
    )
    print(f"created dataset '{DATASET_NAME}' with {len(EXAMPLES)} examples")


def run_agent(inputs: dict) -> dict:
    """Target under test: run the graph, report the answer and tools used."""
    graph = build_graph()
    result = graph.invoke({"messages": [HumanMessage(content=inputs["question"])]})

    tool_calls: list[str] = []
    answer = ""
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            tool_calls.extend(call["name"] for call in message.tool_calls)
            if message.content:
                answer = (
                    message.content
                    if isinstance(message.content, str)
                    else str(message.content)
                )
    return {"answer": answer, "tool_calls": tool_calls}


_judge = None


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge: is the agent's answer consistent with the reference?"""
    global _judge
    if _judge is None:
        _judge = build_chat_model(temperature=0)
    prompt = (
        "You are grading an assistant answer. Reply with only 'CORRECT' or 'INCORRECT'.\n"
        f"Question: {inputs['question']}\n"
        f"Reference answer: {reference_outputs['answer']}\n"
        f"Assistant answer: {outputs['answer']}\n"
        "The assistant is CORRECT if it is factually consistent with the reference, "
        "even if worded differently or more detailed."
    )
    verdict = _judge.invoke(prompt).content.strip().upper()
    return {"key": "correctness", "score": int("INCORRECT" not in verdict)}


def used_expected_tool(outputs: dict, reference_outputs: dict) -> dict:
    """Deterministic check: did the agent call the tool we expected?"""
    expected = reference_outputs.get("expected_tool")
    return {
        "key": "used_expected_tool",
        "score": int(expected in outputs["tool_calls"]),
    }


def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("LANGSMITH_API_KEY is required for the evaluation demo.")

    client = Client()
    ensure_dataset(client)

    results = evaluate(
        run_agent,
        data=DATASET_NAME,
        evaluators=[correctness, used_expected_tool],
        experiment_prefix="react-agent",
        metadata={"model": settings.model},
        client=client,
    )
    print("\ndone. open the experiment in LangSmith:")
    print(getattr(results, "experiment_name", DATASET_NAME))


if __name__ == "__main__":
    main()
