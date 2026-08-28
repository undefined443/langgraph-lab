# langgraph-lab

A minimal **LangGraph + LangSmith** demo: a hand-built ReAct agent with tool
calling, multi-turn memory, streaming output, plus a LangSmith tracing and
evaluation workflow. The goal is to show the core concepts once, with code
short enough to read top to bottom.

## Concepts covered

| Concept                                                        | Where                         |
| -------------------------------------------------------------- | ----------------------------- |
| `StateGraph` / state with a reducer (`add_messages`)           | `src/langgraph_lab/state.py`  |
| Nodes and conditional edges                                    | `src/langgraph_lab/graph.py`  |
| ReAct loop: `agent -> tools -> agent`                          | `src/langgraph_lab/graph.py`  |
| `@tool` to define tools, `ToolNode` to run them                | `src/langgraph_lab/tools.py`  |
| `MemorySaver` checkpointer + `thread_id` for multi-turn memory | `src/langgraph_lab/cli.py`    |
| `graph.stream(stream_mode="updates")` node-by-node streaming   | `src/langgraph_lab/cli.py`    |
| Automatic LangSmith tracing (just set env vars)                | `src/langgraph_lab/config.py` |
| `@traceable` custom span (a retrieval step inside a tool)      | `src/langgraph_lab/tools.py`  |
| LangSmith dataset + `evaluate()` + LLM-as-judge                | `scripts/evaluate.py`         |
| LangGraph Studio (`langgraph dev`)                             | `langgraph.json`              |

## Layout

```
langgraph-lab/
├── main.py                     # uv run main.py -> start the interactive chat
├── langgraph.json              # entry point for LangGraph Studio / langgraph dev
├── .env.example                # copy to .env and fill in
├── src/langgraph_lab/
│   ├── config.py               # load .env, wire up LangSmith
│   ├── llm.py                  # ChatOpenAI factory (defaults to OpenRouter)
│   ├── tools.py                # 3 local tools (calculator / time / KB lookup)
│   ├── state.py                # graph state definition
│   ├── graph.py                # build and compile the graph (build_graph)
│   └── cli.py                  # terminal chat loop
└── scripts/
    └── evaluate.py             # LangSmith evaluation example
```

## Quick start

```bash
# 1. Configure keys
cp .env.example .env
# Edit .env:
#   - LLM_API_KEY   your OpenRouter key (sk-or-v1-...)
#   - LLM_MODEL     a model that supports function calling; default openai/gpt-5.6-luna
#                   (step up to openai/gpt-5.6-terra or openai/gpt-5.6-sol for more power)
#   - for tracing, also set LANGSMITH_API_KEY and LANGSMITH_TRACING=true

# 2. Start the interactive chat (uv installs deps on first run)
uv run main.py
```

The model layer talks to an OpenAI-compatible endpoint (`ChatOpenAI` +
`base_url`), pointing at OpenRouter by default. To use another gateway
(OpenAI directly, DeepSeek, a local vLLM, ...), change `LLM_BASE_URL` and
`LLM_API_KEY` in `.env` — no code changes.

Things to try in the chat:

- `Compute 17 * 23 + 4` — triggers the `calculator` tool
- `What is a checkpointer in LangGraph?` — triggers `search_knowledge_base`
- `What time is it in UTC+8?` — triggers `get_current_time`
- Ask follow-ups to see multi-turn memory; type `/new` for a fresh thread,
  `/exit` to quit

The terminal prints each step: `call tool(args)` -> `result <tool output>` ->
`assistant <final answer>`.

## LangSmith tracing

Set in `.env`:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=langgraph-lab
```

After that, every `uv run main.py` conversation produces a trace tree under the
matching project at https://smith.langchain.com. Every graph node, tool call
and LLM call is a span with inputs, outputs, latency and token counts. The
`_kb_lookup` helper in `tools.py` is decorated with `@traceable`, so it shows up
as a child span under the `search_knowledge_base` tool call.

## LangSmith evaluation

```bash
uv run scripts/evaluate.py
```

The script:

1. creates a dataset `langgraph-lab-react` in LangSmith (first run only), with
   4 "question -> reference answer" examples;
2. runs the agent over each question (`run_agent` is the target);
3. scores each run with two evaluators:
   - `correctness`: an LLM-as-judge deciding whether the answer matches the reference;
   - `used_expected_tool`: a deterministic check that the agent called the expected tool;
4. prints the experiment name so you can inspect per-example scores and traces
   in LangSmith.

## LangGraph Studio (visual debugging)

```bash
uv run langgraph dev
```

Starts the local LangGraph API and Studio. In the browser you can visualise the
graph, step through execution, inspect/edit state, and time-travel. The entry
point in `langgraph.json` points at `graph.py:graph`.

## Notes

- Default model is `openai/gpt-5.6-luna` (cheap, fast, reliable tool calling).
  To switch models, change only `LLM_MODEL` in `.env` — it must support function
  calling, otherwise the ReAct loop never invokes a tool.
- All three tools are local and deterministic, so traces are stable and easy to
  compare across runs.
- Verified working: ReAct tool calling, multi-turn memory (checkpointer +
  thread_id), and `stream_mode="updates"` streaming. `scripts/evaluate.py`
  really creates a dataset and an experiment in your LangSmith account — run it
  when you want that.
