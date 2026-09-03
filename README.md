# Production Multi Agent

A multi-agent research assistant built with **LangGraph** and **Groq**, orchestrated by a supervisor graph that routes work between a research agent, an analyst agent, a writer agent, and a reviewer agent — with a revision loop, LangSmith tracing, and a Streamlit dashboard.

## Architecture

```
                 ┌──────────────┐
        ┌───────▶│  supervisor   │◀───────┐
        │        └──────┬───────┘         │
        │               │ routes to       │
        │        ┌──────┴───────┐         │
        │        ▼      ▼       ▼         │
    ┌────────┐┌────────┐┌────────┐┌──────────┐
    │research││analyst ││ writer ││ reviewer │
    └────────┘└────────┘└────────┘└──────────┘
```

- **`research_agent.py`** — searches Wikipedia, arXiv, and Tavily for source material.
- **`analyst_agent.py`** — runs calculations/statistics on research findings via a Python REPL tool.
- **`writer_agent.py`** — drafts a report from the research + analysis; revises on reviewer feedback.
- **`reviewer_agent.py`** — checks the draft for accuracy/completeness, returns an `approve`/`revise` verdict via structured output.
- **`supervisor.py`** — owns the top-level state machine (`SupervisorState`), routes between the four agents, enforces a max revision count, and compiles the graph with a `SqliteSaver` checkpointer for persistence.

Each sub-agent is its own compiled LangGraph graph, invoked as a node inside the supervisor's graph. Only the supervisor graph owns the checkpointer.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.

```powershell
uv add -r requirement.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=Production_multi_agent
```

`LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY` are what actually enable LangSmith tracing — without them, `@traceable` calls run locally but nothing gets sent to LangSmith.

## Running

**Always run through `uv run`** so commands execute inside this project's `.venv` rather than any globally-installed copy of the same tools.

### CLI

```powershell
uv run python app.py
```

Prompts for a research question in the terminal and prints the final report.

### Streamlit dashboard

```powershell
uv run streamlit run streamlit_app.py
```

Shows live per-agent progress as the graph streams, the final report, and a clickable link to the full run trace in LangSmith.

## Persistence

Conversation state is checkpointed to a local SQLite file (`checkpoints.sqlite` by default, configurable via the `CHECKPOINT_DB` env var) using `SqliteSaver`. Each run uses a `thread_id` — reuse the same `thread_id` to resume a prior conversation instead of starting fresh.

## Notes

- LLM calls use Groq-hosted models (currently `openai/gpt-oss-120b` across agents) — Groq's available model list changes over time, so if you hit a `model_not_found` error, list your account's current models with `client.models.list()` and swap in a valid one.
- `TavilySearchResults` (from `langchain-community`) is deprecated in favor of `langchain-tavily`'s `TavilySearch` — migration pending.
- Custom `@traceable` spans use `run_type="chain"` — LangSmith only accepts `tool`, `chain`, `llm`, `retriever`, `embedding`, `prompt`, or `parser` as valid run types.