import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.tracers.context import collect_runs
from langsmith import Client

from supervisor import graph

load_dotenv()

st.set_page_config(
    page_title="Multi Agent Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.caption("Powered by LangGraph + LangSmith")

STEPS = ["research", "analyst", "writer", "reviewer"]
STEP_LABELS = {
    "research": "🔍 Research",
    "analyst": "📊 Analyst",
    "writer": "✍️ Writer",
    "reviewer": "🧐 Reviewer",
}

query = st.text_input(
    "Research question", placeholder="e.g. What are the latest advances in agentic RAG?"
)
run_clicked = st.button("Run", type="primary", disabled=not query)

if run_clicked:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    st.subheader("Live progress")
    cols = st.columns(len(STEPS))
    placeholders = {}
    for col, step in zip(cols, STEPS):
        with col:
            placeholders[step] = st.empty()
            placeholders[step].info(f"{STEP_LABELS[step]}\n\nwaiting…")

    log_box = st.expander("Run log", expanded=False)

    # collect_runs() captures the LangSmith Run object(s) created during
    # this block, so we can build a trace link afterwards.
    with collect_runs() as run_collector:
        for update in graph.stream(
            {"messages": [{"role": "user", "content": query}]}, config
        ):
            for node_name in update:
                if node_name in placeholders:
                    placeholders[node_name].success(f"{STEP_LABELS[node_name]}\n\n✅ done")
                with log_box:
                    st.write(f"**{node_name}** finished")

    final_state = graph.get_state(config).values
    report = final_state.get("report", "")

    st.subheader("📄 Final report")
    st.markdown(report or "_No report produced._")

    # --- LangSmith trace link ---
    if run_collector.traced_runs:
        root_run = run_collector.traced_runs[0]
        try:
            client = Client()
            trace_url = client.get_run_url(run=root_run)
            st.link_button("🔗 View full trace in LangSmith", trace_url)
        except Exception as e:
            st.caption(f"Trace was captured, but the URL couldn't be built: {e}")
    else:
        st.caption(
            "No LangSmith run was captured — make sure LANGCHAIN_TRACING_V2=true "
            "and LANGCHAIN_API_KEY are set in your .env."
        )