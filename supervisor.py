import os
from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from langsmith import traceable
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from research_agent import research_agent
from analyst_agent import analyst_agent
from writer_agent import writer_agent
from reviewer_agent import reviewer_agent

MAX_REVISIONS = 1

MAX_MESSAGE_CHARS =1200

@traceable(run_type="chain", name="_trimmed")
def _trimmed(messages: list) -> list:
    trimmed = []
    for m in messages:
        content = getattr(m, "content", None)
        if isinstance(content, str) and len(content) > MAX_MESSAGE_CHARS:
            m = m.model_copy(
                update={"content": content[:MAX_MESSAGE_CHARS] + "...[truncated]"}
            )
        trimmed.append(m)
    return trimmed

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    research_done: bool
    analysis_done: bool
    draft_ready: bool
    verdict: str
    revision_count: int
    report: str
    next: str
    
@traceable(run_type="chain", name="Supervisor Node")
def supervisor_node(state: SupervisorState):
    
    if not state.get("research_done"):
        next_ = "research"
    elif not state.get("analysis_done"):
        next_= "analyst"
    elif not state.get("draft_ready"):
        next_= "writer"
    elif not state.get("verdict"):
        next_= "reviewer"
    elif state["verdict"]== "revise" and state.get("revision_count", 0) < MAX_REVISIONS:
        next_= "writer"
    else:
        next_="FINISH"
    
    return {"next": next_}

@traceable(run_type="chain", name="Route")
def route(state: SupervisorState) -> str:
    return END if state["next"]== "FINISH" else state["next"]

@traceable(run_type="chain", name="Call Research Agent")
def call_research_agent(state: SupervisorState):
    result = research_agent.invoke({"messages": _trimmed(state["messages"])})
    return {
        "messages": [result["messages"][-1]],
        "research_done": True,
    }
@traceable(run_type="chain", name="Call Analyst Agent")    
def call_analyst_agent(state: SupervisorState):
    result = analyst_agent.invoke({"messages": _trimmed(state["messages"])})
    return {
        "messages": [result["messages"][-1]],
        "analysis_done": True,
    }

@traceable(run_type="chain", name="Call Writer Agent")
def call_writer_agent(state: SupervisorState):
    result = writer_agent.invoke({"messages": _trimmed(state["messages"])})
    was_revision =state.get("verdict")== "revise"
    draft_text = result["messages"][-1].content
    
    update ={
        "messages": [result["messages"][-1]],
        "draft_ready":True,
        "verdict": "",
        "report": draft_text,
    }
    if was_revision:
        update["revision_count"]= state.get("revision_count", 0) + 1
    return update

@traceable(run_type="chain", name="Call Reviewer Agent")
def call_reviewer_agent(state: SupervisorState):
    result = reviewer_agent.invoke({"messages": _trimmed(state["messages"])})
    return {
        "messages": [result["messages"][-1]],
        "verdict": result["verdict"],
    }
    
builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("research", call_research_agent)
builder.add_node("analyst", call_analyst_agent)
builder.add_node("writer", call_writer_agent)
builder.add_node("reviewer", call_reviewer_agent)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route,
    {
        "research": "research",
        "analyst": "analyst",
        "writer": "writer",
        "reviewer": "reviewer",
        END: END,
    },
)
builder.add_edge("research", "supervisor")
builder.add_edge("analyst", "supervisor")
builder.add_edge("writer", "supervisor")
builder.add_edge("reviewer", "supervisor")


db_path = os.environ.get("CHECKPOINT_DB", "checkpoints.sqlite")
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory)