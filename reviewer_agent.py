from typing import Literal
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import SystemMessage, AIMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

REVIEWER_SYSTEM_PROMPT = (
    "You are the reviewer agent in a multi-agent research assistant."
    "You receive a draft report written by the writer agent, plus the research and analysis it was based on."
    "Check the draft for factual consistency with the supplied research, clearity, and completeness. "
    "Decide wheather to approve it or send it back for revision.\n\n"
    "IMPORTANT: You must respond ONLY by calling the review tool with your verdict and feedback."
    "Do not write the report yourself, do not rewrite or expand the draft, and do not respond with plain text--"
    "your only output is the structured tool call."
)

class ReviewDecision(BaseModel):
    verdict: Literal["approve", "revise"] = Field(
        description="'approve' if the draft is ready to deliver, "
        "'revise' if it needs changes."
    )
    feedback: str = Field(
        description="Specific, actionable feedback. If approving, a brief "
        "note of what's good. If requesting revision, exactly what to fix."
    )

class ReviewerState(TypedDict):
    messages: Annotated[list, add_messages]
    verdict: str
    
@traceable(run_type="chain", name="Reviewer Agent")
def build_reviewer_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b")
    structured_llm = llm.with_structured_output(ReviewDecision)
    
    @traceable(run_type="chain", name="Reviewer Node")
    def reviewer_node(state:ReviewerState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=REVIEWER_SYSTEM_PROMPT)] + messages
            
        decision = None
        for attempt in range(2):
            try:
                decision = structured_llm.invoke(messages)
                break
            except Exception as e:
                
                messages = messages + [
                    SystemMessage(
                        content="Reminder:  call the review tool now. "
                        "Do not write any report text."
                    )
                ]
                last_error = e
        
        if decision is None:
            
            feedback_message = AIMessage(
                content=(
                    "[Reviewer verdict: revise] Automatic fallback -- the "
                    f"reviewer model failde to return a structured verdict "
                    f"({last_error.__class__.__name__}). Please review the draft manually if this keeps happening."
                )
            )
            return {"messages": [feedback_message], "verdict": "revise"}
        
        feedback_message = AIMessage(
            content=f"[Reviewer verdict: {decision.verdict}] {decision.feedback}"
        )
        return {
            "messages": [feedback_message],
            "verdict": decision.verdict,
        }
        
    builder = StateGraph(ReviewerState)
    builder.add_node("reviewer", reviewer_node)
    builder.add_edge(START, "reviewer")
    builder.add_edge("reviewer", END)
    
    return builder.compile()

reviewer_agent = build_reviewer_agent()