from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

WRITER_SYSTEM_PROMPT = (
    "You are the writer agent in a multi-agent research assisitant."
    "You recieve reasearch finding and analysis gathered by other agents in the conversation."
    "Draft a clear, well-structured report based on that material -- do not invent facts that weren't provided. "
    "If earlier messages contain reviewer feedback asking for revisions, revise your previous draft to address that feedback directly."
)

class WriterState(TypedDict):
    messages: Annotated[list, add_messages]

@traceable(run_type="chain", name="Writer Agent")
def build_writer_agent():
    
    llm=ChatGroq(model="openai/gpt-oss-120b")
    
    @traceable(run_type="chain", name="Writer Node")
    def writer_node(state:WriterState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0],SystemMessage):
            messages = [SystemMessage(content=WRITER_SYSTEM_PROMPT)] + messages
            
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    builder = StateGraph(WriterState)
    
    builder.add_node("writer", writer_node)
    builder.add_edge(START, "writer")
    builder.add_edge("writer", END)
    
    return builder.compile()

writer_agent = build_writer_agent() 