from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, START, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

ANALYST_SYSTEM_PROMPT = (
    "You are analyst agent in a multi-agent research assistant."
    "You recieve research findings gathered by another agent. Your job is to:\n"
    "1. Run any calculations, statistics, or data processing needed using the Python REPL tool.\n"
    "2. Summarize the key findings clearly and concisely once your analysis is done.\n"
    "Only use the Python tool when actual computation is needed -- don't"
    "run code just to restate text."
)

class AnalystState(TypedDict):
    messages: Annotated[list, add_messages]
    
@traceable(run_type="chain", name="Analyst Agent")
def build_analyst_agent():
    
    """
    Builds and compiles the analyst sub-agent as its own graph:
    analyst (LLM + Python REPL tool) <-> tools, looping until the LLM
    stops calling tools, then returns to the caller (e.g. a supervisor).
    No own checkpointer -- meant to be invoked as a node/subgraph
    inside a parent graph that owns the checkpointer.
    """
    
    tools=[PythonREPLTool()]
    
    llm = ChatGroq(model="openai/gpt-oss-120b").bind_tools(tools)

    @traceable(run_type="chain", name="Analyst Node")    
    def analyst_node(state:AnalystState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=ANALYST_SYSTEM_PROMPT)] + messages 
        
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    builder = StateGraph(AnalystState)
    
    builder.add_node("analyst", analyst_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "analyst")
    builder.add_conditional_edges("analyst", tools_condition)
    builder.add_edge("tools", "analyst")
    
    return builder.compile()

analyst_agent = build_analyst_agent()
