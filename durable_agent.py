import random
import operator
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import RetryPolicy
from langchain_groq import ChatGroq

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, operator.add]
    attempts: int

def unreliable_node(state: State):
    """Simulates a node that sometimes fails."""
    attempt = state.get("attempt", 0) + 1
    
    if random.random() < 0.4 and attempt < 4:
        raise ConnectionError("API temporarily unavailable (simulated)")
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    response = llm.invoke(state["messages"])
    return {"messages": [response], "attempt": attempt}

def fallback_handler(state: State, error: Exception):
    """Called when all retries are exhausted."""
    return {
        "messages": [{"role": "assistant", "content": f"Sorry, I encountered an error: {error}. Please try again later."}]
    }

builder = StateGraph(State)
builder.add_node("unreliable", unreliable_node, retry=RetryPolicy(
    max_attempts=3,          
    initial_interval=1.0,    
    backoff_factor=2.0,      
    max_interval=60.0        
))
builder.add_edge(START, "unreliable")
builder.add_edge("unreliable", END)

with SqliteSaver.from_conn_string("./checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "durable_test"}}
    
    try:
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Hello, can you help me?"}]},
            config
        )
        print("Success:", result["messages"][-1].content)
    except Exception as e:
        print(f"Failed after retries: {e}")

with SqliteSaver.from_conn_string("./checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "durable_test"}}
    
    checkpoint = graph.get_state(config)
    if checkpoint:
        result = graph.invoke(None, config) 
        print("Resumed:", result)