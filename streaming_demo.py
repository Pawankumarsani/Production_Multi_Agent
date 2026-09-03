import asyncio
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated
from langsmith import traceable
from langchain_groq import ChatGroq

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

@traceable(project_name="Production_multi_agent")
def chatbot(state: State):
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

def run_invoke():
    result = graph.invoke({"messages": [{"role": "user", "content": "What is LangSmith?"}]})
    print(result["messages"][-1].content)

async def run_streaming():
    print("======Event_Streaming======\n\n")
    final_state = None
    async for event in  graph.astream_events(
        {"messages": [{"role": "user", "content": "Write a Haiku about AI"}]},version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)

        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            final_state = event["data"]["output"]
    print("\nFinal State:", final_state)

if __name__ == "__main__":
    run_invoke()
    print()
    asyncio.run(run_streaming())