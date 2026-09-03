import operator
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, operator.add]

def llm_with_messages(state: State):
    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("llm_with_messages", llm_with_messages)
builder.add_edge(START, "llm_with_messages")
builder.add_edge("llm_with_messages", END)

with SqliteSaver.from_conn_string("./checkpoint_test.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable":{"thread_id":"1"}}

    response1 = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ]
        },
        config
    )
    print("Response 1:", response1["messages"][-1].content)

    response2 = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "What is LangSmith?"}
            ]
        },
        config
    )
    print("Response 2:", response2["messages"][-1].content)
