import uuid
from supervisor import graph

def run(query: str) -> str:
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50,
    }
    
    for update in graph.stream({"messages": [{"role": "user", "content":query}]}, config):
        for node_name in update:
            print(f"-> {node_name} finished")
            
    final_state = graph.get_state(config).values
    return final_state.get("report", "")

if __name__== "__main__":
    query = input("Research question: ")
    report = run(query)
    print("\n=== Final Report ===\n")
    print(report)