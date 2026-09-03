import requests
import xml.etree.ElementTree as ET
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
_WIKI_HEADERS = {
    "User-Agent": "multi-agent-research-assistant/1.0"
}

@tool(description="Search Wikipedia for the query and return short summaries of the top matching articles.")
def wiki_search(query: str) -> str:
    
    params={
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 2,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "format": "json",
    }
    
    data =None
    last_error =None
    for attempt in range(2):
        try:
            response = requests.get(
                WIKIPEDIA_API_URL, params=params, headers=_WIKI_HEADERS, timeout=10
            )
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.RequestException as e:
            last_error =f"Wikipedia search failed: {e}"
    if data is None:
        return f"{last_error} Try another source."
    
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return "No relevent Wikipedia articles found"
    
    results =[]
    for page in pages.values():
        title = page.get("title", "")
        extract = page.get("extract", "").strip()
        results.append(f"Tile: {title}\nSummary: {extract[:500]}")
        
    return "\n\n".join(results) 

@tool(description="Search arXiv for papers matching the query. Returns titles, links, and short summaries for the top results.")
def arxiv_search(query: str) -> str:
    
    params ={
        "search_query": f"all:{query}",
        "start":0,
        "max_results": 2, 
    }
    root =None
    last_error =None
    for attempt in range(2):
        try:
            response = requests.get(ARXIV_API_URL, params=params, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            break
        except requests.exceptions.RequestException as e:
            last_error = f"arXiv search failed: {e}."
        except ET.ParseError as e:
            last_error = f"aeXiv search returned unparseable data: {e}."
            break
        
    if root is None:
        return f"{last_error} Try another source."
    
    entries = root.findall("atom:entry", _ATOM_NS)
    if not entries:
        return "No relevant arXiv papers found."
    
    results =[]
    for entry in entries:
        title = entry.findtext("atom:title", default="", namespaces= _ATOM_NS).strip()
        link = entry.findtext("atom:id", default="", namespaces=_ATOM_NS).strip()
        summary = entry.findtext("atom:summary", default="", namespaces=_ATOM_NS).strip()
        results.append(f"Title: {title}\nLink: {link}\nSummary: {summary[:500]}")
        
    return "\n\n".join(results)

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]

@traceable(run_type="chain", name="Research Agent")
def build_research_agent():

    tavily=TavilySearchResults()


    tools=[arxiv_search, wiki_search, tavily]

    llm = ChatGroq(model="openai/gpt-oss-120b").bind_tools(tools)

    @traceable(run_type="chain", name="Research Node")
    def research_node(state:ResearchState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(ResearchState)

    builder.add_node("research", research_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "research")
    builder.add_conditional_edges("research", tools_condition)
    builder.add_edge("tools", "research")
    
    return builder.compile()

research_agent = build_research_agent()