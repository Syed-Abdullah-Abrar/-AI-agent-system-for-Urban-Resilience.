import os
from google.cloud import discoveryengine_v1alpha
from typing import Annotated,TypedDict, List, Dict, Any, Optional
from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent,tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langfuse import Langfuse
from IPython.display import Image, display


os.environ["TAVILY-API-KEY"] = "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"

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 800,
    "repetition_penalty": 1,
    "min_new_tokens":1
}

Watson_llm = ChatWatsonx(
    model_id="ibm/granite-34b-code-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    apikey="YOUR API KEY",
    project_id="YOUR PROJECT ID",
    params=parameters,
)

langfuse = Langfuse(
  secret_key="sk-lf-f7b63763-fee5-46b9-9d93-d26f9aca1f67",
  public_key="pk-lf-9459cbae-a1fd-4a09-8d7a-4690e33ec071",
  host="https://us.cloud.langfuse.com"
)

tavily_tool = TavilySearch (
    max_result = 5,
    topic = "news",
    time_range="week",
    search_depth="advance",
)

search_tool = Tool(
    name = "Web_searcher",
    description = "A web search engine which looks out for risks like natural disasters, wars etc at a given location",
    func = tavily_tool.invoke,
)

tools = [search_tool]
memory = InMemorySaver()


class Risk_state(TypedDict):
    location: Dict[str, Any]
    curr_risk: Dict[str, Any]
    is_high: Optional[bool]
    high_reason: Optional[str]
    risk_category: Optional[str]
    
graph_builder = StateGraph(Risk_state)


def search_classify_risk(state : Risk_state, tools : dict) :
    location = state["location"]
    prompt = f"""
    You are a professional city management and maintanence agent. Your job is to analyze the situation of the city and classify the degree of risk. You have been given the risk and location. 
    
    You have access to following tools {tools}
    
    Analyze based on following factors
    
    Location : {location}
    Scarce resources based on current risk : give the detailed list of scarce resources

    Look out the economic and societal condition of the city before classifying.
    Analyze in three categories:
    HIGH
    MEDIUM
    LOW

    Return the answer containig type of risk, degree of risk and scarce resources      

    If the risk is HIGH, then explain the reason
    """

    messages = [HumanMessage(content=prompt)]
    response = Watson_llm.invoke(messages)

    response_text = response.content.lower()
    print(response_text)


graph_builder.add_node("Risk", search_classify_risk)
graph_builder.add_node("tools", tools)
graph_builder.add_edge(START,"Risk")
graph_builder.add_conditional_edges("Risk",tools_condition)
graph_builder.add_edge("tools","Risk")
graph_builder.add_edge("Risk", END)


graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

display(Image(graph.get_graph().draw_mermaid_png()))
        