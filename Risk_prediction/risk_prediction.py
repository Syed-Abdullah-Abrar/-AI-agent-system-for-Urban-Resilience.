import os
from typing import Annotated,TypedDict, List, Dict, Any, Optional
from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langfuse import Langfuse

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

web_search = TavilySearch (
    max_result = 10,
    topic = "news",
    time_range="week",
    search_depth="advance",
)


class Risk_state(TypedDict):
    location: Dict[str]
    curr_risk: Dict[str, Any]
    is_high: Optional[bool]
    high_reason: Optional[str]
    risk_category: Optional[str]
    messages: List[Dict[str, Any]]

def risk_search(state: Risk_state):
    curr_risk = state["curr_risk"]
    location = state["location"]
    print(f"iChat is calculating risk for your current location : {location}")
    return{}

def classify_risk(state : Risk_state) :
    curr_risk = state["curr_risk"]
    location = state["location"]
    prompt = f"""
    You are a professional city management and maintanence agent. Your job is to analyze the situation of the city and classify the degree of risk. You have been given the risk and location.
    
    Analyze based on following factors
    
    Current risk : {curr_risk}
    Location : {location}
    Scarce resources based on current risk : give the detailed list of scarce resources

    Look out the economic and societal condition of the city before classifying.
    Analyze in three categories:
    HIGH
    MEDIUM
    LOW
    and then return the answer

    If the risk is HIGH, then explain the reason
    """

    messages = [HumanMessage(content=prompt)]
    response = Watson_llm.invoke(messages)

    response_text = response.content.lower()
    print(response_text)

    is_high = "HIGH" in response_text and "MEDIUM" or "LOW" not in response_text
    is_medium = "MEDIUM" in response_text and "HIGH" or "LOW" not in response_text
    is_low = "LOW" in response_text and "HIGH" or "MEDIUM" not in response_text

    if is_high :
        risk_stage = state
        