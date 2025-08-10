import os
from typing import Annotated,TypedDict, List, Dict, Any, Optional
from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langfuse import Langfuse

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

def risk_search() ->Dict:
    return 