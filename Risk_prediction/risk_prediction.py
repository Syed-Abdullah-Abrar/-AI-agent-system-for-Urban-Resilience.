import os
from typing import Annotated,TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langfuse import Langfuse

langfuse = Langfuse(
  secret_key="sk-lf-f7b63763-fee5-46b9-9d93-d26f9aca1f67",
  public_key="pk-lf-9459cbae-a1fd-4a09-8d7a-4690e33ec071",
  host="https://us.cloud.langfuse.com"
)

def risk_search() ->Dict:
    return 