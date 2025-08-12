import os
from typing import Annotated,TypedDict, List, Dict, Any, Optional
from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langfuse import Langfuse
from IPython.display import Image, display



os.environ["TAVILY_API_KEY"] = "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"

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 500,
    "repetition_penalty": 1,
    "min_new_tokens":1
}

Watson_llm = ChatWatsonx(
    model_id="ibm/granite-34b-code-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    apikey="GSK7bM6Zd-xMtm0N42OCZloVEU2AAu-hM4ET9QEptoih",
    project_id="bec7cfaf-a6c9-4f54-be5d-dbf91555ea41",
    params=parameters,
)

web_search_tool = TavilySearch (
    max_result = 5,
    topic = "general",
    search_depth="advance",
)

tools = [web_search_tool]
memory = InMemorySaver()

llm_with_tools = Watson_llm.bind_tools(tools)


class qna_state(TypedDict):
    messages : Annotated[list, add_messages]

graph_builder = StateGraph(qna_state)


def chatbot(state: qna_state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(tools)


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

display(Image(graph.get_graph().draw_mermaid_png()))