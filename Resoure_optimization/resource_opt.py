import os
import googlemaps
from datetime import datetime
from typing import Annotated,TypedDict, List, Dict, Any, Optional
from langchain_google_community import GCSDirectoryLoader
from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool, tool
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langfuse import Langfuse
from IPython.display import Image, display


parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 500,
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

os.environ["TAVILY_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCGEiOW7wwcP0lCnpXOab4B-Da23XwI054"
gmaps = googlemaps.Client(key=os.environ.get("GOOGLE_API_KEY"))

class Risk_state(TypedDict):
    location: Dict[str, Any]
    curr_risk: Dict[str, Any]
    is_high: Optional[bool]
    high_reason: Optional[str]
    risk_category: Optional[str]

graph_builder = StateGraph(Risk_state)


tavily_tool = TavilySearch (
    max_result = 15,
    topic = "general",
    time_range="month",
    search_depth="advance",
)

search_tool = Tool(
    name = "Web_searcher",
    description = "A web search engine which scrapes through web and look out for optimization techniques of resources given the risk in the area",
    func = tavily_tool.invoke,
)

@tool
def find_nearby_shelters(location: str, radius: int = 5000, keyword: str = "shelter") -> str:
    """
    Finds nearby shelters (e.g., flood shelters, refugee camp) near a given location.
    Uses the Google Places API (New) Text Search with a text query.
    Args:
        location (str): The location (address or coordinates) to search near.
        radius (int): The search radius in meters (default: 5000 meters).
        keyword (str): A keyword to search for (default: "shelter").
    """
    try:
        # First, geocode the location to get coordinates
        geocode_result = gmaps.geocode(location)
        if not geocode_result:
            return f"Could not geocode the given location: {location}"

        lat = geocode_result[0]['geometry']['location']['lat']
        lng = geocode_result[0]['geometry']['location']['lng']

        # Use Text Search (New) to search for shelters
        places_result = gmaps.places(query=f"{keyword} near me",
                                     location=(lat, lng),
                                     radius=radius) # Use Text Search instead of Nearby search for keyword based filtering

        shelters = []
        for place in places_result['results']:
            shelters.append({
                "name": place['name'],
                "address": place.get('formatted_address', 'N/A'),
            })

        if shelters:
            formatted_shelters = "\n".join([f"- {s['name']} ({s['address']}, Rating: {s['rating']})" for s in shelters])
            return f"Found the following shelters near {location}:\n{formatted_shelters}"
        else:
            return f"No shelters found near {location} with the keyword '{keyword}' within {radius} meters."

    except googlemaps.exceptions.ApiError as e:
        return f"Error accessing Google Places API: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    

shelter_tool = Tool(
    name = "Shelter finder",
    description= "An search engine which looks out for shelters neraby provided the current location",
    func= find_nearby_shelters.invoke
)

@tool
def route_builder(start: str, end: str, mode: str ) -> str:
    """tool which optimizes and returns the route of the risk vulnerable area from risk zone to safe shelter
       Args:
       start(str) : risk vulnerable are
       end(str) : Safe shelter area
       mode(str) : Mode of travel
    """
    try:
        directions_result = gmaps.directions(start, end, mode=mode, departure_time=datetime.now())

        if directions_result:
            route = directions_result
            leg = route['legs']
            steps = [step['html_instructions'] for step in leg['steps']]
            
            return (
                f"Origin: {leg['start_address']} \n"
                f"Destination: {leg['end_address']} \n"
                f"Distance: {leg['distance']['text']} \n"
                f"Duration: {leg['duration']['text']} \n"
                f"Steps: {'; '.join(steps)}"
            )
        else:
            return "No directions found."

    except googlemaps.exceptions.ApiError as e:
        return f"Error accessing Google Maps Directions API: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

route_tool = Tool(
    name = "Evacuation route",
    description= "A route builder engine that provides a relevent rout from risk vulnerable area to shelter area",
    func= route_builder.invoke
)

tools = [search_tool, shelter_tool, route_tool]
memory = InMemorySaver()

def optimized_route(state : Risk_state, tools : dict) :
    location = state["location"]
    curr_risk = state["curr_risk"]
    risk_category = state["risk_category"]
    prompt = f"""You are an route designing and expert . Your job is to design a route of the given vulnerable area and warn the resident about scarce resources in the given condition and there optimization.
    
    You have been given the following information about resident and area:
    location : {location}
    current casualities : {curr_risk}
    risk grade (HIGH/MEDIUM/LOW) : {risk_category}


    You have been access to following tools
    {tools}
    Use search tool to optimize resources.
    Use shelter tool to find nearby shelters.
    Use route tool to design a route to shelters in the specified travel mode
    
    Proceed in thought-action-observation loop.
    
    Return an optimized result to the query."""

    messages = [HumanMessage(content=prompt)]
    response = Watson_llm.invoke(messages)

    response_text = response.content.lower()
    print(response_text)

graph_builder.add_node("Optimization", optimized_route )
graph_builder.add_node("Toolkit", tools)
graph_builder.add_edge(START, "Optimization")
graph_builder.add_conditional_edges("Optimization",tools_condition)
graph_builder.add_edge("Toolkit","Optimization")
graph_builder.add_edge("Optimization",END)

graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

display(Image(graph.get_graph().draw_mermaid_png()))