import os
import requests
from dotenv import load_dotenv
from typing import TypedDict, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

load_dotenv()

# ─────────────────────────────────────────
# 1. DEFINE STATE
# This is the object that gets passed between nodes
# ─────────────────────────────────────────
class TripState(TypedDict):
    user_input: str                    # raw user message
    location: Optional[str]            # extracted location
    date: Optional[str]                # extracted date
    group_size: Optional[int]          # extracted group size
    budget_per_person: Optional[float] # extracted budget
    preference: Optional[str]          # outdoor/indoor
    weather_data: Optional[dict]       # raw weather API response
    weather_summary: Optional[str]     # human readable weather response
    date_within_forecast: Optional[bool]  # whether the date is within the 5-day forecast window
    events: Optional[list]                # list of events fetched  
    final_response: Optional[str]      # final output to user


# ─────────────────────────────────────────
# 2. INITIALIZE LLM
# ─────────────────────────────────────────
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)


# ─────────────────────────────────────────
# 3. NODE 1 - Parse User Input
# Extracts structured constraints from natural language
# ─────────────────────────────────────────
def parse_user_input(state: TripState) -> TripState:
    print("\n[Node 1] Parsing user input...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that extracts trip planning 
        constraints from a user's message. Extract the following into JSON:
        {{
            "location": "city name only",
            "date": "day or date mentioned",
            "group_size": number,
            "budget_per_person": number in dollars,
            "preference": "outdoor or indoor or both"
        }}
        If something is not mentioned, use null.
        Return ONLY the JSON object, nothing else."""),
        ("user", "{input}")
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    result = chain.invoke({"input": state["user_input"]})

    # After getting the result from the LLM
    preference = result.get("preference")

    # If no preference mentioned, default to "both"
    if not preference or preference == "null":
        preference = "both"
        result.update({"preference": preference})
        print("[Node 1] No preference indicated — defaulting to both indoor and outdoor")

    print(f"[Node 1] Extracted constraints: {result}")

    # Update state with extracted values
    return {
        **state,
        "location": result.get("location"),
        "date": result.get("date"),
        "group_size": result.get("group_size"),
        "budget_per_person": result.get("budget_per_person"),
        "preference": preference
    }


# ─────────────────────────────────────────
# 4. NODE 2 - Fetch Weather
# Calls OpenWeatherMap with the extracted location
# ─────────────────────────────────────────
from datetime import datetime
from dateutil import parser as date_parser


def fetch_weather(state: TripState) -> TripState:
    print(f"\n[Node 2] Fetching forecast for {state['location']} on {state['date']}...")

    API_KEY = os.environ.get("WEATHER_API_KEY")
    location = state["location"] or "Brampton"

    url = f"https://api.openweathermap.org/data/2.5/forecast?q={location},CA&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        try:
            parsed_date = date_parser.parse(state["date"], fuzzy=True)
            target_day = parsed_date.strftime("%A")
            target_date_str = parsed_date.strftime("%A %B %d %Y")
        except:
            target_day = "Saturday"
            target_date_str = "Saturday (date unknown)"
            parsed_date = None

        print(f"[Node 2] Looking for forecast on: {target_date_str}")

        day_forecasts = [
            entry for entry in data["list"]
            if datetime.fromtimestamp(entry["dt"]).strftime("%A") == target_day
        ]

        if day_forecasts:
            midday = day_forecasts[len(day_forecasts) // 2]
            summary = (
                f"Temperature: {midday['main']['temp']}°C, "
                f"Feels like: {midday['main']['feels_like']}°C, "
                f"Conditions: {midday['weather'][0]['description']}, "
                f"Humidity: {midday['main']['humidity']}%, "
                f"Wind: {midday['wind']['speed']} m/s"
            )
            date_within_forecast = True
        else:
            summary = f"No forecast available for {target_day} (may be beyond 5-day window)"
            date_within_forecast = False

        print(f"[Node 2] Forecast: {summary}")
        print(f"[Node 2] Date within forecast window: {date_within_forecast}")

        return {
            **state,
            "weather_data": day_forecasts,
            "weather_summary": summary,
            "date_within_forecast": date_within_forecast
        }
    else:
        print(f"[Node 2] Weather API error: {response.status_code}")
        return {
            **state,
            "weather_data": None,
            "weather_summary": "Weather data unavailable",
            "date_within_forecast": False
        }
    
def fetch_weather(state: TripState) -> TripState:
    print(f"\n[Node 2] Fetching forecast for {state['location']} on {state['date']}...")

    API_KEY = os.environ.get("WEATHER_API_KEY")
    location = state["location"] or "Brampton"

    url = f"https://api.openweathermap.org/data/2.5/forecast?q={location},CA&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Handles inputs like "Saturday", "July 26", "tomorrow", etc.
        # Convert extracted date to a day name e.g. "Saturday July 26"
        try:
            parsed_date = date_parser.parse(state["date"], fuzzy=True)
            target_day = parsed_date.strftime("%A")
            target_date_str = parsed_date.strftime("%A %B %d %Y")
        except:
            target_day = "Saturday"
            target_date_str = "Saturday (date unknown)"

        print(f"[Node 2] Looking for forecast on: {target_date_str}")

        # Filter forecast entries for target day
        day_forecasts = [
            entry for entry in data["list"]
            if datetime.fromtimestamp(entry["dt"]).strftime("%A") == target_day
        ]

        if day_forecasts:
            # Take midday forecast
            midday = day_forecasts[len(day_forecasts) // 2]
            summary = (
                f"Temperature: {midday['main']['temp']}°C, "
                f"Feels like: {midday['main']['feels_like']}°C, "
                f"Conditions: {midday['weather'][0]['description']}, "
                f"Humidity: {midday['main']['humidity']}%, "
                f"Wind: {midday['wind']['speed']} m/s"
            )
            date_within_forecast = True

        else:
            summary = f"No forecast available for {target_day} (may be beyond 5-day window)"
            date_within_forecast = False

        print(f"[Node 2] Forecast: {summary}")
        print(f"[Node 2] Date within forecast window: {date_within_forecast}")

        return {
            **state,
            "weather_data": day_forecasts,
            "weather_summary": summary,
            "date_within_forecast": date_within_forecast
        }
    else:
        print(f"[Node 2] Weather API error: {response.status_code}")
        return {
            **state,
            "weather_data": None,
            "weather_summary": "Weather data unavailable",
            "date_within_forecast": False

        }

# ─────────────────────────────────────────
# 5. NODE 3 - Generate Response
# LLM synthesizes everything into a basic recommendation
# ─────────────────────────────────────────
def generate_response(state: TripState) -> TripState:
    print("\n[Node 3] Generating response...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful group outing planner. 
        Based on the user's constraints and weather conditions, 
        provide a brief recommendation for their outing.
        Be concise but helpful. Explain your reasoning."""),
        ("user", """
        User request: {user_input}
        
        Extracted constraints:
        - Location: {location}
        - Date: {date}
        - Group size: {group_size}
        - Budget per person: ${budget_per_person}
        - Preference: {preference}
        
        Current weather in {location}:
        {weather_summary}
        
        Based on this information, what would you recommend for their outing?
        """)
    ])

    chain = prompt | llm

    result = chain.invoke({
        "user_input": state["user_input"],
        "location": state["location"],
        "date": state["date"],
        "group_size": state["group_size"],
        "budget_per_person": state["budget_per_person"],
        "preference": state["preference"],
        "weather_summary": state["weather_summary"]
    })

    print(f"[Node 3] Response generated")
    return {
        **state,
        "final_response": result.content
    }


# function checks if the date is within the 5-day forecast window
def check_date_in_forecast(state: TripState) -> str:
    """
    Conditional edge — decides which node to go to after weather fetch.
    Returns the name of the next node to visit.
    """
    if state.get("date_within_forecast"):
        print("\n[Conditional] Date is within forecast window → fetching events with weather context")
        return "fetch_events"
    else:
        print("\n[Conditional] Date is beyond forecast window → fetching events without weather context")
        return "fetch_events_no_weather"

# ─────────────────────────────────────────
# 6. BUILD THE GRAPH
# ─────────────────────────────────────────
def build_graph():
    graph = StateGraph(TripState)

    # Add nodes
    graph.add_node("parse_input", parse_user_input)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("generate_response", generate_response)

    # Define edges (flow between nodes)
    # INPUT -> parse_input -> fetch_weather -> generate_response -> END
    graph.set_entry_point("parse_input")
    graph.add_edge("parse_input", "fetch_weather")
    graph.add_edge("fetch_weather", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


# ─────────────────────────────────────────
# 7. RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()

    # Test input
    user_input = "Me and 3 friends want to do something fun in Brampton on Sunday July 26, 2026. Our budget is $20 each."

    print("=" * 50)
    print(f"User: {user_input}")
    print("=" * 50)

    result = app.invoke({
        "user_input": user_input,
        "location": None,
        "date": None,
        "group_size": None,
        "budget_per_person": None,
        "preference": None,
        "weather_data": None,
        "weather_summary": None,
        "final_response": None
    })

    print("\n" + "=" * 50)
    print("FINAL RECOMMENDATION:")
    print("=" * 50)
    print(result["final_response"])