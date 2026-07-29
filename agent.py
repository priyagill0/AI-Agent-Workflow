import os
import requests
from dotenv import load_dotenv
from typing import TypedDict, Optional
from datetime import datetime, timedelta
from dateutil import parser as date_parser
 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

load_dotenv()

# ─────────────────────────────────────────
# DEFINE STATE
# This is the object that gets passed between nodes
# ─────────────────────────────────────────
class TripState(TypedDict):
    user_input: str                    # raw user message
    location: Optional[str]            # extracted location
    date: Optional[str]                # extracted date
    group_size: Optional[int]          # extracted group size
    budget_per_person: Optional[float] # extracted budget
    preference: Optional[str]          # outdoor/indoor
    yelp_categories: Optional[list]    # mapped Yelp categories based on preference
    weather_data: Optional[dict]       # raw weather API response
    weather_summary: Optional[str]     # human readable weather response
    date_within_forecast: Optional[bool]  # whether the date is within the 5-day forecast window
    events: Optional[list]                # list of events fetched  
    yelp_results: Optional[list]       # list of Yelp results fetched 
    final_response: Optional[str]      # final output to user


# ─────────────────────────────────────────
#  INITIALIZE LLM
# ─────────────────────────────────────────
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)


# ─────────────────────────────────────────
#  NODE 1 - Parse User Input
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
            "preference": "outdoor or indoor or both",
            "yelp_categories": ["category1", "category2", "category3"]
        }}
        
        For yelp_categories, suggest 3-5 Yelp business categories based on 
        the user's preference and group context. Choose from these options:
        - Fun activities: "bowling", "arcades", "escapegames", "mini_golf", "go_karts"
        - Food/drinks: "restaurants", "cafes", "breweries", "desserts"
        - Outdoor: "parks", "hiking", "beaches", "gardens"
        - Entertainment: "movietheaters", "museums", "galleries"
        
        If preference is outdoor → prioritize outdoor categories
        If preference is indoor → prioritize fun activities + food
        If preference is both → mix of all three buckets
        
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
    
    yelp_categories = result.get("yelp_categories") or ["bowling", "parks", "restaurants"]

    print(f"[Node 1] Extracted constraints: {result}")
    print(f"[Node 1] Yelp categories: {yelp_categories}")

    # Update state with extracted values
    return {
        **state,
        "location": result.get("location"),
        "date": result.get("date"),
        "group_size": result.get("group_size"),
        "budget_per_person": result.get("budget_per_person"),
        "preference": preference,
        "yelp_categories": yelp_categories  # NEW
    }


# ─────────────────────────────────────────
#  NODE 2 - Fetch Weather
# Calls OpenWeatherMap with the extracted location
# ─────────────────────────────────────────

    
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
            date_str = state["date"].lower().strip()

            if date_str == "tomorrow":
                parsed_date = datetime.now() + timedelta(days=1)
            elif date_str == "today":
                parsed_date = datetime.now()
            elif "this weekend" in date_str:
                # Find next Saturday
                days_until_saturday = (5 - datetime.now().weekday()) % 7
                parsed_date = datetime.now() + timedelta(days=days_until_saturday)
            else:
                parsed_date = date_parser.parse(date_str, fuzzy=True)

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

# NODE 3 - Fetch Events
def fetch_events(state: TripState) -> TripState:
    print(f"\n[Node 3] Fetching Ticketmaster events for {state['location']}...")

    API_KEY = os.environ.get("TICKETMASTER_API_KEY")
    location = state["location"] or "Toronto"

    try:
        parsed_date = date_parser.parse(state["date"], fuzzy=True)
        # Ticketmaster needs date in format 2026-07-26T00:00:00Z
        start_date = parsed_date.strftime("%Y-%m-%dT00:00:00Z")
        end_date = parsed_date.strftime("%Y-%m-%dT23:59:59Z")
    except:
        start_date = None
        end_date = None

    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    params = {
        "apikey": API_KEY,
        "city": location,
        "countryCode": "CA",
        "startDateTime": start_date,
        "endDateTime": end_date,
        "size": 5,  # get top 5 events
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        events_raw = data.get("_embedded", {}).get("events", [])

        # Extract relevant fields
        events = []
        for event in events_raw:
            # Get price range if available
            price_ranges = event.get("priceRanges", [])

            if price_ranges:
                min_price = price_ranges[0].get("min")
                max_price = price_ranges[0].get("max")

                if min_price and max_price:
                    price = f"${min_price:.2f} - ${max_price:.2f}"
                elif min_price:
                    price = f"From ${min_price:.2f}"
                else:
                    price = "Price unavailable"
            else:
                price = "Price unavailable"

            # OUTSIDE the if/else block
            events.append({
                "name": event.get("name"),
                "date": event.get("dates", {}).get("start", {}).get("localDate"),
                "time": event.get("dates", {}).get("start", {}).get("localTime"),
                "venue": event.get("_embedded", {}).get("venues", [{}])[0].get("name"),
                "price": price,
                "url": event.get("url")
            })

        print(f"[Node 3] Found {len(events)} events")
        for e in events:
            # print(f"  - {e['name']} at {e['venue']} (from ${e['min_price']})")
            # price_display = "Price TBD" if e["min_price"] == "N/A" else f"from ${e['min_price']}"
            # print(f"  - {e['name']} at {e['venue']} ({price_display})")
            print(f"  - {e['name']} at {e['venue']} ({e['price']})")
        return {
     
            **state,
            "events": events
        }
    else:
        print(f"[Node 3] Ticketmaster API error: {response.status_code}")
        return {
            **state,
            "events": []
        }

# NODE 4
# Fetch Yelp Activities
def fetch_yelp_activities(state: TripState) -> TripState:
    print(f"\n[Node 3b] Fetching Yelp activities for {state['location']}...")

    API_KEY = os.environ.get("YELP_API_KEY")
    location = state["location"] or "Brampton"
    categories = ",".join(state.get("yelp_categories") or ["bowling", "parks", "restaurants"])

    # Map budget to Yelp price filter
    # Yelp price: 1=$, 2=$$, 3=$$$, 4=$$$$
    budget = state.get("budget_per_person") or 0
    if budget <= 20:
        price_filter = "1,2"
    elif budget <= 50:
        price_filter = "1,2,3"
    else:
        price_filter = "1,2,3,4"

    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {
        "location": f"{location}, ON, Canada",
        "categories": categories,
        "price": price_filter,
        "limit": 5,
        "sort_by": "best_match"
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        businesses = data.get("businesses", [])

        yelp_results = []
        for biz in businesses:
            yelp_results.append({
                "name": biz.get("name"),
                "category": ", ".join([c["title"] for c in biz.get("categories", [])]),
                "rating": biz.get("rating"),
                "price": biz.get("price", "Price unavailable"),
                "address": ", ".join(biz.get("location", {}).get("display_address", [])),
                "url": biz.get("url")
            })

        print(f"[Node 3b] Found {len(yelp_results)} Yelp activities")
        for y in yelp_results:
            print(f"  - {y['name']} ({y['category']}) — {y['rating']}⭐ {y['price']}")

        return {
            **state,
            "yelp_results": yelp_results
        }
    else:
        print(f"[Node 3b] Yelp API error: {response.status_code} {response.text}")
        return {
            **state,
            "yelp_results": []
        }

# Node 5 
# Merge Ticketmaster + Yelp results
def merge_results(state: TripState) -> TripState:
    print("\n[Node 4] Merging Ticketmaster + Yelp results...")

    events = state.get("events") or []
    yelp_results = state.get("yelp_results") or []

    print(f"[Node 4] Ticketmaster events: {len(events)}")
    print(f"[Node 4] Yelp activities: {len(yelp_results)}")

    return {
        **state,
        "events": events,
        "yelp_results": yelp_results
    }

# ─────────────────────────────────────────
#  NODE 6 - Generate Response
# LLM synthesizes everything into a basic recommendation
# ─────────────────────────────────────────
def generate_response(state: TripState) -> TripState:
    print("\n[Node 5] Generating response...")

    # Format Ticketmaster events
    events_text = ""
    if state.get("events"):
        events_text = "\n\n".join([
            f"""Event: {e['name']}
                Venue: {e['venue']}
                Date: {e['date']} {e['time']}
                Price: {e['price']}
                Ticketmaster: {e['url']}"""
            for e in state["events"]
        ])
    else:
        events_text = "No Ticketmaster events found for this date/location"

    # Format Yelp activities
    yelp_text = ""
    if state.get("yelp_results"):
        yelp_text = "\n\n".join([
            f"""Name: {y['name']}
                Category: {y['category']}
                Rating: {y['rating']}⭐
                Price Range: {y['price']}
                Address: {y['address']}
                Yelp: {y['url']}"""
            for y in state["yelp_results"]
        ])
    else:
        yelp_text = "No Yelp activities found"

    # Weather context
    weather_context = ""
    if state.get("date_within_forecast") and state.get("weather_summary"):
        weather_context = f"Weather forecast: {state['weather_summary']}"
    else:
        weather_context = "Weather forecast not available for this date"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful group outing planner.
        You have access to real events from Ticketmaster and real local 
        activities from Yelp. Use both to create a personalized itinerary.
        
        IMPORTANT RULES:
        - Prioritize real data from Ticketmaster and Yelp over your own knowledge
        - Never invent venues, prices, or events
        - If Ticketmaster event price is unavailable, say so and include the URL
        - Use Yelp activities to fill gaps when no events are found
        - Consider weather when recommending outdoor vs indoor options
        - Consider budget when recommending activities
        - Structure your response as a clear itinerary with reasoning
        - Include URLs for everything you recommend
        """),
        ("user", """
        User request: {user_input}

        Constraints:
        - Location: {location}
        - Date: {date}
        - Group size: {group_size}
        - Budget per person: ${budget_per_person}
        - Preference: {preference}

        {weather_context}

        Available Ticketmaster Events:
        {events_text}

        Available Local Activities (Yelp):
        {yelp_text}

        Create a personalized group outing itinerary with your reasoning.
        Structure it clearly with options ranked by fit for this group.
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
        "weather_context": weather_context,
        "events_text": events_text,
        "yelp_text": yelp_text
    })

    print(f"[Node 5] Response generated")
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

def fetch_events_no_weather(state: TripState) -> TripState:
    """Same as fetch_events but acknowledges no weather context"""
    print(f"\n[Node 3b] Fetching events without weather context...")
    # Reuse same logic
    return fetch_events(state)


# ─────────────────────────────────────────
# 6. BUILD THE GRAPH
# ─────────────────────────────────────────
def build_graph():
    graph = StateGraph(TripState)

    graph.add_node("parse_input", parse_user_input)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("fetch_events", fetch_events)
    graph.add_node("fetch_events_no_weather", fetch_events_no_weather)
    graph.add_node("fetch_yelp", fetch_yelp_activities)
    graph.add_node("merge_results", merge_results)
    graph.add_node("generate_response", generate_response)

    graph.set_entry_point("parse_input")
    graph.add_edge("parse_input", "fetch_weather")

    graph.add_conditional_edges(
        "fetch_weather",
        check_date_in_forecast,
        {
            "fetch_events": "fetch_events",
            "fetch_events_no_weather": "fetch_events_no_weather"
        }
    )

    graph.add_edge("fetch_events", "fetch_yelp")
    graph.add_edge("fetch_events_no_weather", "fetch_yelp")
    graph.add_edge("fetch_yelp", "merge_results")
    graph.add_edge("merge_results", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()

# ─────────────────────────────────────────
# 7. RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()

    user_input = "Me and 3 friends want to do something fun in Toronto tomorrow. Our budget is $50 each. Also, we would like to have some good food."

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
        "yelp_categories": None,    
        "weather_data": None,
        "weather_summary": None,
        "date_within_forecast": None,
        "events": None,
        "yelp_results": None,      
        "final_response": None
    })

    print("\n" + "=" * 50)
    print("FINAL RECOMMENDATION:")
    print("=" * 50)
    print(result["final_response"])