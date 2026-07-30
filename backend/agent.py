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
from tavily import TavilyClient

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
    reddit_tips: Optional[list]        # list of Reddit tips fetched

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
    print(f"\n[Node 4] Fetching Yelp activities for {state['location']}...")

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

        print(f"[Node 4] Found {len(yelp_results)} Yelp activities")
        for y in yelp_results:
            print(f"  - {y['name']} ({y['category']}) — {y['rating']}⭐ {y['price']}")

        return {
            **state,
            "yelp_results": yelp_results
        }
    else:
        print(f"[Node 4] Yelp API error: {response.status_code} {response.text}")
        return {
            **state,
            "yelp_results": []
        }

# Node 4b - Fetch Reddit Tips
def fetch_reddit_tips(state: TripState) -> TripState:
    print(f"\n[Node 4b] Fetching Reddit tips for {state['location']}...")

    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    location = state["location"] or "Toronto"

    try:
        # Search Reddit for local tips
        query = f"site:reddit.com things to do {location} hidden gems local tips weekend" #IMPORTANT KEYWORDS SEARCH
        results = client.search(
            query=query,
            max_results=5,
            include_answer=True
        )

        reddit_tips = []
        for result in results.get("results", []):
            # Only include reddit.com results
            if "reddit.com" in result.get("url", ""):
                reddit_tips.append({
                    "title": result.get("title"),
                    "summary": result.get("content", "")[:300],  # first 300 chars
                    "url": result.get("url")
                })

        print(f"[Node 4b] Found {len(reddit_tips)} Reddit tips")
        for r in reddit_tips:
            print(f"  - {r['title']}")

        return {
            **state,
            "reddit_tips": reddit_tips
        }

    except Exception as e:
        print(f"[Node 4b] Reddit search error: {e}")
        return {
            **state,
            "reddit_tips": []
        }
    
# Node 5 
# Merge Ticketmaster + Yelp results
def merge_results(state: TripState) -> TripState:
    print("\n[Node 5] Merging all results...")

    events = state.get("events") or []
    yelp_results = state.get("yelp_results") or []
    reddit_tips = state.get("reddit_tips") or []

    print(f"[Node 5] Ticketmaster events: {len(events)}")
    print(f"[Node 5] Yelp activities: {len(yelp_results)}")
    print(f"[Node 5] Reddit tips: {len(reddit_tips)}")

    return {
        **state,
        "events": events,
        "yelp_results": yelp_results,
        "reddit_tips": reddit_tips
    }

# ─────────────────────────────────────────
#  NODE 6 - Generate Response
# LLM synthesizes everything into a basic recommendation
# ─────────────────────────────────────────
def generate_response(state: TripState) -> TripState:
    print("\n[Node 6] Generating structured response...")

    # Format Ticketmaster events
    events_text = ""
    if state.get("events"):
        events_text = "\n".join([
            f"- {e['name']} at {e['venue']} | "
            f"Price: {e['price']} | "
            f"Link: {e['url']}"
            for e in state["events"]
        ])
    else:
        events_text = "No events found for this date/location"

    # Format Yelp activities
    yelp_text = ""
    if state.get("yelp_results"):
        yelp_text = "\n".join([
            f"- {y['name']} ({y['category']}) | "
            f"⭐ {y['rating']} | "
            f"💰 {y['price']} | "
            f"{y['address']} | "
            f"Link: {y['url']}"
            for y in state["yelp_results"]
        ])
    else:
        yelp_text = "No local activities found"

    # Format Reddit tips
    reddit_text = ""
    if state.get("reddit_tips"):
        reddit_text = "\n".join([
            f"- {r['title']}: {r['summary']} | Link: {r['url']}"
            for r in state["reddit_tips"]
        ])
    else:
        reddit_text = "No Reddit tips found"

    # Weather context
    weather_context = ""
    if state.get("date_within_forecast") and state.get("weather_summary"):
        weather_context = state["weather_summary"]
    else:
        weather_context = "Weather forecast not available for this date"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful group outing planner.
        
        You MUST format your response EXACTLY like this every time, 
        no exceptions. Use the exact separators and emojis shown:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🗓️  OUTING PLAN — {location}, {date}
👥  Group of {group_size} | 💰 Budget: ${budget_per_person}/person
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌤️  WEATHER
[one sentence weather summary and whether it favors indoor or outdoor]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅  SUGGESTED ITINERARY
[3 timed activities using ONLY real data from Yelp and Ticketmaster]
[Format each as: HH:MM AM/PM  Venue Name]
[                📍 Address | ⭐ Rating | 💰 Price]
[                🔗 URL]
[Leave a blank line between each activity]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎟️  EVENTS TODAY
[List each Ticketmaster event on its own line]
[Format: Event Name at Venue | Price | 🔗 URL]
[If price unavailable say: Price unavailable — verify budget before booking]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎳  LOCAL ACTIVITIES
[List top 3 Yelp results]
[Format: Name (Category) | ⭐ Rating | 💰 Price | 📍 Address | 🔗 URL]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬  FROM THE LOCALS
[2-3 Reddit tips summarized in 1-2 sentences each]
[Format: "Tip summary" — source]
[🔗 URL]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        STRICT RULES:
        - Use ONLY real data provided — never invent venues, prices, or events
        - For Ticketmaster events with unavailable prices, always include the URL
        - For the itinerary, pick the best 3 activities from Yelp that fit the budget
        - Weather should influence itinerary choices (rain = indoor, clear = outdoor)
        - Never deviate from the format above
        """),
        ("user", """
        Location: {location}
        Date: {date}
        Group size: {group_size}
        Budget per person: ${budget_per_person}
        Preference: {preference}

        Weather: {weather_context}

        Ticketmaster Events:
        {events_text}

        Yelp Activities:
        {yelp_text}

        Reddit Tips:
        {reddit_text}

        Generate the structured outing plan now.
        """)
    ])

    chain = prompt | llm

    result = chain.invoke({
        "location": state["location"],
        "date": state["date"],
        "group_size": state["group_size"],
        "budget_per_person": state["budget_per_person"],
        "preference": state["preference"],
        "weather_context": weather_context,
        "events_text": events_text,
        "yelp_text": yelp_text,
        "reddit_text": reddit_text
    })

    print(f"[Node 6] Response generated")
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
#  BUILD THE GRAPH
# ─────────────────────────────────────────
def build_graph():
    graph = StateGraph(TripState)

    graph.add_node("parse_input", parse_user_input)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("fetch_events", fetch_events)
    graph.add_node("fetch_events_no_weather", fetch_events_no_weather)
    graph.add_node("fetch_yelp", fetch_yelp_activities)
    graph.add_node("fetch_reddit", fetch_reddit_tips)
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
    graph.add_edge("fetch_yelp", "fetch_reddit")
    graph.add_edge("fetch_reddit", "merge_results")
    graph.add_edge("merge_results", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()

# ─────────────────────────────────────────
# RUN Method
# ─────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()

    # user_input = "Me and 3 friends want to do something fun in Toronto tomorrow. Our budget is $50 each. Also, we would like to have some good food."
    user_input = " Me and 2 friends want to go out in Toronto this Friday. We only have $10 each."
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
        "reddit_tips": None,   
        "final_response": None
    })

    print("\n" + "=" * 50)
    print("FINAL RECOMMENDATION:")
    print("=" * 50)
    print(result["final_response"])