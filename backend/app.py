"""
Session 6 — Backend for the group outing agent.

This does NOT reimplement your agent logic. It imports build_graph()
straight from your existing agent.py and streams the graph's own node-by-node
updates to the browser using Server-Sent Events (SSE).

Why streaming updates instead of a fake "loading..." spinner:
LangGraph's `.stream(inputs, stream_mode="updates")` yields the state
delta returned by each node the moment that node finishes. That trace IS
the agent's real reasoning path — which APIs it called, what came back,
which conditional branch it took. Streaming that straight to the UI is what
makes this "not a black box": the user sees the same thing you saw in your
terminal print statements during Sessions 1-5, just rendered nicely.

Run:
    cd backend
    pip install -r requirements.txt
    python3 app.py
Then open http://localhost:5001 in your browser.

IMPORTANT: place this file (and requirements.txt) in the SAME folder as
your agent.py, or adjust the import path below. Your .env file with the
API keys should also be reachable from here (python-dotenv looks in the
current working directory by default).
"""

import json
import traceback

from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS

from agent import build_graph 
app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Build the graph once at startup (compiling it is cheap; running it isn't).
GRAPH = build_graph()

# ─────────────────────────────────────────
# Friendly labels for each node name, so the UI shows human language
# instead of python function names. Keep this in sync with the node
# names you register in build_graph() in agent.py.
# ─────────────────────────────────────────
NODE_INFO = {
    "parse_input": {"label": "Reading your request", "icon": "📝"},
    "fetch_weather": {"label": "Checking the forecast", "icon": "🌤️"},
    "fetch_events": {"label": "Looking for ticketed events", "icon": "🎟️"},
    "fetch_events_no_weather": {"label": "Looking for ticketed events", "icon": "🎟️"},
    "fetch_yelp": {"label": "Finding local spots", "icon": "🍽️"},
    "fetch_reddit": {"label": "Searching Reddit for local tips", "icon": "💬"},
    "merge_results": {"label": "Combining everything found", "icon": "🔀"},
    "generate_response": {"label": "Writing your itinerary", "icon": "🧭"},
}

# Default order used only as a fallback if the graph reports a node we
# don't recognize (keeps the UI from breaking if you add a node later).
DEFAULT_INFO = {"label": "Working", "icon": "⚙️"}


def summarize(node_name, delta):
    """
    Pull out ONLY the fields worth showing the user for each node, so the
    trail log stays readable instead of dumping the entire state object.
    This is the "grounding" data — real API responses, not LLM prose.
    """
    delta = delta or {}

    if node_name == "parse_input":
        return {
            "location": delta.get("location"),
            "date": delta.get("date"),
            "group_size": delta.get("group_size"),
            "budget_per_person": delta.get("budget_per_person"),
            "preference": delta.get("preference"),
            "yelp_categories": delta.get("yelp_categories"),
        }

    if node_name == "fetch_weather":
        return {
            "weather_summary": delta.get("weather_summary"),
            "date_within_forecast": delta.get("date_within_forecast"),
        }

    if node_name in ("fetch_events", "fetch_events_no_weather"):
        events = delta.get("events") or []
        return {"count": len(events), "events": events}

    if node_name == "fetch_yelp":
        results = delta.get("yelp_results") or []
        return {"count": len(results), "results": results}

    if node_name == "fetch_reddit":
        tips = delta.get("reddit_tips") or []
        return {"count": len(tips), "tips": tips}

    if node_name == "merge_results":
        return {
            "events": len(delta.get("events") or []),
            "yelp_results": len(delta.get("yelp_results") or []),
            "reddit_tips": len(delta.get("reddit_tips") or []),
        }

    if node_name == "generate_response":
        return {"final_response": delta.get("final_response")}

    return {}


def sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


def run_graph_stream(user_input):
    inputs = {
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
        "final_response": None,
    }

    yield sse({"type": "start"})

    try:
        for update in GRAPH.stream(inputs, stream_mode="updates"):
            # `update` looks like {"node_name": {...state fields it returned...}}
            for node_name, delta in update.items():
                info = NODE_INFO.get(node_name, DEFAULT_INFO)
                yield sse({
                    "type": "step",
                    "node": node_name,
                    "label": info["label"],
                    "icon": info["icon"],
                    "data": summarize(node_name, delta),
                })
        yield sse({"type": "done"})

    except Exception as exc:  # surface the real error to the UI instead of hanging
        traceback.print_exc()
        yield sse({"type": "error", "message": str(exc)})


@app.route("/api/plan")
def plan():
    user_input = (request.args.get("input") or "").strip()
    if not user_input:
        return Response(
            sse({"type": "error", "message": "Please describe your outing first."}),
            mimetype="text/event-stream",
        )
    return Response(run_graph_stream(user_input), mimetype="text/event-stream")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001, threaded=True)