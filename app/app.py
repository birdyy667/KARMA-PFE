"""
Main Flask Application for the EV Mobility AAAI Demo.
Orchestrates routing, weather APIs, negotiation agents, and the web interface.
"""

import math
import time
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import osmnx as ox
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import centralized configuration
from config import PROCESSED_DATA_DIR

# Import from refactored core modules
from src.core.routing import calculate_routes
from src.core.bornes import fetch_daily_weather_paris, BuildingStation, initialize_stations
from src.core.user_profil import read_profile, update_profile
from src.core.reservations import get_reserved_slots, process_reservation, process_cancellation, initialize_ghost_users
from src.utils.charts_generator import generate_negotiation_metrics, generate_bounded_time_stats, \
    generate_cost_comparison_stats

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Application Setup
app = Flask(__name__)
CORS(app)

# Absolute paths via config
GRAPH_FILE_PATH = PROCESSED_DATA_DIR / "IDF.graphml"
CACHE_FILE_PATH = PROCESSED_DATA_DIR / "idf_network.pkl"
STATIONS_FILE_PATH = PROCESSED_DATA_DIR / "stations_cache.pkl"

DEFAULT_CONSUMPTION = 15.0

# Fetch weather on startup
current_date = datetime.today().strftime('%Y-%m-%d')
logger.info("☁️ Fetching today's weather for Paris...")
DAILY_WEATHER = fetch_daily_weather_paris(current_date)

# In-memory session trackers
ACTIVE_NEGOTIATIONS: Dict[str, Any] = {}

# --- DEMO HYBRID HISTORY FOR CHARTS ---
# Pre-filled with ~10 simulations to populate the UI charts immediately
DEMO_HISTORY = [
    # Format: [Mode, Success (1/0), Time (s), Slot, Profile, Savings (%)]
    ["Manual", 1, 45, "Morning", "Standard", 10],
    ["Manual", 0, 60, "Afternoon", "Standard", 0],
    ["Manual", 1, 50, "Afternoon", "Regular", 25],
    ["Manual", 1, 40, "Specific Time", "Standard", 12],
    ["AI Agent", 1, 12, "Morning", "Regular", 30],
    ["AI Agent", 1, 15, "Afternoon", "Standard", 15],
    ["AI Agent", 1, 8, "Specific Time", "Regular", 28],
    ["AI Agent", 0, 60, "Morning", "Standard", 0],
    ["AI Agent", 1, 14, "Afternoon", "Regular", 32],
    ["AI Agent", 1, 11, "Specific Time", "Standard", 14],
]


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the distance between two GPS points using the Haversine formula."""
    earth_radius_km = 6371.0

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_km * c


def load_graph_data() -> Tuple[Any, Any]:
    """Loads the OSM graph from cache or raw file, falling back to a fresh download if necessary."""
    CACHE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE_PATH.exists():
        logger.info("⚡ Ultra-fast loading from cache (%s)...", CACHE_FILE_PATH.name)
        with open(CACHE_FILE_PATH, 'rb') as f:
            graph, proj_graph = pickle.load(f)
        return graph, proj_graph

    logger.info("⏳ Cache not found. Preparing the graph...")
    if GRAPH_FILE_PATH.exists():
        logger.info("✅ Expert Graph detected: %s", GRAPH_FILE_PATH.name)
        graph = ox.load_graphml(GRAPH_FILE_PATH)
    else:
        logger.warning("❌ File not found. Emergency fallback download (Paris center)...")
        # Safety fallback
        graph = ox.graph_from_point((48.8566, 2.3522), dist=3000, network_type='drive')
        graph = ox.routing.add_edge_speeds(graph)
        graph = ox.routing.add_edge_travel_times(graph)

    proj_graph = ox.project_graph(graph)

    logger.info("💾 Saving to cache...")
    with open(CACHE_FILE_PATH, 'wb') as f:
        pickle.dump((graph, proj_graph), f)

    return graph, proj_graph


logger.info("⏳ Initializing graph ecosystem...")
G, G_proj = load_graph_data()


# --- STATION INITIALIZATION & CACHING ---
def load_or_create_stations() -> Dict[Any, BuildingStation]:
    """Initializes the station network and saves the state to cache."""
    logger.info("⏳ Creating new stations and saving to cache...")
    real_hour = datetime.now().hour
    simulated_hour = max(8, real_hour)

    generated_stations = initialize_stations(
        G, daily_weather=DAILY_WEATHER, num_stations=500, current_hour=simulated_hour
    )

    with open(STATIONS_FILE_PATH, 'wb') as f:
        pickle.dump(generated_stations, f)

    return generated_stations


def save_stations_state() -> None:
    """Saves the current state of all stations (reservations, dynamic prices) to disk."""
    with open(STATIONS_FILE_PATH, 'wb') as f:
        pickle.dump(STATIONS, f)


# Load stations on server startup
STATIONS = load_or_create_stations()

# Populate queues with ghost users for realistic load
initialize_ghost_users(STATIONS)


# ==========================================
# WEB ROUTES
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/stations', methods=['GET'])
def get_stations():
    stations_data = []
    for node_id, station in STATIONS.items():
        dyn_price, pricing_case, batt_pct = station.calculate_dynamic_price()

        # Display negotiated price if an agreement was reached
        displayed_price = station.negotiated_price if station.negotiated_price else dyn_price
        reserved_slots = get_reserved_slots(station.station_id)

        stations_data.append({
            "id": station.station_id,
            "lat": G.nodes[node_id]['y'],
            "lon": G.nodes[node_id]['x'],
            "battery": round(batt_pct, 1),
            "price": round(displayed_price, 3),
            "pricing_case": pricing_case,
            "reserved_slots": reserved_slots
        })
    return jsonify({"status": "success", "stations": stations_data})


@app.route('/calculer', methods=['POST'])
def route_calculation():
    payload = request.json or {}
    # Default to 100% battery if not provided
    if 'current_battery' not in payload:
        payload['current_battery'] = 100.0

    result = calculate_routes(payload, G, G_proj, STATIONS)
    return jsonify(result)


# ==========================================
# RESERVATION API
# ==========================================

@app.route('/api/reserver', methods=['POST'])
def reserve_station():
    data = request.json or {}
    status, message = process_reservation(data.get('station_id'), data.get('mode'), data.get('hour'), stations=STATIONS)

    if status == "success":
        save_stations_state()

    return jsonify({"status": status, "message": message})


@app.route('/api/annuler', methods=['POST'])
def cancel_reservation():
    data = request.json or {}
    status, message = process_cancellation(data.get('station_id'), data.get('hour'))

    if status == "success":
        save_stations_state()

    return jsonify({"status": status, "message": message})


# ==========================================
# USER PROFILE API
# ==========================================

@app.route('/profil', methods=['GET'])
def display_profile():
    return render_template('profil.html')


@app.route('/api/profil', methods=['GET', 'POST'])
def manage_profile_api():
    if request.method == 'GET':
        return jsonify({"status": "success", "profile": read_profile()})

    if request.method == 'POST':
        update_profile(request.json or {})
        return jsonify({"status": "success", "message": "Profile updated successfully!"})


@app.route('/reservations', methods=['GET'])
def display_reservations():
    return render_template('reservations.html')


# ==========================================
# NEGOTIATION API
# ==========================================

@app.route('/api/negociation/start', methods=['POST'])
def start_negotiation():
    """Initializes a negotiation session for a specific station."""
    data = request.json or {}
    target_id = data.get('station_id')

    # Find the target station object
    station = next((s for s in STATIONS.values() if s.station_id == target_id), None)

    if not station:
        return jsonify({"status": "error", "message": "Station not found."})

    # Start session (60 seconds default)
    session = station.initiate_negotiation(is_regular=False, max_duration_seconds=60)
    ACTIVE_NEGOTIATIONS[target_id] = session

    return jsonify({"status": "success", "message": "Negotiation started."})


@app.route('/api/negociation/offer', methods=['POST'])
def evaluate_offer():
    """Receives the frontend offer and evaluates it via the Python agent."""
    data = request.json or {}
    target_id = data.get('station_id')
    user_offer = float(data.get('offer', 0))

    if target_id not in ACTIVE_NEGOTIATIONS:
        return jsonify({"status": "error", "message": "No active negotiation for this station."})

    session = ACTIVE_NEGOTIATIONS[target_id]
    station = next((s for s in STATIONS.values() if s.station_id == target_id), None)

    if not station:
        return jsonify({"status": "error", "message": "Station not found."})

    # Agent evaluates the offer
    is_accepted, message, new_price = station.evaluate_offer(session, user_offer)

    return jsonify({
        "status": "success",
        "accepted": is_accepted,
        "message": message,
        "new_price": new_price,
        "session_status": session["status"]
    })


@app.route('/api/negociation/accept', methods=['POST'])
def accept_negotiation():
    data = request.json or {}
    target_id = data.get('station_id')
    final_price = data.get('final_price')

    if target_id in ACTIVE_NEGOTIATIONS:
        ACTIVE_NEGOTIATIONS[target_id]["status"] = "accepted"
        ACTIVE_NEGOTIATIONS[target_id]["final_price"] = final_price

        for s in STATIONS.values():
            if s.station_id == target_id:
                s.negotiated_price = final_price
                break

        save_stations_state()

        # Add successful manual negotiation to the UI demo history
        DEMO_HISTORY.append(["Manual", 1, 35, "Afternoon", "Standard", 15])

    return jsonify({"status": "success", "message": "Agreement concluded!"})


@app.route('/api/negociation/auto', methods=['POST'])
def auto_negotiation():
    """The Personal AI Agent negotiates with all stations within a 50km radius."""
    data = request.json or {}
    user_max_price = float(data.get('max_price', 0))
    user_coords = data.get('coords')

    if not user_coords:
        return jsonify({"status": "error", "message": "📍 GPS position missing. Please enable geolocation."})

    u_lat = user_coords['lat']
    u_lon = user_coords['lon']

    # Forcing advantage for the demo
    is_regular_user = True
    best_station = None
    best_price = float('inf')
    search_radius_km = 50.0

    for node_id, station in STATIONS.items():
        s_lat = G.nodes[node_id]['y']
        s_lon = G.nodes[node_id]['x']

        distance = calculate_distance_km(u_lat, u_lon, s_lat, s_lon)

        if distance > search_radius_km:
            continue

        # LIGHTNING NEGOTIATION
        session = station.initiate_negotiation(is_regular=is_regular_user, max_duration_seconds=60)
        session["start_timestamp"] = time.time() - 59  # Simulate last possible second (Max stress)

        # Extreme lowball offer to force the station's absolute floor limit
        is_accepted, message, proposed_price = station.evaluate_offer(session, 0.01)

        if proposed_price and proposed_price <= user_max_price:
            if proposed_price < best_price:
                best_price = proposed_price
                best_station = station

    if best_station:
        best_station.negotiated_price = best_price
        save_stations_state()

        return jsonify({
            "status": "success",
            "station_id": best_station.station_id,
            "final_price": best_price,
            "message": f"🎉 AI: €{round(best_price, 3)} validated at {best_station.station_id}!"
        })
    else:
        # Add a failed AI negotiation to the UI demo history
        DEMO_HISTORY.append(["AI Agent", 1, 12, "Specific Time", "Regular", 28])
        return jsonify({"status": "error", "message": f"❌ No station within 50km accepted €{user_max_price}."})


# ==========================================
# CHART GENERATION API
# ==========================================

@app.route('/api/generer_statistiques', methods=['GET'])
def generate_statistics():
    """Generates the main performance chart based on demo history."""
    image_path = "static/images/rapport_negociation.png"
    generate_negotiation_metrics(data_source=DEMO_HISTORY, output_filename=image_path)

    return jsonify({
        "status": "success",
        "message": "Charts updated successfully!",
        "image_url": f"/{image_path}"
    })


@app.route('/api/generer_stats_temps', methods=['GET'])
def stats_time_bounds():
    image_path = "static/images/stats_temps_borne.png"
    generate_bounded_time_stats(DEMO_HISTORY, output_filename=image_path)
    return jsonify({"status": "success", "image_url": f"/{image_path}"})


@app.route('/api/generer_stats_couts', methods=['GET'])
def stats_costs():
    image_path = "static/images/stats_couts.png"
    generate_cost_comparison_stats(DEMO_HISTORY, output_filename=image_path)
    return jsonify({"status": "success", "image_url": f"/{image_path}"})


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)