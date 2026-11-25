"""
Core routing module for EV mobility planning.
Handles traffic modeling, A* pathfinding, and dynamic EV rerouting (Yield Management).
"""

import math
import logging
import time
import datetime
from typing import Dict, Any, List, Tuple, Optional, Callable
import osmnx as ox
import networkx as nx

# Imports from refactored modules
from src.core.physique import calculate_physical_consumption, get_vehicle_characteristics
from src.core.reservations import get_reserved_slots
# Import from unrefactored module
from src.core.user_profil import read_profile

# Configure module-level logger
logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Converts seconds into a readable string format."""
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60}h {minutes % 60:02d}min"


def parse_hour(hour_str: Optional[str]) -> Optional[float]:
    """Parses a time string (HH:MM) into a decimal float."""
    if not hour_str:
        return None
    try:
        h, m = map(int, hour_str.split(':'))
        return h + (m / 60.0)
    except Exception as e:
        logger.warning("Failed to parse hour string '%s': %s", hour_str, e)
        return None


def calculate_real_time(edge_data: Dict[str, Any], departure_hour: float) -> float:
    """
    Traffic model calibrated on real travel times in Île-de-France.
    Adds time penalties based on road profile and time of day.
    """
    base_time_raw = float(edge_data.get('travel_time_base', edge_data.get('travel_time', 10.0)))

    # Empty road correction: +35% to simulate real average speed vs theoretical max
    empty_time = base_time_raw * 1.35

    profile = str(edge_data.get('traffic_profile', 'local'))
    max_penalty = float(edge_data.get('penalty_max', 0.0))
    traffic_intensity = 0.0

    # Morning rush (07:00 - 09:30)
    if 7.0 <= departure_hour <= 9.5:
        if profile == 'entrant':
            traffic_intensity = 0.15
        elif profile == 'rocade':
            traffic_intensity = 0.10
        elif profile == 'sortant':
            traffic_intensity = 0.02
        elif profile == 'tangentiel':
            traffic_intensity = 0.05

    # Evening rush (16:30 - 19:30)
    elif 16.5 <= departure_hour <= 19.5:
        if profile == 'sortant':
            traffic_intensity = 0.15
        elif profile == 'rocade':
            traffic_intensity = 0.12
        elif profile == 'entrant':
            traffic_intensity = 0.02
        elif profile == 'tangentiel':
            traffic_intensity = 0.05

    # Standard daytime (Off-peak)
    elif 9.5 < departure_hour < 16.5:
        if profile != 'local':
            traffic_intensity = 0.03

    extra_time = empty_time * max_penalty * traffic_intensity
    return empty_time + extra_time


def find_potential_stations(proj_graph: nx.MultiDiGraph, start_node: int, end_node: int, stations: Dict[Any, Any]) -> \
List[Any]:
    """Finds the top 10 best stations, heavily favoring those closer to the departure point."""
    x_start, y_start = proj_graph.nodes[start_node]['x'], proj_graph.nodes[start_node]['y']
    x_end, y_end = proj_graph.nodes[end_node]['x'], proj_graph.nodes[end_node]['y']

    scores = []
    for station_node in stations.keys():
        if station_node in proj_graph.nodes:
            x_s, y_s = proj_graph.nodes[station_node]['x'], proj_graph.nodes[station_node]['y']

            dist_start_to_station = math.hypot(x_s - x_start, y_s - y_start)
            dist_station_to_end = math.hypot(x_end - x_s, y_end - y_s)

            classic_detour = dist_start_to_station + dist_station_to_end

            # Heavily penalize distant stations to survive with low battery
            final_score = classic_detour + (dist_start_to_station * 2.0)
            scores.append((final_score, station_node))

    scores.sort(key=lambda x: x[0])
    return [s[1] for s in scores[:10]]


def _get_street_stats(edge_data: Dict[str, Any]) -> Tuple[float, float, float]:
    """Helper: Extracts length, grade, and speed from a network edge."""
    length_raw = edge_data.get('length', 0)
    length = float(length_raw[0] if isinstance(length_raw, list) else length_raw)

    grade_raw = edge_data.get('grade', 0)
    grade = float(grade_raw[0] if isinstance(grade_raw, list) else grade_raw)

    speed_raw = edge_data.get('maxspeed', edge_data.get('speed_kph', 50.0))
    try:
        speed_str = str(speed_raw[0] if isinstance(speed_raw, list) else speed_raw).lower()
        speed = float(speed_str.replace('km/h', '').replace('mph', '').strip())
    except (ValueError, TypeError):
        speed = 50.0

    return length, grade, speed


def extract_route_stats(graph: nx.MultiDiGraph, proj_graph: nx.MultiDiGraph, path: List[int], label: str,
                        user_profile: Dict[str, Any], departure_hour: float = 8.5, current_battery_pct: float = 100.0,
                        chosen_weight: str = 'length') -> Dict[str, Any]:
    """Iterates through a path to compute cumulative distance, time, and energy consumption."""
    dist_m = 0.0
    time_s = 0.0
    standard_energy_kwh = 0.0
    out_of_battery = False

    vehicle_name = user_profile.get("vehicle", "Unknown")
    vehicle_physics = get_vehicle_characteristics(vehicle_name)
    max_battery_capacity = vehicle_physics["battery"]

    current_battery_kwh = max_battery_capacity * (float(current_battery_pct) / 100.0)

    for u, v in zip(path[:-1], path[1:]):
        edge_data_dict = proj_graph.get_edge_data(u, v)

        # Select the specific edge chosen by the algorithm
        if chosen_weight == 'energy':
            best_edge = min(edge_data_dict.values(),
                            key=lambda x: calculate_physical_consumption(*_get_street_stats(x), vehicle_physics))
        elif chosen_weight == 'time':
            best_edge = min(edge_data_dict.values(), key=lambda x: calculate_real_time(x, departure_hour))
        else:
            best_edge = min(edge_data_dict.values(), key=lambda x: x.get('length', float('inf')))

        length, grade, speed = _get_street_stats(best_edge)

        # Cumulative calculations
        dist_m += length
        time_s += calculate_real_time(best_edge, departure_hour)

        # Step-by-step battery simulation
        segment_consumption = calculate_physical_consumption(length, speed, grade, vehicle_physics)
        standard_energy_kwh += segment_consumption
        current_battery_kwh -= segment_consumption

        # Battery cannot exceed 100% during long downhills
        current_battery_kwh = min(max_battery_capacity, current_battery_kwh)

        if current_battery_kwh <= 0:
            out_of_battery = True
            current_battery_kwh = 0.0

    arrival_percentage = (current_battery_kwh / max_battery_capacity) * 100
    gps_route = [[graph.nodes[n]['y'], graph.nodes[n]['x']] for n in path]

    return {
        "label": label,
        "route": gps_route,
        "distance": round(dist_m / 1000.0, 2),
        "time_text": format_time(time_s),
        "time_s": time_s,
        "energy": round(standard_energy_kwh, 2),
        "arrival_battery": round(arrival_percentage, 1),
        "out_of_battery": out_of_battery
    }


def _evaluate_rerouting(route_plan: Dict[str, Any], proj_graph: nx.MultiDiGraph, graph: nx.MultiDiGraph,
                        stations: Dict[Any, Any], start_node: int, end_node: int, user_profile: Dict[str, Any],
                        departure_hour: float, current_battery_pct: float, max_battery_capacity: float) -> None:
    """Helper: Executes the emergency rerouting logic if the primary route fails."""
    top_10_stations = find_potential_stations(proj_graph, start_node, end_node, stations)
    valid_station_found = False

    for station_node in top_10_stations:
        if valid_station_found:
            break
        try:
            # Leg 1: Start -> Station
            path_leg1 = nx.astar_path(proj_graph, start_node, station_node, weight=route_plan["weight"])
            stats_1 = extract_route_stats(graph, proj_graph, path_leg1, "Leg1", user_profile, departure_hour,
                                          current_battery_pct, route_plan["choice"])

            if stats_1["out_of_battery"]:
                continue  # Battery dead before reaching the station

            # Station assessment
            station_obj = stations[station_node]
            dynamic_price, _, station_batt_pct = station_obj.calculate_dynamic_price()

            if station_batt_pct < 20.0:
                continue  # Station is empty

            # Slot availability check
            arrival_hour_at_station = departure_hour + (stats_1["time_s"] / 3600.0)
            rounded_slot = f"{int(arrival_hour_at_station):02d}:00"
            occupied_slots = get_reserved_slots(station_obj.station_id)

            if rounded_slot in occupied_slots:
                continue  # Slot already booked

            # Cost calculation
            price_per_kwh = getattr(station_obj, 'negotiated_price', None) or dynamic_price
            missing_kwh = ((80.0 - stats_1["arrival_battery"]) / 100.0) * max_battery_capacity
            recharge_cost = missing_kwh * price_per_kwh

            # Leg 2: Station -> End
            path_leg2 = nx.astar_path(proj_graph, station_node, end_node, weight=route_plan["weight"])
            battery_after_charge = 80.0
            departure_hour_leg2 = arrival_hour_at_station + 0.5  # + 30 min charging
            stats_2 = extract_route_stats(graph, proj_graph, path_leg2, "Leg2", user_profile, departure_hour_leg2,
                                          battery_after_charge, route_plan["choice"])

            total_time = stats_1["time_s"] + stats_2["time_s"] + 1800  # 1800s = 30min stop

            # Overwrite failed stats with successful rerouted stats
            route_plan["stats"] = {
                "label": f'{route_plan["name"]} (Via Station)',
                "route": stats_1["route"] + stats_2["route"][1:],
                "distance": round(stats_1["distance"] + stats_2["distance"], 2),
                "time_s": total_time,
                "time_text": format_time(total_time) + " (+30min ⚡)",
                "energy": round(stats_1["energy"] + stats_2["energy"], 2),
                "arrival_battery": stats_2["arrival_battery"],
                "out_of_battery": stats_2["out_of_battery"],
                "recharge_cost": round(recharge_cost, 2),
                "station_id": station_obj.station_id,
                "station_time": rounded_slot,
                "base_kwh_price": round(dynamic_price, 3)
            }
            valid_station_found = True

        except (nx.NetworkXNoPath, Exception) as e:
            logger.debug("Rerouting leg failed: %s", e)
            continue


def calculate_routes(request_data: Dict[str, Any], graph: nx.MultiDiGraph, proj_graph: nx.MultiDiGraph,
                     stations: Dict[Any, Any], sim_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Main pipeline: Resolves coordinates, computes 3 distinct A* routes, and handles EV ablations."""
    try:
        departure_hour_str = request_data.get('departure_hour', None)
        departure_hour = parse_hour(departure_hour_str)
        current_battery_pct = float(request_data.get('current_battery', 100.0))
        t_start_global = time.time()
        if 'start_coords' in request_data:
            coord_start = (request_data['start_coords']['lat'], request_data['start_coords']['lon'])
        else:
            coord_start = ox.geocode(request_data['start_location'])

        if 'end_coords' in request_data:
            coord_end = (request_data['end_coords']['lat'], request_data['end_coords']['lon'])
        else:
            coord_end = ox.geocode(request_data['end_location'])

        node_start = ox.nearest_nodes(graph, X=coord_start[1], Y=coord_start[0])
        node_end = ox.nearest_nodes(graph, X=coord_end[1], Y=coord_end[0])

        if not nx.has_path(proj_graph, node_start, node_end):
            return {"status": "error", "message": "No path found."}

        user_profile = read_profile()
        vehicle_name = user_profile.get("vehicle", "Unknown")
        vehicle_physics = get_vehicle_characteristics(vehicle_name)
        max_battery_capacity = vehicle_physics["battery"]

        # Dynamic weight closures
        def dynamic_fast_weight(u: int, v: int, edges_dict: Dict[Any, Any]) -> float:
            return min([calculate_real_time(edge_data, departure_hour) for edge_data in edges_dict.values()])

        def dynamic_eco_weight(u: int, v: int, edges_dict: Dict[Any, Any]) -> float:
            min_energy = float('inf')
            for edge_data in edges_dict.values():
                length, grade, speed = _get_street_stats(edge_data)
                energy = calculate_physical_consumption(length, speed, grade, vehicle_physics)
                if energy < min_energy:
                    min_energy = energy
            return max(0.001, min_energy)

        t_start_astar = time.time()

        # A* Computations
        path_fast = nx.astar_path(proj_graph, node_start, node_end, weight=dynamic_fast_weight)
        path_short = nx.astar_path(proj_graph, node_start, node_end, weight='length')
        path_eco = nx.astar_path(proj_graph, node_start, node_end, weight=dynamic_eco_weight)

        stats_fast = extract_route_stats(graph, proj_graph, path_fast, "Fast", user_profile, departure_hour,
                                         current_battery_pct, 'time')
        stats_short = extract_route_stats(graph, proj_graph, path_short, "Short", user_profile, departure_hour,
                                          current_battery_pct, 'length')
        stats_eco = extract_route_stats(graph, proj_graph, path_eco, "Eco", user_profile, departure_hour,
                                        current_battery_pct, 'energy')

        times_astar_ms = (time.time() - t_start_astar) * 1000
        route_plans = [
            {"stats": stats_fast, "weight": dynamic_fast_weight, "choice": 'time', "name": "Fast"},
            {"stats": stats_short, "weight": 'length', "choice": 'length', "name": "Short"},
            {"stats": stats_eco, "weight": dynamic_eco_weight, "choice": 'energy', "name": "Eco"}
        ]

        # REROUTING ABLATION
        ai_rerouting_active = sim_config.get("ai_rerouting", True) if sim_config else True

        t_start_reroute = time.time()

        for plan in route_plans:
            if plan["stats"]["out_of_battery"]:
                if not ai_rerouting_active:
                    plan["stats"]["label"] += " (FAILED: Dead Battery)"
                    continue

                _evaluate_rerouting(
                    route_plan=plan, proj_graph=proj_graph, graph=graph, stations=stations,
                    start_node=node_start, end_node=node_end, user_profile=user_profile,
                    departure_hour=departure_hour, current_battery_pct=current_battery_pct,
                    max_battery_capacity=max_battery_capacity
                )
        times_reroute_ms = (time.time() - t_start_reroute) * 1000
        times_total_ms = (time.time() - t_start_global) * 1000
        return {
            "status": "success",
            "fastest": route_plans[0]["stats"],
            "shortest": route_plans[1]["stats"],
            "eco": route_plans[2]["stats"],
            "start_coords": coord_start,
            "end_coords": coord_end,
            "profiling": {
                "total_ms": round(times_total_ms, 2),
                "astar_ms": round(times_astar_ms, 2),
                "reroute_ms": round(times_reroute_ms, 2)
            }
        }

    except Exception as e:
        logger.error("Routing failed: %s", e)
        return {"status": "error", "message": str(e)}