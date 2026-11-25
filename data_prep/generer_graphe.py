"""
Data preparation pipeline for the Île-de-France road network.
Downloads OSM data, merges satellite elevation, and calculates traffic heuristics.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import gc
import math
import logging
from typing import Dict, Tuple, Any, Optional, Set
import osmnx as ox
import networkx as nx
import rasterio
from pathlib import Path

# Add project root to path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths using pathlib standards
ELEVATION_TIF_PATH = RAW_DATA_DIR / "altitude_IDF.tif"
GRAPH_DEST_PATH = PROCESSED_DATA_DIR / "IDF.graphml"

# Major hubs for pendulum traffic calculation (Latitude, Longitude)
TRAFFIC_HUBS: Dict[str, Tuple[float, float]] = {
    "Paris_Centre": (48.8584, 2.3465),
    "La_Defense": (48.8923, 2.2372),
    "Roissy_CDG": (49.0096, 2.5479),
    "Saclay": (48.7110, 2.2192)
}


# --- Mathematical Utilities ---

def calculate_gps_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the real-world distance in meters between two GPS coordinates.
    Uses the Haversine formula to account for Earth's curvature.
    """
    earth_radius = 6371000  # Radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    return earth_radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_numeric_value(raw_value: Any, default_val: float) -> float:
    """
    Cleans raw OSM data. Extracts the first numeric value from a string or list.
    (e.g., '50 km/h' -> 50.0). Returns default if extraction fails.
    """
    if isinstance(raw_value, list):
        raw_value = raw_value[0]
    try:
        digits = ''.join(c for c in str(raw_value) if c.isdigit() or c == '.')
        return float(digits) if digits else default_val
    except (ValueError, TypeError):
        return default_val


# --- Pipeline Helper Functions ---

def _download_road_network(place_name: str) -> nx.MultiDiGraph:
    """Downloads the drive network from OpenStreetMap."""
    logger.info("1/5 - Downloading road network (may take several minutes)...")
    ox.settings.use_cache = True
    graph = ox.graph_from_place(place_name, network_type='drive')
    gc.collect()
    return graph


def _apply_elevation_data(graph: nx.MultiDiGraph, tif_path: Path) -> None:
    """Projects satellite elevation data onto graph nodes."""
    logger.info("2/5 - Projecting satellite elevations onto nodes...")
    with rasterio.open(tif_path) as satellite_image:
        elevation_band = satellite_image.read(1)
        for node, data in graph.nodes(data=True):
            lon, lat = data['x'], data['y']
            try:
                row, col = satellite_image.index(lon, lat)
                data['elevation'] = float(elevation_band[row, col])
            except IndexError:
                data['elevation'] = 0.0
    gc.collect()


def _calculate_topographic_grades(graph: nx.MultiDiGraph) -> None:
    """Calculates road grades (slopes) based on node elevations."""
    logger.info("3/5 - Calculating topographic road grades...")
    proj_graph = ox.project_graph(graph)
    proj_graph = ox.elevation.add_edge_grades(proj_graph)

    for u, v, key, data in graph.edges(keys=True, data=True):
        if (u, v, key) in proj_graph.edges:
            edge_data = proj_graph.edges[u, v, key]
            data['grade'] = edge_data.get('grade', 0.0)
            data['grade_abs'] = edge_data.get('grade_abs', 0.0)

    del proj_graph
    gc.collect()


def _find_penalizing_intersections(graph: nx.MultiDiGraph) -> Set[int]:
    """Identifies nodes that act as penalizing intersections (traffic lights, stops)."""
    penalizing_nodes = set()
    for node, data in graph.nodes(data=True):
        highway = str(data.get('highway', ''))
        if 'traffic_signals' in highway or 'stop' in highway:
            penalizing_nodes.add(node)
    return penalizing_nodes


def _apply_heuristic_traffic_model(graph: nx.MultiDiGraph) -> None:
    """Applies traffic heuristics (speeds, intersection delays, pendulum directions)."""
    logger.info("4/5 - Initializing heuristic traffic model...")
    penalizing_intersections = _find_penalizing_intersections(graph)

    hwy_speeds_fallback = {
        "motorway": 110, "trunk": 90, "primary": 70, "secondary": 50,
        "tertiary": 50, "residential": 30, "living_street": 20, "unclassified": 30
    }

    for u, v, key, data in graph.edges(keys=True, data=True):
        hw_type = data.get('highway', 'unclassified')
        if isinstance(hw_type, list):
            hw_type = hw_type[0]

        # Base empty-road travel time
        real_speed = extract_numeric_value(data.get('maxspeed'), hwy_speeds_fallback.get(hw_type, 30))
        data['speed_kph'] = real_speed
        speed_ms = real_speed / 3.6
        pure_time = data.get('length', 0) / speed_ms if speed_ms > 0 else 0

        if v in penalizing_intersections:
            pure_time += 12.0
        data['travel_time_base'] = pure_time

        # Axis behavioral analysis
        street_name = str(data.get('name', '')).lower()
        street_ref = str(data.get('ref', '')).lower()

        if hw_type in ['residential', 'living_street']:
            data['traffic_profile'] = "local"
            data['penalty_max'] = 0.0
            continue

        if 'periph' in street_name or 'a86' in street_ref or '104' in street_ref:
            data['traffic_profile'] = "rocade"
            data['penalty_max'] = 3.0
            continue

        # Hub proximity evaluation
        lat_u, lon_u = graph.nodes[u]['y'], graph.nodes[u]['x']
        lat_v, lon_v = graph.nodes[v]['y'], graph.nodes[v]['x']

        closest_hub = min(TRAFFIC_HUBS.keys(),
                          key=lambda p: calculate_gps_distance(lat_u, lon_u, TRAFFIC_HUBS[p][0], TRAFFIC_HUBS[p][1]))
        dist_u_hub = calculate_gps_distance(lat_u, lon_u, TRAFFIC_HUBS[closest_hub][0], TRAFFIC_HUBS[closest_hub][1])
        dist_v_hub = calculate_gps_distance(lat_v, lon_v, TRAFFIC_HUBS[closest_hub][0], TRAFFIC_HUBS[closest_hub][1])

        lanes = extract_numeric_value(data.get('lanes'), 1)
        data['penalty_max'] = (2.0 if ('motorway' in hw_type or 'trunk' in hw_type) else 1.0) * (1 + (lanes * 0.1))

        # Assign pendulum direction
        if dist_v_hub < dist_u_hub - 20:
            data['traffic_profile'] = "inbound"
        elif dist_v_hub > dist_u_hub + 20:
            data['traffic_profile'] = "outbound"
        else:
            data['traffic_profile'] = "tangential"

# --- Main Pipeline ---

def generate_complete_graph() -> Optional[nx.MultiDiGraph]:
    """
    Main orchestration function for graph creation.
    Downloads network, adds elevation, calculates slopes, applies traffic, and saves.
    """
    logger.info("Starting graph generation pipeline...")

    if not ELEVATION_TIF_PATH.exists():
        logger.error("Topographic file '%s' not found. Aborting process.", ELEVATION_TIF_PATH)
        return None

    try:
        graph = _download_road_network("Île-de-France, France")
        _apply_elevation_data(graph, ELEVATION_TIF_PATH)
        _calculate_topographic_grades(graph)
        _apply_heuristic_traffic_model(graph)

        logger.info("5/5 - Writing data to '%s'...", GRAPH_DEST_PATH)
        GRAPH_DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, filepath=GRAPH_DEST_PATH)

        logger.info("Process completed. The graph is ready for use.")
        return graph

    except Exception as e:
        logger.error("Graph generation failed: %s", e)
        raise


if __name__ == "__main__":
    generate_complete_graph()