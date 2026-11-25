"""
Charging station module for EV mobility planning.
Handles station state, weather API integration, and negotiation agents.
"""

import math
import time
import random
import logging
from typing import Dict, Tuple, Any, Optional, List
import requests

# Configure module-level logger
logger = logging.getLogger(__name__)


# WEATHER MANAGEMENT (EXTERNAL API)

def fetch_daily_weather_paris(target_date: str) -> Dict[int, float]:
    """
    Fetches hour-by-hour cloud cover for Paris via Open-Meteo.

    Args:
        target_date (str): Target date in "YYYY-MM-DD" format.

    Returns:
        Dict[int, float]: A dictionary mapping the hour (0-23) to cloud cover percentage (0.0 to 1.0).
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude=48.85&longitude=2.35"
        f"&hourly=cloudcover&timezone=Europe%2FBerlin&start_date={target_date}"
        f"&end_date={target_date}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        hours = data['hourly']['time'][:24]
        clouds = data['hourly']['cloudcover'][:24]

        daily_weather: Dict[int, float] = {}
        for i in range(24):
            formatted_hour = int(hours[i].split("T")[1][:2])
            daily_weather[formatted_hour] = clouds[i] / 100.0

        return daily_weather

    except requests.RequestException as e:
        logger.error("Weather API Error: %s. Falling back to default 20%% cloud cover.", e)
        return {h: 0.2 for h in range(24)}



# STATION MODELING (BUILDING + BATTERY + CHARGERS)
class BuildingStation:
    """
    Represents a building acting as a charging hub with its own solar production,
    local battery storage, and dynamic pricing negotiation agent.
    """

    def __init__(self, station_id: str, max_battery_capacity: float = 100.0, base_edf_price: float = 0.25) -> None:
        """
        Initializes the charging station.

        Args:
            station_id (str): Unique identifier for the station.
            max_battery_capacity (float): Maximum battery capacity in kWh.
            base_edf_price (float): Base electricity price in €/kWh.
        """
        self.station_id: str = station_id
        self.max_battery_capacity: float = max_battery_capacity
        self.current_battery_level: float = 50.0  # Starts at 50 kWh (50%)
        self.base_edf_price: float = base_edf_price
        self.negotiated_price: Optional[float] = None

        # The station manages 4 physical charging points
        self.chargers: List[str] = [
            f"Charger_1_{station_id}", f"Charger_2_{station_id}",
            f"Charger_3_{station_id}", f"Charger_4_{station_id}"
        ]

    def simulate_production_and_recharge(self, hour_of_day: int, clouds: float, building_consumption: float) -> Tuple[
        float, float]:
        """
        Calculates the building's solar production. Surplus recharges the local battery.

        Args:
            hour_of_day (int): Current hour of the day (0-23).
            clouds (float): Cloud cover ratio (0.0 to 1.0).
            building_consumption (float): Building energy consumption in kWh.

        Returns:
            Tuple[float, float]: Solar production and the surplus energy generated (in kWh).
        """
        # 1. Solar production (Bell curve centered at 13:00)
        max_solar_production = 60.0
        actual_power = max_solar_production * (1 - clouds)
        solar_production = actual_power * math.exp(-((hour_of_day - 13) ** 2) / (2 * 3 ** 2))

        # 2. Surplus calculation (energy not consumed by the building)
        surplus = max(0.0, solar_production - building_consumption)

        # 3. Local battery recharge using surplus
        self.current_battery_level += surplus

        # Ensure battery does not exceed 100% capacity
        if self.current_battery_level > self.max_battery_capacity:
            self.current_battery_level = self.max_battery_capacity

        return solar_production, surplus

    def calculate_dynamic_price(self, sim_config: Optional[Dict[str, Any]] = None) -> Tuple[float, str, float]:
        """
        Applies the economic model based on current battery level.

        Args:
            sim_config (Optional[Dict[str, Any]]): Simulation configuration for baseline testing.

        Returns:
            Tuple[float, str, float]: The calculated price, the case description, and battery percentage.
        """
        battery_percentage = (self.current_battery_level / self.max_battery_capacity) * 100


        # GREEDY BASELINE
        if sim_config and sim_config.get("baseline") == "Greedy":
            price = self.base_edf_price * 1.60
            pricing_case = "Case D (Greedy Baseline: Price forced +60%)"
            return price, pricing_case, battery_percentage

        # --- Standard classic logic ---
        if battery_percentage > 80:
            price = self.base_edf_price * 0.80
            pricing_case = "Case C (Battery > 80%: Price -20%)"
        elif 30 <= battery_percentage <= 80:
            price = self.base_edf_price
            pricing_case = "Case A (Battery 30-80%: Normal Price)"
        elif 10 <= battery_percentage < 30:
            price = self.base_edf_price * 1.30
            pricing_case = "Case B (Battery 10-30%: Price +30%)"
        else:
            price = self.base_edf_price * 1.60
            pricing_case = "Case D (Battery < 10%: Price +60%)"

        return price, pricing_case, battery_percentage

    def charge_vehicle(self, requested_energy_kwh: float) -> None:
        """
        Simulates an EV plugging in and draining the battery.

        Args:
            requested_energy_kwh (float): Energy requested by the EV in kWh.
        """
        self.current_battery_level -= requested_energy_kwh
        # Safeguard to prevent negative battery levels
        self.current_battery_level = max(0.0, self.current_battery_level)

    def initiate_negotiation(self, is_regular: bool = False, max_duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Starts a negotiation session. Floor price is calculated dynamically each round.

        Args:
            is_regular (bool): True if the user has a loyalty profile.
            max_duration_seconds (int): Maximum negotiation duration in seconds.

        Returns:
            Dict[str, Any]: The negotiation session state.
        """
        current_price, pricing_case, batt_pct = self.calculate_dynamic_price()

        nego_session = {
            "start_price": round(current_price, 3),
            "is_regular": is_regular,
            "last_counter_offer": round(current_price, 3),
            "start_timestamp": time.time(),
            "max_duration": max_duration_seconds,
            "status": "in_progress",
            "history": [{
                "time": 0.0,
                "user_offer": None,
                "counter_offer": round(current_price, 3),
                "secret_floor": round(current_price * (1 - (0.20 if is_regular else 0.10)), 3)
            }]
        }
        return nego_session

    def evaluate_offer(self, nego_session: Dict[str, Any], user_offer: float,
                       sim_config: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, float]:
        """
        Evaluates the user's offer using a dynamic utility function.

        Args:
            nego_session (Dict[str, Any]): Current negotiation session state.
            user_offer (float): Price offered by the user.
            sim_config (Optional[Dict[str, Any]]): Simulation configuration.

        Returns:
            Tuple[bool, str, float]: Success status, agent message, and the new/final price.
        """

        # NEGOTIATION ABLATION
        negotiation_active = sim_config.get("negotiation", True) if sim_config else True
        if not negotiation_active:
            # Without negotiation, the only acceptable price is the starting price
            if user_offer >= nego_session["start_price"]:
                self.negotiated_price = user_offer
                return True, "Fixed price agreement (Negotiation Ablation).", user_offer
            else:
                return False, f"Negotiation Ablation: Fixed price required (€{nego_session['start_price']}).", \
                nego_session["start_price"]

        # Standard logic (Timer etc.)
        if nego_session["status"] != "in_progress":
            return False, "The negotiation session is already concluded.", nego_session.get("final_price", 0.0)

        elapsed_time = time.time() - nego_session["start_timestamp"]
        if elapsed_time > nego_session["max_duration"]:
            nego_session["status"] = "expired"
            return False, f"⏳ Time's up! ({int(elapsed_time)}s elapsed).", 0.0

        # Base margin based on user profile
        max_margin = 0.20 if nego_session["is_regular"] else 0.10

        # Battery level impact
        batt_pct = (self.current_battery_level / self.max_battery_capacity) * 100
        if batt_pct > 80:
            max_margin += 0.05
        elif batt_pct < 30:
            max_margin -= 0.05

        # Time pressure impact
        time_ratio = min(elapsed_time / nego_session["max_duration"], 1.0)
        max_margin += time_ratio * 0.05

        current_floor_price = nego_session["start_price"] * (1 - max_margin)
        current_floor_price = max(current_floor_price, 0.05)

        # Record the current state of the negotiation step
        history_step = {
            "time": round(elapsed_time, 2),
            "user_offer": user_offer,
            "secret_floor": round(current_floor_price, 3)
        }

        # CASE 1: SAFEGUARD (User bids higher than latest counter-offer)
        if user_offer > nego_session["last_counter_offer"]:
            return False, f"❌ The proposed price cannot exceed €{nego_session['last_counter_offer']}.", nego_session[
                "last_counter_offer"]

        # CASE 2: AGREEMENT REACHED
        elif user_offer >= current_floor_price:
            nego_session["status"] = "accepted"
            nego_session["final_price"] = user_offer
            self.negotiated_price = user_offer
            return True, f"✅ Agreement reached! The station accepts your offer at €{user_offer}.", user_offer

        # CASE 3: TACTICAL COUNTER-OFFER
        else:
            remaining_margin = nego_session["last_counter_offer"] - current_floor_price

            # RULE-BASED BASELINE
            if sim_config and sim_config.get("baseline") == "Rule_Based":
                effort_percentage = 0.10  # Fixed 10% concession
            else:
                effort_percentage = 0.20 + (time_ratio * 0.40)  # Time-based AI logic

            effort = remaining_margin * effort_percentage
            counter_offer = nego_session["last_counter_offer"] - effort
            counter_offer = max(current_floor_price, counter_offer)
            counter_offer = round(counter_offer, 3)

            history_step["counter_offer"] = counter_offer
            nego_session["history"].append(history_step)

            if counter_offer >= nego_session["last_counter_offer"]:
                return False, f"❌ This is my final offer. I won't go below €{nego_session['last_counter_offer']}.", \
                nego_session["last_counter_offer"]

            nego_session["last_counter_offer"] = counter_offer
            return False, f"❌ Too low. How about €{counter_offer}?", counter_offer


def initialize_stations(graph: Any, daily_weather: Dict[int, float], num_stations: int = 20, current_hour: int = 14) -> \
Dict[Any, BuildingStation]:
    """
    Creates stations, places them on the graph, and simulates their morning recharge.

    Args:
        graph (nx.DiGraph): The routing network graph (NetworkX).
        daily_weather (Dict[int, float]): Dictionary of hourly cloud cover.
        num_stations (int): Number of stations to initialize.
        current_hour (int): The current hour of the simulation.

    Returns:
        Dict[Any, BuildingStation]: A dictionary mapping graph nodes to station instances.
    """
    logger.info("Placing and simulating weather for %d stations...", num_stations)

    nodes = list(graph.nodes())
    chosen_nodes = random.sample(nodes, num_stations)
    created_stations: Dict[Any, BuildingStation] = {}

    for node in chosen_nodes:
        # 1. Assign a random base price per neighborhood
        base_price = round(random.uniform(0.20, 0.35), 2)

        # Retrieve real street name via OSM graph ---
        street_name = "Unknown Avenue"
        try:
            # Check outgoing edges from this node
            edges = list(graph.out_edges(node, data=True))
            if edges:
                route_data = edges[0][2]
                raw_name = route_data.get('name', 'Unknown Avenue')

                if isinstance(raw_name, list):
                    street_name = raw_name[0]
                else:
                    street_name = raw_name
        except Exception as e:
            logger.debug("Could not extract street name for node %s: %s", node, e)

        station_name = f"Station - {street_name} ({str(node)[-4:]})"

        # 2. Station creation
        station = BuildingStation(station_id=station_name, base_edf_price=base_price)

        # 3. Simulate morning: battery drained overnight
        station.current_battery_level = random.uniform(10.0, 30.0)

        # 4. WEATHER simulation up to current hour
        for h in range(8, current_hour + 1):
            clouds_h = daily_weather.get(h, 0.2)
            building_consumption = random.uniform(10.0, 50.0)
            station.simulate_production_and_recharge(h, clouds_h, building_consumption)

        created_stations[node] = station

    return created_stations



# TERMINAL SIMULATION TEST
if __name__ == "__main__":
    # Basic logging setup for the standalone script
    logging.basicConfig(level=logging.INFO)

    my_station = BuildingStation(station_id="VIP_Charger_01", max_battery_capacity=100.0, base_edf_price=0.25)
    my_station.current_battery_level = 50.0

    print("=======================================")
    print("🤖 WELCOME TO THE NEGOTIATION AGENT")
    print("=======================================")

    is_regular_user = False
    print(f"👤 User Profile: {'Regular (Negotiation Advantage)' if is_regular_user else 'Standard'}")

    session = my_station.initiate_negotiation(is_regular=is_regular_user, max_duration_seconds=30)

    print(f"\n💡 Current displayed price is: €{session['start_price']} / kWh")
    print(f"⏳ You have {session['max_duration']} seconds to negotiate a better rate!")

    # Debug estimation
    estimated_floor = session['start_price'] * (1 - (0.20 if is_regular_user else 0.10))
    print(
        f"   [Secret Debug] The station will start with a floor around: €{round(estimated_floor, 3)} (but this price will decrease over time!)")

    while session["status"] == "in_progress":
        user_input = input("\n👉 Enter your proposed price (e.g., 0.22) or type 'q' to quit: ")

        if user_input.lower() == 'q':
            print("Negotiation cancelled by the user.")
            break

        try:
            offer = float(user_input.replace(',', '.'))
            is_accepted, message, new_price = my_station.evaluate_offer(session, offer)

            print(message)

            if is_accepted or "Time's up" in message:
                break

        except ValueError:
            print("⚠️ Please enter a valid number with a decimal point (e.g., 0.22).")