"""
Reservation and yield management module.
Handles VIP eviction logic, dynamic routing, and karma penalties.
"""

from datetime import datetime
import random
from typing import Dict, Any, List, Tuple, Optional
import logging

from src.core.user_profil import read_profile, save_profile,apply_karma_update

# Configure module-level logger
logger = logging.getLogger(__name__)

# In-memory database for station reservations
RESERVATIONS_DB: Dict[str, Dict[str, Dict[str, Any]]] = {}


def initialize_ghost_users(stations: Dict[Any, Any]) -> None:
    """
    Populates stations with simulated "ghost" users to create realistic load.
    Forces 30% of stations to be fully saturated to facilitate VIP eviction testing.

    Args:
        stations (Dict[Any, Any]): Dictionary of available charging stations.
    """
    if len(RESERVATIONS_DB) > 0:
        return  # Do not reinitialize if already populated

    possible_hours: List[str] = [f"{h:02d}:00" for h in range(8, 21)]

    for station in stations.values():
        station_id = station.station_id
        RESERVATIONS_DB[station_id] = {}

        # Pre-determine if this specific station is "Saturated" (30% chance)
        is_station_saturated = random.random() < 0.30

        for hour in possible_hours:
            # If saturated, fill to 100%. Otherwise, randomly fill to 60%
            if is_station_saturated or random.random() < 0.60:
                # 70% New Users (potential eviction targets), 30% Regulars
                ghost_status = "New" if random.random() < 0.7 else "Regular"

                # Karma penalty score from 0 to 5
                ghost_penalty = round(random.randint(0, 5), 2)

                # Price paid by the ghost user (between €0.15 and €0.20)
                paid_price = round(random.uniform(0.15, 0.20), 2)

                RESERVATIONS_DB[station_id][hour] = {
                    "status": ghost_status,
                    "mode": "precision",
                    "is_ghost": True,
                    "karma_penalty": ghost_penalty,
                    "paid_price": paid_price
                }


def get_reserved_slots(station_id: str) -> List[str]:
    """
    Retrieves the list of reserved hours for a given station.

    Args:
        station_id (str): The ID of the station.

    Returns:
        List[str]: A list of reserved time slots (e.g., ["08:00", "09:00"]).
    """
    return list(RESERVATIONS_DB.get(station_id, {}).keys())


def process_reservation(station_id: str, mode: str, chosen_hour: Optional[str],
                        sim_config: Optional[Dict[str, Any]] = None, stations: Optional[Dict[Any, Any]] = None) -> \
Tuple[str, str]:
    """
    Handles the reservation algorithm and VIP Eviction (Yield Management).

    Args:
        station_id (str): The target station ID.
        mode (str): Reservation mode ('precision', 'morning', 'afternoon').
        chosen_hour (Optional[str]): Specific hour requested (if mode is 'precision').
        sim_config (Optional[Dict[str, Any]]): Simulation parameters.
        stations (Optional[Dict[Any, Any]]): All available stations for routing purposes.

    Returns:
        Tuple[str, str]: A status code ("success" or "error") and a descriptive message.
    """
    if not station_id:
        return "error", "Missing data."

    user_profile = read_profile()

    if user_profile.get("active_reservations") and len(user_profile["active_reservations"]) > 0:
        return "error", "You already have an active reservation. Please cancel it first."

    past_charges = int(user_profile.get("past_charges", 0))
    user_penalty = float(user_profile.get("karma_penalty", 0.0))
    user_status = "Regular" if past_charges >= 5 else "New"

    if station_id not in RESERVATIONS_DB:
        RESERVATIONS_DB[station_id] = {}

    taken_hours = RESERVATIONS_DB[station_id]
    assigned_hour = None

    # Search for an available slot
    time_range: List[str] = []
    if mode == 'precision' and chosen_hour:
        if chosen_hour not in taken_hours:
            assigned_hour = chosen_hour

    elif mode in ['morning', 'afternoon']:
        time_range = ["08:00", "09:00", "10:00", "11:00", "12:00"] if mode == 'morning' else [
            "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00"
        ]
        free_slots = [h for h in time_range if h not in taken_hours]
        if free_slots:
            assigned_hour = free_slots[0]

    # Standard Reservation Assignment
    if assigned_hour:
        taken_hours[assigned_hour] = {
            "status": user_status,
            "mode": mode,
            "karma_penalty": user_penalty,
            "paid_price": 0.15
        }

        user_profile["past_charges"] = past_charges + 1
        if "active_reservations" not in user_profile:
            user_profile["active_reservations"] = []

        user_profile["active_reservations"].append({"station": station_id, "hour": assigned_hour})
        save_profile(user_profile)

        new_penalty = apply_karma_update(0.0)
        return "success", f"Standard reservation confirmed at {assigned_hour}. (Karma: {new_penalty})"

    # FCFS Baseline check
    if sim_config and sim_config.get("baseline") == "FCFS":
        return "error", "[FCFS Baseline] The station is fully booked."


    # CONDITIONAL EVICTION LOGIC
    max_capacity = 13
    if len(taken_hours) < max_capacity:
        return "error", "Operation denied. There are still free slots available."

    if user_status == "New":
        return "error", "The station is 100% booked today."

    eviction_candidates: List[Dict[str, Any]] = []
    search_range = taken_hours.keys() if mode == 'precision' else time_range

    for h in search_range:
        occupant = taken_hours.get(h)
        if occupant and occupant.get("status") == "New":
            eviction_candidates.append({
                "hour": h,
                "karma_penalty": occupant.get("karma_penalty", 0.0),
                "paid_price": occupant.get("paid_price", 0.20)
            })

    if not eviction_candidates:
        return "error", "Slot saturated by other untouchable Regulars!"

    # Sort by highest penalty, then lowest price paid
    eviction_candidates.sort(key=lambda x: (-x["karma_penalty"], x["paid_price"]))
    worst_candidate = eviction_candidates[0]
    freed_hour = worst_candidate["hour"]
    evicted_penalty = float(worst_candidate["karma_penalty"])
    evicted_price = worst_candidate["paid_price"]

    # Karma System Check
    is_karma_active = sim_config.get("karma_active", True) if sim_config else True
    if is_karma_active and (user_penalty >= evicted_penalty):
        return "error", f"PRIVILEGE DENIED: The worst user has a penalty of {evicted_penalty}, while yours is {user_penalty}."

    # EVICTION PREMIUM PRICING FORMULA
    p_start = evicted_price
    gamma = 0.5
    premium_price = round(p_start * (1 + gamma), 2)

    # Lock the slot for the VIP
    taken_hours[freed_hour] = {
        "status": "Regular",
        "mode": "vip_forced",
        "karma_penalty": user_penalty,
        "paid_price": premium_price
    }

    user_profile["past_charges"] = past_charges + 1
    if "active_reservations" not in user_profile:
        user_profile["active_reservations"] = []

    user_profile["active_reservations"].append({"station": station_id, "hour": freed_hour})
    save_profile(user_profile)


    # REROUTING THE EVICTED USER
    backup_station_found = None

    if stations:
        for s_id in stations.keys():
            if s_id != station_id:
                if s_id not in RESERVATIONS_DB:
                    RESERVATIONS_DB[s_id] = {}

                if freed_hour not in RESERVATIONS_DB[s_id]:
                    # User keeps their original price, no compensation applied
                    RESERVATIONS_DB[s_id][freed_hour] = {
                        "status": "New",
                        "mode": "re_routed",
                        "karma_penalty": evicted_penalty,
                        "paid_price": evicted_price
                    }
                    backup_station_found = s_id
                    break

    # Build detailed proof message
    eviction_msg = (
        f"PRIVILEGE ACTIVATED! Slot forced at {freed_hour}.\n"
        f"Premium rate applied: €{premium_price}/kWh.\n\n"
        f"🛡️ Safety net deployed for the evicted user:\n"
        f"- Their Karma remains intact (Immunity)."
    )

    if backup_station_found:
        eviction_msg += f"\n- ✅ They have been automatically rerouted to station {backup_station_found} retaining their €{evicted_price} rate."
    else:
        eviction_msg += f"\n- ⚠️ No backup stations available at this hour."

    return "success", eviction_msg


def process_cancellation(station_id: str, reservation_hour_str: str) -> Tuple[str, str]:
    """
    Handles a reservation cancellation and applies the appropriate karma penalty.

    Args:
        station_id (str): Target station ID.
        reservation_hour_str (str): The reserved hour to cancel (e.g., "14:00").

    Returns:
        Tuple[str, str]: Status code and feedback message.
    """
    user_profile = read_profile()

    try:
        reservation_hour = int(reservation_hour_str.split(':')[0])
    except ValueError:
        return "error", "Invalid hour format."

    current_hour = datetime.now().hour

    # Intelligent Next-Day vs Delay calculation
    raw_diff = reservation_hour - current_hour
    hour_diff = raw_diff + 24 if raw_diff < -6 else raw_diff

    penalty_points = 0
    reason_msg = ""

    if hour_diff >= 24:
        penalty_points = 0
        reason_msg = "Cancelled more than 24h in advance (Free)."
    elif 2 <= hour_diff < 24:
        penalty_points = 1
        reason_msg = "Standard cancellation (Between 2h and 24h in advance)."
    elif 0 <= hour_diff < 2:
        penalty_points = 2
        reason_msg = "Last-minute cancellation (Less than 2h before slot)."
    else:
        penalty_points = 3
        reason_msg = "No-show (Slot time already passed)."

    # 1. Free the station slot
    if station_id in RESERVATIONS_DB and reservation_hour_str in RESERVATIONS_DB[station_id]:
        del RESERVATIONS_DB[station_id][reservation_hour_str]

    # 2. Clean up user profile
    reservations = user_profile.get("active_reservations", [])
    user_profile["active_reservations"] = [
        r for r in reservations
        if not (r["station"] == station_id and r["hour"] == reservation_hour_str)
    ]

    if int(user_profile.get("past_charges", 0)) > 0:
        user_profile["past_charges"] = int(user_profile.get("past_charges", 0)) - 1

    save_profile(user_profile)

    # CALL THE EMA FORMULA (Updates global score)
    new_global_penalty = apply_karma_update(penalty_points)

    final_msg = f"Reservation cancelled.\nReason: {reason_msg}"
    final_msg += f"\n\nAction penalty: +{penalty_points}. Your new (smoothed) global score is {new_global_penalty}."

    return "success", final_msg