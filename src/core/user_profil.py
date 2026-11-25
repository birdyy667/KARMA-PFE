"""
User profile management module.
Handles reading, updating, and storing user preferences and karma scores.
"""

import json
import logging
from typing import Dict, Any

from config import PROCESSED_DATA_DIR

# Configure module-level logger
logger = logging.getLogger(__name__)

# Dynamic absolute path using the architecture standards
PROFILE_FILE_PATH = PROCESSED_DATA_DIR / "user_profile.json"

DEFAULT_PROFILE: Dict[str, Any] = {
    "name": "EV Driver",
    "vehicle": "Peugeot e-208",
    "battery_capacity": 50.0,
    "consumption": 15.0,
    "past_charges": 0,
    "karma_penalty": 0.0,  # Stored as float for the EMA formula
    "active_reservations": []
}


def read_profile() -> Dict[str, Any]:
    """
    Reads the user profile from the JSON file.
    Creates a default profile if the file does not exist.

    Returns:
        Dict[str, Any]: The user profile data.
    """
    PROFILE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not PROFILE_FILE_PATH.exists():
        save_profile(DEFAULT_PROFILE)
        return DEFAULT_PROFILE.copy()

    try:
        with open(PROFILE_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to read profile: %s. Returning default profile.", e)
        return DEFAULT_PROFILE.copy()


def save_profile(data: Dict[str, Any]) -> None:
    """
    Saves the user profile data to the JSON file.

    Args:
        data (Dict[str, Any]): The profile data to save.
    """
    PROFILE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(PROFILE_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except IOError as e:
        logger.error("Failed to save profile: %s", e)


def update_profile(new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates specific fields in the user profile and saves it.

    Args:
        new_data (Dict[str, Any]): The new data to merge into the profile.

    Returns:
        Dict[str, Any]: The fully updated profile.
    """
    current_profile = read_profile()
    current_profile.update(new_data)
    save_profile(current_profile)

    return current_profile


def apply_karma_update(penalty_points: float) -> float:
    """
    Updates the user's Karma penalty using an Exponential Moving Average (EMA).
    Formula: M_u(t+1) = lambda * M_u(t) + (1 - lambda) * P

    Args:
        penalty_points (float): The penalty points incurred from the latest action.

    Returns:
        float: The newly calculated global karma penalty.
    """
    profile = read_profile()

    try:
        current_penalty = float(profile.get("karma_penalty", 0.0))
    except (ValueError, TypeError):
        current_penalty = 0.0

    lambda_ema = 0.8  # 80% weight on the past, 20% on the new action

    new_penalty = (lambda_ema * current_penalty) + ((1.0 - lambda_ema) * penalty_points)
    profile["karma_penalty"] = round(new_penalty, 3)

    save_profile(profile)

    return profile["karma_penalty"]