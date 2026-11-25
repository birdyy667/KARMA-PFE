"""
Core physics engine for EV mobility.
Handles energy consumption based on physical vehicle characteristics.
"""

import math
from typing import Dict

# Physics Constants
RHO: float = 1.225                 # Air density (kg/m³) at 15°C
G: float = 9.81                    # Gravitational acceleration (m/s²)
C_RR: float = 0.01                 # Rolling resistance coefficient (standard asphalt tires)
MOTOR_EFFICIENCY: float = 0.90     # Battery-to-wheel efficiency (90%)
REGEN_EFFICIENCY: float = 0.65     # Regenerative braking efficiency downhill (65%)
JOULES_PER_KWH: float = 3.6e6      # Joules to kWh conversion factor


# Vehicle database (Mass in kg, SCx in m², Battery in kWh)
# SCx = Drag Coefficient (aerodynamics) * Frontal Area
VEHICLE_PHYSICS_DB: Dict[str, Dict[str, float]] = {
    "Dacia Spring": {"mass": 970.0, "scx": 0.74, "battery": 26.8}, # Exact reference figure
    "Peugeot e-208": {"mass": 1455.0, "scx": 0.58, "battery": 46.3},
    "Renault 5 E-Tech 52kWh": {"mass": 1450.0, "scx": 0.61, "battery": 52.0},
    "Renault Zoe R110": {"mass": 1502.0, "scx": 0.69, "battery": 52.0},
    "Fiat 500e": {"mass": 1365.0, "scx": 0.62, "battery": 37.3},
    "Opel Corsa Electric": {"mass": 1455.0, "scx": 0.58, "battery": 46.3},
    "Mini Cooper SE": {"mass": 1365.0, "scx": 0.63, "battery": 28.9},
    "Renault Megane E-Tech EV60": {"mass": 1620.0, "scx": 0.64, "battery": 60.0},
    "MG4 Standard": {"mass": 1655.0, "scx": 0.62, "battery": 51.0},
    "Volkswagen ID.3 Pro": {"mass": 1812.0, "scx": 0.62, "battery": 58.0},
    "Cupra Born 58kWh": {"mass": 1811.0, "scx": 0.62, "battery": 58.0},
    "Citroen e-C4": {"mass": 1561.0, "scx": 0.62, "battery": 46.3},
    "Nissan Leaf 40kWh": {"mass": 1580.0, "scx": 0.62, "battery": 40.0},
    "BMW i4 eDrive40": {"mass": 2125.0, "scx": 0.53, "battery": 80.7},
    "Tesla Model 3 Propulsion": {"mass": 1760.0, "scx": 0.51, "battery": 57.5},
    "Tesla Model 3 Grande Autonomie": {"mass": 1840.0, "scx": 0.51, "battery": 75.0},
    "Tesla Model Y Propulsion": {"mass": 1909.0, "scx": 0.58, "battery": 57.5},
    "Tesla Model Y Grande Autonomie": {"mass": 1979.0, "scx": 0.58, "battery": 75.0},
    "Peugeot e-2008": {"mass": 1548.0, "scx": 0.77, "battery": 46.3},
    "Renault Scenic E-Tech 87kWh": {"mass": 1842.0, "scx": 0.73, "battery": 87.0},
    "Volvo EX30 Single Motor": {"mass": 1830.0, "scx": 0.67, "battery": 49.0},
    "Hyundai Kona Electric 64kWh": {"mass": 1685.0, "scx": 0.64, "battery": 64.0},
    "Kia Niro EV": {"mass": 1739.0, "scx": 0.67, "battery": 64.8},
    "Volkswagen ID.4 Pro": {"mass": 2124.0, "scx": 0.70, "battery": 77.0},
    "Skoda Enyaq iV 80": {"mass": 2149.0, "scx": 0.65, "battery": 77.0},
    "Audi Q4 e-tron 40": {"mass": 2050.0, "scx": 0.70, "battery": 77.0},
    "Hyundai Ioniq 5 RWD": {"mass": 1910.0, "scx": 0.73, "battery": 72.6},
    "Kia EV6 RWD": {"mass": 1910.0, "scx": 0.67, "battery": 77.4},
    "MG ZS EV Long Range": {"mass": 1620.0, "scx": 0.80, "battery": 68.3},
}

# Default vehicle specifications
DEFAULT_VEHICLE: Dict[str, float] = {"mass": 1600.0, "scx": 0.65, "battery": 50.0}


def get_vehicle_characteristics(vehicle_name: str) -> Dict[str, float]:
    """
    Retrieves the physical characteristics of a specific vehicle.

    Args:
        vehicle_name (str): The name of the vehicle to look up.

    Returns:
        Dict[str, float]: A dictionary containing the vehicle's mass, SCx, and battery capacity.
                          Returns the default vehicle configuration if the name is not found.
    """
    return VEHICLE_PHYSICS_DB.get(vehicle_name, DEFAULT_VEHICLE)


def calculate_physical_consumption(distance_m: float, speed_kmh: float, slope_pct: float, vehicle_characteristics: Dict[str, float]) -> float:
    """
    Calculates the energy consumption in kWh based on physical models.

    Args:
        distance_m (float): The travel distance in meters.
        speed_kmh (float): The travel speed in km/h.
        slope_pct (float): The road gradient percentage.
        vehicle_characteristics (Dict[str, float]): The physical characteristics of the vehicle (mass, scx).

    Returns:
        float: The battery energy consumed during the segment in kWh.
    """
    if distance_m <= 0:
        return 0.0

    speed_ms = speed_kmh / 3.6
    angle_rad = math.atan(slope_pct / 100.0)

    mass = vehicle_characteristics["mass"]
    scx = vehicle_characteristics["scx"]

    # Force calculations (in Newtons)
    # Multiplication by frontal area is omitted as it is already included in the SCx value
    f_aero = 0.5 * RHO * scx * (speed_ms ** 2)          # Aerodynamic drag
    f_rolling = mass * G * C_RR * math.cos(angle_rad)   # Rolling resistance
    f_slope = mass * G * math.sin(angle_rad)            # Gravity force component on slope

    f_total = f_aero + f_rolling + f_slope
    mechanical_energy_joules = f_total * distance_m

    if mechanical_energy_joules >= 0:
        battery_energy_joules = mechanical_energy_joules / MOTOR_EFFICIENCY
    else:
        battery_energy_joules = mechanical_energy_joules * REGEN_EFFICIENCY

    return battery_energy_joules / JOULES_PER_KWH