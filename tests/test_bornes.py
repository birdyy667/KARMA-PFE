"""Unit tests for the charging station module."""

import pytest
from src.core.bornes import BuildingStation


@pytest.fixture
def station_test() -> BuildingStation:
    """Provides a fresh station instance for each test."""
    return BuildingStation(station_id="Test_Station", max_battery_capacity=100.0, base_edf_price=0.25)


def test_charge_vehicle(station_test: BuildingStation) -> None:
    """Verifies battery drain and ensures it never drops below 0."""
    station_test.current_battery_level = 50.0

    # Normal drain
    station_test.charge_vehicle(20.0)
    assert station_test.current_battery_level == 30.0

    # Over-drain safeguard check
    station_test.charge_vehicle(50.0)
    assert station_test.current_battery_level == 0.0


def test_calculate_dynamic_price_greedy_baseline(station_test: BuildingStation) -> None:
    """Verifies that the Greedy baseline properly overrides the pricing logic."""
    station_test.current_battery_level = 100.0  # Usually this drops the price
    config = {"baseline": "Greedy"}

    price, pricing_case, pct = station_test.calculate_dynamic_price(sim_config=config)

    # Greedy forces +60% regardless of battery level
    expected_price = 0.25 * 1.60
    assert price == expected_price
    assert "Greedy Baseline" in pricing_case