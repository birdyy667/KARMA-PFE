"""Unit tests for routing helpers."""

import pytest
from src.core.routing import format_time, parse_hour

def test_format_time() -> None:
    """Verifies that seconds are correctly converted to readable strings."""
    assert format_time(1800) == "30 min"
    assert format_time(3600) == "1h 00min"
    assert format_time(3720) == "1h 02min"

def test_parse_hour() -> None:
    """Verifies that HH:MM strings correctly resolve to decimal floats."""
    assert parse_hour("08:30") == 8.5
    assert parse_hour("14:00") == 14.0
    assert parse_hour("invalid_string") is None