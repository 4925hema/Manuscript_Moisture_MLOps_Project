import pytest
from pydantic import ValidationError
from src.api import RoomObservation


def valid_payload():
    return {
        "relative_humidity_pct": 72,
        "room_temperature_c": 20,
        "wall_surface_temperature_c": 17,
        "door_openings_per_hour": 4,
        "ventilation_duration_min_per_hour": 25,
        "dehumidifier_operating_hours_per_day": 12,
        "room_occupancy": 2,
    }


def test_valid_payload_is_accepted():
    assert RoomObservation(**valid_payload()).relative_humidity_pct == 72


@pytest.mark.parametrize("field,value", [
    ("relative_humidity_pct", -1),
    ("relative_humidity_pct", 101),
    ("room_temperature_c", 4.9),
    ("room_temperature_c", 40.1),
])
def test_invalid_humidity_and_temperature_are_rejected(field, value):
    payload = valid_payload()
    payload[field] = value
    with pytest.raises(ValidationError):
        RoomObservation(**payload)
