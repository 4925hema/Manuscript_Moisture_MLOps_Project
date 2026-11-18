import pandas as pd
from src.features import calculate_dew_point, engineer_features


def test_dew_point_is_below_room_temperature():
    assert calculate_dew_point(22.0, 60.0) < 22.0


def test_engineered_features_are_in_valid_ranges():
    sample = pd.DataFrame([{
        "relative_humidity_pct": 80.0,
        "room_temperature_c": 20.0,
        "wall_surface_temperature_c": 17.0,
        "door_openings_per_hour": 5,
        "ventilation_duration_min_per_hour": 20.0,
        "dehumidifier_operating_hours_per_day": 10.0,
        "room_occupancy": 2,
    }])
    result = engineer_features(sample)
    assert 0.0 <= result.loc[0, "condensation_risk_index"] <= 1.0
    assert 0.0 <= result.loc[0, "ventilation_effectiveness_ratio"] <= 1.0
    assert result.loc[0, "dew_point_c"] < 20.0
