"""Shared validation and deterministic feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FEATURE_COLUMNS = [
    "relative_humidity_pct",
    "room_temperature_c",
    "wall_surface_temperature_c",
    "door_openings_per_hour",
    "ventilation_duration_min_per_hour",
    "dehumidifier_operating_hours_per_day",
    "room_occupancy",
]


def calculate_dew_point(temp_c: float, rh_pct: float) -> float:
    """Magnus dew-point approximation (degrees C)."""
    rh = float(np.clip(rh_pct, 1.0, 100.0)) / 100.0
    a, b = 17.625, 243.04
    gamma = np.log(rh) + (a * float(temp_c)) / (b + float(temp_c))
    return float((b * gamma) / (a - gamma))


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return rows with dew point, condensation index and ventilation ratio."""
    result = data.copy()
    required = RAW_FEATURE_COLUMNS
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(f"Missing required raw features: {missing}")

    dew_points = [
        calculate_dew_point(t, rh)
        for t, rh in zip(result["room_temperature_c"], result["relative_humidity_pct"])
    ]
    result["dew_point_c"] = np.round(dew_points, 3)
    result["dew_point_margin_c"] = (
        result["wall_surface_temperature_c"] - result["dew_point_c"]
    ).round(3)

    # Index rises as the dew point approaches / exceeds the wall temperature.
    # It combines a wall-temperature margin component with high-RH exposure.
    margin_pressure = np.clip((3.0 - result["dew_point_margin_c"]) / 6.0, 0.0, 1.0)
    humidity_pressure = np.clip((result["relative_humidity_pct"] - 55.0) / 35.0, 0.0, 1.0)
    result["condensation_risk_index"] = (
        0.70 * margin_pressure + 0.30 * humidity_pressure
    ).clip(0.0, 1.0).round(4)

    # Available ventilation minutes are discounted for door traffic and occupancy.
    denominator = (
        60.0
        + 5.0 * result["door_openings_per_hour"]
        + 2.0 * result["room_occupancy"]
    )
    result["ventilation_effectiveness_ratio"] = (
        result["ventilation_duration_min_per_hour"] / denominator
    ).clip(0.0, 1.0).round(4)
    return result
