"""Generate reproducible synthetic room observations for two dataset versions."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_RAW_DIR, V1_PATH, V2_PATH, RANDOM_STATE


def dew_point_c(temp_c: float, rh_pct: float) -> float:
    """Magnus approximation of dew point in degrees Celsius."""
    rh = float(np.clip(rh_pct, 1.0, 100.0)) / 100.0
    a, b = 17.625, 243.04
    gamma = np.log(rh) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


def _risk_label(row: dict, rng: np.random.Generator) -> str:
    """Create a synthetic 48-hour risk label from physically motivated rules."""
    dp = dew_point_c(row["room_temperature_c"], row["relative_humidity_pct"])
    margin = row["wall_surface_temperature_c"] - dp
    vent_ratio = row["ventilation_duration_min_per_hour"] / (
        60.0 + 5.0 * row["door_openings_per_hour"] + 2.0 * row["room_occupancy"]
    )
    score = 0.0
    score += 2.0 if margin <= 0.5 else 1.2 if margin <= 2.0 else 0.0
    score += 1.2 if row["relative_humidity_pct"] >= 78 else 0.7 if row["relative_humidity_pct"] >= 70 else 0.0
    score += 0.8 if row["room_temperature_c"] < 19 else 0.0
    score += 0.8 if vent_ratio < 0.20 else 0.4 if vent_ratio < 0.30 else 0.0
    score += 0.5 if row["door_openings_per_hour"] >= 8 else 0.0
    score += 0.6 if row["dehumidifier_operating_hours_per_day"] < 5 else 0.0
    score += 0.25 if row["room_occupancy"] >= 4 else 0.0
    # Small noise avoids a perfectly deterministic threshold boundary.
    score += rng.normal(0.0, 0.16)
    return "Moisture Risk" if score >= 1.9 else "Storage Conditions Acceptable"


def generate_version(n_rows: int, version: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    room_ids = [f"MSR-{i:02d}" for i in range(1, 7)]

    for i in range(n_rows):
        room_id = room_ids[i % len(room_ids)]
        if version == "v1":
            # Normal / controlled storage season.
            humidity = np.clip(rng.normal(64, 10), 35, 82)
            room_temp = np.clip(rng.normal(21.5, 2.2), 15, 28)
            vent = np.clip(rng.normal(34, 9), 8, 58)
            dehum = np.clip(rng.normal(9.5, 4.0), 0, 20)
        else:
            # Monsoon domain shift: higher RH, slightly cooler rooms, less effective ventilation.
            humidity = np.clip(rng.normal(76, 9), 48, 98)
            room_temp = np.clip(rng.normal(20.2, 2.1), 14, 27)
            vent = np.clip(rng.normal(20, 8), 2, 46)
            dehum = np.clip(rng.normal(12.0, 4.5), 1, 24)

        door_openings = int(np.clip(rng.poisson(3.2 if version == "v1" else 4.8), 0, 18))
        occupancy = int(np.clip(rng.poisson(1.1 if version == "v1" else 1.4), 0, 8))
        # A cold exterior wall, with extra cooling during monsoon observations.
        wall_temp = room_temp - rng.uniform(1.0, 4.5) - (rng.uniform(0.3, 1.4) if version == "v2" else 0.0)

        row = {
            "observation_id": f"{version.upper()}-{i + 1:04d}",
            "dataset_version": "Version 1 - Baseline" if version == "v1" else "Version 2 - Monsoon",
            "room_id": room_id,
            "observed_at": (base_time + timedelta(hours=4 * i)).isoformat(),
            "relative_humidity_pct": round(float(humidity), 2),
            "room_temperature_c": round(float(room_temp), 2),
            "wall_surface_temperature_c": round(float(wall_temp), 2),
            "door_openings_per_hour": door_openings,
            "ventilation_duration_min_per_hour": round(float(vent), 2),
            "dehumidifier_operating_hours_per_day": round(float(dehum), 2),
            "room_occupancy": occupancy,
        }
        row["moisture_risk_status"] = _risk_label(row, rng)
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    v1 = generate_version(300, "v1", RANDOM_STATE)
    v2 = generate_version(300, "v2", RANDOM_STATE + 100)
    v1.to_csv(V1_PATH, index=False)
    v2.to_csv(V2_PATH, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_STATE,
        "versions": {
            "Version 1 - Baseline": {
                "file": str(V1_PATH.relative_to(V1_PATH.parents[2])),
                "rows": len(v1),
                "risk_rate": round(float((v1.moisture_risk_status == "Moisture Risk").mean()), 4),
                "mean_relative_humidity_pct": round(float(v1.relative_humidity_pct.mean()), 2),
                "mean_ventilation_min_per_hour": round(float(v1.ventilation_duration_min_per_hour.mean()), 2),
            },
            "Version 2 - Monsoon": {
                "file": str(V2_PATH.relative_to(V2_PATH.parents[2])),
                "rows": len(v2),
                "risk_rate": round(float((v2.moisture_risk_status == "Moisture Risk").mean()), 4),
                "mean_relative_humidity_pct": round(float(v2.relative_humidity_pct.mean()), 2),
                "mean_ventilation_min_per_hour": round(float(v2.ventilation_duration_min_per_hour.mean()), 2),
            },
        },
    }
    (DATA_RAW_DIR / "dataset_versions.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
