"""SQLite feature store: materialize offline features and retrieve them for training."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import FEATURE_STORE_PATH
from .features import engineer_features


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS room_features (
    observation_id TEXT PRIMARY KEY,
    dataset_version TEXT NOT NULL,
    room_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    relative_humidity_pct REAL NOT NULL,
    room_temperature_c REAL NOT NULL,
    wall_surface_temperature_c REAL NOT NULL,
    door_openings_per_hour REAL NOT NULL,
    ventilation_duration_min_per_hour REAL NOT NULL,
    dehumidifier_operating_hours_per_day REAL NOT NULL,
    room_occupancy REAL NOT NULL,
    dew_point_c REAL NOT NULL,
    dew_point_margin_c REAL NOT NULL,
    condensation_risk_index REAL NOT NULL,
    ventilation_effectiveness_ratio REAL NOT NULL,
    moisture_risk_status TEXT NOT NULL
)
"""


def materialize_features(raw_csv_paths: list[Path], db_path: Path = FEATURE_STORE_PATH) -> int:
    """Compute and upsert features for all supplied dataset versions."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(path) for path in raw_csv_paths]
    raw = pd.concat(frames, ignore_index=True)
    enriched = engineer_features(raw)
    columns = [
        "observation_id", "dataset_version", "room_id", "observed_at",
        "relative_humidity_pct", "room_temperature_c", "wall_surface_temperature_c",
        "door_openings_per_hour", "ventilation_duration_min_per_hour",
        "dehumidifier_operating_hours_per_day", "room_occupancy", "dew_point_c",
        "dew_point_margin_c", "condensation_risk_index",
        "ventilation_effectiveness_ratio", "moisture_risk_status",
    ]
    with sqlite3.connect(db_path) as connection:
        connection.execute(CREATE_SQL)
        placeholders = ",".join(["?"] * len(columns))
        update_cols = columns[1:]
        update_sql = ",".join([f"{col}=excluded.{col}" for col in update_cols])
        sql = (
            f"INSERT INTO room_features ({','.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(observation_id) DO UPDATE SET {update_sql}"
        )
        connection.executemany(sql, enriched[columns].itertuples(index=False, name=None))
        connection.commit()
    return len(enriched)


def get_features(db_path: Path = FEATURE_STORE_PATH) -> pd.DataFrame:
    """Read the persisted offline feature table."""
    if not db_path.exists():
        raise FileNotFoundError(f"Feature store not found: {db_path}. Run feature_pipeline first.")
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query("SELECT * FROM room_features ORDER BY observed_at", connection)


def get_feature_by_id(observation_id: str, db_path: Path = FEATURE_STORE_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(
            "SELECT * FROM room_features WHERE observation_id = ?", connection,
            params=(observation_id,),
        )
