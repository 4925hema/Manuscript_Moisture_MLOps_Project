"""Shared project paths and model configuration."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"

V1_PATH = DATA_RAW_DIR / "manuscript_rooms_v1.csv"
V2_PATH = DATA_RAW_DIR / "manuscript_rooms_v2_monsoon.csv"
FEATURE_STORE_PATH = DATA_PROCESSED_DIR / "feature_store.sqlite"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"

RANDOM_STATE = 42
TARGET_COLUMN = "moisture_risk_status"
TARGET_MAP = {"Storage Conditions Acceptable": 0, "Moisture Risk": 1}
FEATURE_COLUMNS = [
    "relative_humidity_pct",
    "room_temperature_c",
    "wall_surface_temperature_c",
    "door_openings_per_hour",
    "ventilation_duration_min_per_hour",
    "dehumidifier_operating_hours_per_day",
    "room_occupancy",
    "dew_point_c",
    "dew_point_margin_c",
    "condensation_risk_index",
    "ventilation_effectiveness_ratio",
]
