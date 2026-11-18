"""FastAPI inference service for the registered/local moisture-risk model."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .config import BEST_MODEL_PATH, MODEL_METADATA_PATH, FEATURE_COLUMNS
from .features import engineer_features


class RoomObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_humidity_pct: float = Field(ge=0, le=100, description="Relative humidity (%)")
    room_temperature_c: float = Field(ge=5, le=40, description="Room temperature (C)")
    wall_surface_temperature_c: float = Field(ge=0, le=45, description="Wall surface temperature (C)")
    door_openings_per_hour: int = Field(ge=0, le=60)
    ventilation_duration_min_per_hour: float = Field(ge=0, le=60)
    dehumidifier_operating_hours_per_day: float = Field(ge=0, le=24)
    room_occupancy: int = Field(ge=0, le=30)


app = FastAPI(
    title="Manuscript Storage Moisture-Risk API",
    version="1.0.0",
    description="Predicts whether current room conditions indicate moisture-related damage risk within 48 hours.",
)


def load_model_and_metadata():
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8")) if MODEL_METADATA_PATH.exists() else {"threshold": 0.35}
    model_uri = os.getenv("MLFLOW_MODEL_URI")
    if model_uri:
        try:
            import mlflow.sklearn
            model = mlflow.sklearn.load_model(model_uri)
            return model, metadata
        except Exception as exc:
            raise RuntimeError(f"Unable to load MLFLOW_MODEL_URI={model_uri}: {exc}") from exc
    if not BEST_MODEL_PATH.exists():
        raise RuntimeError("Model artifact not found. Run `python -m src.train` first.")
    return joblib.load(BEST_MODEL_PATH), metadata


try:
    MODEL, MODEL_METADATA = load_model_and_metadata()
    MODEL_LOAD_ERROR = None
except Exception as load_exc:  # app can still expose /health for diagnostics
    MODEL, MODEL_METADATA = None, {"threshold": 0.35}
    MODEL_LOAD_ERROR = str(load_exc)


@app.get("/health")
def health():
    ready = MODEL is not None
    return {
        "status": "healthy" if ready else "model_not_loaded",
        "model_loaded": ready,
        "model_name": MODEL_METADATA.get("model_name", "ManuscriptMoistureRiskModel"),
        "model_version": MODEL_METADATA.get("model_version", "unknown"),
        "detail": MODEL_LOAD_ERROR,
    }


@app.post("/predict")
def predict(observation: RoomObservation):
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Model is not ready: {MODEL_LOAD_ERROR}")
    raw = pd.DataFrame([observation.model_dump()])
    features = engineer_features(raw)[FEATURE_COLUMNS]
    threshold = float(MODEL_METADATA.get("threshold", 0.35))
    try:
        probability = float(MODEL.predict_proba(features)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    risk = probability >= threshold
    return {
        "prediction": "Moisture Risk" if risk else "Storage Conditions Acceptable",
        "moisture_risk_probability": round(probability, 4),
        "decision_threshold": threshold,
        "forecast_horizon": "Next 48 hours",
        "engineered_features": {
            "condensation_risk_index": float(features.iloc[0]["condensation_risk_index"]),
            "ventilation_effectiveness_ratio": float(features.iloc[0]["ventilation_effectiveness_ratio"]),
            "dew_point_c": float(features.iloc[0]["dew_point_c"]),
        },
    }
