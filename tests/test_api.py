from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint_exists():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_predict_endpoint_returns_expected_label_or_clear_not_ready():
    payload = {
        "relative_humidity_pct": 91,
        "room_temperature_c": 18,
        "wall_surface_temperature_c": 14,
        "door_openings_per_hour": 10,
        "ventilation_duration_min_per_hour": 8,
        "dehumidifier_operating_hours_per_day": 2,
        "room_occupancy": 4,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert response.json()["prediction"] in ("Moisture Risk", "Storage Conditions Acceptable")
