"""
backend/tests/test_predict.py
Unit tests for the prediction endpoint.

Run with:
    pytest backend/tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "vehicle_type":    "car",
    "temperature_c":   26.0,
    "humidity_pct":    78.0,
    "rainfall_mm":     5.0,
    "wind_speed_kmh":  15.0,
    "visibility_km":   7.0,
    "gradient_pct":    8.0,
    "curvature":       0.4,
    "road_surface":    "asphalt",
    "lane_count":      2,
    "hour_of_day":     14,
    "day_of_week":     2,
    "is_holiday":      False,
    "traffic_density": 0.5,
}


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_risk_level():
    r = client.post("/api/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] in ("Low", "Medium", "High")
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["vehicle_type"] == "car"
    assert body["recommended_speed_kmh"] is not None


def test_predict_all_vehicle_types():
    for vtype in ("car", "motorcycle", "bus", "lorry", "three-wheeler"):
        payload = {**VALID_PAYLOAD, "vehicle_type": vtype}
        r = client.post("/api/predict", json=payload)
        assert r.status_code == 200, f"Failed for vehicle_type={vtype}"


def test_predict_high_risk_conditions():
    payload = {
        **VALID_PAYLOAD,
        "rainfall_mm":   80.0,
        "visibility_km": 0.5,
        "gradient_pct":  25.0,
        "curvature":     0.9,
        "hour_of_day":   2,
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200


def test_predict_invalid_vehicle_type():
    payload = {**VALID_PAYLOAD, "vehicle_type": "spaceship"}
    r = client.post("/api/predict", json=payload)
    # Should still return 200 — unknown vehicle defaults to car encoding
    assert r.status_code == 200


def test_predict_out_of_range_value():
    payload = {**VALID_PAYLOAD, "temperature_c": 999}
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 422  # Pydantic validation error
