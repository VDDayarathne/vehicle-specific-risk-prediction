"""
backend/tests/test_integration.py
End-to-end smoke tests that cover telemetry ingestion and prediction.
"""
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_telemetry_ingest_and_predict_smoke():
    telemetry_payload = {
        "device_id": "integration-device-001",
        "vehicle_type": "bus",
        "latitude": 7.2500,
        "longitude": 80.5300,
        "speed_kmh": 34.5,
        "heading_deg": 142.0,
        "accel_m_s2": -1.8,
        "rpm": 2100,
    }

    telemetry_res = client.post("/api/telemetry", json=telemetry_payload)
    assert telemetry_res.status_code == 200
    assert telemetry_res.json()["stored"] is True

    predict_payload = {
        "device_id": telemetry_payload["device_id"],
        "vehicle_type": "bus",
        "temperature_c": 26.0,
        "humidity_pct": 76.0,
        "rainfall_mm": 8.0,
        "wind_speed_kmh": 15.0,
        "visibility_km": 4.0,
        "gradient_pct": 9.0,
        "curvature": 0.5,
        "road_surface": "asphalt",
        "lane_count": 2,
        "hour_of_day": 18,
        "day_of_week": 2,
        "is_holiday": False,
        "traffic_density": 0.6,
    }

    predict_res = client.post("/api/predict", json=predict_payload)
    assert predict_res.status_code == 200

    body = predict_res.json()
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert body["vehicle_type"] == "bus"
    assert body["recommended_speed_kmh"] is not None
