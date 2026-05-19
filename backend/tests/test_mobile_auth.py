"""
backend/tests/test_mobile_auth.py
Smoke tests for auth and mobile endpoints added in step 2 and step 3.
"""
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from backend.config import database as database_module
from backend.config.database import Base
from backend.main import app
from backend.models import db_models  # noqa: F401 - ensure models are registered


client = TestClient(app)


@pytest.fixture(autouse=True)
def sqlite_test_db(monkeypatch, tmp_path):
    db_path = tmp_path / "kaduguard_test.sqlite3"
    test_engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr(database_module, "engine", test_engine)
    monkeypatch.setattr(database_module, "SessionLocal", test_session_local)
    monkeypatch.setattr("backend.main.engine", test_engine)

    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


def _register_user() -> tuple[str, str]:
    email = f"driver-{uuid4().hex[:8]}@example.com"
    password = "TestPass123!"
    response = client.post(
        "/api/auth/register",
        json={
            "phone": "+94111222333",
            "email": email,
            "password": password,
            "vehicle_type": "car",
        },
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return email, token


def test_auth_register_login_and_me():
    email, token = _register_user()

    login_response = client.post(
        "/api/auth/login",
        json={"email": email, "password": "TestPass123!"},
    )
    assert login_response.status_code == 200
    body = login_response.json()
    assert body["access_token"]
    assert body["refresh_token"]

    me_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_response.status_code == 200
    assert me_response.json()["email"] == email


def test_device_registration_and_trip_flow():
    _, token = _register_user()
    device_id = f"device-{uuid4().hex[:10]}"

    device_response = client.post(
        "/api/auth/device-register",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "device_id": device_id,
            "fcm_token": "fcm-token-test-123",
            "device_name": "Test Device",
        },
    )
    assert device_response.status_code == 200
    assert device_response.json()["device_id"] == device_id

    start_response = client.post(
        "/api/mobile/trip/start",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "device_id": device_id,
            "vehicle_type": "car",
            "latitude": 7.25,
            "longitude": 80.53,
        },
    )
    assert start_response.status_code == 200
    trip_id = start_response.json()["trip_id"]

    end_response = client.post(
        "/api/mobile/trip/end",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "trip_id": trip_id,
            "latitude": 7.251,
            "longitude": 80.531,
            "distance_km": 2.5,
            "duration_minutes": 12,
        },
    )
    assert end_response.status_code == 200
    assert end_response.json()["trip_id"] == trip_id

    summary_response = client.get(
        "/api/mobile/trip-summary?days=7",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert summary_response.status_code == 200
    summary_body = summary_response.json()
    assert summary_body["total_trips"] >= 1
    assert isinstance(summary_body["trips"], list)


def test_mobile_prediction_and_risk_zones():
    _, token = _register_user()

    prediction_response = client.post(
        "/api/mobile/predict",
        json={
            "vehicle_type": "car",
            "temperature_c": 26.0,
            "humidity_pct": 78.0,
            "rainfall_mm": 5.0,
            "wind_speed_kmh": 15.0,
            "visibility_km": 7.0,
            "gradient_pct": 8.0,
            "curvature": 0.4,
            "road_surface": "asphalt",
            "lane_count": 2,
            "hour_of_day": 14,
            "day_of_week": 2,
            "is_holiday": False,
            "traffic_density": 0.5,
        },
    )
    assert prediction_response.status_code == 200
    prediction_body = prediction_response.json()
    assert prediction_body["risk_level"] in {"Low", "Medium", "High"}
    assert prediction_body["recommended_speed_kmh"] is not None

    zones_response = client.get(
        "/api/mobile/risk-zones",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert zones_response.status_code == 200
    zones_body = zones_response.json()
    assert zones_body["count"] >= 0
    assert isinstance(zones_body["zones"], list)


def test_high_risk_prediction_triggers_notification(monkeypatch):
    _, token = _register_user()
    device_id = f"device-{uuid4().hex[:10]}"

    device_register = client.post(
        "/api/auth/device-register",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "device_id": device_id,
            "fcm_token": "fcm-token-high-risk",
            "device_name": "Alert Device",
        },
    )
    assert device_register.status_code == 200

    class FakeModelService:
        def predict(self, data):
            return "High", 0.99, {"Low": 0.0, "Medium": 0.0, "High": 1.0}

    class FakeNotificationService:
        def __init__(self):
            self.sent = []

        def send_device_alert(self, **kwargs):
            self.sent.append(kwargs)
            return "message-id-123"

    fake_notification_service = FakeNotificationService()

    monkeypatch.setattr("backend.api.mobile.ModelService.get_instance", lambda: FakeModelService())
    monkeypatch.setattr("backend.api.mobile.NotificationService.get_instance", lambda: fake_notification_service)

    response = client.post(
        "/api/mobile/predict",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "device_id": device_id,
            "vehicle_type": "car",
            "temperature_c": 30.0,
            "humidity_pct": 90.0,
            "rainfall_mm": 100.0,
            "wind_speed_kmh": 40.0,
            "visibility_km": 0.5,
            "gradient_pct": 20.0,
            "curvature": 0.9,
            "road_surface": "gravel",
            "lane_count": 1,
            "hour_of_day": 2,
            "day_of_week": 6,
            "is_holiday": True,
            "traffic_density": 0.9,
        },
    )

    assert response.status_code == 200
    assert response.json()["risk_level"] == "High"
    assert len(fake_notification_service.sent) == 1