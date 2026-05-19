"""
backend/tests/test_weather.py
Tests for the weather endpoint.
"""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_weather_default_location():
    r = client.get("/api/weather")
    assert r.status_code in (200, 502)   # 502 if no internet in CI
    if r.status_code == 200:
        body = r.json()
        assert "temperature_c" in body
        assert "humidity_pct" in body
        assert "rainfall_mm" in body
        assert "icon" in body


def test_weather_custom_location():
    r = client.get("/api/weather?lat=6.9271&lon=79.8612")   # Colombo
    assert r.status_code in (200, 502)
