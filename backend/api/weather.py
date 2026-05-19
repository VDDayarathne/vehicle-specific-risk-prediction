"""
backend/api/weather.py
GET /api/weather  — live weather for Kadugannawa from OpenMeteo.
"""
from fastapi import APIRouter, HTTPException, Query

from backend.models.schemas import WeatherResponse
from backend.services.weather_service import fetch_current_weather
from backend.config.settings import KADUGANNAWA_LAT, KADUGANNAWA_LON

router = APIRouter(prefix="/api", tags=["Weather"])


@router.get("/weather", response_model=WeatherResponse, summary="Live weather conditions")
async def get_weather(
    lat: float = Query(KADUGANNAWA_LAT, description="Latitude"),
    lon: float = Query(KADUGANNAWA_LON, description="Longitude"),
) -> WeatherResponse:
    """
    Returns current weather at the given coordinates (default: Kadugannawa).
    Uses OpenMeteo — no API key required.
    """
    try:
        data = await fetch_current_weather(lat, lon)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weather fetch failed: {exc}")

    return WeatherResponse(**data)
