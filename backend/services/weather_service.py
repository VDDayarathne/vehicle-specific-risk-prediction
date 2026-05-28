"""
backend/services/weather_service.py
Fetches live weather from OpenMeteo API (no API key required).
"""
import httpx
from backend.config.settings import (
    OPENMETEO_BASE_URL, KADUGANNAWA_LAT, KADUGANNAWA_LON
)
from backend.services.cache_service import write_weather_cache


WMO_DESCRIPTIONS = {
    0:  ("Clear sky",        "☀️"),
    1:  ("Mainly clear",     "🌤️"),
    2:  ("Partly cloudy",    "⛅"),
    3:  ("Overcast",         "☁️"),
    45: ("Foggy",            "🌫️"),
    48: ("Icy fog",          "🌫️"),
    51: ("Light drizzle",    "🌦️"),
    53: ("Moderate drizzle", "🌦️"),
    55: ("Dense drizzle",    "🌧️"),
    61: ("Slight rain",      "🌧️"),
    63: ("Moderate rain",    "🌧️"),
    65: ("Heavy rain",       "🌧️"),
    71: ("Slight snow",      "🌨️"),
    73: ("Moderate snow",    "❄️"),
    75: ("Heavy snow",       "❄️"),
    80: ("Rain showers",     "🌦️"),
    81: ("Rain showers",     "🌦️"),
    82: ("Violent showers",  "⛈️"),
    95: ("Thunderstorm",     "⛈️"),
    99: ("Thunderstorm+hail","⛈️"),
}


async def fetch_current_weather(lat: float = KADUGANNAWA_LAT,
                                lon: float = KADUGANNAWA_LON) -> dict:
    """Return current weather conditions from OpenMeteo."""
    params = {
        "latitude":  lat,
        "longitude": lon,
        "current":   [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "visibility",
            "weather_code",
        ],
        "timezone": "Asia/Colombo",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(OPENMETEO_BASE_URL, params=params)
        r.raise_for_status()
        data = r.json()

    c = data["current"]
    code = int(c.get("weather_code", 0))
    desc, icon = WMO_DESCRIPTIONS.get(code, ("Unknown", "🌡️"))

    # visibility comes in metres from OpenMeteo → convert to km
    vis_raw = c.get("visibility", 10000)
    vis_km  = round(vis_raw / 1000, 1) if vis_raw else 10.0

    prcp = round(c.get("precipitation", 0.0), 2)
    wind_spd = round(c.get("wind_speed_10m", 0.0), 1)

    # Calculate weather risk score (from 0 to 6)
    import datetime
    risk = 0
    if prcp > 5.0:
        risk += 3
    elif prcp > 0.5:
        risk += 1

    if vis_km < 0.5:
        risk += 2
    elif vis_km < 1.5:
        risk += 1

    if wind_spd > 40.0: # high winds
        risk += 1

    # Night risk
    current_hour = datetime.datetime.now().hour
    if current_hour < 6 or current_hour >= 20:
        risk += 1

    weather_risk_score = min(6, risk)

    payload = {
        "temperature_c":  round(c.get("temperature_2m", 25.0), 1),
        "humidity_pct":   round(c.get("relative_humidity_2m", 70.0), 1),
        "rainfall_mm":    prcp,
        "precipitation_mm": prcp,
        "wind_speed_kmh": wind_spd,
        "visibility_km":  vis_km,
        "description":    desc,
        "icon":           icon,
        "weather_risk_score": weather_risk_score,
    }

    # Write to cache for offline fallback (key by lat:lon)
    try:
        key = f"{lat:.6f}:{lon:.6f}"
        write_weather_cache(key, {**payload, "fetched_at": __import__("datetime").datetime.utcnow().isoformat()})
    except Exception:
        pass

    return payload

