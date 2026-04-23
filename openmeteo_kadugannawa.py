"""
openmeteo_kadugannawa.py
========================
Complete Open-Meteo API integration for Kadugannawa Pass accident
prediction system. Covers:
  1. Real-time current conditions
  2. 7-day forecast (hourly)
  3. Historical data (any past date range — no API key needed)
  4. Parse → ML-ready DataFrame
  5. Live alert generation per vehicle type
  6. Merge with accident dataset for training

Requirements:  pip install requests pandas numpy
Open-Meteo:    https://open-meteo.com  (free, no API key, no rate limit for research)
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

# ── Constants ──────────────────────────────────────────────────────────────────
KADUGANNAWA = {"lat": 7.2500, "lon": 80.5300}  # Centre of A1 pass
TIMEZONE = "Asia/Colombo"
FORECAST_API  = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"

# Variables to request — all relevant to accident prediction
HOURLY_VARS = [
    "temperature_2m",           # °C   — temp_avg_c
    "relative_humidity_2m",     # %    — humidity
    "precipitation",            # mm   — rainfall
    "rain",                     # mm   — rain only (no snow)
    "wind_speed_10m",           # km/h — wind speed
    "wind_direction_10m",       # °    — wind direction
    "wind_gusts_10m",           # km/h — gusts (dangerous for motorcycles/buses)
    "visibility",               # m    — visibility (critical for fog/mist)
    "surface_pressure",         # hPa  — barometric pressure
    "cloud_cover",              # %    — cloud cover
    "weather_code",             # WMO  — standardised condition code
    "is_day",                   # 0/1  — daylight flag
]

# WMO weather interpretation codes
WMO_CODES = {
    0: "Clear sky",          1: "Mainly clear",       2: "Partly cloudy",
    3: "Overcast",           45: "Foggy",             48: "Depositing rime fog",
    51: "Light drizzle",     53: "Moderate drizzle",  55: "Dense drizzle",
    61: "Slight rain",       63: "Moderate rain",     65: "Heavy rain",
    71: "Slight snow fall",  73: "Moderate snow",     75: "Heavy snow",
    77: "Snow grains",       80: "Slight showers",    81: "Moderate showers",
    82: "Violent showers",   85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm",      96: "Thunderstorm + hail", 99: "Thunderstorm + heavy hail",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. API CALL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_current_weather(lat: float = KADUGANNAWA["lat"],
                        lon: float = KADUGANNAWA["lon"]) -> dict:
    """
    Fetch current conditions + 24h forecast for Kadugannawa.
    Returns parsed dict ready to feed into the prediction API.

    Example return:
        {
          "temperature_c": 26.4,
          "precipitation_mm": 2.1,
          "weather_condition": "light_rain",
          "wind_speed_kmh": 18.0,
          "visibility_km": 3.2,
          "pressure_hpa": 1010.8,
          "weather_risk_score": 2,
          "wmo_description": "Moderate rain",
          "timestamp": "2024-06-15T14:00"
        }
    """
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    ",".join(HOURLY_VARS),
        "timezone":  TIMEZONE,
        "forecast_days": 1,
    }
    resp = _get(FORECAST_API, params)
    df   = _parse_hourly(resp)

    # Get the row closest to now
    now = pd.Timestamp.now()
    idx = (df["time"] - now).abs().idxmin()
    row = df.iloc[idx]

    return {
        "timestamp":          row["time"].strftime("%Y-%m-%dT%H:%M"),
        "temperature_c":      row["temp_avg_c"],
        "humidity_pct":       row["humidity_pct"],
        "precipitation_mm":   row["precipitation_mm"],
        "wind_speed_kmh":     row["wind_speed_kmh"],
        "wind_gusts_kmh":     row.get("wind_gusts_kmh", None),
        "visibility_km":      row["visibility_km"],
        "pressure_hpa":       row["pressure_hpa"],
        "cloud_cover_pct":    row["cloud_cover_pct"],
        "weather_condition":  row["weather_condition"],
        "wmo_description":    row["wmo_description"],
        "weather_risk_score": int(row["weather_risk_score"]),
        "is_night":           int(row["is_night"]),
    }


def get_forecast(days: int = 7,
                 lat: float = KADUGANNAWA["lat"],
                 lon: float = KADUGANNAWA["lon"]) -> pd.DataFrame:
    """
    Fetch hourly forecast for next `days` days (max 16).
    Returns ML-ready DataFrame — one row per hour.
    """
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "hourly":        ",".join(HOURLY_VARS),
        "timezone":      TIMEZONE,
        "forecast_days": min(days, 16),
    }
    resp = _get(FORECAST_API, params)
    return _parse_hourly(resp)


def get_historical(start_date: str, end_date: str,
                   lat: float = KADUGANNAWA["lat"],
                   lon: float = KADUGANNAWA["lon"]) -> pd.DataFrame:
    """
    Fetch historical hourly data for any past date range (back to 1940).
    Free — no API key needed.

    Args:
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"  (can be yesterday)

    Example:
        df = get_historical("2023-01-01", "2024-12-31")
        # Returns ~17,520 hourly rows (2 years)
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     ",".join([
            # Historical API has slightly different available vars
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "visibility",
            "surface_pressure",
            "cloud_cover",
            "weather_code",
            "is_day",
        ]),
        "timezone": TIMEZONE,
    }
    resp = _get(HISTORICAL_API, params)
    return _parse_hourly(resp)


def get_historical_daily(start_date: str, end_date: str,
                         lat: float = KADUGANNAWA["lat"],
                         lon: float = KADUGANNAWA["lon"]) -> pd.DataFrame:
    """
    Fetch daily aggregated historical data — useful to merge with
    your export.xlsx daily weather data or police accident records.

    Columns: date, temp_max, temp_min, temp_mean, precipitation_sum,
             wind_speed_max, wind_gusts_max, weather_code
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "weather_code",
            "sunrise",
            "sunset",
        ]),
        "timezone": TIMEZONE,
    }
    resp = _get(HISTORICAL_API, params)

    d = resp["daily"]
    df = pd.DataFrame(d)
    df = df.rename(columns={
        "time":                  "date",
        "temperature_2m_max":    "tmax",
        "temperature_2m_min":    "tmin",
        "temperature_2m_mean":   "tavg",
        "precipitation_sum":     "prcp",
        "rain_sum":              "rain",
        "wind_speed_10m_max":    "wind_speed_max_kmh",
        "wind_gusts_10m_max":    "wind_gusts_max_kmh",
        "weather_code":          "wmo_code",
    })
    df["wmo_description"] = df["wmo_code"].map(WMO_CODES).fillna("Unknown")
    df["weather_condition"] = df["wmo_code"].apply(_wmo_to_condition)
    df["temp_range"] = (df["tmax"] - df["tmin"]).round(1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PARSER & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _get(url: str, params: dict, retries: int = 3) -> dict:
    """HTTP GET with retry logic and friendly error messages."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30,
                             headers={"User-Agent": "VehicleAlertResearch/1.0"})
            if r.status_code == 200:
                return r.json()
            if r.status_code == 400:
                raise ValueError(f"Bad request: {r.json().get('reason', r.text)}")
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            if attempt == retries - 1:
                raise ConnectionError(
                    "Cannot reach api.open-meteo.com. "
                    "Check internet connection or try archive-api.open-meteo.com"
                )
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed after {retries} attempts")


def _wmo_to_condition(code: int) -> str:
    """Map WMO code → your accident dataset's weather_condition categories."""
    if code == 0:                           return "clear"
    if code in (1, 2, 3):                  return "clear"
    if code in (45, 48):                   return "foggy"
    if code in (51, 53, 55, 61, 63,
                77, 80, 81, 85):           return "light_rain"
    if code in (55, 65, 82, 86,
                95, 96, 99):               return "heavy_rain"
    return "clear"


def _parse_hourly(response: dict) -> pd.DataFrame:
    """
    Convert raw Open-Meteo JSON → clean ML-ready DataFrame.
    Column names deliberately match kadugannawa_accidents_mock.csv
    so a merge on 'date' + 'hour' works directly.
    """
    h = response["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])

    # Rename to match accident dataset
    rename = {
        "temperature_2m":        "temp_avg_c",
        "relative_humidity_2m":  "humidity_pct",
        "precipitation":         "precipitation_mm",
        "wind_speed_10m":        "wind_speed_kmh",
        "wind_direction_10m":    "wind_dir_deg",
        "wind_gusts_10m":        "wind_gusts_kmh",
        "visibility":            "visibility_m",
        "surface_pressure":      "pressure_hpa",
        "cloud_cover":           "cloud_cover_pct",
        "weather_code":          "wmo_code",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Derived columns
    df["date"]             = df["time"].dt.date.astype(str)
    df["hour"]             = df["time"].dt.hour
    df["visibility_km"]    = (df["visibility_m"] / 1000).round(2)
    df["wmo_description"]  = df["wmo_code"].map(WMO_CODES).fillna("Unknown")

    # weather_condition: prefer visibility-based override for fog/mist
    def classify(row):
        if "visibility_km" in row and row["visibility_km"] < 0.3:
            return "foggy"
        if "visibility_km" in row and row["visibility_km"] < 1.0:
            return "mist"
        return _wmo_to_condition(int(row["wmo_code"]))

    df["weather_condition"]   = df.apply(classify, axis=1)
    df["is_rainy"]            = (df["precipitation_mm"] > 0.5).astype(int)
    df["rain_intensity"]      = pd.cut(
        df["precipitation_mm"],
        bins=[-0.01, 0.5, 5.0, 999],
        labels=["none", "light", "heavy"]
    )
    df["is_night"]            = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)
    df["is_peak_hour"]        = df["hour"].isin([7, 8, 17, 18, 19]).astype(int)
    df["pressure_drop"]       = df["pressure_hpa"].diff().fillna(0).round(2)
    df["temp_feels_risky"]    = (
        (df["precipitation_mm"] > 2) |
        (df.get("wind_gusts_kmh", pd.Series(0, index=df.index)) > 50) |
        (df["visibility_km"] < 1.0)
    ).astype(int)

    # Weather risk score (matches risk_score logic in accident dataset)
    df["weather_risk_score"] = (
        df["precipitation_mm"].apply(lambda x: 3 if x > 5 else (1 if x > 0.5 else 0)) +
        df["visibility_km"].apply(lambda x: 2 if x < 0.5 else (1 if x < 1.5 else 0)) +
        df.get("wind_gusts_kmh", pd.Series(0, index=df.index)).apply(lambda x: 1 if x > 50 else 0) +
        df["is_night"]
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. ALERT ENGINE — real-time per vehicle type
# ══════════════════════════════════════════════════════════════════════════════

VEHICLE_RISK_WEIGHTS = {
    "motorcycle":   4,   # most vulnerable — no cabin, skid risk
    "lorry":        3,   # brake fade on slopes, wide load on curves
    "bus":          3,   # passenger duty, poor wet stopping distance
    "car":          1,
    "three-wheeler": 2,
}

ALERT_MESSAGES = {
    "motorcycle": {
        3: ("EXTREME DANGER",  "DO NOT PROCEED. Heavy rain + poor visibility is fatal for motorcyclists on mountain bends."),
        2: ("HIGH RISK",       "Extreme caution. Wet road + curves. Reduce speed to 20 km/h. Consider shelter."),
        1: ("MODERATE RISK",   "Slippery when wet. Widen following distance. Slow down on bends."),
        0: ("LOW RISK",        "Conditions acceptable. Normal care required on sharp bends."),
    },
    "lorry": {
        3: ("EXTREME DANGER",  "LORRIES PROHIBITED IN CURRENT CONDITIONS. Risk of jackknife and brake failure on descent."),
        2: ("HIGH RISK",       "Steep gradient + wet surface. Engage low gear. Do NOT use brakes on long descent."),
        1: ("MODERATE RISK",   "Reduce speed to 30 km/h on descent. Check brakes before Kadugannawa Pass."),
        0: ("LOW RISK",        "Normal conditions. Use low gear on steep sections as per regulation."),
    },
    "bus": {
        3: ("HIGH RISK",       "Passenger safety alert. Consider halting until rain reduces. Max 20 km/h."),
        2: ("MODERATE RISK",   "Reduce speed. Increase stopping distance. Alert passengers."),
        1: ("CAUTION",         "Wet road advisory. Reduce speed 10 km/h below limit on curves."),
        0: ("LOW RISK",        "Normal conditions. Maintain regulated speeds."),
    },
    "car": {
        3: ("HIGH RISK",       "Do not drive unless essential. Flash flooding possible on A1 near Kadugannawa."),
        2: ("MODERATE RISK",   "Reduce speed. Use headlights. Avoid overtaking on curves."),
        1: ("CAUTION",         "Light rain — slow down and increase following distance."),
        0: ("LOW RISK",        "Clear conditions. Normal care required."),
    },
    "three-wheeler": {
        3: ("HIGH RISK",       "Three-wheelers MUST NOT attempt descent in heavy rain. Overturn risk is extreme."),
        2: ("MODERATE RISK",   "Avoid sharp curves at speed. Max 25 km/h. Consider alternate lower route."),
        1: ("CAUTION",         "Reduce speed on bends. Three-wheelers have high tipping risk when wet."),
        0: ("LOW RISK",        "Normal conditions. Take curves slowly as standard."),
    },
}


def generate_alert(vehicle_type: str, weather: dict) -> dict:
    """
    Generate a real-time driver alert from current weather conditions.

    Args:
        vehicle_type: "car" | "bus" | "lorry" | "motorcycle" | "three-wheeler"
        weather:      dict from get_current_weather()

    Returns:
        Alert dict with level, message, recommended_speed_kmh
    """
    base_score = weather["weather_risk_score"]
    vehicle_weight = VEHICLE_RISK_WEIGHTS.get(vehicle_type, 1)

    # Adjusted score: clamp 0–3
    adjusted = min(3, base_score + (1 if vehicle_weight >= 3 else 0))

    messages = ALERT_MESSAGES.get(vehicle_type, ALERT_MESSAGES["car"])
    level, message = messages[adjusted]

    # Recommended speed
    speed_map = {0: None, 1: 40, 2: 25, 3: 0}
    rec_speed = speed_map[adjusted]

    return {
        "vehicle_type":          vehicle_type,
        "alert_level":           level,
        "alert_score":           adjusted,
        "message":               message,
        "recommended_speed_kmh": rec_speed,
        "conditions_summary": (
            f"{weather['weather_condition'].replace('_',' ').title()} | "
            f"Temp {weather['temperature_c']}°C | "
            f"Rain {weather['precipitation_mm']}mm | "
            f"Visibility {weather['visibility_km']}km | "
            f"Wind {weather['wind_speed_kmh']}km/h"
        ),
        "timestamp": weather["timestamp"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. MERGE WITH ACCIDENT TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════════

def merge_weather_with_accidents(accidents_csv: str,
                                  weather_start: str,
                                  weather_end: str) -> pd.DataFrame:
    """
    Fetch historical hourly weather for the accident dataset date range
    and merge on date + hour. Adds all weather features to accident rows.

    Usage:
        df = merge_weather_with_accidents(
            "kadugannawa_accidents_mock.csv",
            "2019-01-01",
            "2024-12-31"
        )
        df.to_csv("accidents_with_weather.csv", index=False)
    """
    print(f"Fetching historical weather {weather_start} → {weather_end}...")
    weather_df = get_historical(weather_start, weather_end)

    accidents  = pd.read_csv(accidents_csv)
    accidents["date"] = pd.to_datetime(accidents["date"]).dt.date.astype(str)
    accidents["hour"] = pd.to_datetime(
        accidents["date"] + " " + accidents["time"]
    ).dt.hour

    merged = accidents.merge(
        weather_df[["date", "hour", "precipitation_mm", "temp_avg_c",
                    "wind_speed_kmh", "visibility_km", "pressure_hpa",
                    "weather_condition", "is_rainy", "rain_intensity",
                    "weather_risk_score", "pressure_drop"]],
        on=["date", "hour"],
        how="left",
        suffixes=("_mock", "_osm")
    )

    print(f"Merged shape: {merged.shape}")
    print(f"Weather match rate: {merged['precipitation_mm_osm'].notna().mean():.1%}")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN — demo all functions
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Open-Meteo Integration — Kadugannawa Pass")
    print("=" * 60)

    # ── A: Current conditions ──────────────────────────────────────
    print("\n[1] Current weather at Kadugannawa")
    current = get_current_weather()
    for k, v in current.items():
        print(f"    {k:<25} {v}")

    # ── B: 7-day forecast → save CSV ──────────────────────────────
    print("\n[2] 7-day hourly forecast")
    forecast_df = get_forecast(days=7)
    forecast_df.to_csv("kadugannawa_forecast_7d.csv", index=False)
    print(f"    Saved: kadugannawa_forecast_7d.csv ({len(forecast_df)} rows)")
    print("    High-risk hours:")
    risky = forecast_df[forecast_df["weather_risk_score"] >= 3]
    if len(risky):
        print(risky[["time", "weather_condition", "precipitation_mm",
                      "visibility_km", "weather_risk_score"]].head(5).to_string())
    else:
        print("    (none in current 7-day window)")

    # ── C: Historical data → matches your accident CSV date range ──
    print("\n[3] Historical daily data (2019-2024)")
    hist_daily = get_historical_daily("2019-01-01", "2024-12-31")
    hist_daily.to_csv("kadugannawa_historical_daily.csv", index=False)
    print(f"    Saved: kadugannawa_historical_daily.csv ({len(hist_daily)} rows)")
    print("    Rain stats:")
    print(f"      Mean daily precip : {hist_daily['prcp'].mean():.2f} mm")
    print(f"      Max daily precip  : {hist_daily['prcp'].max():.1f} mm")
    print(f"      Rainy days (>1mm) : {(hist_daily['prcp']>1).sum()} days")

    # ── D: Real-time alert per vehicle type ───────────────────────
    print("\n[4] Vehicle alerts for current conditions")
    for vtype in ["motorcycle", "lorry", "bus", "car", "three-wheeler"]:
        alert = generate_alert(vtype, current)
        print(f"    {vtype:<14} → [{alert['alert_level']}] {alert['message'][:60]}...")

    # ── E: Merge with accident dataset ────────────────────────────
    print("\n[5] Merging with accident data")
    merged = merge_weather_with_accidents(
        "kadugannawa_accidents_mock.csv",
        "2019-01-01", "2024-12-31"
    )
    merged.to_csv("accidents_with_realweather.csv", index=False)
    print(f"    Final dataset: {merged.shape[0]} rows × {merged.shape[1]} cols")
    print("    Ready for model training.")