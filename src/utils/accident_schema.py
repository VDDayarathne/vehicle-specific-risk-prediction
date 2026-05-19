"""Helpers to normalize accident CSVs into the legacy ML training schema."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


_LEGACY_COLUMNS = [
    "accident_id", "date", "time", "day_of_week", "is_weekend", "hour",
    "location_name", "latitude", "longitude", "elevation_m", "vehicle_type",
    "num_vehicles", "accident_type", "driver_age", "driver_gender",
    "driving_exp_yrs", "estimated_speed_kmh", "speed_limit_kmh", "speeding",
    "weather_condition", "precipitation_mm", "temp_avg_c", "visibility_km",
    "wind_speed_kmh", "pressure_hpa", "road_curvature", "road_slope_deg",
    "surface_condition", "severity", "injuries", "fatalities", "risk_score",
    "reporting_officer",
]

_TERRAIN_RULES = {
    "hairpin bends / steep gradient": {
        "road_curvature": "sharp",
        "road_slope_deg": 13.0,
        "speed_limit_kmh": 50,
        "elevation_m": 620,
    },
    "steep winding road": {
        "road_curvature": "sharp",
        "road_slope_deg": 11.0,
        "speed_limit_kmh": 50,
        "elevation_m": 560,
    },
    "moderate bends / slope": {
        "road_curvature": "mild",
        "road_slope_deg": 7.0,
        "speed_limit_kmh": 60,
        "elevation_m": 520,
    },
    "flat urban / town center": {
        "road_curvature": "straight",
        "road_slope_deg": 1.0,
        "speed_limit_kmh": 50,
        "elevation_m": 470,
    },
}

_VEHICLE_PROFILES = {
    "motorcycle": {"age": 27, "exp": 5, "gender": "male"},
    "car": {"age": 36, "exp": 11, "gender": "male"},
    "bus": {"age": 43, "exp": 16, "gender": "male"},
    "lorry": {"age": 40, "exp": 14, "gender": "male"},
    "van": {"age": 34, "exp": 10, "gender": "male"},
    "three-wheeler": {"age": 32, "exp": 8, "gender": "male"},
    "pick-up": {"age": 35, "exp": 10, "gender": "male"},
}


def _slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _terrain_defaults(terrain: Any) -> dict[str, Any]:
    key = str(terrain).strip().lower()
    return _TERRAIN_RULES.get(key, {
        "road_curvature": "mild",
        "road_slope_deg": 6.0,
        "speed_limit_kmh": 50,
        "elevation_m": 500,
    })


def _weather_defaults(is_wet_road: Any, risk_probability: Any) -> dict[str, Any]:
    wet = int(pd.to_numeric(pd.Series([is_wet_road]), errors="coerce").fillna(0).iloc[0])
    risk = pd.to_numeric(pd.Series([risk_probability]), errors="coerce").fillna(0.5).iloc[0]

    if not wet:
        return {
            "weather_condition": "clear",
            "precipitation_mm": 0.0,
            "temp_avg_c": 27.0,
            "visibility_km": 8.0,
            "wind_speed_kmh": 12.0,
            "pressure_hpa": 1011.0,
            "surface_condition": "dry",
        }

    if risk >= 0.75:
        return {
            "weather_condition": "heavy_rain",
            "precipitation_mm": 8.0,
            "temp_avg_c": 25.5,
            "visibility_km": 1.5,
            "wind_speed_kmh": 18.0,
            "pressure_hpa": 1007.0,
            "surface_condition": "very_wet",
        }

    return {
        "weather_condition": "light_rain",
        "precipitation_mm": 2.0,
        "temp_avg_c": 26.0,
        "visibility_km": 4.0,
        "wind_speed_kmh": 15.0,
        "pressure_hpa": 1009.0,
        "surface_condition": "wet",
    }


def normalize_accident_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with the legacy accident schema expected by the pipeline."""
    if df.empty:
        return df.copy()

    frame = df.copy()

    rename_map = {
        "Date": "date",
        "Time": "time",
        "Vehicle_Type": "vehicle_type",
        "Location": "location_name",
        "Terrain": "terrain",
        "Cause": "accident_type",
        "Severity": "severity",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Is_Wet_Road": "is_wet_road",
        "Risk_Probability": "risk_probability",
        "Description": "description",
    }
    frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})

    if "date" not in frame.columns:
        raise ValueError("Accident data must contain a date column")

    if "vehicle_type" in frame.columns:
        frame["vehicle_type"] = frame["vehicle_type"].astype(str).str.strip().str.lower()

    parsed_date = pd.to_datetime(frame["date"], errors="coerce")

    if "time" in frame.columns:
        parsed_time = pd.to_datetime(
            frame["time"].astype(str).str.strip(),
            format="%H:%M:%S",
            errors="coerce",
        )
    else:
        parsed_time = pd.Series(pd.NaT, index=frame.index)
        frame["time"] = "00:00:00"

    if "hour" not in frame.columns:
        frame["hour"] = parsed_time.dt.hour.fillna(0).astype(int)

    if "day_of_week" not in frame.columns:
        frame["day_of_week"] = parsed_date.dt.day_name()

    if "is_weekend" not in frame.columns:
        frame["is_weekend"] = parsed_date.dt.weekday.ge(5).astype(int)

    if "location_name" not in frame.columns and "Location" in df.columns:
        frame["location_name"] = df["Location"]

    if "latitude" not in frame.columns and "Latitude" in df.columns:
        frame["latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    frame["latitude"] = pd.to_numeric(frame.get("latitude"), errors="coerce")

    if "longitude" not in frame.columns and "Longitude" in df.columns:
        frame["longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame.get("longitude"), errors="coerce")

    if "terrain" not in frame.columns:
        frame["terrain"] = "Flat Urban / Town Center"

    terrain_defaults = frame["terrain"].map(_terrain_defaults)
    frame["road_curvature"] = terrain_defaults.map(lambda x: x["road_curvature"])
    frame["road_slope_deg"] = terrain_defaults.map(lambda x: x["road_slope_deg"])
    frame["speed_limit_kmh"] = terrain_defaults.map(lambda x: x["speed_limit_kmh"])
    frame["elevation_m"] = terrain_defaults.map(lambda x: x["elevation_m"])

    if "is_wet_road" not in frame.columns:
        frame["is_wet_road"] = 0
    if "risk_probability" not in frame.columns:
        frame["risk_probability"] = 0.5

    weather_defaults = frame.apply(
        lambda row: _weather_defaults(row.get("is_wet_road", 0), row.get("risk_probability", 0.5)),
        axis=1,
        result_type="expand",
    )
    for column in weather_defaults.columns:
        frame[column] = weather_defaults[column]

    if "accident_type" in frame.columns:
        frame["accident_type"] = frame["accident_type"].astype(str).map(_slugify)
    else:
        frame["accident_type"] = "unknown"

    if "severity" in frame.columns:
        severity_map = {
            "damage only": "low",
            "minor injury": "medium",
            "non-grievous": "medium",
            "grievous": "high",
            "fatal": "high",
        }
        frame["severity"] = frame["severity"].astype(str).str.strip().str.lower().map(severity_map).fillna("medium")
    else:
        frame["severity"] = "medium"

    if "injuries" not in frame.columns:
        frame["injuries"] = frame["severity"].map({"low": 0, "medium": 1, "high": 2}).fillna(1).astype(int)
    if "fatalities" not in frame.columns:
        frame["fatalities"] = frame["severity"].map({"low": 0, "medium": 0, "high": 1}).fillna(0).astype(int)

    if "num_vehicles" not in frame.columns:
        frame["num_vehicles"] = 1

    if "driver_age" not in frame.columns:
        frame["driver_age"] = frame["vehicle_type"].map(lambda v: _VEHICLE_PROFILES.get(v, {"age": 35})["age"])
    if "driving_exp_yrs" not in frame.columns:
        frame["driving_exp_yrs"] = frame["vehicle_type"].map(lambda v: _VEHICLE_PROFILES.get(v, {"exp": 8})["exp"])
    if "driver_gender" not in frame.columns:
        frame["driver_gender"] = frame["vehicle_type"].map(lambda v: _VEHICLE_PROFILES.get(v, {"gender": "male"})["gender"])

    if "estimated_speed_kmh" not in frame.columns:
        frame["estimated_speed_kmh"] = frame.apply(
            lambda row: max(
                20,
                row["speed_limit_kmh"] + (
                    18 if row["accident_type"] == "excessive_speed"
                    else 8 if row["accident_type"] == "overtaking"
                    else 3 if row["accident_type"] in {"brake_failure", "mechanical_error"}
                    else -8 if row["accident_type"] == "slipped"
                    else 0
                ),
            ),
            axis=1,
        )

    frame["speeding"] = (frame["estimated_speed_kmh"] > frame["speed_limit_kmh"]).astype(int)
    frame["risk_score"] = pd.to_numeric(frame.get("risk_probability"), errors="coerce").fillna(0.5).mul(10).round(0).astype(int)
    frame["reporting_officer"] = [f"SLP-{1000 + i:04d}" for i in range(len(frame))]

    if "precipitation_mm" not in frame.columns:
        frame["precipitation_mm"] = 0.0
    if "temp_avg_c" not in frame.columns:
        frame["temp_avg_c"] = 26.0
    if "visibility_km" not in frame.columns:
        frame["visibility_km"] = 8.0
    if "wind_speed_kmh" not in frame.columns:
        frame["wind_speed_kmh"] = 12.0
    if "pressure_hpa" not in frame.columns:
        frame["pressure_hpa"] = 1011.0

    if "accident_id" not in frame.columns:
        frame["accident_id"] = [f"KDG-{i + 1:04d}" for i in range(len(frame))]

    frame["date"] = parsed_date.dt.strftime("%Y-%m-%d")
    frame["time"] = frame["time"].astype(str).str.strip()

    for column in _LEGACY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    ordered = frame[_LEGACY_COLUMNS + [c for c in frame.columns if c not in _LEGACY_COLUMNS]]
    return ordered