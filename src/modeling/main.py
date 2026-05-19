"""
main.py  —  KaduGuard: Real-Time Road Risk Prediction & Warning System
======================================================================
FastAPI backend combining:
  1. ML model inference  (XGBoost / Random Forest, pkl loaded at startup)
  2. Live weather        (Open-Meteo — no API key needed)
  3. Alert engine        (per vehicle-type, per-segment risk messages)
  4. Route hazard API    (Kadugannawa->Hingula segment risk scoring)
  5. Navigation session  (ride start/stop, real-time position updates)
  6. 7-day forecast      (hourly risk scores for dashboard)
  7. Static HTML serving (driver navigation UI)

Usage:
  pip install fastapi uvicorn scikit-learn xgboost imbalanced-learn \\
              requests numpy pandas python-multipart
  python main.py
  # or: uvicorn main:app --reload --port 8000

Endpoints:
  GET  /                          -> HTML navigation dashboard
  POST /predict                   -> ML severity prediction
  POST /predict/segment           -> Predict risk for a specific route segment
  GET  /route/segments            -> All route segments with static risk data
  POST /route/session/start       -> Start a ride session
  POST /route/session/update      -> Update vehicle position during ride
  GET  /route/session/{sid}       -> Get session status
  GET  /weather/current           -> Live Kadugannawa conditions
  GET  /weather/forecast          -> 7-day hourly forecast
  GET  /alert/{vehicle_type}      -> Quick alert by vehicle type
  GET  /alert                     -> All 5 vehicle alerts
  GET  /health                    -> System status + model name
  GET  /docs                      -> Swagger UI (auto-generated)
"""

from __future__ import annotations

import os
import time
import uuid
import pickle
import logging
import warnings
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("kaduAlert")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

KADUGANNAWA   = {"lat": 7.2500, "lon": 80.5300}
TIMEZONE      = "Asia/Colombo"
FORECAST_API  = "https://api.open-meteo.com/v1/forecast"
_ROOT         = Path(__file__).resolve().parents[2]
MODEL_DIR     = _ROOT / "models" / "saved_models"
DATA_DIR      = _ROOT / "data" / "processed"
SEVERITY_NAMES = ["Low", "Medium", "High"]
VERSION       = "2.0.0"

WMO_CODES = {
    0: "Clear sky",        1: "Mainly clear",     2: "Partly cloudy",
    3: "Overcast",         45: "Foggy",           48: "Rime fog",
    51: "Light drizzle",   53: "Moderate drizzle",55: "Dense drizzle",
    61: "Slight rain",     63: "Moderate rain",   65: "Heavy rain",
    80: "Rain showers",    81: "Moderate showers",82: "Violent showers",
    95: "Thunderstorm",    96: "Thunderstorm+hail",
}

HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "rain",
    "wind_speed_10m", "wind_gusts_10m", "visibility", "surface_pressure",
    "cloud_cover", "weather_code", "is_day",
]

# ── Route definition: Kadugannawa -> Hingula ───────────────────────────────────
ROUTE_SEGMENTS = [
    {"id":1,  "name":"Peradeniya Junction",       "lat":7.2488,"lon":80.4748,"dist_km":0.0,  "slope":1.7,  "curvature":"sharp",    "speed_limit":60,"risk":"low",    "alert_km":0.4},
    {"id":2,  "name":"A1 Entry Climb",            "lat":7.2510,"lon":80.4850,"dist_km":1.8,  "slope":2.1,  "curvature":"mild",     "speed_limit":60,"risk":"low",    "alert_km":0.3},
    {"id":3,  "name":"Kadugannawa–Pottepitiya",   "lat":7.2569,"lon":80.5190,"dist_km":4.2,  "slope":2.5,  "curvature":"sharp",    "speed_limit":50,"risk":"medium", "alert_km":0.5},
    {"id":4,  "name":"Sensation Rock Bend",        "lat":7.2585,"lon":80.5218,"dist_km":5.5,  "slope":8.0,  "curvature":"sharp",    "speed_limit":50,"risk":"high",   "alert_km":0.8},
    {"id":5,  "name":"Salgala Descent",            "lat":7.2620,"lon":80.5108,"dist_km":7.1,  "slope":12.0, "curvature":"sharp",    "speed_limit":40,"risk":"high",   "alert_km":1.0},
    {"id":6,  "name":"Queens Hotel Hairpin",       "lat":7.2649,"lon":80.5221,"dist_km":8.9,  "slope":6.0,  "curvature":"sharp",    "speed_limit":40,"risk":"high",   "alert_km":0.7},
    {"id":7,  "name":"Tunnel Bend",               "lat":7.2663,"lon":80.5520,"dist_km":10.4, "slope":3.5,  "curvature":"mild",     "speed_limit":50,"risk":"medium", "alert_km":0.5},
    {"id":8,  "name":"Kadugannawa Town",          "lat":7.2657,"lon":80.5524,"dist_km":11.8, "slope":1.0,  "curvature":"straight", "speed_limit":50,"risk":"medium", "alert_km":0.4},
    {"id":9,  "name":"Danture Road Junction",     "lat":7.2821,"lon":80.5343,"dist_km":13.5, "slope":0.3,  "curvature":"sharp",    "speed_limit":50,"risk":"medium", "alert_km":0.4},
    {"id":10, "name":"Balana Road Descent",       "lat":7.2618,"lon":80.5107,"dist_km":15.2, "slope":2.8,  "curvature":"sharp",    "speed_limit":40,"risk":"high",   "alert_km":0.8},
    {"id":11, "name":"Gampola Road Merge",        "lat":7.2479,"lon":80.5269,"dist_km":17.0, "slope":2.6,  "curvature":"mild",     "speed_limit":50,"risk":"medium", "alert_km":0.4},
    {"id":12, "name":"Hingula Approach",          "lat":7.2437,"lon":80.5437,"dist_km":19.5, "slope":0.5,  "curvature":"mild",     "speed_limit":60,"risk":"low",    "alert_km":0.3},
    {"id":13, "name":"Hingula Junction",          "lat":7.2437,"lon":80.5437,"dist_km":22.0, "slope":0.0,  "curvature":"straight", "speed_limit":60,"risk":"low",    "alert_km":0.0},
]
TOTAL_ROUTE_KM = 22.0

# In-memory ride sessions {session_id: {...}}
_sessions: Dict[str, dict] = {}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

_model_cache: dict = {}

def load_model() -> object:
    if "model" in _model_cache:
        return _model_cache["model"]
    for name in ["xgb_model.pkl", "rf_tuned_final.pkl", "rf_model.pkl"]:
        path = MODEL_DIR / name
        if path.exists():
            try:
                with open(path, "rb") as f:
                    m = pickle.load(f)
                _model_cache["model"] = m
                _model_cache["name"]  = name
                log.info(f"Model loaded: {name}")
                return m
            except Exception as e:
                log.warning(f"Could not load {name}: {e}")
    log.warning("No pre-trained model found — training fallback RF from CSVs...")
    return _train_fallback_model()


def _train_fallback_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    X = pd.read_csv(DATA_DIR / "engineered_features.csv")
    y = pd.read_csv(DATA_DIR / "target_severity.csv").squeeze()
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_sm, y_sm = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_tr, y_tr)

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_sm, y_sm)
    _model_cache["model"] = rf
    _model_cache["name"]  = "fallback_rf (trained at startup)"
    log.info("Fallback RF trained and ready.")
    return rf

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

VEHICLE_RISK_WEIGHT = {
    "motorcycle": 4, "lorry": 4, "bus": 3, "car": 2, "three-wheeler": 3,
}
VEHICLE_TYPES = list(VEHICLE_RISK_WEIGHT.keys())

def build_feature_vector(req: "PredictionRequest") -> pd.DataFrame:
    vt   = req.vehicle_type
    hour = req.hour
    prcp = req.precipitation_mm
    vis  = req.visibility_km
    slope= req.road_slope_deg
    curv = req.road_curvature
    surf = req.surface_condition

    weather_enc   = {"clear":0,"light_rain":1,"mist":2,"foggy":3,"heavy_rain":4}.get(req.weather_condition, 0)
    road_curv_enc = {"straight":0,"mild":1,"sharp":2}.get(curv, 0)
    surface_enc   = {"dry":0,"wet":1,"very_wet":2,"oily":3}.get(surf, 0)
    vis_class     = 3 if vis<0.3 else (2 if vis<1.0 else (1 if vis<3.0 else 0))
    rain_intensity= 2 if prcp>5.0 else (1 if prcp>0.5 else 0)
    slope_cat     = 0 if slope<=3 else (1 if slope<=10 else 2)

    vehicle_risk_weight = VEHICLE_RISK_WEIGHT.get(vt, 2)
    is_heavy_vehicle = int(vt in ["bus","lorry"])
    is_two_wheeler   = int(vt in ["motorcycle","three-wheeler"])

    is_night        = int(hour < 6 or hour >= 20)
    is_morning_peak = int(hour in [7,8,9])
    is_evening_peak = int(hour in [16,17,18,19])
    hour_sin        = float(np.sin(2*np.pi*hour/24))
    hour_cos        = float(np.cos(2*np.pi*hour/24))
    month           = req.month
    is_monsoon      = int(month in [5,6,7,8,9])
    season_enc      = 2 if month in [5,6,7,8,9] else (1 if month in [10,11,12,1] else 0)

    is_sharp      = int(curv == "sharp")
    is_steep      = int(slope > 10)
    slope_x_curve = round(slope * road_curv_enc, 2)

    young_driver  = int(req.driver_age < 25)
    senior_driver = int(req.driver_age > 55)
    inexperienced = int(req.driving_exp_yrs < 3)
    exp_ratio     = round(req.driving_exp_yrs / max(1, req.driver_age - 17), 3)

    speed_ratio   = round(req.estimated_speed_kmh / max(1, req.speed_limit_kmh), 3)
    speed_excess  = max(0.0, req.estimated_speed_kmh - req.speed_limit_kmh)
    is_over_limit = int(req.estimated_speed_kmh > req.speed_limit_kmh)

    is_low_vis  = int(vis < 1.0)
    is_fog_risk = int(req.weather_condition in ["mist","foggy"])

    is_wet_surface  = int(surf in ["wet","very_wet","oily"])
    high_risk_combo = int(curv=="sharp" and slope>10 and prcp>1.0)
    wet_night       = int(is_wet_surface and is_night)
    lorry_steep     = int(vt in ["lorry","bus"] and slope>10)
    moto_rain       = int(vt in ["motorcycle","three-wheeler"] and prcp>1.0)
    vis_rain_prod   = float(vis_class * rain_intensity)
    speed_wet       = round(speed_excess * is_wet_surface, 1)
    rain_x_curve    = float(rain_intensity * road_curv_enc)
    vis_x_night     = float(vis_class * is_night)

    veh_bus           = int(vt=="bus")
    veh_car           = int(vt=="car")
    veh_lorry         = int(vt=="lorry")
    veh_motorcycle    = int(vt=="motorcycle")
    veh_three_wheeler = int(vt=="three-wheeler")

    acc_types = ["acc_fell_off_road","acc_head_on","acc_hit_barrier",
                 "acc_overturn","acc_pedestrian_hit","acc_rear_end",
                 "acc_sideswipe","acc_vehicle_fire"]

    wx_tavg = req.temp_avg_c; wx_tmin = req.temp_avg_c-3; wx_tmax = req.temp_avg_c+3
    wx_prcp = prcp; wx_wspd = req.wind_speed_kmh; wx_pres = req.pressure_hpa
    wx_temp_range = 6.0; wx_is_rainy = int(prcp>1.0); wx_heavy_rain = int(prcp>5.0)
    wx_rain_3day = prcp*1.5; wx_pressure_drop = 0.0

    feat = {
        "is_weekend": int(req.is_weekend), "hour": hour,
        "elevation_m": req.elevation_m, "num_vehicles": req.num_vehicles,
        "driver_age": req.driver_age, "driving_exp_yrs": req.driving_exp_yrs,
        "estimated_speed_kmh": req.estimated_speed_kmh,
        "speed_limit_kmh": req.speed_limit_kmh, "speeding": is_over_limit,
        "precipitation_mm": prcp, "temp_avg_c": req.temp_avg_c,
        "visibility_km": vis, "wind_speed_kmh": req.wind_speed_kmh,
        "road_slope_deg": slope,
        "wx_tavg": wx_tavg, "wx_tmin": wx_tmin, "wx_tmax": wx_tmax,
        "wx_prcp": wx_prcp, "wx_wspd": wx_wspd, "wx_pres": wx_pres,
        "wx_temp_range": wx_temp_range, "wx_is_rainy": wx_is_rainy,
        "wx_heavy_rain": wx_heavy_rain, "wx_rain_3day": wx_rain_3day,
        "wx_pressure_drop": wx_pressure_drop,
        "vehicle_risk_weight": vehicle_risk_weight,
        "is_heavy_vehicle": is_heavy_vehicle, "is_two_wheeler": is_two_wheeler,
        "veh_bus": veh_bus, "veh_car": veh_car, "veh_lorry": veh_lorry,
        "veh_motorcycle": veh_motorcycle, "veh_three-wheeler": veh_three_wheeler,
        "speed_ratio": speed_ratio, "speed_excess_kmh": speed_excess,
        "is_over_limit": is_over_limit,
        "hour_sin": hour_sin, "hour_cos": hour_cos, "is_night": is_night,
        "is_morning_peak": is_morning_peak, "is_evening_peak": is_evening_peak,
        "is_monsoon": is_monsoon, "season_enc": season_enc,
        "road_curv_enc": road_curv_enc, "surface_enc": surface_enc,
        "is_sharp": is_sharp, "slope_cat": float(slope_cat),
        "is_steep": is_steep, "slope_x_curve": slope_x_curve,
        **{k: 0 for k in acc_types},
        "weather_enc": weather_enc, "vis_class": float(vis_class),
        "rain_intensity": float(rain_intensity), "is_low_vis": is_low_vis,
        "is_fog_risk": is_fog_risk,
        "young_driver": young_driver, "senior_driver": senior_driver,
        "inexperienced": inexperienced, "exp_ratio": exp_ratio,
        "high_risk_combo": high_risk_combo, "wet_night": wet_night,
        "lorry_steep": lorry_steep, "moto_rain": moto_rain,
        "vis_rain_prod": vis_rain_prod, "speed_wet": speed_wet,
        "rain_x_curve": rain_x_curve, "vis_x_night": vis_x_night,
        "location_risk": 1.0,
    }

    try:
        expected = list(pd.read_csv(DATA_DIR / "engineered_features.csv", nrows=0).columns)
        for col in expected:
            if col not in feat:
                feat[col] = 0
        df = pd.DataFrame([feat])[expected]
    except FileNotFoundError:
        df = pd.DataFrame([feat])

    return df

# ══════════════════════════════════════════════════════════════════════════════
# ALERT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

ALERT_MESSAGES = {
    "motorcycle": {
        3: ("EXTREME DANGER",  "DO NOT PROCEED. Heavy rain + poor visibility is fatal for motorcyclists on mountain bends.", "#7F1D1D"),
        2: ("HIGH RISK",       "Extreme caution. Wet road + sharp curves ahead. Reduce speed to 20 km/h. Consider shelter.", "#991B1B"),
        1: ("MODERATE RISK",   "Slippery when wet. Widen following distance. Slow down on bends.", "#D97706"),
        0: ("LOW RISK",        "Conditions acceptable. Normal care required on sharp bends.", "#065F46"),
    },
    "lorry": {
        3: ("EXTREME DANGER",  "LORRIES PROHIBITED IN CURRENT CONDITIONS. Risk of jackknife and brake failure on descent.", "#7F1D1D"),
        2: ("HIGH RISK",       "Steep gradient + wet surface. Engage low gear. Do NOT use brakes on long descent.", "#991B1B"),
        1: ("MODERATE RISK",   "Reduce speed to 30 km/h on descent. Check brakes before Kadugannawa Pass.", "#D97706"),
        0: ("LOW RISK",        "Normal conditions. Use low gear on steep sections as per regulation.", "#065F46"),
    },
    "bus": {
        3: ("HIGH RISK",       "Passenger safety alert. Consider halting until rain reduces. Max 20 km/h.", "#991B1B"),
        2: ("MODERATE RISK",   "Reduce speed. Increase stopping distance. Alert passengers before sharp bends.", "#D97706"),
        1: ("CAUTION",         "Wet road advisory. Reduce speed 10 km/h below limit on curves.", "#B45309"),
        0: ("LOW RISK",        "Normal conditions. Maintain regulated speeds.", "#065F46"),
    },
    "car": {
        3: ("HIGH RISK",       "Do not drive unless essential. Flash flooding possible on A1 near Kadugannawa.", "#991B1B"),
        2: ("MODERATE RISK",   "Reduce speed. Use headlights. Avoid overtaking on curves.", "#D97706"),
        1: ("CAUTION",         "Light rain — slow down and increase following distance on bends.", "#B45309"),
        0: ("LOW RISK",        "Clear conditions. Normal care required on mountain curves.", "#065F46"),
    },
    "three-wheeler": {
        3: ("HIGH RISK",       "Three-wheelers MUST NOT attempt descent in heavy rain. Overturn risk is extreme.", "#991B1B"),
        2: ("MODERATE RISK",   "Avoid sharp curves at speed. Max 25 km/h. Consider alternate lower route.", "#D97706"),
        1: ("CAUTION",         "Reduce speed on bends. Three-wheelers have high tipping risk on wet road.", "#B45309"),
        0: ("LOW RISK",        "Normal conditions. Take curves slowly as standard.", "#065F46"),
    },
}

VEHICLE_RISK_WEIGHTS = {
    "motorcycle": 4, "lorry": 3, "bus": 3, "car": 1, "three-wheeler": 2,
}
SPEED_RECOMMENDATIONS = {0: None, 1: 40, 2: 25, 3: 0}

# Segment-specific alert descriptions (matched by segment name keywords)
SEGMENT_ALERTS = {
    "Sensation Rock": "Most dangerous bend on the pass. High fatality rate. Reduce speed drastically.",
    "Salgala":        "Steep downhill — brake fade risk. Engage low gear. No overtaking.",
    "Hairpin":        "Tight hairpin. Large vehicles must swing wide. Oncoming traffic risk.",
    "Tunnel":         "Visibility reduced near tunnel exit. Wet road likely from seepage.",
    "Town":           "Pedestrian zone. Bus stops ahead. Expect sudden stops.",
    "Junction":       "Merging traffic. Check mirrors. Sound horn on blind corners.",
    "Descent":        "Winding descent. Rain significantly increases slide risk.",
    "Balana":         "Winding descent toward Balana. Reduce speed well in advance.",
}


def compute_alert(vehicle_type: str, weather: dict, segment: dict | None = None) -> dict:
    base   = weather.get("weather_risk_score", 0)
    weight = VEHICLE_RISK_WEIGHTS.get(vehicle_type, 1)
    score  = min(3, base + (1 if weight >= 3 else 0))

    # Boost score for high-risk segments
    if segment and segment.get("risk") == "high":
        score = min(3, score + 1)
    elif segment and segment.get("risk") == "medium":
        score = min(3, score)

    msgs  = ALERT_MESSAGES.get(vehicle_type, ALERT_MESSAGES["car"])
    level, message, color = msgs[score]

    # Override message with segment-specific text if applicable
    if segment:
        seg_name = segment.get("name","")
        for key, desc in SEGMENT_ALERTS.items():
            if key.lower() in seg_name.lower():
                message = desc
                break

    return {
        "vehicle_type":          vehicle_type,
        "alert_level":           level,
        "alert_score":           score,
        "alert_color":           color,
        "message":               message,
        "recommended_speed_kmh": SPEED_RECOMMENDATIONS.get(score),
        "segment":               segment.get("name") if segment else None,
        "conditions_summary": (
            f"{weather.get('weather_condition','').replace('_',' ').title()} | "
            f"{weather.get('temperature_c',0):.0f}°C | "
            f"Rain {weather.get('precipitation_mm',0):.1f}mm | "
            f"Vis {weather.get('visibility_km',0):.1f}km | "
            f"Wind {weather.get('wind_speed_kmh',0):.0f}km/h"
        ),
        "timestamp": weather.get("timestamp", datetime.now().isoformat()),
    }

# ══════════════════════════════════════════════════════════════════════════════
# WEATHER CLIENT
# ══════════════════════════════════════════════════════════════════════════════

_weather_cache: dict = {"data": None, "ts": 0}
WEATHER_TTL = 600

def _wmo_to_condition(code: int) -> str:
    if code == 0:                        return "clear"
    if code in (1,2,3):                  return "clear"
    if code in (45,48):                  return "foggy"
    if code in (51,53,61,80,85):         return "light_rain"
    if code in (55,65,82,86,95,96,99):   return "heavy_rain"
    return "clear"


def fetch_current_weather() -> dict:
    now = time.time()
    if _weather_cache["data"] and now - _weather_cache["ts"] < WEATHER_TTL:
        return _weather_cache["data"]

    params = {
        "latitude": KADUGANNAWA["lat"], "longitude": KADUGANNAWA["lon"],
        "hourly": ",".join(HOURLY_VARS), "timezone": TIMEZONE,
        "forecast_days": 1,
    }
    try:
        r = requests.get(FORECAST_API, params=params, timeout=10,
                         headers={"User-Agent": "KaduGuard/2.0"})
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame(h)
        df["time"] = pd.to_datetime(df["time"])

        idx = (df["time"] - pd.Timestamp.now()).abs().idxmin()
        row = df.iloc[idx]

        vis_m  = row.get("visibility", 10000)
        vis_km = round(vis_m / 1000, 2)
        prcp   = float(row.get("precipitation", 0) or 0)
        wmo    = int(row.get("weather_code", 0) or 0)
        is_day = int(row.get("is_day", 1) or 1)
        hour_  = row["time"].hour

        condition = "foggy" if vis_km < 0.3 else ("mist" if vis_km < 1.0 else _wmo_to_condition(wmo))

        risk = (
            (3 if prcp > 5 else (1 if prcp > 0.5 else 0)) +
            (2 if vis_km < 0.5 else (1 if vis_km < 1.5 else 0)) +
            (1 if float(row.get("wind_gusts_10m", 0) or 0) > 50 else 0) +
            (1 if hour_ < 6 or hour_ >= 20 else 0)
        )

        data = {
            "timestamp":         row["time"].strftime("%Y-%m-%dT%H:%M"),
            "temperature_c":     round(float(row.get("temperature_2m", 26) or 26), 1),
            "humidity_pct":      int(row.get("relative_humidity_2m", 75) or 75),
            "precipitation_mm":  round(prcp, 2),
            "wind_speed_kmh":    round(float(row.get("wind_speed_10m", 0) or 0), 1),
            "wind_gusts_kmh":    round(float(row.get("wind_gusts_10m", 0) or 0), 1),
            "visibility_km":     vis_km,
            "pressure_hpa":      round(float(row.get("surface_pressure", 1010) or 1010), 1),
            "cloud_cover_pct":   int(row.get("cloud_cover", 0) or 0),
            "weather_condition": condition,
            "wmo_code":          wmo,
            "wmo_description":   WMO_CODES.get(wmo, "Unknown"),
            "weather_risk_score":min(6, risk),
            "is_day":            is_day,
        }
        _weather_cache["data"] = data
        _weather_cache["ts"]   = now
        return data

    except Exception as e:
        log.warning(f"Weather API unavailable: {e}. Using last cache or fallback.")
        if _weather_cache["data"]:
            return _weather_cache["data"]
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M"),
            "temperature_c": 26.0, "humidity_pct": 70, "precipitation_mm": 0.0,
            "wind_speed_kmh": 10.0, "wind_gusts_kmh": 15.0,
            "visibility_km": 8.0, "pressure_hpa": 1010.0, "cloud_cover_pct": 30,
            "weather_condition": "clear", "wmo_code": 0,
            "wmo_description": "Clear sky (offline fallback)",
            "weather_risk_score": 0, "is_day": 1,
        }


def fetch_forecast(days: int = 7) -> list[dict]:
    params = {
        "latitude": KADUGANNAWA["lat"], "longitude": KADUGANNAWA["lon"],
        "hourly": ",".join(HOURLY_VARS), "timezone": TIMEZONE,
        "forecast_days": min(days, 7),
    }
    try:
        r = requests.get(FORECAST_API, params=params, timeout=15,
                         headers={"User-Agent": "KaduGuard/2.0"})
        r.raise_for_status()
        h  = r.json()["hourly"]
        df = pd.DataFrame(h)
        df["time"] = pd.to_datetime(df["time"])

        rows = []
        for _, row in df.iterrows():
            prcp   = float(row.get("precipitation", 0) or 0)
            vis_m  = float(row.get("visibility", 10000) or 10000)
            vis_km = round(vis_m / 1000, 2)
            wmo    = int(row.get("weather_code", 0) or 0)
            gusts  = float(row.get("wind_gusts_10m", 0) or 0)
            hour_  = row["time"].hour
            risk = (
                (3 if prcp>5 else (1 if prcp>0.5 else 0)) +
                (2 if vis_km<0.5 else (1 if vis_km<1.5 else 0)) +
                (1 if gusts>50 else 0) +
                (1 if hour_<6 or hour_>=20 else 0)
            )
            rows.append({
                "datetime":         row["time"].strftime("%Y-%m-%dT%H:%M"),
                "date":             row["time"].strftime("%Y-%m-%d"),
                "hour":             hour_,
                "temperature_c":    round(float(row.get("temperature_2m",26) or 26), 1),
                "precipitation_mm": round(prcp, 2),
                "wind_speed_kmh":   round(float(row.get("wind_speed_10m",0) or 0), 1),
                "visibility_km":    vis_km,
                "weather_condition":"foggy" if vis_km<0.3 else ("mist" if vis_km<1.0 else _wmo_to_condition(wmo)),
                "wmo_description":  WMO_CODES.get(wmo, "Unknown"),
                "risk_score":       min(6, risk),
                "is_day":           int(row.get("is_day", 1) or 1),
            })
        return rows
    except Exception as e:
        log.warning(f"Forecast fetch failed: {e}")
        return []

# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PredictionRequest(BaseModel):
    vehicle_type:        str   = Field(..., example="motorcycle")
    hour:                int   = Field(14, ge=0, le=23)
    month:               int   = Field(6,  ge=1, le=12)
    is_weekend:          bool  = Field(False)
    driver_age:          float = Field(30.0, ge=16, le=80)
    driving_exp_yrs:     float = Field(5.0,  ge=0,  le=50)
    estimated_speed_kmh: float = Field(50.0, ge=0,  le=150)
    speed_limit_kmh:     int   = Field(50,   ge=20, le=100)
    num_vehicles:        int   = Field(1,    ge=1,  le=10)
    elevation_m:         int   = Field(478,  ge=400, le=700)
    road_curvature:      str   = Field("mild")
    road_slope_deg:      float = Field(5.0, ge=0, le=20)
    surface_condition:   str   = Field("dry")
    weather_condition:   str   = Field("clear")
    precipitation_mm:    float = Field(0.0, ge=0)
    temp_avg_c:          float = Field(26.0, ge=15, le=40)
    visibility_km:       float = Field(8.0,  ge=0.1, le=20)
    wind_speed_kmh:      float = Field(10.0, ge=0,  le=80)
    pressure_hpa:        float = Field(1010.0, ge=950, le=1040)
    use_live_weather:    bool  = Field(False)

    @validator("vehicle_type")
    def validate_vehicle(cls, v):
        if v not in VEHICLE_TYPES:
            raise ValueError(f"vehicle_type must be one of {VEHICLE_TYPES}")
        return v

    @validator("road_curvature")
    def validate_curvature(cls, v):
        if v not in ["straight","mild","sharp"]:
            raise ValueError("road_curvature must be straight|mild|sharp")
        return v

    @validator("surface_condition")
    def validate_surface(cls, v):
        if v not in ["dry","wet","very_wet","oily"]:
            raise ValueError("surface_condition must be dry|wet|very_wet|oily")
        return v


class SegmentPredictionRequest(BaseModel):
    """Predict risk for a specific named route segment."""
    vehicle_type:        str   = Field(..., example="car")
    segment_id:          int   = Field(..., ge=1, le=13, description="Route segment ID (1–13)")
    estimated_speed_kmh: float = Field(50.0, ge=0, le=150)
    driver_age:          float = Field(30.0, ge=16, le=80)
    driving_exp_yrs:     float = Field(5.0,  ge=0,  le=50)
    use_live_weather:    bool  = Field(True)
    # Optional manual weather override
    precipitation_mm:    float = Field(0.0, ge=0)
    visibility_km:       float = Field(8.0, ge=0.1, le=20)


class RideSessionStart(BaseModel):
    vehicle_type: str  = Field(..., example="car")
    driver_age:   float= Field(30.0)
    driver_exp:   float= Field(5.0)


class RidePositionUpdate(BaseModel):
    session_id:   str
    dist_km:      float = Field(..., ge=0, le=TOTAL_ROUTE_KM)
    speed_kmh:    float = Field(..., ge=0, le=200)
    lat:          Optional[float] = None
    lon:          Optional[float] = None


class PredictionResponse(BaseModel):
    severity:          str
    severity_code:     int
    confidence:        float
    probabilities:     dict
    alert:             dict
    weather_used:      dict
    model_name:        str
    inference_time_ms: float
    timestamp:         str

# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="KaduGuard — Real-Time Road Risk Prediction & Warning System",
    description=(
        "Predictive safety navigator for Kadugannawa Pass (A1 Highway, Sri Lanka). "
        "Combines ML model inference with live Open-Meteo weather, GPS tracking, "
        "and per-segment proactive hazard alerts for drivers."
    ),
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    log.info("Loading ML model...")
    load_model()
    log.info("Warming up weather cache...")
    try:
        fetch_current_weather()
    except Exception:
        pass
    log.info(f"KaduGuard Navigation System v{VERSION} ready.")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "model": _model_cache.get("name", "not loaded"),
        "active_sessions": len(_sessions),
        "weather_cache_age_s": round(time.time() - _weather_cache["ts"], 0),
        "timestamp": datetime.now().isoformat(),
    }


# ── Route segments ─────────────────────────────────────────────────────────────
@app.get("/route/segments", tags=["Route"])
def get_route_segments():
    """
    Return all Kadugannawa->Hingula route segments with static risk data.
    Includes GPS coordinates, speed limits, curvature, slope, and risk level.
    """
    weather = fetch_current_weather()
    enriched = []
    for seg in ROUTE_SEGMENTS:
        alert = compute_alert("car", weather, seg)   # reference vehicle: car
        enriched.append({
            **seg,
            "weather_alert_level": alert["alert_level"],
            "weather_risk_score":  weather["weather_risk_score"],
        })
    return {
        "route": "Kadugannawa -> Hingula",
        "total_km": TOTAL_ROUTE_KM,
        "segment_count": len(ROUTE_SEGMENTS),
        "weather": weather,
        "segments": enriched,
    }


# ── Predict risk for a specific segment ──────────────────────────────────────
@app.post("/predict/segment", tags=["Prediction"])
def predict_segment(req: SegmentPredictionRequest):
    """
    Run ML severity prediction for a specific named route segment.
    Uses live weather by default.  Returns vehicle-specific alert + speed recommendation.
    """
    seg = next((s for s in ROUTE_SEGMENTS if s["id"] == req.segment_id), None)
    if not seg:
        raise HTTPException(404, f"Segment {req.segment_id} not found")

    weather = fetch_current_weather()
    prcp = weather["precipitation_mm"] if req.use_live_weather else req.precipitation_mm
    vis  = weather["visibility_km"]    if req.use_live_weather else req.visibility_km
    surf = "wet" if prcp > 1.0 else "dry"

    pred_req = PredictionRequest(
        vehicle_type        = req.vehicle_type,
        hour                = datetime.now().hour,
        month               = datetime.now().month,
        is_weekend          = datetime.now().weekday() >= 5,
        driver_age          = req.driver_age,
        driving_exp_yrs     = req.driving_exp_yrs,
        estimated_speed_kmh = req.estimated_speed_kmh,
        speed_limit_kmh     = seg["speed_limit"],
        num_vehicles        = 1,
        elevation_m         = 530,
        road_curvature      = seg["curvature"],
        road_slope_deg      = seg["slope"],
        surface_condition   = surf,
        weather_condition   = weather["weather_condition"],
        precipitation_mm    = prcp,
        temp_avg_c          = weather["temperature_c"],
        visibility_km       = vis,
        wind_speed_kmh      = weather["wind_speed_kmh"],
        pressure_hpa        = weather["pressure_hpa"],
        use_live_weather    = False,
    )

    t0 = time.perf_counter()
    X  = build_feature_vector(pred_req)
    model = load_model()
    proba = model.predict_proba(X)[0]
    pred_cls = int(np.argmax(proba))
    elapsed  = round((time.perf_counter()-t0)*1000, 2)

    alert_weather = {
        "weather_condition":  pred_req.weather_condition,
        "temperature_c":      pred_req.temp_avg_c,
        "precipitation_mm":   pred_req.precipitation_mm,
        "visibility_km":      pred_req.visibility_km,
        "wind_speed_kmh":     pred_req.wind_speed_kmh,
        "weather_risk_score": min(6, pred_cls * 2),
        "timestamp":          datetime.now().strftime("%Y-%m-%dT%H:%M"),
    }
    alert = compute_alert(req.vehicle_type, alert_weather, seg)

    return {
        "segment":            seg,
        "severity":           SEVERITY_NAMES[pred_cls],
        "severity_code":      pred_cls,
        "confidence":         round(float(proba[pred_cls]), 4),
        "probabilities":      {"Low": round(float(proba[0]),4), "Medium": round(float(proba[1]),4), "High": round(float(proba[2]),4)},
        "alert":              alert,
        "weather_used":       weather,
        "model_name":         _model_cache.get("name","unknown"),
        "inference_time_ms":  elapsed,
        "timestamp":          datetime.now().isoformat(),
    }


# ── Ride session management ───────────────────────────────────────────────────
@app.post("/route/session/start", tags=["Navigation"])
def start_session(req: RideSessionStart):
    """
    Start a new ride session. Returns session_id to use in position updates.
    """
    if req.vehicle_type not in VEHICLE_TYPES:
        raise HTTPException(400, f"vehicle_type must be one of {VEHICLE_TYPES}")

    sid = str(uuid.uuid4())[:8]
    weather = fetch_current_weather()
    _sessions[sid] = {
        "session_id":    sid,
        "vehicle_type":  req.vehicle_type,
        "driver_age":    req.driver_age,
        "driver_exp":    req.driver_exp,
        "started_at":    datetime.now().isoformat(),
        "current_dist":  0.0,
        "current_speed": 0.0,
        "alert_count":   0,
        "alerts_fired":  set(),
        "weather_start": weather,
        "status":        "active",
    }
    log.info(f"Session {sid} started: {req.vehicle_type}")
    return {"session_id": sid, "route": "Kadugannawa -> Hingula", "total_km": TOTAL_ROUTE_KM, "weather": weather}


@app.post("/route/session/update", tags=["Navigation"])
def update_position(req: RidePositionUpdate):
    """
    Update vehicle position. Returns upcoming hazard alerts if within warning distance.
    This is the core real-time endpoint called by the navigation app.
    """
    sid = req.session_id
    if sid not in _sessions:
        raise HTTPException(404, f"Session {sid} not found")

    sess = _sessions[sid]
    sess["current_dist"]  = req.dist_km
    sess["current_speed"] = req.speed_kmh
    vt = sess["vehicle_type"]

    weather = fetch_current_weather()
    upcoming_alerts = []

    for seg in ROUTE_SEGMENTS:
        ahead = seg["dist_km"] - req.dist_km
        if seg["id"] in sess["alerts_fired"]:
            continue
        if 0 < ahead <= seg["alert_km"]:
            # Run ML prediction for this segment
            prcp = weather["precipitation_mm"]
            surf = "wet" if prcp > 1.0 else "dry"
            pred_req = PredictionRequest(
                vehicle_type=vt, hour=datetime.now().hour,
                month=datetime.now().month, is_weekend=datetime.now().weekday()>=5,
                driver_age=sess["driver_age"], driving_exp_yrs=sess["driver_exp"],
                estimated_speed_kmh=req.speed_kmh, speed_limit_kmh=seg["speed_limit"],
                num_vehicles=1, elevation_m=530,
                road_curvature=seg["curvature"], road_slope_deg=seg["slope"],
                surface_condition=surf, weather_condition=weather["weather_condition"],
                precipitation_mm=prcp, temp_avg_c=weather["temperature_c"],
                visibility_km=weather["visibility_km"],
                wind_speed_kmh=weather["wind_speed_kmh"],
                pressure_hpa=weather["pressure_hpa"], use_live_weather=False,
            )
            try:
                X = build_feature_vector(pred_req)
                proba = load_model().predict_proba(X)[0]
                pred_cls = int(np.argmax(proba))
            except Exception:
                pred_cls = 1

            alert_weather = {
                **weather,
                "weather_risk_score": min(6, pred_cls * 2),
            }
            alert = compute_alert(vt, alert_weather, seg)

            upcoming_alerts.append({
                "segment_id":   seg["id"],
                "segment_name": seg["name"],
                "dist_ahead_km":round(ahead, 2),
                "severity":     SEVERITY_NAMES[pred_cls],
                "alert":        alert,
            })
            sess["alerts_fired"].add(seg["id"])
            sess["alert_count"] += 1

    # Current segment
    cur_seg = next((s for s in reversed(ROUTE_SEGMENTS) if s["dist_km"] <= req.dist_km + 0.1), ROUTE_SEGMENTS[0])
    pct = round((req.dist_km / TOTAL_ROUTE_KM) * 100, 1)
    eta = round(((TOTAL_ROUTE_KM - req.dist_km) / req.speed_kmh * 60)) if req.speed_kmh > 0 else None

    return {
        "session_id":      sid,
        "dist_km":         req.dist_km,
        "pct_complete":    pct,
        "eta_minutes":     eta,
        "current_segment": cur_seg["name"],
        "current_speed":   req.speed_kmh,
        "speed_limit":     cur_seg["speed_limit"],
        "speeding":        req.speed_kmh > cur_seg["speed_limit"],
        "upcoming_alerts": upcoming_alerts,
        "alert_count":     sess["alert_count"],
        "weather":         weather,
    }


@app.get("/route/session/{session_id}", tags=["Navigation"])
def get_session(session_id: str):
    """Get current state of a ride session."""
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    sess = _sessions[session_id].copy()
    sess.pop("alerts_fired", None)
    return sess


# ── Current weather ────────────────────────────────────────────────────────────
@app.get("/weather/current", tags=["Weather"])
def current_weather():
    return fetch_current_weather()


# ── 7-day forecast ─────────────────────────────────────────────────────────────
@app.get("/weather/forecast", tags=["Weather"])
def forecast(days: int = 7):
    rows = fetch_forecast(days)
    if not rows:
        raise HTTPException(503, "Weather forecast unavailable")
    return {"count": len(rows), "forecast": rows}


# ── Vehicle alert (weather-only, no ML) ───────────────────────────────────────
@app.get("/alert/{vehicle_type}", tags=["Alerts"])
def vehicle_alert(vehicle_type: str):
    if vehicle_type not in VEHICLE_TYPES:
        raise HTTPException(400, f"vehicle_type must be one of {VEHICLE_TYPES}")
    weather = fetch_current_weather()
    return compute_alert(vehicle_type, weather)


# ── All-vehicle alert summary ─────────────────────────────────────────────────
@app.get("/alert", tags=["Alerts"])
def all_alerts():
    weather = fetch_current_weather()
    return {
        "weather": weather,
        "alerts":  {vt: compute_alert(vt, weather) for vt in VEHICLE_TYPES},
    }


# ── Full ML prediction (original endpoint, preserved) ─────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: PredictionRequest):
    """
    Full ML-based accident severity prediction.
    Set `use_live_weather=true` to pull current Open-Meteo conditions automatically.
    """
    t0 = time.perf_counter()
    weather_used = {}
    if req.use_live_weather:
        live = fetch_current_weather()
        req.precipitation_mm  = live["precipitation_mm"]
        req.temp_avg_c        = live["temperature_c"]
        req.visibility_km     = live["visibility_km"]
        req.wind_speed_kmh    = live["wind_speed_kmh"]
        req.pressure_hpa      = live["pressure_hpa"]
        req.weather_condition = live["weather_condition"]
        req.hour              = datetime.now().hour
        weather_used          = live
    else:
        weather_used = {
            "source": "manual input",
            "precipitation_mm": req.precipitation_mm, "temp_avg_c": req.temp_avg_c,
            "visibility_km": req.visibility_km, "wind_speed_kmh": req.wind_speed_kmh,
            "weather_condition": req.weather_condition,
        }

    try:
        X = build_feature_vector(req)
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Feature engineering failed: {e}")

    model = load_model()
    try:
        proba    = model.predict_proba(X)[0]
        pred_cls = int(np.argmax(proba))
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Model inference failed: {e}")

    severity_label = SEVERITY_NAMES[pred_cls]
    alert_weather = {
        "weather_condition":  req.weather_condition,
        "temperature_c":      req.temp_avg_c,
        "precipitation_mm":   req.precipitation_mm,
        "visibility_km":      req.visibility_km,
        "wind_speed_kmh":     req.wind_speed_kmh,
        "weather_risk_score": min(6, pred_cls * 2),
        "timestamp":          datetime.now().strftime("%Y-%m-%dT%H:%M"),
    }
    alert   = compute_alert(req.vehicle_type, alert_weather)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    return PredictionResponse(
        severity          = severity_label,
        severity_code     = pred_cls,
        confidence        = round(float(proba[pred_cls]), 4),
        probabilities     = {"Low": round(float(proba[0]),4), "Medium": round(float(proba[1]),4), "High": round(float(proba[2]),4)},
        alert             = alert,
        weather_used      = weather_used,
        model_name        = _model_cache.get("name","unknown"),
        inference_time_ms = elapsed,
        timestamp         = datetime.now().isoformat(),
    )


# ── Driver navigation dashboard ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["UI"])
def dashboard():
    """
    Serve the KaduGuard navigation dashboard.
    Uses Leaflet.js with CartoDB dark tiles for real map rendering.
    Features: GPS tracking, predictive hazard alerts, simulation mode.
    """
    for name in ["dashboard.html"]:
        html_path = Path(name)
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("""
    <html><body style="background:#07090f;color:#e2e8f5;font-family:sans-serif;padding:2rem">
    <h2>⛰ KaduGuard — Dashboard not found</h2>
    <p>Place <code>dashboard.html</code> in the same directory as <code>main.py</code>.</p>
    <p>API is running — try <a href="/docs" style="color:#f0a500">/docs</a> for Swagger UI.</p>
    </body></html>
    """)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")