"""
backend/api/predict.py
POST /api/predict  — vehicle-specific accident risk prediction endpoint.
"""
from fastapi import APIRouter, HTTPException

from backend.models.schemas import PredictionRequest, PredictionResponse
from backend.services.model_service import (
    ModelService,
    get_recommended_speed,
    get_alert_message,
)
from backend.services.cache_service import write_device_context, read_device_context

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse, summary="Predict accident risk")
async def predict_risk(payload: PredictionRequest) -> PredictionResponse:
    """
    Accepts road + weather + vehicle parameters and returns
    a Low / Medium / High risk level with confidence score.
    """
    svc = ModelService.get_instance()

    data = payload.model_dump(exclude_none=True)

    # Merge cached device context when caller sends only partial telemetry/GPS.
    if payload.device_id:
        cached = read_device_context(payload.device_id)
        if cached:
            for key, value in cached.items():
                data.setdefault(key, value)

    # Persist last known context so the next partial request can be completed.
    if payload.device_id:
        write_device_context(payload.device_id, {
            key: data.get(key)
            for key in ["latitude", "longitude", "speed_kmh", "heading_deg", "accel_m_s2", "rpm", "vehicle_type"]
            if data.get(key) is not None
        })

    # Keep downstream model input compatible with the current model schema.
    data.setdefault("temperature_c", 25.0)
    data.setdefault("humidity_pct", 70.0)
    data.setdefault("rainfall_mm", 0.0)
    data.setdefault("wind_speed_kmh", 0.0)
    data.setdefault("visibility_km", 10.0)
    data.setdefault("gradient_pct", 0.0)
    data.setdefault("curvature", 0.0)
    data.setdefault("road_surface", "asphalt")
    data.setdefault("lane_count", 2)
    data.setdefault("hour_of_day", 12)
    data.setdefault("day_of_week", 0)
    data.setdefault("is_holiday", False)
    data.setdefault("traffic_density", 0.5)

    risk_label, confidence, prob_dict = svc.predict(data)

    speed = get_recommended_speed(risk_label, payload.vehicle_type)
    msg   = get_alert_message(risk_label)

    return PredictionResponse(
        risk_level=risk_label,
        risk_score=round(confidence, 4),
        probabilities=prob_dict,
        vehicle_type=payload.vehicle_type,
        recommended_speed_kmh=speed,
        alert_message=msg,
    )


@router.get("/alert/{vehicle_type}", summary="Lightweight alert preview for a vehicle type")
async def alert_preview(vehicle_type: str):
    """Return a lightweight alert preview used by the dashboard UI.

    This endpoint uses the same ModelService prediction logic but with
    conservative defaults so the frontend can request a quick alert
    without sending a full prediction payload.
    """
    svc = ModelService.get_instance()

    # Defaults kept in sync with predict_risk defaults
    data = {
        "vehicle_type": vehicle_type,
        "temperature_c": 25.0,
        "humidity_pct": 70.0,
        "rainfall_mm": 0.0,
        "wind_speed_kmh": 0.0,
        "visibility_km": 10.0,
        "gradient_pct": 0.0,
        "curvature": 0.0,
        "road_surface": "asphalt",
        "lane_count": 2,
        "hour_of_day": 12,
        "day_of_week": 0,
        "is_holiday": False,
        "traffic_density": 0.5,
    }

    risk_label, confidence, _ = svc.predict(data)
    recommended = get_recommended_speed(risk_label, vehicle_type)
    message = get_alert_message(risk_label)

    # Small numeric alert score expected by the dashboard (0=low,1=medium,2=high)
    score_map = {"Low": 0, "Medium": 1, "High": 2}
    alert_score = score_map.get(risk_label, 0)

    return {
        "alert_level": risk_label.upper(),
        "alert_score": alert_score,
        "message": message,
        "recommended_speed_kmh": recommended,
    }
