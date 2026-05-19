"""
backend/api/mobile.py
Mobile-friendly endpoints for the Android app.
"""
import csv
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.api.auth import get_current_user
from backend.config.database import get_db
from backend.config.settings import BASE_DIR
from backend.models.db_models import Device, Trip, User
from backend.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    RiskZone,
    RiskZonesResponse,
    TripEndRequest,
    TripEndResponse,
    TripHistoryResponse,
    TripStartRequest,
    TripStartResponse,
    TripSummary,
)
from backend.services.model_service import ModelService, get_alert_message, get_recommended_speed
from backend.services.notification_service import NotificationService

router = APIRouter(prefix="/api/mobile", tags=["Mobile"])

SEGMENTS_CSV = BASE_DIR / "data" / "raw" / "kadugannawa_road_segments.csv"


@router.post("/predict", response_model=PredictionResponse, summary="Mobile prediction endpoint")
async def predict_mobile(payload: PredictionRequest, db: Session = Depends(get_db)) -> PredictionResponse:
    svc = ModelService.get_instance()
    data = payload.model_dump(exclude_none=True)

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
    message = get_alert_message(risk_label)

    if payload.device_id and risk_label == "High":
        device = db.query(Device).filter(Device.device_id == payload.device_id, Device.is_active.is_(True)).first()
        if device and device.fcm_token:
            notification_service = NotificationService.get_instance()
            notification_service.send_device_alert(
                token=device.fcm_token,
                title="High risk detected",
                body=message,
                risk_level=risk_label,
                recommended_speed_kmh=speed,
            )

    return PredictionResponse(
        risk_level=risk_label,
        risk_score=round(confidence, 4),
        probabilities=prob_dict,
        vehicle_type=payload.vehicle_type,
        recommended_speed_kmh=speed,
        alert_message=message,
    )


@router.post("/trip/start", response_model=TripStartResponse, summary="Start a trip")
def start_trip(
    payload: TripStartRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TripStartResponse:
    trip_id = str(uuid4())
    trip = Trip(
        trip_id=trip_id,
        driver_id=current_user.driver_id,
        start_time=datetime.utcnow(),
        start_lat=payload.latitude,
        start_lon=payload.longitude,
        vehicle_type=payload.vehicle_type,
    )
    db.add(trip)
    db.commit()
    return TripStartResponse(trip_id=trip_id, status="ok", message="Trip started")


@router.post("/trip/end", response_model=TripEndResponse, summary="End a trip")
def end_trip(
    payload: TripEndRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TripEndResponse:
    trip = (
        db.query(Trip)
        .filter(Trip.trip_id == payload.trip_id, Trip.driver_id == current_user.driver_id)
        .first()
    )
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trip not found")

    trip.end_time = datetime.utcnow()
    trip.end_lat = payload.latitude
    trip.end_lon = payload.longitude
    if payload.distance_km is not None:
        trip.distance_km = payload.distance_km
    if payload.duration_minutes is not None:
        trip.duration_minutes = payload.duration_minutes

    db.commit()

    return TripEndResponse(
        trip_id=payload.trip_id,
        status="ok",
        avg_risk_score=trip.avg_risk_score,
        max_risk_score=trip.max_risk_score,
        high_risk_count=trip.high_risk_count,
    )


@router.get("/trip-summary", response_model=TripHistoryResponse, summary="Get trip history summary")
def trip_summary(
    days: int = Query(7, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TripHistoryResponse:
    cutoff = datetime.utcnow() - timedelta(days=days)
    trips = (
        db.query(Trip)
        .filter(Trip.driver_id == current_user.driver_id, Trip.start_time >= cutoff)
        .order_by(Trip.start_time.desc())
        .all()
    )

    summaries = [
        TripSummary(
            trip_id=trip.trip_id,
            start_time=trip.start_time,
            end_time=trip.end_time,
            start_lat=trip.start_lat,
            start_lon=trip.start_lon,
            end_lat=trip.end_lat,
            end_lon=trip.end_lon,
            distance_km=trip.distance_km or 0.0,
            duration_minutes=trip.duration_minutes,
            avg_risk_score=trip.avg_risk_score or 0.0,
            max_risk_score=trip.max_risk_score or 0.0,
            high_risk_count=trip.high_risk_count or 0,
            medium_risk_count=trip.medium_risk_count or 0,
            low_risk_count=trip.low_risk_count or 0,
            vehicle_type=trip.vehicle_type,
        )
        for trip in trips
    ]

    total_distance = sum(item.distance_km for item in summaries)
    avg_risk = (sum(item.avg_risk_score for item in summaries) / len(summaries)) if summaries else 0.0

    return TripHistoryResponse(
        trips=summaries,
        total_trips=len(summaries),
        avg_risk=round(avg_risk, 4),
        total_distance_km=round(total_distance, 3),
    )


@router.get("/risk-zones", response_model=RiskZonesResponse, summary="Get road risk zones")
def risk_zones() -> RiskZonesResponse:
    if not SEGMENTS_CSV.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Road segments data not found")

    zones: list[RiskZone] = []
    with open(SEGMENTS_CSV, newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            zones.append(
                RiskZone(
                    segment_id=row.get("segment_id", ""),
                    name=row.get("name", ""),
                    start_lat=float(row.get("lat_start", 0)),
                    start_lon=float(row.get("lon_start", 0)),
                    end_lat=float(row.get("lat_end", 0)),
                    end_lon=float(row.get("lon_end", 0)),
                    gradient_pct=float(row.get("gradient_pct", 0)),
                    curvature=float(row.get("curvature", 0)),
                    base_risk_level=row.get("base_risk", "Low"),
                )
            )

    return RiskZonesResponse(zones=zones, count=len(zones))