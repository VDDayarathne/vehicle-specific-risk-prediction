"""
backend/api/telemetry.py
POST /api/telemetry  — ingestion endpoint for device telemetry
"""
from fastapi import APIRouter
from backend.models.schemas import TelemetryRequest, TelemetryResponse
from backend.config.settings import BASE_DIR
import pathlib
import csv
from datetime import datetime

router = APIRouter(prefix="/api", tags=["Telemetry"])

RAW_DIR = BASE_DIR / "data" / "raw" / "telemetry"
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV = RAW_DIR / "telemetry_raw.csv"


@router.post("/telemetry", response_model=TelemetryResponse, summary="Ingest device telemetry")
async def ingest_telemetry(payload: TelemetryRequest) -> TelemetryResponse:
    now = datetime.utcnow().isoformat()
    ts = payload.timestamp or now

    row = {
        "device_id": payload.device_id,
        "vehicle_type": payload.vehicle_type,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "speed_kmh": payload.speed_kmh,
        "heading_deg": payload.heading_deg,
        "accel_m_s2": payload.accel_m_s2,
        "rpm": payload.rpm,
        "timestamp": ts,
        "received_at": now,
    }

    # Append to CSV (create header if missing)
    write_header = not RAW_CSV.exists()
    with open(RAW_CSV, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return TelemetryResponse(status="ok", stored=True, message="Telemetry stored")
