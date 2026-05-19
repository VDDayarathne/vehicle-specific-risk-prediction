"""
backend/api/segments.py
GET /api/segments  — returns road segment risk data for the map overlay.
"""
import csv
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from backend.config.settings import BASE_DIR

router = APIRouter(prefix="/api", tags=["Segments"])

SEGMENTS_CSV = BASE_DIR / "data" / "raw" / "kadugannawa_road_segments.csv"


class RoadSegment(BaseModel):
    segment_id:  str
    name:        str
    lat_start:   float
    lon_start:   float
    lat_end:     float
    lon_end:     float
    gradient_pct: float
    curvature:   float
    base_risk:   str   # Low | Medium | High


@router.get("/segments", response_model=List[RoadSegment], summary="Road segments with base risk")
async def get_segments() -> List[RoadSegment]:
    """Returns all Kadugannawa road segments with pre-computed base-risk levels."""
    if not SEGMENTS_CSV.exists():
        raise HTTPException(status_code=404, detail="Road segments data not found.")

    segments: List[RoadSegment] = []
    try:
        with open(SEGMENTS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                segments.append(RoadSegment(
                    segment_id=row.get("segment_id", ""),
                    name=row.get("name", ""),
                    lat_start=float(row.get("lat_start", 0)),
                    lon_start=float(row.get("lon_start", 0)),
                    lat_end=float(row.get("lat_end", 0)),
                    lon_end=float(row.get("lon_end", 0)),
                    gradient_pct=float(row.get("gradient_pct", 0)),
                    curvature=float(row.get("curvature", 0)),
                    base_risk=row.get("base_risk", "Low"),
                ))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read segments: {exc}")

    return segments
