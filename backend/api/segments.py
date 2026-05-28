"""
backend/api/segments.py
GET /api/segments  — returns road segment risk data for the map overlay.
"""
import csv
import json
import random
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from backend.config.settings import BASE_DIR

router = APIRouter(prefix="/api", tags=["Segments"])

SEGMENTS_CSV = BASE_DIR / "data" / "raw" / "kadugannawa_road_segments.csv"
ROUTE_DETAILS_JSON = BASE_DIR / "data" / "processed" / "route_details.json"



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


def classify_bend(angle, dist_m):
    if angle > 80:
        risk = 'high'; curv = 'sharp'; lim = 40; slope = 10.0
    elif angle > 45:
        risk = 'medium'; curv = 'sharp'; lim = 45; slope = 7.0
    elif angle > 30:
        risk = 'medium'; curv = 'mild'; lim = 50; slope = 4.0
    else:
        risk = 'low'; curv = 'mild'; lim = 55; slope = 2.0
    if dist_m < 2000:
        slope += 4.0
        risk = 'high'; lim = 40
    elif dist_m < 4000:
        slope += 2.0
        if risk == 'low': risk = 'medium'
    return risk, curv, lim, slope


NAMES = [
    'Kadugannawa Pass Entry – Blind Hairpin',
    'Upper Pass – Sharp Right Bend',
    'Upper Pass – Sharp Left Bend',
    'Pahala Kadugannawa – Main Hairpin',
    'Pahala Kadugannawa – U-Turn Bend',
    'Lower Pass – Counter Curve',
    'Descent Curve A',
    'Salgala – Steep Descent Bend A',
    'Salgala – Steep Descent Bend B',
    'Salgala – River Curve',
    'Salgala – Mid Bend',
    'Salgala Lower – Curve',
    'Ganethenna – Upper Hairpin',
    'Ganethenna – Sharp Left',
    'Ganethenna – Double Curve A',
    'Ganethenna – Double Curve B',
    'Ganethenna – Tight Hairpin',
    'Ganethenna – Lower Curve',
    'Hingula – Long Bend',
    'Hingula – Junction Curve',
    'Hingula – Descent Bend',
    'Pre-Mawanella – Upper Curve',
    'Pre-Mawanella – Sweeping Bend',
    'Pre-Mawanella – Final Hairpin',
    'Mawanella Approach – Entry Bend',
    'Mawanella – Town Entry Curve',
]

DESCS = {
    'high': [
        'Extremely sharp blind bend. Reduce to 30 km/h before the curve. Sound horn. No overtaking.',
        'Steep descent with tight curve ahead. Brake early. Rollover risk for lorries and buses.',
        'Critical hairpin bend. Large vehicles must swing wide. Oncoming traffic risk is very high.',
        'Blind corner on steep gradient. Stay far left. Lorries engage low gear well in advance.',
    ],
    'medium': [
        'Sharp curve ahead. Reduce speed to 40 km/h. Stay in lane. Visibility is restricted.',
        'Winding section with multiple curves. Keep left. No overtaking allowed on this stretch.',
        'Moderately sharp bend. Rain significantly increases slide risk. Reduce speed.',
        'Hidden curve after straight section. Do NOT accelerate before bend is fully visible.',
    ],
    'low': [
        'Gentle curve ahead. Maintain safe speed. Stay alert for unexpected secondary bends.',
        'Mild bend. Keep steady speed. Watch for pedestrians near roadside stalls.',
        'Sweeping bend. Maintain lane discipline. Watch for merging vehicles from side roads.',
    ]
}


@router.get("/route/segments", summary="High-resolution route segments and path")
async def get_route_segments():
    """Returns the full 308-point high-resolution curved road path and 26 bend warning segments."""
    if not ROUTE_DETAILS_JSON.exists():
        raise HTTPException(status_code=404, detail="Route details JSON not found.")

    try:
        with open(ROUTE_DETAILS_JSON, "r", encoding="utf-8") as f:
            d = json.load(f)

        bends = d.get('bends', [])
        road_path = d.get('road_path', [])
        total_m = d.get('total_distance_m', 0.0)

        # Seed random to ensure deterministic descriptions
        rng = random.Random(42)

        segments = []
        for i, b in enumerate(bends):
            risk, curv, lim, slope = classify_bend(b['angle_diff'], b['dist_m'])
            name = NAMES[i] if i < len(NAMES) else f"Bend {i+1}"
            desc = rng.choice(DESCS[risk])
            dist_km = round(b['dist_m'] / 1000, 2)
            angle = round(b['angle_diff'], 1)

            segments.append({
                "id": i + 1,
                "name": name,
                "lat": b['lat'],
                "lon": b['lon'],
                "dist": dist_km,
                "slope": slope,
                "curv": curv,
                "lim": lim,
                "risk": risk,
                "desc": desc,
                "angle": angle
            })

        return {
            "segments": segments,
            "total_km": round(total_m / 1000, 1),
            "road_path": road_path
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process route segments: {exc}")

