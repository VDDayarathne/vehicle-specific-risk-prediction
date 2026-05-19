"""
osm_road_geometry.py
====================
Scrape OpenStreetMap road geometry for Kadugannawa Pass (A1 highway)
and compute curvature + slope features for ML accident prediction.

Requirements:
    pip install osmnx geopandas shapely requests numpy pandas

Usage:
    python osm_road_geometry.py
    
Output:
    kadugannawa_road_segments.csv  — one row per OSM way segment
    kadugannawa_road_nodes.csv     — one row per GPS node (for mapping)

Author: Generated for Vehicle Alert System research project
"""

import math
import warnings
import numpy as np
import pandas as pd
import requests

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.accident_schema import normalize_accident_frame

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
BBOX = {
    "north": 7.270,
    "south": 7.230,
    "east":  80.550,
    "west":  80.510,
}

# Overpass API endpoints (tried in order — use first that responds)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Highway types to include
HIGHWAY_FILTER = ["trunk", "primary", "secondary", "tertiary", "trunk_link"]


# ── Step 1: Query Overpass API ─────────────────────────────────────────────────
def build_overpass_query(bbox: dict, highway_types: list) -> str:
    """Build Overpass QL query for all road ways + their nodes in the bbox."""
    hw_filter = "|".join(highway_types)
    return f"""
[out:json][timeout:60];
(
  way["highway"~"^({hw_filter})$"]
    ({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
);
out body geom;
"""


def fetch_osm_data(bbox: dict, highway_types: list) -> dict:
    """Try each Overpass endpoint until one succeeds."""
    query = build_overpass_query(bbox, highway_types)

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"Trying {endpoint} ...")
            resp = requests.post(
                endpoint,
                data={"data": query},
                timeout=60,
                headers={"User-Agent": "AccidentPredictionResearch/1.0"},
            )
            if resp.status_code == 200:
                data = resp.json()
                print(f"  [OK] Got {len(data['elements'])} OSM elements")
                return data
            else:
                print(f"  [X] HTTP {resp.status_code}")
        except Exception as e:
            print(f"  [X] {e}")

    raise RuntimeError("All Overpass endpoints failed. Try later or use osmnx.")


# ── Step 2: Parse OSM response into node list per way ─────────────────────────
def parse_ways(data: dict) -> list[dict]:
    """Extract way metadata and GPS node sequences from Overpass response."""
    ways = []
    for el in data["elements"]:
        if el["type"] != "way":
            continue

        tags = el.get("tags", {})
        geometry = el.get("geometry", [])

        if len(geometry) < 2:
            continue  # need at least 2 points to compute anything

        nodes = [(pt["lat"], pt["lon"]) for pt in geometry]

        # Parse maxspeed (OSM stores as "50", "50 mph", "LK:urban", etc.)
        raw_speed = tags.get("maxspeed", "")
        speed_kmh = parse_speed(raw_speed)

        ways.append({
            "osm_way_id":   el["id"],
            "road_name":    tags.get("name", tags.get("ref", "Unnamed")),
            "highway_type": tags.get("highway", ""),
            "speed_limit_kmh": speed_kmh,
            "surface":      tags.get("surface", "unknown"),
            "lanes":        int(tags.get("lanes", 1)),
            "oneway":       1 if tags.get("oneway") == "yes" else 0,
            "nodes":        nodes,
        })
    return ways


def parse_speed(raw: str) -> int:
    """Convert OSM maxspeed tag to integer km/h."""
    if not raw:
        return 50  # Sri Lanka default urban/mountain
    raw = raw.strip().lower()
    if raw in ("lk:urban", "urban"):   return 50
    if raw in ("lk:rural", "rural"):   return 70
    if raw in ("lk:trunk",):           return 70
    try:
        if "mph" in raw:
            return int(float(raw.replace("mph", "").strip()) * 1.609)
        return int(float(raw.split()[0]))
    except ValueError:
        return 50


# ── Step 3: Geometry calculations ─────────────────────────────────────────────
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS points in metres."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth (0–360°) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlam))
    return math.degrees(math.atan2(x, y)) % 360


def bearing_change(b1: float, b2: float) -> float:
    """Smallest angle between two bearings (0–180°)."""
    diff = abs(b2 - b1) % 360
    return diff if diff <= 180 else 360 - diff


def compute_curvature_metrics(nodes: list[tuple]) -> dict:
    """
    Compute three curvature metrics from a node sequence:
      - max_bearing_change_deg : sharpest single turn (degrees)
      - total_bearing_change_deg: sum of all turns (tortuosity)
      - curvature_per_km       : total_bearing / length (deg/km)
      - road_curvature         : classified label
    """
    if len(nodes) < 2:
        return {"max_bearing_change_deg": 0, "total_bearing_change_deg": 0,
                "curvature_per_km": 0, "road_curvature": "straight"}

    segment_lengths = []
    bearings = []
    for i in range(len(nodes) - 1):
        segment_lengths.append(haversine_m(*nodes[i], *nodes[i + 1]))
        bearings.append(bearing(*nodes[i], *nodes[i + 1]))

    total_length_m = sum(segment_lengths)

    turn_angles = []
    for i in range(len(bearings) - 1):
        turn_angles.append(bearing_change(bearings[i], bearings[i + 1]))

    max_turn   = max(turn_angles) if turn_angles else 0.0
    total_turn = sum(turn_angles) if turn_angles else 0.0
    curv_per_km = (total_turn / (total_length_m / 1000)) if total_length_m > 0 else 0

    # Classification thresholds (calibrated for mountain roads)
    if max_turn < 15:
        label = "straight"
    elif max_turn < 45:
        label = "mild"
    else:
        label = "sharp"

    return {
        "max_bearing_change_deg":   round(max_turn, 1),
        "total_bearing_change_deg": round(total_turn, 1),
        "curvature_per_km":         round(curv_per_km, 1),
        "road_curvature":           label,
    }


# ── Step 4: Elevation & slope from OpenTopoData (SRTM 30m) ───────────────────
def get_elevations_batch(lats: list, lons: list,
                         dataset: str = "srtm30m") -> list[float | None]:
    """
    Query OpenTopoData API for elevation at multiple points.
    Free tier: 100 points/request, 1 req/s.
    API docs: https://www.opentopodata.org/
    """
    points_str = "|".join(f"{la},{lo}" for la, lo in zip(lats, lons))
    url = f"https://api.opentopodata.org/v1/{dataset}?locations={points_str}"
    try:
        r = requests.get(url, timeout=30,
                         headers={"User-Agent": "AccidentPredictionResearch/1.0"})
        if r.status_code == 200:
            results = r.json().get("results", [])
            return [res.get("elevation") for res in results]
    except Exception as e:
        print(f"  Elevation API error: {e}")
    return [None] * len(lats)


def compute_slope_deg(elev_start: float, elev_end: float,
                      dist_m: float) -> float:
    """Slope angle in degrees from elevation change and horizontal distance."""
    if dist_m == 0 or elev_start is None or elev_end is None:
        return 0.0
    rise = abs(elev_end - elev_start)
    return round(math.degrees(math.atan(rise / dist_m)), 2)


# ── Step 5: Assemble feature rows ─────────────────────────────────────────────
def ways_to_dataframe(ways: list[dict],
                      fetch_elevation: bool = True) -> pd.DataFrame:
    """
    Convert parsed OSM ways to a flat DataFrame with all ML features.
    Set fetch_elevation=False to skip the SRTM API call (faster, for testing).
    """
    rows = []
    all_start_lats, all_start_lons = [], []
    all_end_lats,   all_end_lons   = [], []

    for w in ways:
        nodes = w["nodes"]
        all_start_lats.append(nodes[0][0]);  all_start_lons.append(nodes[0][1])
        all_end_lats.append(nodes[-1][0]);   all_end_lons.append(nodes[-1][1])

    # Batch elevation fetch (start + end points)
    start_elevs = end_elevs = [None] * len(ways)
    if fetch_elevation:
        print("Fetching start-point elevations from OpenTopoData...")
        start_elevs = get_elevations_batch(all_start_lats, all_start_lons)
        print("Fetching end-point elevations from OpenTopoData...")
        end_elevs   = get_elevations_batch(all_end_lats, all_end_lons)

    for i, w in enumerate(ways):
        nodes  = w["nodes"]
        curv   = compute_curvature_metrics(nodes)

        # Total segment length
        length_m = sum(haversine_m(*nodes[j], *nodes[j+1])
                       for j in range(len(nodes) - 1))

        # Midpoint
        mid_lat = np.mean([n[0] for n in nodes])
        mid_lon = np.mean([n[1] for n in nodes])

        # Slope
        s_elev  = start_elevs[i]
        e_elev  = end_elevs[i]
        slope   = compute_slope_deg(s_elev, e_elev, length_m)

        rows.append({
            "osm_way_id":              w["osm_way_id"],
            "road_name":               w["road_name"],
            "highway_type":            w["highway_type"],
            "speed_limit_kmh":         w["speed_limit_kmh"],
            "surface":                 w["surface"],
            "lanes":                   w["lanes"],
            "oneway":                  w["oneway"],
            "length_m":                round(length_m, 1),
            "num_nodes":               len(nodes),
            "start_lat":               nodes[0][0],
            "start_lon":               nodes[0][1],
            "end_lat":                 nodes[-1][0],
            "end_lon":                 nodes[-1][1],
            "midpoint_lat":            round(mid_lat, 6),
            "midpoint_lon":            round(mid_lon, 6),
            "start_elevation_m":       s_elev,
            "end_elevation_m":         e_elev,
            "road_slope_deg":          slope,
            **curv,
        })

    return pd.DataFrame(rows)


# ── Step 6: Spatial join with accident data ────────────────────────────────────
def join_accidents_to_roads(accidents_csv: str,
                             roads_df: pd.DataFrame,
                             radius_m: float = 200.0) -> pd.DataFrame:
    """
    Match each accident GPS point to the nearest OSM road segment.
    Adds road_curvature, road_slope_deg, speed_limit_kmh to accident rows.
    
    radius_m: max distance to consider a match (200m is reasonable for A1)
    """
    accidents = pd.read_csv(accidents_csv)
    accidents = normalize_accident_frame(accidents)

    # For each accident, find closest road midpoint
    road_curvatures = []
    road_slopes     = []
    road_speed_lims = []
    road_names      = []
    dist_to_road    = []

    for _, acc in accidents.iterrows():
        a_lat, a_lon = acc["latitude"], acc["longitude"]
        min_dist = float("inf")
        best_row  = roads_df.iloc[0]  # fallback

        for _, road in roads_df.iterrows():
            d = haversine_m(a_lat, a_lon,
                            road["midpoint_lat"], road["midpoint_lon"])
            if d < min_dist:
                min_dist = d
                best_row = road

        road_curvatures.append(best_row["road_curvature"] if min_dist < radius_m else "unknown")
        road_slopes.append(best_row["road_slope_deg"]     if min_dist < radius_m else np.nan)
        road_speed_lims.append(best_row["speed_limit_kmh"] if min_dist < radius_m else np.nan)
        road_names.append(best_row["road_name"]            if min_dist < radius_m else "unknown")
        dist_to_road.append(round(min_dist, 1))

    accidents["osm_road_name"]      = road_names
    accidents["osm_road_curvature"] = road_curvatures
    accidents["osm_road_slope_deg"] = road_slopes
    accidents["osm_speed_limit"]    = road_speed_lims
    accidents["dist_to_road_m"]     = dist_to_road

    return accidents


# ── Main pipeline ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  OSM Road Geometry Pipeline — Kadugannawa Pass")
    print("=" * 55)

    # 1. Fetch from Overpass API
    osm_data = fetch_osm_data(BBOX, HIGHWAY_FILTER)

    # 2. Parse
    ways = parse_ways(osm_data)
    print(f"\nParsed {len(ways)} road segments")

    # 3. Compute geometry + elevation
    roads_df = ways_to_dataframe(ways, fetch_elevation=True)

    import pathlib
    _root = pathlib.Path(__file__).resolve().parents[2]
    out_roads = str(_root / "data" / "raw" / "kadugannawa_road_segments.csv")
    roads_df.to_csv(out_roads, index=False)
    print(f"\nSaved: {out_roads} ({len(roads_df)} rows, {len(roads_df.columns)} cols)")
    print(roads_df[["road_name","road_curvature","road_slope_deg",
                     "length_m","speed_limit_kmh"]].to_string())

    # 5. Join with accident data
    print("\nJoining with accident data...")
    merged = join_accidents_to_roads(
        str(_root / "data" / "raw" / "kadugannawa_accidents_mock.csv"),
        roads_df
    )
    merged.to_csv(_root / "data" / "processed" / "accidents_with_osm_features.csv", index=False)
    print(f"Saved: data/processed/accidents_with_osm_features.csv")
    print("\nCurvature match rate:",
          (merged["osm_road_curvature"] != "unknown").mean().round(3))

    print("\nDone.")