"""
Simple telemetry simulator for development.
Sends POSTs to /api/telemetry with randomized OBD-like fields.

Usage:
    python scripts/telemetry_simulator.py --count 10 --interval 1
"""
import requests
import time
import uuid
import random
from datetime import datetime

API_URL = "http://localhost:8000/api/telemetry"


def make_payload(device_id=None):
    if device_id is None:
        device_id = str(uuid.uuid4())
    lat = 7.251 + random.uniform(-0.01, 0.01)
    lon = 80.533 + random.uniform(-0.01, 0.01)
    return {
        "device_id": device_id,
        "vehicle_type": random.choice(["car", "motorcycle", "bus", "lorry", "three-wheeler"]),
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "speed_kmh": round(random.uniform(0, 80), 1),
        "heading_deg": round(random.uniform(0, 359), 1),
        "accel_m_s2": round(random.uniform(-3, 3), 2),
        "rpm": random.randint(500, 3500),
        "timestamp": datetime.utcnow().isoformat(),
    }


def run(count=10, interval=1.0):
    did = str(uuid.uuid4())
    for i in range(count):
        p = make_payload(device_id=did)
        try:
            r = requests.post(API_URL, json=p, timeout=5)
            print(i, r.status_code, r.json())
        except Exception as e:
            print("Error sending telemetry:", e)
        time.sleep(interval)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    run(count=args.count, interval=args.interval)
