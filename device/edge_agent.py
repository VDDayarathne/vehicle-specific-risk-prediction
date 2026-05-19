"""
device/edge_agent.py
Small edge runtime that reads telemetry and posts it to the backend.

Features:
  - OBD-II adapter when available
  - deterministic simulator fallback
  - offline JSONL queue for failed uploads
  - simple retry loop for queued samples

Usage:
  python device/edge_agent.py --backend http://localhost:8000 --vehicle car --count 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from device.obd_adapter import OBDAdapter


QUEUE_DIR = Path(__file__).resolve().parent / "queue"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_FILE = QUEUE_DIR / "telemetry_queue.jsonl"


def enqueue(sample: dict) -> None:
    with open(QUEUE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")


def drain_queue(backend_url: str) -> None:
    if not QUEUE_FILE.exists():
        return

    lines = QUEUE_FILE.read_text(encoding="utf-8").splitlines()
    if not lines:
        return

    remaining = []
    for raw in lines:
        if not raw.strip():
            continue
        payload = json.loads(raw)
        try:
            r = requests.post(f"{backend_url.rstrip('/')}/api/telemetry", json=payload, timeout=8)
            if r.status_code >= 400:
                remaining.append(raw)
        except Exception:
            remaining.append(raw)

    if remaining:
        QUEUE_FILE.write_text("\n".join(remaining) + "\n", encoding="utf-8")
    else:
        QUEUE_FILE.unlink(missing_ok=True)


def post_sample(backend_url: str, payload: dict) -> bool:
    try:
        r = requests.post(f"{backend_url.rstrip('/')}/api/telemetry", json=payload, timeout=8)
        return r.ok
    except Exception:
        return False


def run(backend_url: str, vehicle_type: str, device_id: str, count: int, interval: float, force_simulator: bool) -> None:
    adapter = OBDAdapter(device_id=device_id, vehicle_type=vehicle_type, use_simulator=force_simulator)

    for _ in range(count):
        sample = adapter.read().as_dict()
        if not post_sample(backend_url, sample):
            enqueue(sample)
        drain_queue(backend_url)
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="KaduGuard edge telemetry agent")
    parser.add_argument("--backend", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--vehicle", default="car", choices=["car", "motorcycle", "bus", "lorry", "three-wheeler"])
    parser.add_argument("--device-id", default="edge-demo-device")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--simulator", action="store_true", help="Force simulated telemetry")
    args = parser.parse_args()

    run(
        backend_url=args.backend,
        vehicle_type=args.vehicle,
        device_id=args.device_id,
        count=args.count,
        interval=args.interval,
        force_simulator=args.simulator,
    )


if __name__ == "__main__":
    main()
