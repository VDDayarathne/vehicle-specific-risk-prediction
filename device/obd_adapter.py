"""
device/obd_adapter.py
Best-effort OBD-II adapter with a deterministic simulator fallback.

If python-obd is installed and a compatible adapter is connected, this module
can read live values. Otherwise it produces realistic hill-road telemetry for
development, demos, and offline testing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import random
from typing import Optional


@dataclass
class TelemetrySample:
    device_id: str
    vehicle_type: str
    latitude: float
    longitude: float
    speed_kmh: float
    heading_deg: float
    accel_m_s2: float
    rpm: int
    timestamp: str

    def as_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "vehicle_type": self.vehicle_type,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "speed_kmh": self.speed_kmh,
            "heading_deg": self.heading_deg,
            "accel_m_s2": self.accel_m_s2,
            "rpm": self.rpm,
            "timestamp": self.timestamp,
        }


class OBDAdapter:
    """Read live OBD telemetry or simulate it when hardware is unavailable."""

    def __init__(self, device_id: str, vehicle_type: str = "car", use_simulator: bool = True) -> None:
        self.device_id = device_id
        self.vehicle_type = vehicle_type
        self.use_simulator = use_simulator
        self._obd = None
        self._sim_phase = random.random() * math.tau

        if not use_simulator:
            self._obd = self._connect_obd()

    def _connect_obd(self):
        try:
            import obd  # type: ignore

            return obd.OBD()
        except Exception:
            return None

    def read(self) -> TelemetrySample:
        """Return one telemetry sample."""
        live = self._read_live() if self._obd is not None else None
        if live is not None:
          return live
        return self._simulate()

    def _read_live(self) -> Optional[TelemetrySample]:
        try:
            # OBD reads are optional and device-specific; degrade gracefully.
            speed = self._get_obd_value("SPEED", default=0.0)
            rpm = int(self._get_obd_value("RPM", default=900.0))
            accel = self._estimate_accel(speed)
            return TelemetrySample(
                device_id=self.device_id,
                vehicle_type=self.vehicle_type,
                latitude=7.2500,
                longitude=80.5300,
                speed_kmh=round(speed, 1),
                heading_deg=round((speed * 3.1) % 360, 1),
                accel_m_s2=round(accel, 2),
                rpm=rpm,
                timestamp=datetime.utcnow().isoformat(),
            )
        except Exception:
            return None

    def _get_obd_value(self, pid_name: str, default: float = 0.0) -> float:
        try:
            cmd = getattr(__import__("obd").commands, pid_name)
            response = self._obd.query(cmd)
            if response.is_null():
                return default
            value = response.value
            return float(value.magnitude) if hasattr(value, "magnitude") else float(value)
        except Exception:
            return default

    def _estimate_accel(self, speed_kmh: float) -> float:
        base = math.sin(self._sim_phase + speed_kmh / 35.0) * 0.9
        return base + random.uniform(-0.25, 0.25)

    def _simulate(self) -> TelemetrySample:
        self._sim_phase += 0.3
        speed = 32 + 18 * math.sin(self._sim_phase) + random.uniform(-2.5, 2.5)
        speed = max(0.0, min(90.0, speed))
        accel = self._estimate_accel(speed)
        rpm = int(900 + speed * 45 + random.uniform(-120, 120))
        lat = 7.2500 + 0.01 * math.sin(self._sim_phase / 2)
        lon = 80.5300 + 0.01 * math.cos(self._sim_phase / 2)
        heading = (self._sim_phase * 37) % 360
        return TelemetrySample(
            device_id=self.device_id,
            vehicle_type=self.vehicle_type,
            latitude=round(lat, 6),
            longitude=round(lon, 6),
            speed_kmh=round(speed, 1),
            heading_deg=round(heading, 1),
            accel_m_s2=round(accel, 2),
            rpm=max(500, rpm),
            timestamp=datetime.utcnow().isoformat(),
        )
