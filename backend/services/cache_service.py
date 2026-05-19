"""
backend/services/cache_service.py
Simple file-based cache for recent weather and small artifacts.
"""
import json
from pathlib import Path
from backend.config.settings import BASE_DIR
from threading import Lock

_CACHE_DIR = BASE_DIR / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_WEATHER_CACHE = _CACHE_DIR / "weather_cache.json"
_MODEL_CACHE_DIR = _CACHE_DIR / "models"
_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MODEL_CACHE_PATH = _MODEL_CACHE_DIR / "rf_tuned_final.pkl"
_DEVICE_CACHE = _CACHE_DIR / "device_context.json"
_lock = Lock()


def write_weather_cache(key: str, payload: dict) -> None:
    """Write or update a weather payload under `key` (e.g., 'lat:lon')."""
    with _lock:
        data = {}
        if _WEATHER_CACHE.exists():
            try:
                data = json.loads(_WEATHER_CACHE.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data[key] = payload
        _WEATHER_CACHE.write_text(json.dumps(data), encoding="utf-8")


def read_weather_cache(key: str) -> dict | None:
    """Return cached payload for key, or None."""
    if not _WEATHER_CACHE.exists():
        return None
    with _lock:
        try:
            data = json.loads(_WEATHER_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return data.get(key)


def write_model_cache(src_path: str) -> None:
    """Copy model file into cache dir for offline use."""
    try:
        import shutil
        src = Path(src_path)
        if src.exists():
            shutil.copy(src, _MODEL_CACHE_PATH)
    except Exception:
        pass


def read_model_cache() -> str | None:
    """Return path to cached model if exists, else None."""
    if _MODEL_CACHE_PATH.exists():
        return str(_MODEL_CACHE_PATH)
    return None


def write_device_context(device_id: str, payload: dict) -> None:
    """Persist last known GPS/telemetry context for a device."""
    if not device_id:
        return
    with _lock:
        data = {}
        if _DEVICE_CACHE.exists():
            try:
                data = json.loads(_DEVICE_CACHE.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data[device_id] = payload
        _DEVICE_CACHE.write_text(json.dumps(data), encoding="utf-8")


def read_device_context(device_id: str) -> dict | None:
    """Read last known payload for a device."""
    if not device_id or not _DEVICE_CACHE.exists():
        return None
    with _lock:
        try:
            data = json.loads(_DEVICE_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return data.get(device_id)
