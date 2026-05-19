"""
backend/services/model_service.py
Loads the trained ML model and provides prediction logic.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from backend.config.settings import (
    MODEL_PATH, SCALER_PATH, KNN_IMP_PATH, FEATURES_PATH,
    RISK_LABELS, RISK_COLORS
)
from backend.services.cache_service import read_weather_cache, write_model_cache, read_model_cache
from pathlib import Path

# ── Vehicle-type encoding ─────────────────────────────────────────────────────
VEHICLE_ENCODING = {
    "car":           0,
    "motorcycle":    1,
    "bus":           2,
    "lorry":         3,
    "three-wheeler": 4,
}

# ── Recommended speeds per risk & vehicle ─────────────────────────────────────
SPEED_MAP: Dict[str, Dict[str, int]] = {
    "Low":    {"car": 60, "motorcycle": 50, "bus": 50, "lorry": 45, "three-wheeler": 40},
    "Medium": {"car": 40, "motorcycle": 30, "bus": 35, "lorry": 30, "three-wheeler": 25},
    "High":   {"car": 20, "motorcycle": 15, "bus": 20, "lorry": 15, "three-wheeler": 10},
}

ALERT_MESSAGES: Dict[str, str] = {
    "Low":    "Conditions are acceptable. Maintain safe following distance on hill sections.",
    "Medium": "Moderate risk detected. Reduce speed and watch for sharp bends.",
    "High":   "HIGH RISK — Hazardous conditions ahead. Slow down immediately and consider stopping.",
}


class ModelService:
    """Singleton ML model service."""

    _instance: "ModelService | None" = None

    def __init__(self) -> None:
        self.model   = None
        self.scaler  = None
        self.feature_names: list[str] = []
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Load model and pre-processing artifacts from disk."""
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            print(f"[ModelService] Model loaded from {MODEL_PATH}")
            # cache the model for offline fallback
            try:
                write_model_cache(str(MODEL_PATH))
            except Exception:
                pass
        except FileNotFoundError:
            # try cached model
            cached = read_model_cache()
            if cached:
                try:
                    with open(cached, "rb") as f:
                        self.model = pickle.load(f)
                    print(f"[ModelService] Loaded cached model from {cached}")
                except Exception:
                    print(f"[ModelService] WARNING: cached model load failed")
            else:
                print(f"[ModelService] WARNING: model not found at {MODEL_PATH}")
                return

        try:
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError:
            print("[ModelService] WARNING: scaler not found — predictions will use raw values")

        if FEATURES_PATH.exists():
            df = pd.read_csv(FEATURES_PATH)
            self.feature_names = df.iloc[:, 0].tolist()

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, data: dict) -> Tuple[str, float, dict]:
        """
        Returns (risk_label, confidence, class_probabilities).
        Falls back to a rule-based estimate if model is not loaded.
        """
        if not self._loaded:
            return self._rule_based_fallback(data)

        X = self._build_feature_vector(data)

        if self.scaler is not None:
            try:
                if hasattr(self.scaler, "n_features_in_") and self.scaler.n_features_in_ == X.shape[1]:
                    X = self.scaler.transform(X)
                else:
                    print(
                        f"[ModelService] WARNING: scaler expects {getattr(self.scaler, 'n_features_in_', 'unknown')} features, "
                        f"but prediction input has {X.shape[1]}; skipping scaling"
                    )
            except Exception as exc:
                print(f"[ModelService] WARNING: scaler transform failed; using raw features ({exc})")

        expected_features = getattr(self.model, "n_features_in_", X.shape[1])
        if X.shape[1] != expected_features:
            if X.shape[1] < expected_features:
                pad = expected_features - X.shape[1]
                X = np.pad(X, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
                print(
                    f"[ModelService] WARNING: padded prediction vector from {X.shape[1] - pad} to {expected_features} features"
                )
            else:
                X = X[:, :expected_features]
                print(
                    f"[ModelService] WARNING: trimmed prediction vector to {expected_features} features"
                )

        probs  = self.model.predict_proba(X)[0]
        cls_id = int(np.argmax(probs))
        label  = RISK_LABELS[cls_id]
        conf   = float(probs[cls_id])

        prob_dict = {RISK_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)}
        return label, conf, prob_dict

    # ── Feature engineering ───────────────────────────────────────────────────

    def _build_feature_vector(self, d: dict) -> np.ndarray:
        veh_enc   = VEHICLE_ENCODING.get(d["vehicle_type"], 0)
        road_enc  = {"asphalt": 0, "gravel": 1, "dirt": 2}.get(d.get("road_surface", "asphalt"), 0)

        row = {
            "vehicle_type_enc":  veh_enc,
            "temperature_c":     d["temperature_c"],
            "humidity_pct":      d["humidity_pct"],
            "rainfall_mm":       d["rainfall_mm"],
            "wind_speed_kmh":    d["wind_speed_kmh"],
            "visibility_km":     d["visibility_km"],
            "gradient_pct":      d["gradient_pct"],
            "curvature":         d["curvature"],
            "road_surface_enc":  road_enc,
            "lane_count":        d.get("lane_count", 2),
            "hour_of_day":       d["hour_of_day"],
            "day_of_week":       d["day_of_week"],
            "is_holiday":        int(d.get("is_holiday", False)),
            "traffic_density":   d.get("traffic_density", 0.5),
            # Derived features
            "road_wetness_index":  min(d["rainfall_mm"] / 50.0, 1.0),
            "visibility_score":    1.0 - min(d["visibility_km"] / 10.0, 1.0),
            "gradient_severity":   abs(d["gradient_pct"]) / 30.0,
            "curvature_risk":      d["curvature"],
            "is_night":            int(d["hour_of_day"] < 6 or d["hour_of_day"] >= 20),
            "is_peak_hour":        int(d["hour_of_day"] in range(7, 10) or d["hour_of_day"] in range(16, 20)),
        }

        if self.feature_names:
            ordered = [row.get(f, 0.0) for f in self.feature_names]
            return np.array([ordered])

        return np.array([list(row.values())])

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _rule_based_fallback(self, d: dict) -> Tuple[str, float, dict]:
        # If weather fields missing, try reading cached weather by lat:lon
        if (d.get("temperature_c") is None or d.get("visibility_km") is None) and d.get("latitude") and d.get("longitude"):
            key = f"{float(d['latitude']):.6f}:{float(d['longitude']):.6f}"
            cached = read_weather_cache(key)
            if cached:
                # populate any missing keys with cached values
                d.setdefault("temperature_c", cached.get("temperature_c", 25.0))
                d.setdefault("humidity_pct", cached.get("humidity_pct", 70.0))
                d.setdefault("rainfall_mm", cached.get("rainfall_mm", 0.0))
                d.setdefault("wind_speed_kmh", cached.get("wind_speed_kmh", 0.0))
                d.setdefault("visibility_km", cached.get("visibility_km", 10.0))

        score = 0.0
        score += min(d.get("rainfall_mm", 0) / 50.0, 1.0) * 0.3
        score += min(abs(d.get("gradient_pct", 0)) / 20.0, 1.0) * 0.25
        score += d.get("curvature", 0) * 0.2
        score += (1.0 - min(d.get("visibility_km", 10) / 10.0, 1.0)) * 0.15
        score += d.get("traffic_density", 0) * 0.1

        if score < 0.33:
            label = "Low"
        elif score < 0.66:
            label = "Medium"
        else:
            label = "High"

        return label, round(score, 4), {"Low": 0.0, "Medium": 0.0, "High": 0.0, label: 1.0}


def get_recommended_speed(risk_label: str, vehicle_type: str) -> int:
    return SPEED_MAP.get(risk_label, {}).get(vehicle_type, 40)


def get_alert_message(risk_label: str) -> str:
    return ALERT_MESSAGES.get(risk_label, "")
