"""
backend/config/settings.py
Application configuration loaded from environment variables.
"""
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR = BASE_DIR / "models" / "saved_models"
PIPELINE_DIR = BASE_DIR / "models" / "pipeline_objects"

MODEL_PATH       = MODELS_DIR / "rf_tuned_final.pkl"
SCALER_PATH      = PIPELINE_DIR / "scaler.pkl"
KNN_IMP_PATH     = PIPELINE_DIR / "knn_imp.pkl"
MED_IMP_PATH     = PIPELINE_DIR / "med_imp.pkl"
FEATURES_PATH    = BASE_DIR / "data" / "processed" / "feature_names.csv"

# ── External API keys (set in .env) ───────────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
GOOGLE_MAPS_API_KEY  = os.getenv("GOOGLE_MAPS_API_KEY", "")

# ── OpenMeteo (no key required) ───────────────────────────────────────────────
OPENMETEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# ── Kadugannawa fixed coordinates ────────────────────────────────────────────
KADUGANNAWA_LAT = 7.2500
KADUGANNAWA_LON = 80.5333

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {0: "#2ed573", 1: "#ff8c42", 2: "#ff4757"}

# ── Server ────────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── Database (PostgreSQL) ──────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kaduguard:kaduguard@db:5432/kaduguard"
)

# ── JWT Authentication ────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-12345")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))

# ── Firebase Cloud Messaging (FCM) ────────────────────────────────────────────
FIREBASE_SERVICE_ACCOUNT_KEY = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY", "")  # Path or JSON string
FCM_DEFAULT_TOPIC = os.getenv("FCM_DEFAULT_TOPIC", "kaduguard-alerts")
