"""
backend/models/schemas.py
Pydantic request/response schemas for the API.

Schemas are organized by feature:
  - Auth: Register, Login, RefreshToken
  - User: UserProfile, UpdateProfile
  - Trip: TripStart, TripEnd, TripSummary
  - Prediction: PredictionRequest, PredictionResponse
  - Telemetry: TelemetryRequest, TelemetryResponse
  - Weather: WeatherResponse
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime


# ── Prediction Request ────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    vehicle_type: str = Field(
        ...,
        description="Vehicle type: car | motorcycle | bus | lorry | three-wheeler",
        examples=["car"]
    )
    # Weather inputs
    temperature_c: float   = Field(..., ge=-10, le=50,  description="Air temperature (°C)")
    humidity_pct:  float   = Field(..., ge=0,   le=100, description="Relative humidity (%)")
    rainfall_mm:   float   = Field(..., ge=0,   le=500, description="Rainfall in mm")
    wind_speed_kmh: float  = Field(..., ge=0,   le=200, description="Wind speed (km/h)")
    visibility_km: float   = Field(..., ge=0,   le=50,  description="Visibility (km)")

    # Road inputs
    gradient_pct:  float   = Field(..., ge=-30, le=30,  description="Road gradient (%)")
    curvature:     float   = Field(..., ge=0,   le=1,   description="Curvature index 0–1")
    road_surface:  str     = Field("asphalt", description="asphalt | gravel | dirt")
    lane_count:    int     = Field(2, ge=1, le=6,       description="Number of lanes")

    # Temporal inputs
    hour_of_day:   int     = Field(..., ge=0, le=23,    description="Hour (0–23)")
    day_of_week:   int     = Field(..., ge=0, le=6,     description="Day 0=Mon … 6=Sun")
    is_holiday:    bool    = Field(False,               description="Sri Lankan public holiday")

    # Traffic
    traffic_density: float = Field(0.5, ge=0, le=1,    description="Traffic density 0–1")

    # Optional telemetry/GPS context
    device_id: Optional[str] = Field(None, description="Optional hashed device identifier")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Optional GPS latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Optional GPS longitude")
    speed_kmh: Optional[float] = Field(None, ge=0, description="Optional current speed from telemetry")
    heading_deg: Optional[float] = Field(None, ge=0, le=360, description="Optional heading from telemetry")
    accel_m_s2: Optional[float] = Field(None, description="Optional acceleration from telemetry")
    rpm: Optional[int] = Field(None, description="Optional engine RPM from telemetry")


# ── Prediction Response ───────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    risk_level:    str   = Field(..., description="Low | Medium | High")
    risk_score:    float = Field(..., description="Probability of the predicted class (0–1)")
    probabilities: dict  = Field(..., description="Class probabilities {Low, Medium, High}")
    vehicle_type:  str
    recommended_speed_kmh: Optional[int] = None
    alert_message: str


# ── Weather Response ─────────────────────────────────────────────────────────

class WeatherResponse(BaseModel):
    temperature_c:  float
    humidity_pct:   float
    rainfall_mm:    float
    wind_speed_kmh: float
    visibility_km:  float
    description:    str
    icon:           str


# ── Telemetry Request / Response ───────────────────────────────────────────
class TelemetryRequest(BaseModel):
    device_id: str = Field(..., description="Hashed device identifier")
    vehicle_type: str = Field(..., description="Vehicle type: car | motorcycle | bus | lorry | three-wheeler")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    speed_kmh: Optional[float] = Field(None, ge=0)
    heading_deg: Optional[float] = Field(None, ge=0, le=360)
    accel_m_s2: Optional[float] = Field(None, description="Approx lateral/longitudinal accel")
    rpm: Optional[int] = None
    timestamp: Optional[str] = Field(None, description="ISO8601 timestamp; server will use now if missing")


class TelemetryResponse(BaseModel):
    status: str
    stored: bool
    message: Optional[str] = None


# ── Health Response ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    version:     str = "1.0.0"


# ── Authentication Requests & Responses ────────────────────────────────────────

class RegisterRequest(BaseModel):
    """Request body for driver registration."""
    phone: Optional[str] = Field(None, min_length=10, max_length=20, description="Phone number")
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=6, max_length=128, description="Password (min 6 chars)")
    vehicle_type: str = Field(default="car", description="Vehicle type: car | motorcycle | bus | lorry | three-wheeler")
    

class LoginRequest(BaseModel):
    """Request body for driver login."""
    email: str = Field(..., description="Email address")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Response with access and refresh tokens."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer")
    expires_in: int = Field(..., description="Seconds until access token expires")


class RefreshTokenRequest(BaseModel):
    """Request to refresh access token."""
    refresh_token: str = Field(..., description="Valid refresh token")


class DeviceRegisterRequest(BaseModel):
    """Request to register a device (mobile app) with FCM token."""
    device_id: str = Field(..., description="Unique device identifier")
    fcm_token: str = Field(..., description="Firebase Cloud Messaging token")
    device_name: Optional[str] = Field(None, description="Optional device name")


class DeviceRegisterResponse(BaseModel):
    """Response after device registration."""
    status: str
    device_id: str
    message: Optional[str] = None


# ── User Profile ────────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    """User/Driver profile information."""
    driver_id: str
    email: str
    phone: Optional[str]
    vehicle_type: str
    created_at: datetime
    is_active: bool


class UpdateProfileRequest(BaseModel):
    """Request to update user profile."""
    phone: Optional[str] = Field(None, description="Phone number")
    vehicle_type: Optional[str] = Field(None, description="Vehicle type")
    password: Optional[str] = Field(None, min_length=6, description="New password (optional)")


# ── Trip Requests & Responses ───────────────────────────────────────────────────

class TripStartRequest(BaseModel):
    """Request to mark the start of a trip."""
    device_id: str = Field(..., description="Device identifier")
    vehicle_type: str = Field(..., description="Vehicle type for this trip")
    latitude: float = Field(..., ge=-90, le=90, description="Starting GPS latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Starting GPS longitude")


class TripStartResponse(BaseModel):
    """Response after trip start."""
    trip_id: str
    status: str
    message: Optional[str] = None


class TripEndRequest(BaseModel):
    """Request to mark the end of a trip."""
    trip_id: str = Field(..., description="Trip ID to end")
    latitude: float = Field(..., ge=-90, le=90, description="Ending GPS latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Ending GPS longitude")
    distance_km: Optional[float] = Field(None, ge=0, description="Distance traveled")
    duration_minutes: Optional[int] = Field(None, ge=0, description="Trip duration")


class TripEndResponse(BaseModel):
    """Response after trip end."""
    trip_id: str
    status: str
    avg_risk_score: Optional[float] = None
    max_risk_score: Optional[float] = None
    high_risk_count: Optional[int] = None


class TripSummary(BaseModel):
    """Summary of a single trip."""
    trip_id: str
    start_time: datetime
    end_time: Optional[datetime]
    start_lat: float
    start_lon: float
    end_lat: Optional[float]
    end_lon: Optional[float]
    distance_km: float
    duration_minutes: Optional[int]
    avg_risk_score: float
    max_risk_score: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    vehicle_type: str


class TripHistoryResponse(BaseModel):
    """Response with trip summaries."""
    trips: list[TripSummary] = Field(..., description="List of trips")
    total_trips: int = Field(..., description="Total number of trips")
    avg_risk: float = Field(..., description="Average risk score across all trips")
    total_distance_km: float = Field(..., description="Total distance across all trips")


# ── Risk Zone (Segment) Response ────────────────────────────────────────────────

class RiskZone(BaseModel):
    """A road segment with base risk level for map visualization."""
    segment_id: str = Field(..., description="Unique segment identifier")
    name: str = Field(..., description="Segment name")
    start_lat: float = Field(..., description="Start latitude")
    start_lon: float = Field(..., description="Start longitude")
    end_lat: float = Field(..., description="End latitude")
    end_lon: float = Field(..., description="End longitude")
    gradient_pct: float = Field(..., description="Road gradient (%)")
    curvature: float = Field(..., description="Curvature index (0–1)")
    base_risk_level: str = Field(..., description="Base risk: Low | Medium | High")


class RiskZonesResponse(BaseModel):
    """Response with risk zones for map overlay."""
    zones: list[RiskZone] = Field(..., description="List of risk zones")
    count: int = Field(..., description="Total number of zones")


# ── Error Response ──────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
