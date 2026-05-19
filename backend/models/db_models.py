"""
backend/models/db_models.py
SQLAlchemy ORM database models for User, Device, and Trip management.

These models define the database schema for storing driver accounts, device registrations,
and trip history.
"""
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from backend.config.database import Base


# ── User (Driver) Model ────────────────────────────────────────────────────────
class User(Base):
    """
    Represents a driver/user in the KaduGuard system.
    
    Attributes:
        driver_id: Unique driver identifier (primary key, UUID)
        phone: Phone number (optional)
        email: Email address
        vehicle_type: Default vehicle type (car, motorcycle, bus, lorry, three-wheeler)
        password_hash: Bcrypt hashed password
        created_at: Account creation timestamp
        updated_at: Last profile update timestamp
        is_active: Soft delete flag
    """
    __tablename__ = "users"

    driver_id = Column(String(36), primary_key=True, index=True, doc="UUID primary key")
    phone = Column(String(20), nullable=True, doc="Phone number")
    email = Column(String(255), unique=True, index=True, nullable=False, doc="Email address")
    vehicle_type = Column(String(50), default="car", doc="car | motorcycle | bus | lorry | three-wheeler")
    password_hash = Column(String(255), nullable=False, doc="Bcrypt hashed password")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, doc="Soft delete flag")

    # Relationships
    devices = relationship("Device", back_populates="user", cascade="all, delete-orphan")
    trips = relationship("Trip", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(driver_id='{self.driver_id}', email='{self.email}')>"


# ── Device Model ───────────────────────────────────────────────────────────────
class Device(Base):
    """
    Represents a mobile device registered with a driver.
    
    Attributes:
        device_id: Unique device identifier (primary key)
        driver_id: Foreign key to User
        fcm_token: Firebase Cloud Messaging token for push notifications
        device_name: Optional device name (e.g., "John's Samsung A12")
        last_ping: Last time device checked in with the server
        created_at: Device registration timestamp
        is_active: Whether device is currently active
    """
    __tablename__ = "devices"

    device_id = Column(String(255), primary_key=True, index=True, doc="Unique device identifier")
    driver_id = Column(String(36), ForeignKey("users.driver_id", ondelete="CASCADE"), nullable=False, index=True)
    fcm_token = Column(String(255), nullable=True, doc="Firebase Cloud Messaging token")
    device_name = Column(String(255), nullable=True, doc="Optional device name")
    last_ping = Column(DateTime, default=datetime.utcnow, doc="Last server check-in")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="devices")

    def __repr__(self) -> str:
        return f"<Device(device_id='{self.device_id}', driver_id='{self.driver_id}')>"


# ── Trip Model ─────────────────────────────────────────────────────────────────
class Trip(Base):
    """
    Represents a trip (drive session) by a driver.
    
    Attributes:
        trip_id: Unique trip identifier (primary key, UUID)
        driver_id: Foreign key to User
        start_time: When the trip started
        end_time: When the trip ended (nullable if ongoing)
        start_lat / start_lon: GPS coordinates at trip start
        end_lat / end_lon: GPS coordinates at trip end (nullable)
        distance_km: Estimated distance traveled
        duration_minutes: Trip duration
        avg_risk_score: Average risk score during trip
        max_risk_score: Maximum risk score during trip
        high_risk_count: Number of high-risk predictions during trip
        medium_risk_count: Number of medium-risk predictions during trip
        low_risk_count: Number of low-risk predictions during trip
        vehicle_type: Vehicle type during trip
        notes: Optional driver notes
    """
    __tablename__ = "trips"

    trip_id = Column(String(36), primary_key=True, index=True, doc="UUID primary key")
    driver_id = Column(String(36), ForeignKey("users.driver_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Time
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True)
    
    # Location
    start_lat = Column(Float, nullable=False)
    start_lon = Column(Float, nullable=False)
    end_lat = Column(Float, nullable=True)
    end_lon = Column(Float, nullable=True)
    
    # Metrics
    distance_km = Column(Float, default=0.0, doc="Estimated distance traveled")
    duration_minutes = Column(Integer, nullable=True, doc="Trip duration in minutes")
    avg_risk_score = Column(Float, default=0.0, doc="Average risk score (0–1)")
    max_risk_score = Column(Float, default=0.0, doc="Maximum risk score (0–1)")
    
    # Risk counts
    high_risk_count = Column(Integer, default=0, doc="Number of high-risk predictions")
    medium_risk_count = Column(Integer, default=0, doc="Number of medium-risk predictions")
    low_risk_count = Column(Integer, default=0, doc="Number of low-risk predictions")
    
    # Details
    vehicle_type = Column(String(50), nullable=False, doc="Vehicle type during trip")
    notes = Column(String(500), nullable=True, doc="Optional driver notes")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="trips")

    def __repr__(self) -> str:
        return f"<Trip(trip_id='{self.trip_id}', driver_id='{self.driver_id}', duration={self.duration_minutes}m)>"
