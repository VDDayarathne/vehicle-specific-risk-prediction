"""
backend/config/database.py
SQLAlchemy database configuration and session management.

Usage:
    from backend.config.database import engine, SessionLocal, Base
    
    # Create tables on app startup
    Base.metadata.create_all(bind=engine)
    
    # Get database session in route
    db: Session = Depends(get_db)
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

# ── Database URL ──────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kaduguard:kaduguard@db:5432/kaduguard"
)

# ── SQLAlchemy Engine & Session ────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("DEBUG", "false").lower() == "true",  # Log SQL queries in debug mode
    pool_pre_ping=True,  # Verify connections before using them
    pool_size=5,  # Connection pool size
    max_overflow=10,  # Max overflow connections
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Base class for all models ──────────────────────────────────────────────────
Base = declarative_base()


# ── Dependency for FastAPI routes ──────────────────────────────────────────────
def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to provide database session to FastAPI routes.
    
    Usage in routes:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
