"""
backend/utils/auth.py
JWT token generation, validation, and password hashing utilities.

Usage:
    from backend.utils.auth import create_access_token, verify_password, get_password_hash
    
    # Hash password during registration
    hashed = get_password_hash("user_password")
    
    # Create JWT token after login
    token = create_access_token(data={"sub": "driver_123"})
    
    # Verify token in protected routes
    from fastapi import Depends
    current_user = Depends(get_current_user)
"""
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── Configuration ─────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS = 7  # 7 days

# ── Password hashing ──────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Pydantic models for token responses ────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    driver_id: Optional[str] = None


# ── Password utilities ─────────────────────────────────────────────────────────
def get_password_hash(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


# ── JWT token utilities ────────────────────────────────────────────────────────
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary to encode in the token (typically {"sub": driver_id})
        expires_delta: Optional custom expiration time (defaults to ACCESS_TOKEN_EXPIRE_MINUTES)
    
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create a JWT refresh token with extended expiration.
    
    Args:
        data: Dictionary to encode in the token (typically {"sub": driver_id})
    
    Returns:
        JWT refresh token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        TokenData with driver_id if valid, None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        driver_id: str = payload.get("sub")
        if driver_id is None:
            return None
        return TokenData(driver_id=driver_id)
    except JWTError:
        return None


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode a JWT and return the raw payload if valid."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def verify_refresh_token(token: str) -> Optional[TokenData]:
    """Verify a refresh token and ensure it is marked as type=refresh."""
    payload = decode_token(token)
    if not payload or payload.get("type") != "refresh":
        return None
    driver_id = payload.get("sub")
    if not driver_id:
        return None
    return TokenData(driver_id=driver_id)
