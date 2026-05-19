"""
backend/api/auth.py
Authentication endpoints for the mobile app.
"""
from datetime import timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.config.database import get_db
from backend.models.db_models import Device, User
from backend.models.schemas import (
    DeviceRegisterRequest,
    DeviceRegisterResponse,
    LoginRequest,
    RegisterRequest,
    RefreshTokenRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserProfile,
)
from backend.utils.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
    TokenData,
    verify_password,
    verify_refresh_token,
    verify_token,
)

router = APIRouter(prefix="/api/auth", tags=["Auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def _build_token_response(driver_id: str) -> TokenResponse:
    access_token = create_access_token({"sub": driver_id}, expires_delta=timedelta(minutes=60))
    refresh_token = create_refresh_token({"sub": driver_id})
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=60 * 60,
    )


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    payload: TokenData | None = verify_token(token)
    if payload is None or not payload.driver_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user = db.query(User).filter(User.driver_id == payload.driver_id, User.is_active.is_(True)).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


@router.post("/register", response_model=TokenResponse, summary="Register a new driver")
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user = User(
        driver_id=str(uuid4()),
        phone=payload.phone,
        email=payload.email.lower(),
        vehicle_type=payload.vehicle_type,
        password_hash=get_password_hash(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _build_token_response(user.driver_id)


@router.post("/login", response_model=TokenResponse, summary="Login and receive tokens")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.email == payload.email.lower(), User.is_active.is_(True)).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    return _build_token_response(user.driver_id)


@router.post("/refresh", response_model=TokenResponse, summary="Refresh access token")
def refresh(payload: RefreshTokenRequest, db: Session = Depends(get_db)) -> TokenResponse:
    token_data = verify_refresh_token(payload.refresh_token)
    if token_data is None or not token_data.driver_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    user = db.query(User).filter(User.driver_id == token_data.driver_id, User.is_active.is_(True)).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return _build_token_response(user.driver_id)


@router.post("/device-register", response_model=DeviceRegisterResponse, summary="Register device for notifications")
def register_device(
    payload: DeviceRegisterRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeviceRegisterResponse:
    device = db.query(Device).filter(Device.device_id == payload.device_id).first()
    if device is None:
        device = Device(
            device_id=payload.device_id,
            driver_id=current_user.driver_id,
            fcm_token=payload.fcm_token,
            device_name=payload.device_name,
        )
        db.add(device)
    else:
        device.driver_id = current_user.driver_id
        device.fcm_token = payload.fcm_token
        device.device_name = payload.device_name
        device.is_active = True
    db.commit()

    return DeviceRegisterResponse(
        status="ok",
        device_id=payload.device_id,
        message="Device registered successfully",
    )


@router.get("/me", response_model=UserProfile, summary="Get current driver profile")
def me(current_user: User = Depends(get_current_user)) -> UserProfile:
    return UserProfile(
        driver_id=current_user.driver_id,
        email=current_user.email,
        phone=current_user.phone,
        vehicle_type=current_user.vehicle_type,
        created_at=current_user.created_at,
        is_active=current_user.is_active,
    )