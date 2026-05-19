"""
backend/api/health.py
GET /api/health  — server + model liveness check.
"""
from fastapi import APIRouter

from backend.models.schemas import HealthResponse
from backend.services.model_service import ModelService

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    svc = ModelService.get_instance()
    return HealthResponse(
        status="ok",
        model_loaded=svc.is_loaded,
    )
