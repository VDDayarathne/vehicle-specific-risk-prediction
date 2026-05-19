"""
backend/main.py
FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.config.settings import ALLOWED_ORIGINS, BASE_DIR
from backend.config.database import engine, Base
from backend.api import predict, weather, health, segments, telemetry, auth, mobile
from backend.models import db_models  # noqa: F401 - ensure SQLAlchemy models are registered
from backend.services.model_service import ModelService

# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="KaduGuard — Vehicle Risk Prediction API",
    description=(
        "Real-time accident risk prediction for Kadugannawa hill-country roads. "
        "Provides vehicle-specific Low / Medium / High risk levels using a trained "
        "Random Forest model combined with live weather data."
    ),
    version="1.0.0",
    contact={
        "name": "H.G.V.D. Dayarathne",
        "email": "20apse4843@sliit.lk",
    },
)

# ── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routers ───────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(weather.router)
app.include_router(predict.router)
app.include_router(segments.router)
app.include_router(telemetry.router)
app.include_router(auth.router)
app.include_router(mobile.router)

# ── Serve frontend static files ───────────────────────────────────────────────

FRONTEND_PUBLIC = BASE_DIR / "frontend" / "public"
FRONTEND_DIST   = BASE_DIR / "frontend" / "dist"

# Prefer built dist/ if it exists, otherwise serve from public/
_static_dir = FRONTEND_DIST if FRONTEND_DIST.exists() else FRONTEND_PUBLIC

if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index = _static_dir / "index.html"
        if not index.exists():
            index = _static_dir / "dashboard.html"
        return FileResponse(str(index))

    @app.get("/{path:path}", include_in_schema=False)
    async def serve_file(path: str):
        # Serve files from the frontend static directory at top-level paths
        target = _static_dir / path
        if target.exists() and target.is_file():
            return FileResponse(str(target))
        # Fallback to index for single-page navigation
        index = _static_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(status_code=404, detail="Not Found")


# ── Startup event — pre-load model ───────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("[KaduGuard] Initialising ML model...")
    Base.metadata.create_all(bind=engine)
    ModelService.get_instance()
    print("[KaduGuard] API ready. Docs at http://localhost:8000/docs")


# ── Root redirect for health check ───────────────────────────────────────────

@app.get("/ping", tags=["Health"], include_in_schema=False)
async def ping():
    return {"pong": True}
