from fastapi import APIRouter
from .auth import router as auth_router
from .image_inference import router as image_inference_router
from .image_analysis import router as image_analysis_router  # ← AJOUTER
from .stats import router as stats_router
from .ingest import router as ingest_router
from .ingest_files import router as ingest_files_router

api_router = APIRouter()

api_router.include_router(
    auth_router,
    prefix="/auth",
    tags=["Authentification et utilisateurs"]
)

api_router.include_router(
    image_inference_router,
    tags=["Analyse d'images"]
)

api_router.include_router(
    image_analysis_router,
    prefix="/image-analysis",   
    tags=["Analyse d'images IA"]
)

api_router.include_router(
    stats_router,
    tags=["Statistiques et rapports"]
)

api_router.include_router(
    ingest_router,
    tags=["Anonymisation et traitement d'images"]
)

api_router.include_router(
    ingest_files_router,
)