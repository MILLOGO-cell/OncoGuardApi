from fastapi import APIRouter
from .auth import router as auth_router

api_router = APIRouter()

# Authentification et gestion des utilisateurs
api_router.include_router(auth_router, prefix="/auth", tags=["Authentification"])
