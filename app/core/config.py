from __future__ import annotations
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Paramètres globaux de l'application chargés depuis l'environnement."""
    DEBUG: bool = False
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7)
    SMTP_SERVER: str
    SMTP_PORT: int
    SMTP_USER: str
    SMTP_PASSWORD: str
    FROM_EMAIL: str
    FRONTEND_URL: str
    API_KEY: str
    email_use_tls: bool = Field(default=True)
    
    class Config:
        env_file = ".env.development"
        extra = "ignore"

settings = Settings()

def get_database_url() -> str:
    """Retourne l'URL de connexion à la base de données."""
    if settings.DEBUG:
        return "sqlite:///./dev.db"
    return settings.DATABASE_URL

# CORRECTION ICI : Les données sont dans app/data/ pas dans data/
BASE_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = BASE_DIR / "app" / "data"  # ← CHANGEMENT ICI
MIAS_DATA_DIR: Path = DATA_DIR / "mias"
CBIS_CSV_DIR: Path = DATA_DIR / "cbis-ddsm" / "csv"
CBIS_JPEG_DIR: Path = DATA_DIR / "cbis-ddsm" / "jpeg"
UNIFIED_DIR: Path = DATA_DIR / "unified"
UNIFIED_CSV_PATH: Path = UNIFIED_DIR / "unified_annotations.csv"
ML_DIR: Path = BASE_DIR / "app" / "ml"
MODELS_DIR: Path = ML_DIR / "models"
ARTIFACTS_DIR: Path = ML_DIR / "artifacts_mias"