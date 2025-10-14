import os
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DEBUG: bool = False
    DATABASE_URL: str
    SECRET_KEY: str
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


def get_database_url():
    if settings.DEBUG:
        return "sqlite:///./dev.db"
    return settings.DATABASE_URL


# 📂 Chemin absolu vers le dossier MIAS (portable)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIAS_DATA_DIR = os.path.join(BASE_DIR, "data", "mias")
