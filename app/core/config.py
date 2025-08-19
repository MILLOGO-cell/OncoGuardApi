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


# Chemin absolu local vers le dossier data/mias  
MIAS_DATA_DIR = r"C:\Users\XPS\Documents\Perso\Memoire\data\mias"