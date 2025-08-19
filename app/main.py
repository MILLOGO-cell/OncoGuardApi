from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.database import init_db  

from app.api.v1.routes import auth         

app = FastAPI(
    title="OncoGuardAPI",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes publiques
app.include_router(auth.router, prefix="/auth", tags=["auth"])

@app.get("/")
async def root():
    return {"message": "Bienvenue sur OncoGuardAPI"}

@app.on_event("startup")
async def startup_event():
    if settings.DEBUG:
        print("Initialisation de la base de données SQLite...")
        init_db()
