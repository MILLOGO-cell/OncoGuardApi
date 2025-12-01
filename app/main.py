from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.database import init_db
from app.api.v1.routes import api_router

app = FastAPI(title="OncoGuardAPI", version="1.0.0")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://oncoguard.millogo-studio.com",
]

if getattr(settings, "FRONTEND_URL", None):
    origins.append(settings.FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Bienvenue sur OncoGuardAPI"}

@app.on_event("startup")
async def startup_event():
    if settings.DEBUG:
        print("Initialisation de la base de données SQLite...")
        init_db()
        for r in app.routes:
            try:
                print(f"[route] {r.path} -> {','.join(r.methods)}")
            except Exception:
                pass