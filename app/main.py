from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.database import init_db, SessionLocal
from app.api.v1.routes import api_router
from app.api.v1.models.user import User
from passlib.context import CryptContext

app = FastAPI(title="OncoGuardAPI", version="1.0.0")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
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
    print("Initialisation de la base de données...")
    init_db()
    
    print("Vérification de l'utilisateur admin...")
    db = SessionLocal()
    try:
        admin_email = "nicolasmillogo3@gmail.com"
        existing_user = db.query(User).filter(User.email == admin_email).first()
        
        if not existing_user:
            print("Création de l'utilisateur admin...")
            pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            hashed_password = pwd_context.hash("Admin@2025")
            
            admin_user = User(
                email=admin_email,
                full_name="Nicolas MILLOGO",
                hashed_password=hashed_password,
                is_active=True,
                is_verified=True
            )
            
            db.add(admin_user)
            db.commit()
            print("✅ Utilisateur admin créé avec succès!")
        else:
            print(f"✅ Utilisateur admin existe déjà (ID: {existing_user.id})")
    except Exception as e:
        print(f"❌ Erreur lors de la création de l'utilisateur admin: {e}")
        db.rollback()
    finally:
        db.close()
    
    if settings.DEBUG:
        print("Mode DEBUG activé - Affichage des routes:")
        for r in app.routes:
            try:
                print(f"[route] {r.path} -> {','.join(r.methods)}")
            except Exception:
                pass