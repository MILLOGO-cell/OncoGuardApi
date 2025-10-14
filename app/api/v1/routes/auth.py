from typing import Annotated, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, StringConstraints

from app.api.v1.models.password_reset import PasswordResetCode
from app.api.v1.models.user import User
from app.api.v1.schemas.password_reset import PasswordResetConfirm, PasswordResetRequest
from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.db.database import SessionLocal
from app.api.v1.schemas.user import Token, UserCreate, UserLogin, UserOut
from app.api.v1.services.user_service import get_user_by_email, create_user
from app.core.email import send_email
from app.core.config import settings
import random

router = APIRouter()

# -------- Schemas locaux pour read/update/delete --------
class VerifyCodePayload(BaseModel):
    email: EmailStr
    code: Annotated[str, StringConstraints(min_length=6, max_length=6)]

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    password: Optional[str] = None  # si fourni, sera hashé

# -------- DB dependency --------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------- Helpers --------
def generate_otp() -> str:
    return f"{random.randint(100000, 999999)}"

def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()

def email_exists(db: Session, email: str, exclude_user_id: int | None = None) -> bool:
    q = db.query(User).filter(User.email == email)
    if exclude_user_id is not None:
        q = q.filter(User.id != exclude_user_id)
    return db.query(q.exists()).scalar()

# ===================== Auth & Password reset =====================

@router.post("/register", response_model=UserOut, status_code=201)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, user.email)
    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email déjà enregistré.")
    new_user = create_user(db, user)
    send_email(
        to_email=new_user.email,
        subject="Bienvenue sur OncoGuardAPI",
        content=(
            f"Bonjour {new_user.full_name or new_user.email},\n\n"
            f"Votre compte a bien été créé.\n\n"
            f"Identifiants de connexion :\n"
            f"Email : {new_user.email}\n"
            f"Mot de passe : {new_user.plain_password}\n\n"
            f"Accédez à l'interface ici : {settings.FRONTEND_URL}\n\n"
            f"--\nL'équipe OncoGuardAPI"
        )
    )
    return new_user

@router.post("/login", response_model=Token)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user: User = get_user_by_email(db, payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Email ou mot de passe invalide")
    access_token = create_access_token({"sub": str(user.id)})
    refresh_token = create_refresh_token({"sub": str(user.id)})
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@router.post("/request-password-reset")
def request_password_reset(payload: PasswordResetRequest, db: Session = Depends(get_db)):
    user = get_user_by_email(db, payload.email)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    code = generate_otp()
    reset = PasswordResetCode(email=payload.email, code=code)
    db.add(reset)
    db.commit()
    send_email(
        to_email=payload.email,
        subject="Code de réinitialisation de mot de passe",
        content=(
            f"Bonjour,\n\nVoici votre code de vérification : {code}\n"
            f"Ce code expirera dans 10 minutes.\n\n--\nL'équipe OncoGuardAPI"
        )
    )
    return {"message": "Code de réinitialisation envoyé par email"}

@router.post("/verify-reset-code")
def verify_reset_code(payload: VerifyCodePayload, db: Session = Depends(get_db)):
    record = db.query(PasswordResetCode).filter_by(
        email=payload.email, code=payload.code, used=False
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Code invalide ou inexistant")
    if record.is_expired():
        raise HTTPException(status_code=410, detail="Code expiré")
    record.used = True
    db.commit()
    return {"message": "Code vérifié avec succès. Vous pouvez maintenant définir un nouveau mot de passe."}

@router.post("/reset-password")
def reset_password(payload: PasswordResetConfirm, db: Session = Depends(get_db)):
    if payload.new_password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Les mots de passe ne correspondent pas.")
    code_record = db.query(PasswordResetCode).filter_by(
        email=payload.email, code=payload.code, used=True
    ).first()
    if not code_record:
        raise HTTPException(status_code=404, detail="Code de vérification introuvable ou non validé.")
    user = get_user_by_email(db, payload.email)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé.")
    user.hashed_password = hash_password(payload.new_password)
    db.commit()
    return {"message": "Mot de passe mis à jour avec succès."}

# ===================== Users: list / read / update / delete =====================

@router.get("/users", response_model=List[UserOut])
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

@router.get("/users/{user_id}", response_model=UserOut)
def retrieve_user(user_id: int, db: Session = Depends(get_db)):
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    return user

@router.put("/users/{user_id}", response_model=UserOut)
def update_user_endpoint(user_id: int, payload: UserUpdate, db: Session = Depends(get_db)):
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

    if payload.email is not None:
        if email_exists(db, payload.email, exclude_user_id=user.id):
            raise HTTPException(status_code=400, detail="Email déjà utilisé.")
        user.email = payload.email
    if payload.full_name is not None:
        user.full_name = payload.full_name
    if payload.is_active is not None:
        user.is_active = payload.is_active
    if payload.is_verified is not None:
        user.is_verified = payload.is_verified
    if payload.password:
        user.hashed_password = hash_password(payload.password)

    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_endpoint(user_id: int, db: Session = Depends(get_db)):
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    db.delete(user)
    db.commit()
    return None
