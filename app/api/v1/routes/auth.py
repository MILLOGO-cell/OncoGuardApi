from typing import Annotated
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

def generate_otp() -> str:
    """
    Génère un code OTP (One-Time Password) à 6 chiffres.
    
    Returns:
        str: Code OTP sous forme de chaîne de 6 chiffres.
    """
    return f"{random.randint(100000, 999999)}"

def get_db():
    """
    Fournit une session SQLAlchemy pour les routes FastAPI.

    Cette fonction est utilisée comme dépendance dans les routes pour
    injecter une session de base de données. Elle garantit que la session
    est correctement fermée après l'utilisation.
    
    Yields:
        Session: Session de base de données SQLAlchemy.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/register", response_model=UserOut, status_code=201)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """
    Enregistre un nouvel utilisateur.

    - Vérifie si l'email est déjà utilisé.
    - Crée un nouvel utilisateur avec mot de passe généré.
    - Envoie un email de bienvenue contenant les identifiants.

    Args:
        user (UserCreate): Données d'inscription.
        db (Session): Session de base de données injectée.

    Returns:
        UserOut: Données de l'utilisateur créé (hors mot de passe).
    
    Raises:
        HTTPException: Si l'email est déjà enregistré.
    """
    db_user = get_user_by_email(db, user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email déjà enregistré."
        )

    # Création du nouvel utilisateur (inclut mot de passe temporaire)
    new_user = create_user(db, user)

    # Envoi de l'email avec identifiants
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
    """
    Authentifie un utilisateur avec email et mot de passe.

    - Vérifie les identifiants.
    - Retourne un token JWT d'accès et un refresh token.

    Args:
        payload (UserLogin): Email et mot de passe fournis par l'utilisateur.
        db (Session): Session de base de données injectée.

    Returns:
        Token: Contient les tokens JWT d'authentification.
    
    Raises:
        HTTPException: Si identifiants invalides.
    """
    user: User = get_user_by_email(db, payload.email)

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe invalide"
        )

    access_token = create_access_token({"sub": str(user.id)})
    refresh_token = create_refresh_token({"sub": str(user.id)})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@router.post("/request-password-reset")
def request_password_reset(payload: PasswordResetRequest, db: Session = Depends(get_db)):
    """
    Étape 1 : Envoie un code OTP par email pour la réinitialisation du mot de passe.

    Args:
        payload (PasswordResetRequest): Contient l'email utilisateur.
        db (Session): Session de base de données injectée.

    Returns:
        dict: Message informant que le code a été envoyé.
    
    Raises:
        HTTPException: Si l'utilisateur n'existe pas.
    """
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
            f"Bonjour,\n\n"
            f"Voici votre code de vérification pour réinitialiser votre mot de passe : {code}\n"
            f"Ce code expirera dans 10 minutes.\n\n"
            f"--\nL'équipe OncoGuardAPI"
        )
    )

    return {"message": "Code de réinitialisation envoyé par email"}

class VerifyCodePayload(BaseModel):
    """
    Modèle Pydantic pour vérifier le code OTP envoyé par email.
    """
    email: EmailStr
    code: Annotated[str, StringConstraints(min_length=6, max_length=6)]

@router.post("/verify-reset-code")
def verify_reset_code(payload: VerifyCodePayload, db: Session = Depends(get_db)):
    """
    Vérifie la validité du code OTP fourni pour réinitialisation du mot de passe.

    - Le code doit exister, être lié à l'email, non expiré et pas encore utilisé.
    - Retourne HTTP 200 en cas de succès, sinon code d’erreur approprié.

    Args:
        payload (VerifyCodePayload): Email et code OTP à vérifier.
        db (Session): Session de base de données injectée.

    Returns:
        dict: Message confirmant la vérification du code.
    
    Raises:
        HTTPException: Si le code est invalide, expiré ou inexistant.
    """
    record = db.query(PasswordResetCode).filter_by(
        email=payload.email,
        code=payload.code,
        used=False
    ).first()

    if not record:
        raise HTTPException(status_code=404, detail="Code invalide ou inexistant")

    if record.is_expired():
        raise HTTPException(status_code=410, detail="Code expiré")

    # Marquer le code comme utilisé
    record.used = True
    db.commit()

    return {"message": "Code vérifié avec succès. Vous pouvez maintenant définir un nouveau mot de passe."}

@router.post("/reset-password")
def reset_password(payload: PasswordResetConfirm, db: Session = Depends(get_db)):
    """
    Étape 3 : Change le mot de passe après vérification du code OTP.

    - Vérifie que le code est valide et utilisé.
    - Vérifie que les deux nouveaux mots de passe correspondent.
    - Met à jour le mot de passe de l’utilisateur dans la base.

    Args:
        payload (PasswordResetConfirm): Email, code OTP et nouveaux mots de passe.
        db (Session): Session de base de données injectée.

    Returns:
        dict: Message confirmant la mise à jour du mot de passe.
    
    Raises:
        HTTPException: Si mots de passe non concordants, utilisateur non trouvé ou code invalide.
    """
    if payload.new_password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Les mots de passe ne correspondent pas.")

    code_record = db.query(PasswordResetCode).filter_by(
        email=payload.email,
        code=payload.code,
        used=True
    ).first()

    if not code_record:
        raise HTTPException(status_code=404, detail="Code de vérification introuvable ou non validé.")

    user = get_user_by_email(db, payload.email)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé.")

    user.hashed_password = hash_password(payload.new_password)
    db.commit()

    return {"message": "Mot de passe mis à jour avec succès."}
