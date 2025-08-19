from pydantic import BaseModel, EmailStr
from typing import Optional

class UserBase(BaseModel):
    """
    Base commune pour tous les schémas utilisateur.

    Attributes:
        email (EmailStr): Adresse email de l'utilisateur.
        full_name (Optional[str]): Nom complet de l'utilisateur (facultatif).
    """
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    """
    Schéma pour la création d'un utilisateur.

    Le mot de passe n'est pas fourni ici, il est généré automatiquement.
    """
    pass

class UserOut(UserBase):
    """
    Schéma retourné par l'API pour décrire un utilisateur.

    Attributes:
        id (int): Identifiant unique de l'utilisateur.
        is_active (bool): Indique si l'utilisateur est actif.
        is_verified (bool): Indique si l'utilisateur a vérifié son compte.
    """
    id: int
    is_active: bool
    is_verified: bool

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    """
    Schéma pour la connexion utilisateur.

    Contient les informations nécessaires à la connexion.
    
    Attributes:
        email (EmailStr): Adresse email utilisée pour la connexion.
        password (str): Mot de passe de l'utilisateur.
    """
    email: EmailStr
    password: str

class Token(BaseModel):
    """
    Schéma du token JWT retourné après authentification.

    Attributes:
        access_token (str): Token d'accès JWT.
        refresh_token (str): Token de rafraîchissement JWT.
        token_type (str): Type de token, par défaut 'bearer'.
    """
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    """
    Schéma contenant les données extraites du token JWT.

    Attributes:
        user_id (Optional[int]): Identifiant de l'utilisateur extrait du token.
                                 Peut être None si non présent.
    """
    user_id: Optional[int] = None
