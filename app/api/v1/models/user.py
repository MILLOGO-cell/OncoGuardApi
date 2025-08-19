from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from app.db.database import Base

class User(Base):
    """
    Modèle ORM représentant un utilisateur dans la base de données.

    Attributs de la table 'users' :
    
    Attributes:
        id (int): Identifiant unique (clé primaire).
        email (str): Adresse email unique et indexée.
        full_name (str | None): Nom complet de l'utilisateur (optionnel).
        hashed_password (str): Mot de passe hashé, obligatoire.
        is_active (bool): Statut actif de l'utilisateur (par défaut True).
        is_verified (bool): Indique si l'utilisateur a vérifié son compte (défaut False).
        mfa_enabled (bool): Activation de l'authentification à deux facteurs par email (défaut True).
        temporary_token (str | None): Token temporaire utilisé pour MFA ou autres fonctionnalités (optionnel).
        created_at (datetime): Date de création, automatique lors de l'insertion.
        updated_at (datetime | None): Date de dernière mise à jour, mise à jour automatique.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)

    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
