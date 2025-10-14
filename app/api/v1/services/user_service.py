import secrets
import string
from sqlalchemy.orm import Session

from app.api.v1.models.user import User
from app.api.v1.schemas.user import UserCreate, UserUpdate
from app.core.security import hash_password

def get_user_by_email(db: Session, email: str) -> User | None:
    """
    Recherche un utilisateur dans la base par son adresse email.

    Args:
        db (Session): Session SQLAlchemy de la base de données.
        email (str): Adresse email de l'utilisateur recherché.

    Returns:
        User | None: L'objet User trouvé ou None si aucun utilisateur ne correspond.
    """
    return db.query(User).filter(User.email == email).first()

def generate_random_password(length: int = 12) -> str:
    """
    Génère un mot de passe aléatoire sécurisé.

    Utilise des lettres majuscules et minuscules, chiffres et caractères spéciaux.

    Args:
        length (int, optional): Longueur du mot de passe. Par défaut 12.

    Returns:
        str: Mot de passe aléatoire généré.
    """
    chars = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    return ''.join(secrets.choice(chars) for _ in range(length))

def create_user(db: Session, user_data: UserCreate) -> User:
    """
    Crée un nouvel utilisateur en base avec un mot de passe généré automatiquement.

    Le mot de passe généré est hashé avant stockage.

    Args:
        db (Session): Session SQLAlchemy de la base de données.
        user_data (UserCreate): Données utilisateur à créer (email, full_name).

    Returns:
        User: L'objet User créé en base. L'attribut `plain_password` est ajouté temporairement
              contenant le mot de passe non-hashé (pour envoi par email par exemple).
    """
    raw_password = generate_random_password()
    hashed_pwd = hash_password(raw_password)

    new_user = User(
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=hashed_pwd,
        is_active=True,
        is_verified=False,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    new_user.plain_password = raw_password

    return new_user

def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()

def email_exists(db: Session, email: str, exclude_user_id: int | None = None) -> bool:
    q = db.query(User).filter(User.email == email)
    if exclude_user_id:
        q = q.filter(User.id != exclude_user_id)
    return db.query(q.exists()).scalar()

def update_user(db: Session, user: User, payload: UserUpdate) -> User:
    if payload.email is not None:
        if email_exists(db, payload.email, exclude_user_id=user.id):
            raise ValueError("Email déjà utilisé.")
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

def delete_user(db: Session, user: User) -> None:
    db.delete(user)
    db.commit()