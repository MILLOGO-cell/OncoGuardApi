from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from app.core.config import settings

# 🔐 Initialisation du contexte de hachage avec bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """
    Hash un mot de passe en utilisant bcrypt.

    Args:
        password (str): Le mot de passe en clair.

    Returns:
        str: Le mot de passe haché.
    """
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Vérifie si un mot de passe correspond à son hash.

    Args:
        plain_password (str): Mot de passe en clair saisi par l'utilisateur.
        hashed_password (str): Mot de passe haché stocké dans la base.

    Returns:
        bool: True si les mots de passe correspondent, False sinon.
    """
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=30)) -> str:
    """
    Génère un JWT d'accès (access token).

    Args:
        data (dict): Données à inclure dans le token (ex: {"sub": user_id}).
        expires_delta (timedelta, optional): Durée de validité du token. Défaut : 30 minutes.

    Returns:
        str: Token JWT signé.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")

def create_refresh_token(data: dict, expires_delta: timedelta = timedelta(days=7)) -> str:
    """
    Génère un JWT de rafraîchissement (refresh token).

    Args:
        data (dict): Données à inclure dans le token.
        expires_delta (timedelta, optional): Durée de validité du refresh token. Défaut : 7 jours.

    Returns:
        str: Token JWT signé.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")

def decode_access_token(token: str):
    """
    Décode un token d'accès JWT. Vérifie la validité et la signature.

    Args:
        token (str): Le token JWT à décoder.

    Returns:
        dict | None: Le payload décodé si valide, sinon None.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except JWTError:
        return None
