from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import get_database_url

# Récupère l'URL de connexion à la base de données depuis la configuration
SQLALCHEMY_DATABASE_URL = get_database_url()

# Paramètres spécifiques pour SQLite
connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}

# Crée un moteur SQLAlchemy pour se connecter à la base de données
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args)

# Initialise la session pour gérer les connexions à la base de données
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Déclaration de la base pour les modèles SQLAlchemy
Base = declarative_base()


def init_db():
    """
    Initialise la base de données en créant toutes les tables définies
    dans les modèles importés.

    Cette fonction doit être appelée une fois au démarrage de l'application
    pour s'assurer que la base est bien configurée.
    """
    # Importer TOUS les modèles utilisés pour qu'ils soient enregistrés
    from app.api.v1.models import user, password_reset   
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Fournit une session SQLAlchemy pour les routes FastAPI.

    Cette fonction est conçue pour être utilisée comme dépendance FastAPI.
    Elle garantit que la session est correctement ouverte et fermée
    pour chaque requête HTTP.

    Yields:
        Session: Une instance de session de base de données active.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
