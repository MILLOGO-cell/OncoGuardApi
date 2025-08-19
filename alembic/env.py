from logging.config import fileConfig
import os
from dotenv import load_dotenv

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# Charger les variables d'environnement (.env.development)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env.development'))

# --- Alembic configuration ---
config = context.config

# Met à jour l'URL de connexion depuis ta config FastAPI
from app.core.config import get_database_url
config.set_main_option("sqlalchemy.url", get_database_url())

# Logging (facultatif)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import du Base et des modèles
from app.db.database import Base
from app.api.v1.models import user, password_reset  # autres modèles si besoin

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Exécute les migrations en mode 'offline' (sans connexion DB)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Exécute les migrations en mode 'online' (avec une connexion DB active)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
