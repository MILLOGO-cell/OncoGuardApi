# tools/export_schema.py
from sqlalchemy import create_engine

from app.db.database import Base
# from app.api.v1.models import *  # IMPORTANT: importe tes modèles pour les enregistrer sur Base

engine = create_engine("sqlite:///./dev.db", echo=True)  # ou ta vraie URL PostgreSQL/MySQL
Base.metadata.create_all(bind=engine)
