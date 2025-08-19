from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base

class Patient(Base):
    """
    Modèle représentant un patient.

    Les informations personnelles et médicales sont optionnelles,
    respectant ainsi la confidentialité et la flexibilité de saisie.

    Attributes:
        id (int): Identifiant unique du patient.
        birth_date (datetime | None): Date de naissance (facultative).
        gender (str | None): Sexe (optionnel).
        medical_history (str | None): Antécédents médicaux (optionnel).
        created_at (datetime): Date de création de l'enregistrement.
        updated_at (datetime): Date de dernière mise à jour.
        analyses (List[ImageAnalysis]): Liste des analyses associées à ce patient.
    """
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    birth_date = Column(DateTime, nullable=True)
    gender = Column(String, nullable=True)
    medical_history = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    analyses = relationship("ImageAnalysis", back_populates="patient")
