from sqlalchemy import Column, Integer, String, DateTime, Float, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base
from app.api.v1.models.enums import BiradsCategory, AnalysisStatus  


class ImageAnalysis(Base):
    """
    Représente une analyse d'image mammaire enregistrée en base.

    Attributes:
        id (int): Identifiant unique de l'analyse (clé primaire).
        filename (str): Nom du fichier image analysé.
        patient_id (int | None): Référence optionnelle vers le patient lié.
        result_class (BiradsCategory | None): Catégorie BI-RADS attribuée à l'image.
        confidence (float | None): Score indiquant la confiance de la prédiction.
        description (str | None): Commentaires ou notes additionnels.
        status (AnalysisStatus): Statut du traitement de l'analyse (ex: pending, completed).
        submitted_at (datetime): Horodatage automatique de la soumission de l'image.
    """
    
    __tablename__ = "image_analyses"

    id: int = Column(Integer, primary_key=True, index=True)
    filename: str = Column(String, nullable=False)
    # patient_id: int | None = Column(Integer, ForeignKey("patients.id"), nullable=True)
    result_class: BiradsCategory | None = Column(Enum(BiradsCategory), nullable=True)
    confidence: float | None = Column(Float, nullable=True)
    description: str | None = Column(String, nullable=True)
    status: AnalysisStatus = Column(
        Enum(AnalysisStatus),
        default=AnalysisStatus.PENDING,
        nullable=False
    )
    submitted_at: DateTime = Column(DateTime(timezone=True), server_default=func.now())

    # patient = relationship("Patient", back_populates="analyses")
