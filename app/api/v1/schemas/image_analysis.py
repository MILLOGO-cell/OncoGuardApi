from datetime import datetime
from typing import Optional,Annotated
from decimal import Decimal
from pydantic import BaseModel, Field
from app.api.v1.models.enums import BiradsCategory, AnalysisStatus

class ImageAnalysisBase(BaseModel):
    filename: str = Field(..., description="Nom du fichier image analysé")
    patient_id: Optional[int] = Field(None, description="ID optionnel du patient lié")
    description: Optional[str] = Field(None, description="Commentaires additionnels")

class ImageAnalysisCreate(ImageAnalysisBase):
    pass  

class ImageAnalysisPrediction(BaseModel):
    result_class: Optional[BiradsCategory] = Field(None, description="Catégorie BI-RADS prédite")
    confidence: Optional[Annotated[Decimal, Field(gt=0, lt=1)]] = Field(None, description="Score de confiance entre 0 et 1")

class ImageAnalysis(ImageAnalysisBase):
    id: int
    result_class: Optional[BiradsCategory]
    confidence: Optional[float]
    status: AnalysisStatus
    submitted_at: datetime

    class Config:
        orm_mode = True
