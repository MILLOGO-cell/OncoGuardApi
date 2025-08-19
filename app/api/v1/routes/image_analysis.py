from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.v1.schemas.image_analysis import (
    ImageAnalysisCreate,
    ImageAnalysisPrediction,
    ImageAnalysis,
)
from app.api.v1.models.enums import BiradsCategory, AnalysisStatus
from app.api.v1.models.image_analysis import ImageAnalysis as ImageAnalysisModel
import shutil
import uuid
import os

from app.db.database import get_db
from app.ml.predictor import predict_birads

router = APIRouter()


@router.post("/predict", response_model=ImageAnalysis)
async def create_and_predict_image_analysis(
    patient_id: int | None = None,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Format de fichier non supporté. Utiliser JPEG ou PNG.",
        )

    # Sauvegarder le fichier uploadé localement
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = f"./uploads/{unique_filename}"
    os.makedirs("./uploads", exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Prédire avec ton modèle ML (factice ici)
    prediction = predict_birads(file_path)

    # Créer l’analyse à enregistrer en base
    analysis_db = ImageAnalysisModel(
        filename=unique_filename,
        patient_id=patient_id,
        result_class=prediction.result_class,
        confidence=float(prediction.confidence) if prediction.confidence else None,
        status=AnalysisStatus.COMPLETED,
        description=None,
    )

    db.add(analysis_db)
    await db.commit()
    await db.refresh(analysis_db)

    # Optionnel : supprimer fichier après usage ou garder selon besoin

    return analysis_db
