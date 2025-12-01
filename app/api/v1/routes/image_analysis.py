from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session
from app.api.v1.schemas.image_analysis import ImageAnalysis
from app.api.v1.models.enums import BiradsCategory, AnalysisStatus
from app.api.v1.models.image_analysis import ImageAnalysis as ImageAnalysisModel
import shutil
import uuid
import os
import logging
from app.db.database import get_db
from app.ml.predictor import predict

logger = logging.getLogger(__name__)

router = APIRouter()


def _label_to_birads(label: str) -> BiradsCategory:
    if not label:
        return BiradsCategory.BI_RADS_0
    
    lab = label.strip().lower()
    
    if lab.startswith("norm"):
        return BiradsCategory.BI_RADS_1
    elif lab.startswith("ben"):
        return BiradsCategory.BI_RADS_2
    elif lab.startswith("mal"):
        return BiradsCategory.BI_RADS_5
    else:
        logger.warning(f"Label inconnu '{label}', retour BI-RADS 0")
        return BiradsCategory.BI_RADS_0


@router.post("/predict", response_model=ImageAnalysis)
def create_and_predict_image_analysis(
    patient_id: int | None = None,
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.60, ge=0.0, le=1.0),
    uncertainty_threshold: float = Query(0.35, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Format de fichier non supporté. Utiliser JPEG ou PNG.",
        )
    
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = f"./uploads/{unique_filename}"
    os.makedirs("./uploads", exist_ok=True)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        prediction = predict(
            file_path, 
            confidence_threshold=confidence_threshold,
            uncertainty_threshold=uncertainty_threshold
        )
        
        birads_category = _label_to_birads(prediction.label)
        
        if prediction.reclassified_as_normal:
            logger.info(f"Image {unique_filename} reclassée NORMAL (confiance: {prediction.confidence:.2%})")
        else:
            logger.info(f"Image {unique_filename} classée {prediction.label.upper()} (confiance: {prediction.confidence:.2%})")
        
        analysis_db = ImageAnalysisModel(
            filename=unique_filename,
            patient_id=patient_id,
            result_class=birads_category,
            confidence=float(prediction.confidence) if prediction.confidence else None,
            status=AnalysisStatus.COMPLETED,
            description=(
                f"Pred: {prediction.label} ({prediction.birads}) - "
                f"Model: {prediction.model_type}"
                + (f" - Reclassifié NORMAL" if prediction.reclassified_as_normal else "")
            ),
        )
        
        db.add(analysis_db)
        db.commit()
        db.refresh(analysis_db)
        
        logger.info(f"Analyse enregistrée : ID={analysis_db.id}, BI-RADS={birads_category.value}")
        
        return analysis_db
    
    except Exception as e:
        logger.exception(f"Erreur lors de l'analyse de {unique_filename}")
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse de l'image : {str(e)}"
        )