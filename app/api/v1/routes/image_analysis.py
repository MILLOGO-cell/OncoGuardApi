# app/api/v1/routes/image_analysis.py
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
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
    """
    Convertit un label de prédiction en enum BiradsCategory.
    
    Args:
        label: Label prédit ("normal", "benign" ou "malignant")
    
    Returns:
        Enum BiradsCategory correspondant
    """
    if not label:
        return BiradsCategory.BI_RADS_0
    
    lab = label.strip().lower()
    
    if lab.startswith("norm"):
        return BiradsCategory.BI_RADS_1  # Normal
    elif lab.startswith("ben"):
        return BiradsCategory.BI_RADS_2  # Bénin
    elif lab.startswith("mal"):
        return BiradsCategory.BI_RADS_5  # Malin
    else:
        logger.warning(f"Label inconnu '{label}', retour BI-RADS 0")
        return BiradsCategory.BI_RADS_0


@router.post("/predict", response_model=ImageAnalysis)
async def create_and_predict_image_analysis(
    patient_id: int | None = None,
    file: UploadFile = File(...),
    confidence_threshold: float = Query(
        0.60, 
        ge=0.0, 
        le=1.0,
        description="Seuil de confiance minimum (défaut: 0.60)"
    ),
    uncertainty_threshold: float = Query(
        0.35,
        ge=0.0,
        le=1.0,
        description="Seuil d'écart benign/malignant pour normal (défaut: 0.35)"
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Analyse une image mammographique et retourne la classification BI-RADS.
    
    Logique de classification :
    - Si confiance < confidence_threshold → NORMAL (BI-RADS 1)
    - Si incertitude benign/malignant (écart < uncertainty_threshold) → NORMAL
    - Sinon → prédiction du modèle (benign=BI-RADS 2, malignant=BI-RADS 5)
    
    Args:
        patient_id: ID optionnel du patient
        file: Image à analyser (JPEG ou PNG)
        confidence_threshold: Seuil de confiance (0.0-1.0)
        uncertainty_threshold: Seuil d'incertitude (0.0-1.0)
        db: Session base de données
    
    Returns:
        ImageAnalysis avec le résultat de la classification
    
    Example:
        POST /image-analysis/predict?confidence_threshold=0.65&uncertainty_threshold=0.30
    """
    # 1. Validation du format de fichier
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Format de fichier non supporté. Utiliser JPEG ou PNG.",
        )
    
    # 2. Sauvegarde du fichier uploadé
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = f"./uploads/{unique_filename}"
    os.makedirs("./uploads", exist_ok=True)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Prédiction avec le modèle ML (avec détection de "normal")
        prediction = predict(
            file_path, 
            confidence_threshold=confidence_threshold,
            uncertainty_threshold=uncertainty_threshold
        )
        
        # 4. Conversion du label en BiradsCategory
        birads_category = _label_to_birads(prediction.label)
        
        # 5. Log pour traçabilité
        if prediction.reclassified_as_normal:
            logger.info(
                f"✅ Image {unique_filename} reclassée NORMAL "
                f"(confiance: {prediction.confidence:.2%})"
            )
        else:
            logger.info(
                f"✅ Image {unique_filename} classée {prediction.label.upper()} "
                f"(confiance: {prediction.confidence:.2%})"
            )
        
        # 6. Création de l'analyse en base de données
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
        await db.commit()
        await db.refresh(analysis_db)
        
        logger.info(f"💾 Analyse enregistrée : ID={analysis_db.id}, BI-RADS={birads_category.value}")
        
        return analysis_db
    
    except Exception as e:
        logger.exception(f"❌ Erreur lors de l'analyse de {unique_filename}")
        
        # Nettoyage du fichier en cas d'erreur
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse de l'image : {str(e)}"
        )