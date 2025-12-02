# app/api/v1/endpoints/image_analysis.py

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session
from app.api.v1.schemas.image_analysis import ImageAnalysis
from app.api.v1.models.enums import BiradsCategory, AnalysisStatus
from app.api.v1.models.image_analysis import ImageAnalysis as ImageAnalysisModel
import shutil
import uuid
import os
import logging
import cv2
import numpy as np
from app.db.database import get_db
from app.ml.predictor import predict

logger = logging.getLogger(__name__)
router = APIRouter()

# Types MIME supportés
SUPPORTED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "application/dicom": ".dcm",
    "image/x-portable-graymap": ".pgm",
    "application/octet-stream": None,  # Pour PGM qui peut être mal détecté
}


def convert_medical_image_to_grayscale(file_path: str) -> np.ndarray:
    """
    Convertit une image médicale (DICOM, PGM, JPEG, PNG) en niveaux de gris.
    
    Args:
        file_path: Chemin vers le fichier image
    
    Returns:
        Image en niveaux de gris (numpy array)
    
    Raises:
        ValueError: Si le format n'est pas supporté
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # DICOM
    if ext in ['.dcm', '.dicom']:
        try:
            import pydicom
            dcm = pydicom.dcmread(file_path)
            img_array = dcm.pixel_array
            
            # Normaliser l'image DICOM
            img_array = img_array.astype(np.float32)
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            img_array = (img_array * 255).astype(np.uint8)
            
            # S'assurer que c'est en niveaux de gris
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            logger.info(f"✅ Image DICOM convertie : {img_array.shape}")
            return img_array
            
        except ImportError:
            raise ValueError(
                "pydicom n'est pas installé. "
                "Installez-le avec: pip install pydicom"
            )
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture du fichier DICOM : {str(e)}")
    
    # PGM, JPEG, PNG
    elif ext in ['.pgm', '.jpg', '.jpeg', '.png']:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Impossible de lire l'image : {file_path}")
        
        logger.info(f"✅ Image {ext} lue : {img.shape}")
        return img
    
    else:
        raise ValueError(
            f"Extension non supportée : {ext}. "
            f"Formats acceptés : .dcm, .pgm, .jpg, .jpeg, .png"
        )


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
    """
    Analyse une image mammographique et retourne la classification BI-RADS.
    
    Formats supportés :
    - JPEG/PNG : Images standard
    - DICOM (.dcm) : Format médical standard
    - PGM (.pgm) : Format d'images en niveaux de gris
    """
    # Vérifier l'extension du fichier
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Validation plus souple du type de contenu
    is_valid = False
    if file.content_type in SUPPORTED_CONTENT_TYPES:
        is_valid = True
    elif file_extension in ['.dcm', '.dicom', '.pgm']:
        # Accepter DICOM et PGM même si le content_type est mal détecté
        is_valid = True
        logger.info(
            f"Fichier accepté par extension ({file_extension}) "
            f"malgré content_type={file.content_type}"
        )
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Format de fichier non supporté : {file.content_type} ({file_extension}). "
                f"Formats acceptés : JPEG, PNG, DICOM (.dcm), PGM (.pgm)"
            ),
        )
    
    # Générer un nom de fichier unique
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = f"./uploads/{unique_filename}"
    os.makedirs("./uploads", exist_ok=True)
    
    temp_png_path = None
    
    try:
        # Sauvegarder le fichier uploadé
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"📁 Fichier sauvegardé : {file_path}")
        
        # Si c'est DICOM ou PGM, convertir en PNG pour le modèle
        if file_extension in ['.dcm', '.dicom', '.pgm']:
            img_gray = convert_medical_image_to_grayscale(file_path)
            
            # Créer un fichier PNG temporaire
            temp_png_path = f"./uploads/{uuid.uuid4()}_converted.png"
            cv2.imwrite(temp_png_path, img_gray)
            
            logger.info(f"🔄 Conversion en PNG : {temp_png_path}")
            prediction_path = temp_png_path
        else:
            prediction_path = file_path
        
        # Effectuer la prédiction
        prediction = predict(
            prediction_path, 
            confidence_threshold=confidence_threshold,
            uncertainty_threshold=uncertainty_threshold
        )
        
        birads_category = _label_to_birads(prediction.label)
        
        if prediction.reclassified_as_normal:
            logger.info(
                f"Image {unique_filename} reclassée NORMAL "
                f"(confiance: {prediction.confidence:.2%})"
            )
        else:
            logger.info(
                f"Image {unique_filename} classée {prediction.label.upper()} "
                f"(confiance: {prediction.confidence:.2%})"
            )
        
        # Enregistrer en base de données
        analysis_db = ImageAnalysisModel(
            filename=unique_filename,
            patient_id=patient_id,
            result_class=birads_category,
            confidence=float(prediction.confidence) if prediction.confidence else None,
            status=AnalysisStatus.COMPLETED,
            description=(
                f"Format: {file_extension.upper()} | "
                f"Pred: {prediction.label} ({prediction.birads}) | "
                f"Model: {prediction.model_type}"
                + (f" | Reclassifié NORMAL" if prediction.reclassified_as_normal else "")
            ),
        )
        
        db.add(analysis_db)
        db.commit()
        db.refresh(analysis_db)
        
        logger.info(
            f"✅ Analyse enregistrée : ID={analysis_db.id}, "
            f"BI-RADS={birads_category.value}"
        )
        
        return analysis_db
    
    except ValueError as ve:
        logger.error(f"❌ Erreur de validation : {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    
    except Exception as e:
        logger.exception(f"❌ Erreur lors de l'analyse de {unique_filename}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse de l'image : {str(e)}"
        )
    
    finally:
        # Nettoyer les fichiers temporaires
        if temp_png_path and os.path.exists(temp_png_path):
            try:
                os.remove(temp_png_path)
                logger.info(f"🗑️ Fichier temporaire supprimé : {temp_png_path}")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de supprimer {temp_png_path}: {e}")