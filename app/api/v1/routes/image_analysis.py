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
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "application/dicom": ".dcm",
    "image/x-portable-graymap": ".pgm",
    "application/octet-stream": None,
}

ROOT_DIR = Path(__file__).resolve().parents[3]
TAGGED_DIR = ROOT_DIR / "tagged"
TAGGED_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLORS = {
    "normal": (0, 220, 0),
    "benign": (0, 170, 255),
    "malignant": (38, 38, 220),
    "default": (128, 128, 128)
}


def convert_medical_image_to_grayscale(file_path: str) -> np.ndarray:
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.dcm', '.dicom']:
        try:
            import pydicom
            dcm = pydicom.dcmread(file_path)
            img_array = dcm.pixel_array
            
            img_array = img_array.astype(np.float32)
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            img_array = (img_array * 255).astype(np.uint8)
            
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            logger.info(f"Image DICOM convertie : {img_array.shape}")
            return img_array
            
        except ImportError:
            raise ValueError("pydicom n'est pas installé")
        except Exception as e:
            raise ValueError(f"Erreur lecture DICOM : {str(e)}")
    
    elif ext in ['.pgm', '.jpg', '.jpeg', '.png']:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Impossible de lire l'image : {file_path}")
        
        logger.info(f"Image {ext} lue : {img.shape}")
        return img
    
    else:
        raise ValueError(f"Extension non supportée : {ext}")


def create_tagged_image(
    src_path: str,
    base_filename: str,
    label: str,
    birads: str,
    confidence: float
) -> str:
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {src_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    txt = f"{label.upper()} • {birads} • {confidence * 100:.1f}%"
    
    label_lower = label.lower().strip()
    color = LABEL_COLORS.get(label_lower, LABEL_COLORS["default"])
    
    pad_y = 8
    bar_h = 36 + pad_y * 2
    
    cv2.rectangle(img_rgb, (0, 0), (img_rgb.shape[1], bar_h), (0, 0, 0), -1)
    
    cv2.putText(
        img_rgb,
        txt,
        (12, 28 + pad_y // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA
    )
    
    tagged_filename = f"{base_filename}__tag.png"
    out_path = TAGGED_DIR / tagged_filename
    
    if not cv2.imwrite(str(out_path), img_rgb):
        raise ValueError("Impossible d'écrire l'image annotée")
    
    logger.info(f"Image annotée créée : {out_path}")
    return tagged_filename


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
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.60, ge=0.0, le=1.0),
    uncertainty_threshold: float = Query(0.35, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    is_valid = False
    if file.content_type in SUPPORTED_CONTENT_TYPES:
        is_valid = True
    elif file_extension in ['.dcm', '.dicom', '.pgm']:
        is_valid = True
        logger.info(f"Fichier accepté par extension ({file_extension})")
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {file.content_type} ({file_extension})"
        )
    
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = f"./uploads/{unique_filename}"
    os.makedirs("./uploads", exist_ok=True)
    
    temp_png_path = None
    tagged_filename = None
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Fichier sauvegarde : {file_path}")
        
        if file_extension in ['.dcm', '.dicom', '.pgm']:
            img_gray = convert_medical_image_to_grayscale(file_path)
            temp_png_path = f"./uploads/{uuid.uuid4()}_converted.png"
            cv2.imwrite(temp_png_path, img_gray)
            logger.info(f"Conversion en PNG : {temp_png_path}")
            prediction_path = temp_png_path
        else:
            prediction_path = file_path
        
        prediction = predict(
            prediction_path,
            confidence_threshold=confidence_threshold,
            uncertainty_threshold=uncertainty_threshold
        )
        
        birads_category = _label_to_birads(prediction.label)
        
        base_name = os.path.splitext(unique_filename)[0]
        tagged_filename = create_tagged_image(
            prediction_path,
            base_name,
            prediction.label,
            prediction.birads,
            float(prediction.confidence)
        )
        
        if prediction.reclassified_as_normal:
            logger.info(f"Image {unique_filename} reclassee NORMAL")
        else:
            logger.info(f"Image {unique_filename} classee {prediction.label.upper()}")
        
        analysis_db = ImageAnalysisModel(
            filename=unique_filename,
            result_class=birads_category,
            confidence=float(prediction.confidence) if prediction.confidence else None,
            status=AnalysisStatus.COMPLETED,
            description=(
                f"Format: {file_extension.upper()} | "
                f"Pred: {prediction.label} ({prediction.birads}) | "
                f"Model: {prediction.model_type} | "
                f"Tagged: {tagged_filename}"
                + (f" | Reclassifie NORMAL" if prediction.reclassified_as_normal else "")
            ),
        )
        
        db.add(analysis_db)
        db.commit()
        db.refresh(analysis_db)
        
        logger.info(f"Analyse enregistree : ID={analysis_db.id}")
        
        return analysis_db
    
    except ValueError as ve:
        logger.error(f"Erreur de validation : {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    
    except Exception as e:
        logger.exception(f"Erreur lors de l'analyse de {unique_filename}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse : {str(e)}"
        )
    
    finally:
        if temp_png_path and os.path.exists(temp_png_path):
            try:
                os.remove(temp_png_path)
                logger.info(f"Fichier temporaire supprime : {temp_png_path}")
            except Exception as e:
                logger.warning(f"Impossible de supprimer {temp_png_path}: {e}")


@router.get("/tagged/{filename}")
def download_tagged_image(filename: str):
    p = TAGGED_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Image introuvable : {filename}")
    
    from fastapi.responses import FileResponse
    return FileResponse(str(p), media_type="image/png", filename=filename)