"""
Routes API pour l'inférence et l'analyse d'images mammographiques.

Ce module fournit des endpoints pour :
- Prédire la classification d'images individuelles ou par lot
- Annoter les images avec les résultats (tagged images)
- Exporter les résultats et les images annotées
- Gérer l'historique des analyses en base de données

MODÈLE SUPPORTÉ :
Le modèle peut classifier en 2 ou 3 classes selon l'entraînement :
- Version 2 classes (CBIS-DDSM) : benign → BI-RADS 2, malignant → BI-RADS 5
- Version 3 classes (CBIS-DDSM + MIAS) : normal → BI-RADS 1, benign → BI-RADS 2, malignant → BI-RADS 5

Le nombre de classes est détecté automatiquement depuis le modèle chargé.
"""
import os
import io
import csv
import uuid
import shutil
import zipfile
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.v1.schemas.image_inference import (
    InferenceTaggedResponse,
    ResultsResponse,
    ResultItem,
)
from app.ml.predictor import predict as ml_predict, get_model_info
from app.db.database import get_db
from app.api.v1.models.image_analysis import ImageAnalysis
from app.api.v1.models.enums import BiradsCategory, AnalysisStatus
from app.core.config import UPLOAD_DIR, TAGGED_DIR

try:
    import pydicom
    _HAS_PYDICOM = True
except ImportError:
    _HAS_PYDICOM = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/image-inference", tags=["Analyse d'images"])

ALLOWED_IMG = {
    "image/jpeg",
    "image/png",
    "image/pgm",
    "application/dicom",
    "application/dicom+json",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pgm", ".dcm"}

LABEL_COLORS = {
    "normal": (0, 220, 0),
    "benign": (0, 170, 255),
    "malignant": (38, 38, 220),
    "default": (128, 128, 128)
}


class PredictFromUploadedRequest(BaseModel):
    kind: str = Field(..., description="Type de fichiers: 'dicom' ou 'pgm'")
    filenames: List[str] = Field(..., description="Liste des noms de fichiers")


class ModelInfoResponse(BaseModel):
    classes: List[str]
    n_classes: int
    feature_dim: str
    model_type: str
    input_size: str
    dataset: str
    note: str


def _save_upload(file: UploadFile) -> str:
    ext = (os.path.splitext(file.filename or "")[1].lower() or ".png")[:10]
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".png"
    fname = f"{uuid.uuid4()}{ext}"
    dest = UPLOAD_DIR / fname
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.exception("Erreur lors de la sauvegarde du fichier")
        raise HTTPException(
            status_code=500,
            detail=f"Impossible de sauvegarder le fichier : {e}"
        )
    logger.debug(f"Fichier sauvegardé : {dest}")
    return str(dest)


def _dicom_to_png(dcm_path: str) -> str:
    if not _HAS_PYDICOM:
        raise HTTPException(
            status_code=415, 
            detail="DICOM reçu mais pydicom non installé. Installez avec: pip install pydicom"
        )
    try:
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype("float32")
    except Exception as e:
        logger.exception("Échec lecture DICOM")
        raise HTTPException(
            status_code=400, 
            detail=f"Impossible de lire le DICOM: {e}"
        )
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-6:
        arr = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr = 255.0 * (arr - mn) / (mx - mn)
    arr = np.clip(arr, 0, 255).astype("uint8")
    png_path = f"{dcm_path}.__tmp.png"
    if not cv2.imwrite(png_path, arr):
        raise HTTPException(
            status_code=500, 
            detail="Impossible d'écrire l'image PNG temporaire"
        )
    logger.debug(f"DICOM converti en PNG temporaire : {png_path}")
    return png_path


def _label_to_birads(label: Optional[str]) -> BiradsCategory:
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


def _extract_birads_code(birads_category: Optional[BiradsCategory]) -> Optional[str]:
    if not birads_category:
        return None
    value = birads_category.value
    parts = value.split(" - ", 1)
    return parts[0].strip() if parts else "0"


def _overlay_tag(src_path: str, out_base: str, label: str, birads: str, confidence: float) -> str:
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(
            status_code=500, 
            detail=f"Impossible de lire l'image pour annotation : {src_path}"
        )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    txt = f"{label.upper()} • {birads} • {confidence * 100:.1f}%"
    label_lower = label.lower().strip()
    color = LABEL_COLORS.get(label_lower, LABEL_COLORS["default"])
    pad_y = 8
    bar_h = 36 + pad_y * 2
    cv2.rectangle(img_rgb, (0, 0), (img_rgb.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(img_rgb, txt, (12, 28 + pad_y // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    out_path = TAGGED_DIR / f"{out_base}__tag.png"
    if not cv2.imwrite(str(out_path), img_rgb):
        raise HTTPException(
            status_code=500, 
            detail="Impossible d'écrire l'image annotée"
        )
    logger.info(f"Image annotée créée : {out_path}")
    return str(out_path)


def _is_allowed_upload(file: UploadFile) -> bool:
    if file.content_type in ALLOWED_IMG:
        return True
    if file.filename:
        ext = file.filename.lower()
        return any(ext.endswith(e) for e in ALLOWED_EXTENSIONS)
    return False


def _cleanup_temp_file(file_path: str) -> None:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.debug(f"Fichier temporaire supprimé : {file_path}")
        except Exception as e:
            logger.warning(f"Impossible de supprimer le fichier temporaire {file_path} : {e}")


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_information():
    try:
        info = get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.exception("Erreur récupération infos modèle")
        raise HTTPException(
            status_code=500, 
            detail=f"Impossible de charger les infos du modèle: {e}"
        )


@router.post("/predict", response_model=InferenceTaggedResponse)
async def predict_single_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not _is_allowed_upload(file):
        raise HTTPException(
            status_code=400, 
            detail="Formats autorisés: JPEG, PNG, PGM, DICOM"
        )
    saved_path = _save_upload(file)
    work_path = saved_path
    is_dicom = (
        (file.content_type in {"application/dicom", "application/dicom+json"}) or 
        saved_path.lower().endswith(".dcm")
    )
    if is_dicom:
        work_path = _dicom_to_png(saved_path)
    res = None
    tagged_path = None
    try:
        res = ml_predict(work_path)
        base = os.path.splitext(os.path.basename(saved_path))[0]
        tagged_path = _overlay_tag(work_path, base, res.label, res.birads, float(res.confidence))
    except Exception as e:
        logger.exception("Erreur lors de la prédiction ou de l'annotation")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse de l'image : {e}"
        )
    finally:
        if is_dicom and work_path != saved_path:
            _cleanup_temp_file(work_path)
    try:
        birads_enum = _label_to_birads(res.label)
        row = ImageAnalysis(
            filename=os.path.basename(saved_path),
            result_class=birads_enum,
            confidence=float(res.confidence),
            description=f"Pred: {res.label} ({res.birads}) - Model: {res.model_type}",
            status=AnalysisStatus.COMPLETED,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        logger.info(f"Analyse enregistrée en DB : ID={row.id}, Label={res.label}, BI-RADS={birads_enum.value}")
    except Exception as e:
        db.rollback()
        logger.exception(f"Échec enregistrement DB: {e}")
    return InferenceTaggedResponse(
        label=res.label,
        birads=res.birads,
        confidence=float(res.confidence),
        filename=os.path.basename(saved_path),
        tagged_filename=os.path.basename(tagged_path),
    )


@router.post("/predict-batch", response_model=List[InferenceTaggedResponse])
async def predict_multiple_images(
    files: List[UploadFile] = File(...), 
    persist: bool = Query(True, description="Enregistrer les résultats en DB"),
    db: Session = Depends(get_db)
):
    results: List[InferenceTaggedResponse] = []
    for idx, file in enumerate(files):
        if not _is_allowed_upload(file):
            logger.warning(f"Fichier {idx+1}/{len(files)} ignoré (format non autorisé)")
            continue
        saved_path = None
        work_path = None
        try:
            saved_path = _save_upload(file)
            work_path = saved_path
            is_dicom = (
                (file.content_type in {"application/dicom", "application/dicom+json"}) or 
                saved_path.lower().endswith(".dcm")
            )
            if is_dicom:
                work_path = _dicom_to_png(saved_path)
            res = ml_predict(work_path)
            base = os.path.splitext(os.path.basename(saved_path))[0]
            tagged_path = _overlay_tag(work_path, base, res.label, res.birads, float(res.confidence))
            if is_dicom and work_path != saved_path:
                _cleanup_temp_file(work_path)
            results.append(
                InferenceTaggedResponse(
                    label=res.label,
                    birads=res.birads,
                    confidence=float(res.confidence),
                    filename=os.path.basename(saved_path),
                    tagged_filename=os.path.basename(tagged_path),
                )
            )
            if persist:
                try:
                    birads_enum = _label_to_birads(res.label)
                    row = ImageAnalysis(
                        filename=os.path.basename(saved_path),
                        result_class=birads_enum,
                        confidence=float(res.confidence),
                        description=f"Batch pred: {res.label} ({res.birads})",
                        status=AnalysisStatus.COMPLETED,
                    )
                    db.add(row)
                    db.commit()
                except Exception as e:
                    db.rollback()
                    logger.exception(f"Échec enregistrement DB (batch, fichier {idx+1}): {e}")
        except Exception as e:
            logger.exception(f"Erreur traitement fichier {idx+1}/{len(files)}: {e}")
            continue
    if not results:
        raise HTTPException(
            status_code=400,
            detail="Aucun fichier valide n'a pu être traité"
        )
    logger.info(f"Batch prediction terminé : {len(results)}/{len(files)} fichiers traités avec succès")
    return results


@router.get("/tagged/{filename}")
def download_tagged_image(filename: str):
    p = TAGGED_DIR / filename
    if not p.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Image annotée introuvable : {filename}"
        )
    return FileResponse(str(p), media_type="image/png", filename=filename)


@router.get("/results", response_model=ResultsResponse)
def list_analysis_results(
    limit: int = Query(200, ge=1, le=5000, description="Nombre max de résultats"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Ordre de tri"),
    db: Session = Depends(get_db)
):
    rows = db.query(ImageAnalysis).all()
    rows_sorted = sorted(rows, key=lambda r: r.submitted_at or datetime.min, reverse=(order == "desc"))
    items: List[ResultItem] = []
    for r in rows_sorted[:limit]:
        base = os.path.splitext(os.path.basename(r.filename or ""))[0]
        tagged = f"{base}__tag.png" if base else None
        tagged_path = TAGGED_DIR / tagged if tagged else None
        tagged_exists = tagged_path and tagged_path.exists()
        items.append(
            ResultItem(
                id=r.id,
                filename=r.filename,
                label=(r.result_class.value if r.result_class else None),
                birads=_extract_birads_code(r.result_class),
                confidence=(float(r.confidence) if r.confidence is not None else None),
                tagged_filename=(tagged if tagged_exists else None),
                submitted_at=(r.submitted_at.isoformat() if r.submitted_at else None),
            )
        )
    logger.info(f"Retour de {len(items)} résultats sur {len(rows_sorted)} total (limit={limit})")
    return ResultsResponse(items=items, total=len(rows_sorted))


@router.get("/report.csv")
def export_results_csv(db: Session = Depends(get_db)):
    rows = db.query(ImageAnalysis).all()
    def stream():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "filename", "birads", "confidence", "submitted_at"])
        for r in rows:
            w.writerow([
                r.id,
                r.filename,
                r.result_class.value if r.result_class else "",
                f"{float(r.confidence):.4f}" if r.confidence is not None else "",
                r.submitted_at.isoformat() if r.submitted_at else "",
            ])
        buf.seek(0)
        yield buf.read()
        buf.close()
    headers = {"Content-Disposition": 'attachment; filename="report.csv"'}
    logger.info(f"Export CSV de {len(rows)} résultats")
    return StreamingResponse(stream(), media_type="text/csv; charset=utf-8", headers=headers)


@router.get("/export-tagged.zip")
def export_all_tagged_images():
    files = [f for f in os.listdir(TAGGED_DIR) if f.lower().endswith(".png")]
    if not files:
        raise HTTPException(status_code=404, detail="Aucune image annotée disponible")
    def stream():
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            for name in files:
                full_path = os.path.join(TAGGED_DIR, name)
                z.write(full_path, arcname=name)
        mem.seek(0)
        chunk_size = 256 * 1024
        for chunk in iter(lambda: mem.read(chunk_size), b""):
            yield chunk
        mem.close()
    headers = {"Content-Disposition": 'attachment; filename="tagged_images.zip"'}
    logger.info(f"Export ZIP de {len(files)} images annotées")
    return StreamingResponse(stream(), media_type="application/zip", headers=headers)