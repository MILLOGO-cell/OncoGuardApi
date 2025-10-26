# app/api/v1/routes/image_inference.py - CORRIGÉ

import os
import io
import csv
import uuid
import shutil
import zipfile
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.v1.schemas.image_inference import (
    InferenceResponse,  # si utilisé ailleurs
    InferenceTaggedResponse,
    ResultsResponse,
    ResultItem,
)
from app.ml.predictor import predict as ml_predict
from app.db.database import get_db
from app.api.v1.models.image_analysis import ImageAnalysis
from app.api.v1.models.enums import BiradsCategory, AnalysisStatus

# Optionnel : bases de dossiers existants dans le projet (si présents)
try:
    from app.ingest.config import DERIVED_IMG_DIR as _DERIVED_IMG_DIR, NORMALIZED_DICOM_DIR as _NORMALIZED_DICOM_DIR
except Exception:
    _DERIVED_IMG_DIR, _NORMALIZED_DICOM_DIR = None, None

# pydicom facultatif
try:
    import pydicom

    _HAS_PYDICOM = True
except Exception:
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
ROOT_DIR = Path(__file__).resolve().parents[5] if len(Path(__file__).parents) >= 6 else Path.cwd()
UPLOAD_DIR = ROOT_DIR / "uploads"
TAGGED_DIR = ROOT_DIR / "tagged"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TAGGED_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(file: UploadFile) -> str:
    ext = (os.path.splitext(file.filename or "")[1].lower() or ".png")[:10]
    if ext not in {".jpg", ".jpeg", ".png", ".pgm", ".dcm"}:
        ext = ".png"
    fname = f"{uuid.uuid4()}{ext}"
    dest = UPLOAD_DIR / fname
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return str(dest)


def _dicom_to_png(dcm_path: str) -> str:
    if not _HAS_PYDICOM:
        raise HTTPException(status_code=415, detail="DICOM reçu mais pydicom non installé")
    try:
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype("float32")
    except Exception as e:
        logger.exception("Lecture DICOM échouée")
        raise HTTPException(status_code=400, detail=f"Impossible de lire le DICOM: {e}")

    # Normalisation robuste
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-6:
        arr = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr = 255.0 * (arr - mn) / (mx - mn)
    arr = np.clip(arr, 0, 255).astype("uint8")

    png_path = f"{dcm_path}.__tmp.png"
    if not cv2.imwrite(png_path, arr):
        raise HTTPException(status_code=500, detail="Impossible d'écrire l'image PNG")
    return png_path


def _label_to_birads(label: Optional[str]) -> BiradsCategory:
    lab = (label or "").strip().lower()
    if lab == "normal":
        return BiradsCategory.BI_RADS_1
    if lab == "benign":
        return BiradsCategory.BI_RADS_2
    if lab == "malignant":
        return BiradsCategory.BI_RADS_5
    return BiradsCategory.BI_RADS_0


def _overlay_tag(src_path: str, out_base: str, label: str, birads: str, confidence: float) -> str:
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=500, detail="Impossible de lire l'image pour annotation")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    txt = f"{label} • {birads} • {confidence * 100:.1f}%"

    # Couleur : vert normal, ambre bénin, rouge malin
    color = (
        (36, 180, 58)
        if label.lower() == "normal"
        else ((0, 170, 255) if label.lower() == "benign" else (38, 38, 220))
    )

    pad_y = 8
    bar_h = 36 + pad_y * 2
    cv2.rectangle(img_rgb, (0, 0), (img_rgb.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(
        img_rgb, txt, (12, 28 + pad_y // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )

    out_path = TAGGED_DIR / f"{out_base}__tag.png"
    if not cv2.imwrite(str(out_path), img_rgb):
        raise HTTPException(status_code=500, detail="Impossible d'écrire l'image annotée")
    return str(out_path)


def _is_allowed_upload(file: UploadFile) -> bool:
    return (file.content_type in ALLOWED_IMG) or any(
        (file.filename or "").lower().endswith(e) for e in (".jpg", ".jpeg", ".png", ".pgm", ".dcm")
    )


@router.post("/predict", response_model=InferenceTaggedResponse)
async def predict_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not _is_allowed_upload(file):
        raise HTTPException(status_code=400, detail="Formats autorisés: JPEG, PNG, PGM, DICOM")

    saved_path = _save_upload(file)
    work_path = saved_path
    if (file.content_type in {"application/dicom", "application/dicom+json"}) or saved_path.lower().endswith(
        ".dcm"
    ):
        work_path = _dicom_to_png(saved_path)

    try:
        res = ml_predict(work_path)  # doit renvoyer un objet avec label, birads, confidence
        base = os.path.splitext(os.path.basename(saved_path))[0]
        tagged_path = _overlay_tag(work_path, base, res.label, res.birads, float(res.confidence))
    finally:
        if work_path != saved_path and os.path.exists(work_path):
            try:
                os.remove(work_path)
            except Exception:
                pass

    # Persist
    try:
        birads_enum = _label_to_birads(res.label)
        row = ImageAnalysis(
            filename=os.path.basename(saved_path),
            result_class=birads_enum,
            confidence=float(res.confidence),
            description=f"Pred: {res.label} ({res.birads})",
            status=AnalysisStatus.COMPLETED,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    except Exception as e:
        db.rollback()
        logger.exception("[image-inference] Échec enregistrement DB: %s", e)

    return InferenceTaggedResponse(
        label=res.label,
        birads=res.birads,
        confidence=float(res.confidence),
        filename=os.path.basename(saved_path),
        tagged_filename=os.path.basename(tagged_path),
    )


@router.post("/predict-batch", response_model=List[InferenceTaggedResponse])
async def predict_images(
    files: List[UploadFile] = File(...), persist: bool = True, db: Session = Depends(get_db)
):
    results: List[InferenceTaggedResponse] = []

    for file in files:
        if not _is_allowed_upload(file):
            # on ignore juste les fichiers non valides
            continue

        saved_path = _save_upload(file)
        work_path = saved_path
        if (file.content_type in {"application/dicom", "application/dicom+json"}) or saved_path.lower().endswith(
            ".dcm"
        ):
            work_path = _dicom_to_png(saved_path)

        try:
            res = ml_predict(work_path)
            base = os.path.splitext(os.path.basename(saved_path))[0]
            tagged_path = _overlay_tag(work_path, base, res.label, res.birads, float(res.confidence))
        finally:
            if work_path != saved_path and os.path.exists(work_path):
                try:
                    os.remove(work_path)
                except Exception:
                    pass

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
                    description=f"Pred: {res.label} ({res.birads})",
                    status=AnalysisStatus.COMPLETED,
                )
                db.add(row)
                db.commit()
            except Exception as e:
                db.rollback()
                logger.exception("[image-inference] Échec enregistrement DB (batch): %s", e)

    return results


@router.get("/tagged/{filename}")
def download_tagged(filename: str):
    p = TAGGED_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Annotée introuvable")
    return FileResponse(str(p), media_type="image/png", filename=filename)


@router.get("/results", response_model=ResultsResponse)
def list_results(
    limit: int = Query(200, ge=1, le=5000),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    db: Session = Depends(get_db),
):
    rows = db.query(ImageAnalysis).all()
    rows_sorted = sorted(
        rows, key=lambda r: r.submitted_at or datetime.min, reverse=(order == "desc")
    )
    items: List[ResultItem] = []

    for r in rows_sorted[:limit]:
        base = os.path.splitext(os.path.basename(r.filename or ""))[0]
        tagged = f"{base}__tag.png" if base else None
        tagged_path = TAGGED_DIR / tagged if tagged else None

        items.append(
            ResultItem(
                id=r.id,
                filename=r.filename,
                label=(r.result_class.value if r.result_class else None),
                birads=(r.result_class.value.split(" - ")[0] if r.result_class else None),
                confidence=(float(r.confidence) if r.confidence is not None else None),
                tagged_filename=(tagged if (tagged_path and tagged_path.exists()) else None),
                submitted_at=(r.submitted_at.isoformat() if r.submitted_at else None),
            )
        )
    return ResultsResponse(items=items, total=len(rows_sorted))


@router.get("/report.csv")
def export_report_csv(db: Session = Depends(get_db)):
    rows = db.query(ImageAnalysis).all()

    def stream():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "filename", "birads", "confidence", "submitted_at"])
        for r in rows:
            w.writerow(
                [
                    r.id,
                    r.filename,
                    r.result_class.value if r.result_class else "",
                    f"{float(r.confidence):.4f}" if r.confidence is not None else "",
                    r.submitted_at.isoformat() if r.submitted_at else "",
                ]
            )
        buf.seek(0)
        yield buf.read()
        buf.close()

    headers = {"Content-Disposition": 'attachment; filename="report.csv"'}
    return StreamingResponse(stream(), media_type="text/csv", headers=headers)


@router.get("/export-tagged.zip")
def export_tagged_zip():
    """Streaming ZIP en mémoire par chunks (évite les reads 1 byte)."""
    files = [f for f in os.listdir(TAGGED_DIR) if f.lower().endswith(".png")]
    if not files:
        raise HTTPException(status_code=404, detail="Aucune image annotée")

    def stream():
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            for name in files:
                z.write(os.path.join(TAGGED_DIR, name), arcname=name)
        mem.seek(0)
        # chunks de 256KB
        for chunk in iter(lambda: mem.read(256 * 1024), b""):
            yield chunk
        mem.close()

    headers = {"Content-Disposition": 'attachment; filename="tagged_images.zip"'}
    return StreamingResponse(stream(), media_type="application/zip", headers=headers)


# --------- Prédiction depuis fichiers déjà présents ---------

class PredictFromUploadedRequest(BaseModel):
    """Schéma pour prédire depuis des fichiers uploadés/anonymisés."""
    kind: str  # "dicom" ou "pgm"
    filenames: List[str]


def _safe_base_dir(kind: str) -> Path:
    kind = kind.lower()
    if kind == "dicom":
        if _NORMALIZED_DICOM_DIR:
            return Path(_NORMALIZED_DICOM_DIR)
        return UPLOAD_DIR
    if kind == "pgm":
        if _DERIVED_IMG_DIR:
            return Path(_DERIVED_IMG_DIR)
        return UPLOAD_DIR
    raise HTTPException(status_code=400, detail="Type invalide: utilisez 'dicom' ou 'pgm'")


def _resolve_within(base_dir: Path, filename: str) -> Path:
    """
    Résout un chemin et garantit qu'il reste à l'intérieur de base_dir
    (protection contre les traversals).
    """
    p = (base_dir / filename).resolve()
    if not str(p).startswith(str(base_dir.resolve())):
        raise HTTPException(status_code=400, detail=f"Accès interdit: {filename}")
    return p


@router.post("/predict-from-uploaded")
async def predict_from_uploaded_files(payload: PredictFromUploadedRequest, db: Session = Depends(get_db)):
    """
    Prédit les classifications depuis des fichiers déjà présents (DICOM ou images),
    annote et persiste les résultats de manière homogène avec /predict.
    """
    base_dir = _safe_base_dir(payload.kind)
    if not base_dir.exists():
        raise HTTPException(status_code=500, detail=f"Répertoire introuvable: {base_dir}")

    results: List[InferenceTaggedResponse] = []

    for name in payload.filenames:
        try:
            src = _resolve_within(base_dir, name)
            if not src.exists():
                raise HTTPException(status_code=404, detail=f"Fichier introuvable: {name}")

            # Prépare un chemin exploitable pour le modèle
            work_path = str(src)
            tmp_to_cleanup: Optional[str] = None

            if payload.kind == "dicom" or src.suffix.lower() == ".dcm":
                work_path = _dicom_to_png(str(src))
                tmp_to_cleanup = work_path

            # Prédiction via le même pipeline
            res = ml_predict(work_path)  # -> label, birads, confidence

            base = os.path.splitext(os.path.basename(str(src)))[0]
            tagged_path = _overlay_tag(work_path, base, res.label, res.birads, float(res.confidence))

            # Persist
            try:
                birads_enum = _label_to_birads(res.label)
                row = ImageAnalysis(
                    filename=os.path.basename(str(src)),
                    result_class=birads_enum,
                    confidence=float(res.confidence),
                    description=f"Pred: {res.label} ({res.birads})",
                    status=AnalysisStatus.COMPLETED,
                )
                db.add(row)
                db.commit()
            except Exception as e:
                db.rollback()
                logger.exception("[image-inference] Échec enregistrement DB (uploaded): %s", e)

            results.append(
                InferenceTaggedResponse(
                    label=res.label,
                    birads=res.birads,
                    confidence=float(res.confidence),
                    filename=os.path.basename(str(src)),
                    tagged_filename=os.path.basename(tagged_path),
                )
            )
        finally:
            # Nettoyage PNG temporaire si conversion DICOM → PNG
            if 'tmp_to_cleanup' in locals() and tmp_to_cleanup and os.path.exists(tmp_to_cleanup):
                try:
                    os.remove(tmp_to_cleanup)
                except Exception:
                    pass

    return results

