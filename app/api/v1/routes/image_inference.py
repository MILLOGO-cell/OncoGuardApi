import os, uuid, shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app.api.v1.schemas.image_inference import InferenceResponse
from app.ml.predictor import predict as ml_predict
from app.db.database import get_db
from app.api.v1.models.image_analysis import ImageAnalysis
from app.api.v1.models.enums import BiradsCategory, AnalysisStatus

try:
    import pydicom
    _HAS_PYDICOM = True
except Exception:
    _HAS_PYDICOM = False

router = APIRouter(prefix="/image-inference", tags=["Analyse d'images"])

ALLOWED_IMG = {"image/jpeg", "image/png", "image/pgm", "application/dicom", "application/dicom+json"}
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _save_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename or "")[1].lower() or ".png"
    fname = f"{uuid.uuid4()}{ext}"
    dest = os.path.join(UPLOAD_DIR, fname)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest

def _dicom_to_png(dcm_path: str) -> str:
    if not _HAS_PYDICOM:
        raise HTTPException(status_code=415, detail="Fichier DICOM reçu mais pydicom n'est pas installé.")
    import numpy as np, cv2, pydicom
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype("float32")
    arr = 255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr = arr.clip(0, 255).astype("uint8")
    png_path = dcm_path + ".__tmp.png"
    cv2.imwrite(png_path, arr)
    return png_path

def _label_to_birads(label: str) -> BiradsCategory:
    lab = (label or "").strip().lower()
    if lab == "normal":
        return BiradsCategory.BI_RADS_1
    if lab == "benign":
        return BiradsCategory.BI_RADS_2
    if lab == "malignant":
        return BiradsCategory.BI_RADS_5
    return BiradsCategory.BI_RADS_0

@router.post("/predict", response_model=InferenceResponse)
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if file.content_type not in ALLOWED_IMG and not any((file.filename or "").lower().endswith(e) for e in (".jpg",".jpeg",".png",".pgm",".dcm")):
        raise HTTPException(status_code=400, detail="Formats autorisés: JPEG, PNG, PGM, DICOM")

    saved_path = _save_upload(file)
    work_path = saved_path

    if (file.content_type in {"application/dicom","application/dicom+json"}) or saved_path.lower().endswith(".dcm"):
        work_path = _dicom_to_png(saved_path)

    res = None
    try:
        res = ml_predict(work_path)
    finally:
        if work_path != saved_path and os.path.exists(work_path):
            try: os.remove(work_path)
            except: pass

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
        print("[image-inference] DB persist failed:", e)

    return InferenceResponse(
        label=res.label,
        birads=res.birads,
        confidence=float(res.confidence),
        filename=os.path.basename(saved_path),
    )
