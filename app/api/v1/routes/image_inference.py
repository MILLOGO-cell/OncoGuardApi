# app/api/v1/routes/image_inference.py
import os, uuid, shutil, io, csv, zipfile
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from datetime import datetime
from app.api.v1.schemas.image_inference import InferenceResponse, InferenceTaggedResponse, ResultsResponse, ResultItem
from app.ml.predictor import predict as ml_predict
from app.db.database import get_db
from app.api.v1.models.image_analysis import ImageAnalysis
from app.api.v1.models.enums import BiradsCategory, AnalysisStatus

try:
    import pydicom
    _HAS_PYDICOM = True
except Exception:
    _HAS_PYDICOM = False

import cv2
import numpy as np

router = APIRouter(prefix="/image-inference", tags=["Analyse d'images"])

ALLOWED_IMG = {"image/jpeg", "image/png", "image/pgm", "application/dicom", "application/dicom+json"}
UPLOAD_DIR = "./uploads"
TAGGED_DIR = "./tagged"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TAGGED_DIR, exist_ok=True)

def _save_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename or "")[1].lower() or ".png"
    fname = f"{uuid.uuid4()}{ext}"
    dest = os.path.join(UPLOAD_DIR, fname)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest

def _dicom_to_png(dcm_path: str) -> str:
    if not _HAS_PYDICOM:
        raise HTTPException(status_code=415, detail="DICOM reçu mais pydicom non installé")
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype("float32")
    arr = 255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr = arr.clip(0, 255).astype("uint8")
    png_path = dcm_path + ".__tmp.png"
    cv2.imwrite(png_path, arr)
    return png_path

def _label_to_birads(label: str) -> BiradsCategory:
    lab = (label or "").strip().lower()
    if lab == "normal": return BiradsCategory.BI_RADS_1
    if lab == "benign": return BiradsCategory.BI_RADS_2
    if lab == "malignant": return BiradsCategory.BI_RADS_5
    return BiradsCategory.BI_RADS_0

def _overlay_tag(src_path: str, out_base: str, label: str, birads: str, confidence: float) -> str:
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=500, detail="Impossible de lire l'image pour annotation")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    txt = f"{label} • {birads} • {confidence*100:.1f}%"
    color = (36, 180, 58) if label.lower()=="normal" else ((255, 170, 0) if label.lower()=="benign" else (220, 38, 38))
    pad = 8
    cv2.rectangle(img_rgb, (0,0), (img_rgb.shape[1], 36+pad*2), (0,0,0), -1)
    cv2.putText(img_rgb, txt, (12, 28+pad//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    out_path = os.path.join(TAGGED_DIR, f"{out_base}__tag.png")
    cv2.imwrite(out_path, img_rgb)
    return out_path

@router.post("/predict", response_model=InferenceTaggedResponse)
async def predict_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED_IMG and not any((file.filename or "").lower().endswith(e) for e in (".jpg",".jpeg",".png",".pgm",".dcm")):
        raise HTTPException(status_code=400, detail="Formats autorisés: JPEG, PNG, PGM, DICOM")
    saved_path = _save_upload(file)
    work_path = saved_path
    if (file.content_type in {"application/dicom","application/dicom+json"}) or saved_path.lower().endswith(".dcm"):
        work_path = _dicom_to_png(saved_path)

    try:
        res = ml_predict(work_path)
        base = os.path.splitext(os.path.basename(saved_path))[0]
        tagged_path = _overlay_tag(work_path, base, res.label, res.birads, float(res.confidence))
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
        db.add(row); db.commit(); db.refresh(row)
    except Exception as e:
        print("[image-inference] DB persist failed:", e)

    return InferenceTaggedResponse(
        label=res.label,
        birads=res.birads,
        confidence=float(res.confidence),
        filename=os.path.basename(saved_path),
        tagged_filename=os.path.basename(tagged_path),
    )

@router.post("/predict-batch", response_model=List[InferenceTaggedResponse])
async def predict_images(files: List[UploadFile] = File(...), persist: bool = True, db: Session = Depends(get_db)):
    results: List[InferenceTaggedResponse] = []
    for file in files:
        if file.content_type not in ALLOWED_IMG and not any((file.filename or "").lower().endswith(e) for e in (".jpg",".jpeg",".png",".pgm",".dcm")):
            continue
        saved_path = _save_upload(file)
        work_path = saved_path
        if (file.content_type in {"application/dicom","application/dicom+json"}) or saved_path.lower().endswith(".dcm"):
            work_path = _dicom_to_png(saved_path)

        try:
            res = ml_predict(work_path)
            base = os.path.splitext(os.path.basename(saved_path))[0]
            tagged_path = _overlay_tag(work_path, base, res.label, res.birads, float(res.confidence))
        finally:
            if work_path != saved_path and os.path.exists(work_path):
                try: os.remove(work_path)
                except: pass

        results.append(InferenceTaggedResponse(
            label=res.label,
            birads=res.birads,
            confidence=float(res.confidence),
            filename=os.path.basename(saved_path),
            tagged_filename=os.path.basename(tagged_path),
        ))

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
                db.add(row); db.commit()
            except Exception as e:
                db.rollback()
                print("[image-inference] DB persist failed:", e)
    return results

@router.get("/tagged/{filename}")
def download_tagged(filename: str):
    p = os.path.join(TAGGED_DIR, filename)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Annotée introuvable")
    return FileResponse(p, media_type="image/png", filename=filename)

@router.get("/results", response_model=ResultsResponse)
def list_results(limit: int = Query(200, ge=1, le=5000), order: str = Query("desc", regex="^(asc|desc)$"), db: Session = Depends(get_db)):
    q = db.query(ImageAnalysis)
    rows = q.all()
    rows_sorted = sorted(rows, key=lambda r: r.submitted_at or datetime.min, reverse=(order=="desc"))
    items: List[ResultItem] = []
    for r in rows_sorted[:limit]:
        base = os.path.splitext(os.path.basename(r.filename))[0]
        tagged = f"{base}__tag.png"
        tagged_path = os.path.join(TAGGED_DIR, tagged)
        items.append(ResultItem(
            id=r.id,
            filename=r.filename,
            label=(r.result_class.value if r.result_class else None),
            birads=(r.result_class.value.split(" - ")[0] if r.result_class else None),
            confidence=(float(r.confidence) if r.confidence is not None else None),
            tagged_filename=(tagged if os.path.exists(tagged_path) else None),
            submitted_at=(r.submitted_at.isoformat() if r.submitted_at else None),
        ))
    return ResultsResponse(items=items, total=len(rows_sorted))

@router.get("/report.csv")
def export_report_csv(db: Session = Depends(get_db)):
    rows = db.query(ImageAnalysis).all()
    def stream():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id","filename","birads","confidence","submitted_at"])
        for r in rows:
            w.writerow([
                r.id,
                r.filename,
                r.result_class.value if r.result_class else "",
                f"{r.confidence:.4f}" if r.confidence is not None else "",
                r.submitted_at.isoformat() if r.submitted_at else "",
            ])
        buf.seek(0)
        yield buf.read()
        buf.close()
    headers = {"Content-Disposition": 'attachment; filename="report.csv"'}
    return StreamingResponse(stream(), media_type="text/csv", headers=headers)

@router.get("/export-tagged.zip")
def export_tagged_zip():
    files = [f for f in os.listdir(TAGGED_DIR) if f.lower().endswith(".png")]
    if not files:
        raise HTTPException(status_code=404, detail="Aucune image annotée")
    def stream():
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            for name in files:
                z.write(os.path.join(TAGGED_DIR, name), arcname=name)
        mem.seek(0)
        yield from mem
        mem.close()
    headers = {"Content-Disposition": 'attachment; filename="tagged_images.zip"'}
    return StreamingResponse(stream(), media_type="application/zip", headers=headers)
