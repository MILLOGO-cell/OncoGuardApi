import os
import uuid
from typing import List, Tuple, Set
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from app.api.v1.schemas.ingest import IngestItem, IngestResponse, PredResult
from app.db.database import get_db
from app.api.v1.models.image_analysis import ImageAnalysis
from app.api.v1.models.enums import BiradsCategory, AnalysisStatus
from app.ingest.config import RAW_IMG_DIR, RAW_DICOM_DIR, IMG_DIR, ANON_DICOM_DIR, AGES_CSV
from app.ingest.pipeline import run as run_pipeline
import csv
from app.ml.predictor import predict as ml_predict

router = APIRouter(prefix="/ingest", tags=["Anonymisation et traitement d’images"])

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DICOM_EXT = {".dcm", ".DCM"}

def _save_uploaded_files(files: List[UploadFile]) -> List[Tuple[str, str, str]]:
    saved = []
    RAW_IMG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DICOM_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = f.filename or "file"
        ext = os.path.splitext(name)[1].lower()
        if ext in PHOTO_EXT:
            dest_dir = RAW_IMG_DIR; kind = "photo"
        elif ext in DICOM_EXT:
            dest_dir = RAW_DICOM_DIR; kind = "dicom"
        else:
            ct = (f.content_type or "").lower()
            if "dicom" in ct:
                dest_dir = RAW_DICOM_DIR; kind = "dicom"; ext = ".dcm"
            elif "image/" in ct:
                dest_dir = RAW_IMG_DIR; kind = "photo"; ext = ".png"
            else:
                raise HTTPException(status_code=400, detail=f"Format non supporté: {name}")
        fname = f"{uuid.uuid4().hex}{ext}"
        dest = dest_dir / fname
        with dest.open("wb") as out:
            out.write(f.file.read())
        saved.append((name, str(dest), kind))
    return saved

def _snapshot_images() -> Set[str]:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    return {p.name for p in IMG_DIR.glob("*.png")}

def _read_age_for_ids(ids: List[str]) -> dict:
    out = {}
    if not AGES_CSV.exists():
        return out
    with AGES_CSV.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rid = (row.get("id") or "").strip()
            if rid in ids:
                val = row.get("age"); src = row.get("source") or ""
                try:
                    age_val = int(val) if val else None
                except Exception:
                    age_val = None
                out[rid] = {"age": age_val, "source": src}
    return out

def _map_label_to_birads(label: str) -> BiradsCategory:
    lab = (label or "").strip().lower()
    if lab == "normal": return BiradsCategory.BI_RADS_1
    if lab == "benign": return BiradsCategory.BI_RADS_2
    if lab == "malignant": return BiradsCategory.BI_RADS_5
    return BiradsCategory.BI_RADS_0

@router.post("/anonymize", response_model=IngestResponse)
def anonymize_and_optionally_predict(
    files: List[UploadFile] = File(...),
    run_inference: bool = Query(False),
    persist: bool = Query(True),
    db: Session = Depends(get_db),
):
    saved = _save_uploaded_files(files)
    before = _snapshot_images()
    run_pipeline()
    after = _snapshot_images()
    new_pngs = sorted(list(after - before))
    new_ids = [os.path.splitext(n)[0] for n in new_pngs]
    ages_map = _read_age_for_ids(new_ids)

    processed_items: List[IngestItem] = []
    for name, saved_path, kind in saved:
        processed_items.append(IngestItem(
            original_filename=name,
            saved_as=saved_path,
            kind=kind,
        ))

    for nid, png_name in zip(new_ids, new_pngs):
        age_entry = ages_map.get(nid, {})
        processed_items.append(IngestItem(
            original_filename="(generated)",
            saved_as=str(IMG_DIR / png_name),
            kind="photo",
            anonymized_image_id=nid,
            anonymized_png=str(IMG_DIR / png_name),
            anonymized_dicom=None,
            age_value=age_entry.get("age"),
            age_source=age_entry.get("source"),
        ))

    pred_count = 0
    if run_inference:
        for it in processed_items:
            if it.anonymized_png:
                res = ml_predict(it.anonymized_png)
                it.prediction = PredResult(label=res.label, birads=res.birads, confidence=float(res.confidence))
                pred_count += 1
                if persist:
                    try:
                        birads_enum = _map_label_to_birads(res.label)
                        row = ImageAnalysis(
                            filename=os.path.basename(it.anonymized_png),
                            result_class=birads_enum,
                            confidence=float(res.confidence),
                            description=f"Pred: {res.label} ({res.birads})",
                            status=AnalysisStatus.COMPLETED,
                        )
                        db.add(row)
                        db.commit()
                    except Exception:
                        db.rollback()

    n_photos = sum(1 for _, _, k in saved if k == "photo")
    n_dicoms = sum(1 for _, _, k in saved if k == "dicom")
    counts = {
        "uploaded_photos": n_photos,
        "uploaded_dicoms": n_dicoms,
        "new_anonymized_png": len(new_pngs),
        "predictions_done": pred_count,
    }

    return IngestResponse(processed=processed_items, counts=counts)
