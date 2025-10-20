# app/api/v1/routes/ingest.py
import os
import uuid
import csv
import shutil
from pathlib import Path
from typing import List, Tuple, Literal, Dict

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from urllib.parse import quote

from app.api.v1.schemas.ingest import IngestItem, IngestResponse, PredResult
from app.db.database import get_db
from app.api.v1.models.image_analysis import ImageAnalysis
from app.api.v1.models.enums import BiradsCategory, AnalysisStatus
from app.ingest.config import RAW_IMG_DIR, RAW_DICOM_DIR, IMG_DIR, ANON_DICOM_DIR, AGES_CSV
from app.ingest.pipeline import run as run_pipeline
from app.ml.predictor import predict as ml_predict

try:
    from app.ingest.overlay import overlay_tag, TAGGED_DIR
except Exception:
    overlay_tag = None
    TAGGED_DIR = None

router = APIRouter(prefix="/ingest", tags=["Anonymisation et traitement d’images"])

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DICOM_EXT = {".dcm", ".DCM"}


def _save_uploaded_files(files: List[UploadFile]) -> List[Tuple[str, str, str]]:
    saved: List[Tuple[str, str, str]] = []
    RAW_IMG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DICOM_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = f.filename or "file"
        ext = os.path.splitext(name)[1].lower()
        if ext in PHOTO_EXT:
            dest_dir, kind = RAW_IMG_DIR, "photo"
        elif ext in DICOM_EXT:
            dest_dir, kind = RAW_DICOM_DIR, "dicom"
        else:
            ct = (f.content_type or "").lower()
            if "dicom" in ct:
                dest_dir, kind, ext = RAW_DICOM_DIR, "dicom", ".dcm"
            elif ct.startswith("image/"):
                dest_dir, kind = RAW_IMG_DIR, "photo"
                ext = ext or ".png"
            else:
                raise HTTPException(status_code=400, detail=f"Format non supporté: {name}")
        fname = f"{uuid.uuid4().hex}{ext}"
        dest = dest_dir / fname
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out, length=1024 * 1024)
        saved.append((name, str(dest), kind))
    return saved


def _read_age_for_ids(ids: List[str]) -> dict:
    out = {}
    if not ids or not AGES_CSV.exists():
        return out
    needed = set(ids)
    with AGES_CSV.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rid = (row.get("id") or "").strip()
            if rid in needed:
                val = row.get("age")
                src = row.get("source") or ""
                try:
                    age_val = int(val) if val else None
                except Exception:
                    age_val = None
                out[rid] = {"age": age_val, "source": src}
    return out


def _map_label_to_birads(label: str) -> BiradsCategory:
    lab = (label or "").strip().lower()
    if lab == "normal":
        return BiradsCategory.BI_RADS_1
    if lab == "benign":
        return BiradsCategory.BI_RADS_2
    if lab == "malignant":
        return BiradsCategory.BI_RADS_5
    return BiradsCategory.BI_RADS_0


@router.get("/download/{kind}/{filename}", name="ingest_download")
def ingest_download(kind: Literal["png", "dicom", "tagged"], filename: str):
    if kind == "png":
        path = IMG_DIR / filename
    elif kind == "dicom":
        path = ANON_DICOM_DIR / filename
    else:
        if TAGGED_DIR is None:
            raise HTTPException(status_code=404, detail="Fichier introuvable")
        path = TAGGED_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    return FileResponse(path)


@router.post("/anonymize", response_model=IngestResponse)
def anonymize_and_optionally_predict(
    request: Request,
    files: List[UploadFile] = File(...),
    run_inference: bool = Query(False),
    persist: bool = Query(True),
    db: Session = Depends(get_db),
):
    saved = _save_uploaded_files(files)
    input_paths = [Path(p) for _, p, _ in saved]
    out = run_pipeline(input_paths, out_img_dir=IMG_DIR, out_dicom_dir=ANON_DICOM_DIR)
    new_ids = [img_id for img_id, _, _ in out]
    ages_map = _read_age_for_ids(new_ids)

    def _file_url(kind: Literal["png", "dicom", "tagged"], filename: str) -> str:
        return str(request.url_for("ingest_download", kind=kind, filename=quote(filename)))

    processed_items: List[IngestItem] = [
        IngestItem(original_filename=name, saved_as=path, kind=kind) for (name, path, kind) in saved
    ]

    id_to_item: Dict[str, IngestItem] = {}
    for (nid, png_path, dicom_path) in out:
        age_entry = ages_map.get(nid, {})
        png_name = os.path.basename(str(png_path))
        dcm_name = os.path.basename(str(dicom_path)) if dicom_path else None
        item = IngestItem(
            original_filename="(generated)",
            saved_as=str(png_path),
            kind="photo",
            anonymized_image_id=nid,
            anonymized_png=_file_url("png", png_name),
            anonymized_dicom=_file_url("dicom", dcm_name) if dcm_name else None,
            age_value=age_entry.get("age"),
            age_source=age_entry.get("source"),
        )
        processed_items.append(item)
        id_to_item[nid] = item

    pred_count = 0
    if run_inference:
        to_persist: List[ImageAnalysis] = []
        for (nid, png_path, _) in out:
            local_png_path = str(png_path)
            res = ml_predict(local_png_path)
            it = id_to_item.get(nid)
            if it:
                it.prediction = PredResult(label=res.label, birads=res.birads, confidence=float(res.confidence))
            pred_count += 1
            if overlay_tag is not None:
                base = os.path.splitext(os.path.basename(local_png_path))[0]
                try:
                    tagged = overlay_tag(local_png_path, base, res.label, res.birads, float(res.confidence))
                    if TAGGED_DIR is not None and tagged and it:
                        tagged_name = os.path.basename(tagged)
                        try:
                            it.prediction.tagged_url = _file_url("tagged", tagged_name)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                except Exception:
                    pass
            if persist:
                try:
                    birads_enum = _map_label_to_birads(res.label)
                    to_persist.append(
                        ImageAnalysis(
                            filename=os.path.basename(local_png_path),
                            result_class=birads_enum,
                            confidence=float(res.confidence),
                            description=f"Pred: {res.label} ({res.birads})",
                            status=AnalysisStatus.COMPLETED,
                        )
                    )
                except Exception:
                    pass
        if persist and to_persist:
            try:
                db.bulk_save_objects(to_persist)
                db.commit()
            except Exception:
                db.rollback()

    n_photos = sum(1 for _, _, k in saved if k == "photo")
    n_dicoms = sum(1 for _, _, k in saved if k == "dicom")
    counts = {
        "uploaded_photos": n_photos,
        "uploaded_dicoms": n_dicoms,
        "new_anonymized_png": len(out),
        "predictions_done": pred_count,
    }

    return IngestResponse(processed=processed_items, counts=counts)
