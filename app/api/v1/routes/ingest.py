# app/api/v1/routes/ingest.py - AVEC PREVIEW
from __future__ import annotations

import os
import io
import time
import zipfile
import uuid
import shutil
from pathlib import Path
from typing import List, Optional, Iterable, Dict

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse, Response

from app.ingest.config import DERIVED_IMG_DIR, NORMALIZED_DICOM_DIR
from app.api.v1.schemas.files import FileItem

try:
    import pydicom
    import cv2
    import numpy as np
    _HAS_DICOM_PREVIEW = True
except Exception:
    _HAS_DICOM_PREVIEW = False

try:
    from app.ingest.overlay import TAGGED_DIR  # type: ignore
except Exception:
    TAGGED_DIR = None  # type: ignore[assignment]

router = APIRouter(prefix="/ingest", tags=["Fichiers et exportations"])

def _safe_roots() -> Dict[str, Path]:
    roots = {
        "pgm": DERIVED_IMG_DIR,
        "dicom": NORMALIZED_DICOM_DIR,
    }
    if TAGGED_DIR is not None:
        roots["tagged"] = TAGGED_DIR
    return roots

def _safe_resolve(kind: str, filename: str) -> Path:
    kind = kind.lower()
    roots = _safe_roots()
    if kind not in roots:
        allowed = "', '".join(roots.keys())
        raise HTTPException(status_code=400, detail=f"kind doit être parmi: '{allowed}'")
    base = roots[kind]
    base.mkdir(parents=True, exist_ok=True)

    p = (base / filename).resolve()
    try:
        p.relative_to(base.resolve())
    except Exception:
        raise HTTPException(status_code=400, detail="Chemin invalide")

    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    return p

def _iter_files(root: Path, pattern: Optional[str] = None) -> Iterable[Path]:
    if not root.exists():
        return []
    if pattern:
        yield from root.glob(pattern)
    else:
        yield from root.glob("*")

def _dicom_to_png_bytes(dcm_path: Path) -> bytes:
    """Convertit DICOM en PNG (bytes) pour preview"""
    if not _HAS_DICOM_PREVIEW:
        raise HTTPException(status_code=501, detail="pydicom/opencv non disponible")
    
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype("float32")
    
    # Normalisation
    arr = 255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr = arr.clip(0, 255).astype("uint8")
    
    # Encoder en PNG
    success, buffer = cv2.imencode('.png', arr)
    if not success:
        raise HTTPException(status_code=500, detail="Erreur conversion PNG")
    
    return buffer.tobytes()

@router.get("/files", response_model=List[FileItem])
def list_files(
    kind: Optional[str] = Query(None, description="pgm | dicom | tagged (si None → tous disponibles)"),
    q: Optional[str] = Query(None, description="pattern glob ex: '*.png' ou 'bfa0*.png'"),
    limit: int = Query(200, ge=1, le=5000),
    order: str = Query("desc", regex="^(asc|desc)$", description="tri par date de modification"),
):
    items: List[FileItem] = []
    roots = _safe_roots()

    if kind is not None:
        kind = kind.lower()
        if kind not in roots:
            allowed = "', '".join(roots.keys())
            raise HTTPException(status_code=400, detail=f"kind doit être parmi: '{allowed}'")
        kinds = [kind]
    else:
        kinds = list(roots.keys())

    for k in kinds:
        root = roots[k]
        files = list(_iter_files(root, q))
        for f in files:
            if f.is_file():
                stat = f.stat()
                items.append(FileItem(
                    kind=k,
                    filename=f.name,
                    size_bytes=stat.st_size,
                    created_at=stat.st_mtime,
                    download_url=f"/api/v1/ingest/download/{k}/{f.name}",
                ))

    items.sort(key=lambda x: x.created_at, reverse=(order == "desc"))
    return items[:limit]

@router.get("/preview/{kind}/{filename}")
def preview_file(kind: str, filename: str):
    """Retourne une image PNG pour preview (convertit DICOM si nécessaire)"""
    p = _safe_resolve(kind, filename)
    kind = kind.lower()
    
    if kind == "dicom":
        # Convertir DICOM en PNG à la volée
        png_bytes = _dicom_to_png_bytes(p)
        return Response(content=png_bytes, media_type="image/png")
    
    elif kind == "pgm" or kind == "tagged":
        # Retourner directement l'image
        return FileResponse(path=str(p), media_type="image/png")
    
    else:
        raise HTTPException(status_code=400, detail="Preview non supporté pour ce type")

@router.get("/download/{kind}/{filename}")
def download_file(kind: str, filename: str):
    p = _safe_resolve(kind, filename)
    kind = kind.lower()
    if kind == "pgm" or (kind == "tagged" and p.suffix.lower() == ".png"):
        media_type = "image/png"
    elif kind == "dicom":
        media_type = "application/dicom"
    else:
        media_type = "application/octet-stream"
    return FileResponse(path=str(p), media_type=media_type, filename=p.name)

@router.post("/upload", response_model=FileItem)
async def upload_file(
    file: UploadFile = File(...),
    kind: str = Query("pgm", description="pgm | dicom"),
):
    kind = kind.lower()
    roots = _safe_roots()
    
    if kind not in ["pgm", "dicom"]:
        raise HTTPException(status_code=400, detail="kind doit être 'pgm' ou 'dicom'")
    
    if kind == "pgm":
        if not any(file.filename.lower().endswith(ext) for ext in [".png", ".pgm", ".jpg", ".jpeg"]):
            raise HTTPException(status_code=400, detail="Format PGM: .png, .pgm, .jpg, .jpeg autorisés")
    elif kind == "dicom":
        if not file.filename.lower().endswith((".dcm", ".dicom")):
            raise HTTPException(status_code=400, detail="Format DICOM: .dcm, .dicom autorisés")
    
    root = roots[kind]
    root.mkdir(parents=True, exist_ok=True)
    
    ext = Path(file.filename).suffix
    base_name = Path(file.filename).stem
    final_name = f"{base_name}_{uuid.uuid4().hex[:8]}{ext}"
    dest_path = root / final_name
    
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    stat = dest_path.stat()
    return FileItem(
        kind=kind,
        filename=final_name,
        size_bytes=stat.st_size,
        created_at=stat.st_mtime,
        download_url=f"/api/v1/ingest/download/{kind}/{final_name}",
    )

@router.post("/upload-batch", response_model=List[FileItem])
async def upload_files_batch(
    files: List[UploadFile] = File(...),
    kind: str = Query("pgm", description="pgm | dicom"),
):
    kind = kind.lower()
    roots = _safe_roots()
    
    if kind not in ["pgm", "dicom"]:
        raise HTTPException(status_code=400, detail="kind doit être 'pgm' ou 'dicom'")
    
    root = roots[kind]
    root.mkdir(parents=True, exist_ok=True)
    
    results: List[FileItem] = []
    
    for file in files:
        if kind == "pgm":
            if not any(file.filename.lower().endswith(ext) for ext in [".png", ".pgm", ".jpg", ".jpeg"]):
                continue
        elif kind == "dicom":
            if not file.filename.lower().endswith((".dcm", ".dicom")):
                continue
        
        ext = Path(file.filename).suffix
        base_name = Path(file.filename).stem
        final_name = f"{base_name}_{uuid.uuid4().hex[:8]}{ext}"
        dest_path = root / final_name
        
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        stat = dest_path.stat()
        results.append(FileItem(
            kind=kind,
            filename=final_name,
            size_bytes=stat.st_size,
            created_at=stat.st_mtime,
            download_url=f"/api/v1/ingest/download/{kind}/{final_name}",
        ))
    
    if not results:
        raise HTTPException(status_code=400, detail="Aucun fichier valide uploadé")
    
    return results

@router.post("/export/zip")
def export_zip(
    filenames: Optional[List[str]] = Query(None),
    kind: str = Query("pgm", description="pgm | dicom | tagged"),
    all_files: bool = Query(False),
):
    roots = _safe_roots()
    kind = kind.lower()
    root = roots.get(kind)
    if root is None:
        allowed = "', '".join(roots.keys())
        raise HTTPException(status_code=400, detail=f"kind doit être parmi: '{allowed}'")
    root.mkdir(parents=True, exist_ok=True)

    to_zip: List[Path] = []
    if all_files:
        to_zip = [p for p in root.glob("*") if p.is_file()]
    else:
        if not filenames:
            raise HTTPException(status_code=400, detail="fournir 'filenames' ou bien 'all_files=true'")
        for name in filenames:
            p = _safe_resolve(kind, name)
            to_zip.append(p)

    if not to_zip:
        raise HTTPException(status_code=404, detail="Aucun fichier à zipper")

    def stream():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pth in to_zip:
                zf.write(pth, arcname=pth.name)
        buf.seek(0)
        while True:
            chunk = buf.read(1024 * 256)
            if not chunk:
                break
            yield chunk
        buf.close()

    ts = int(time.time())
    fname = f"export_{kind}_{ts}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return StreamingResponse(stream(), media_type="application/zip", headers=headers)

@router.delete("/delete/{kind}/{filename}")
def delete_file(kind: str, filename: str):
    p = _safe_resolve(kind, filename)
    try:
        p.unlink()
        return {"message": f"Fichier {filename} supprimé", "kind": kind, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression: {str(e)}")