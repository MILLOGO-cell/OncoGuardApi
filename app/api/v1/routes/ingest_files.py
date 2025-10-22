import os
import io
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Iterable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from app.ingest.config import IMG_DIR, ANON_DICOM_DIR
from app.api.v1.schemas.files import FileItem

router = APIRouter(prefix="/ingest", tags=["Fichiers et exportations"])

SAFE_ROOTS = {
    "png": IMG_DIR,
    "dicom": ANON_DICOM_DIR,
}

def _safe_resolve(kind: str, filename: str) -> Path:
    kind = kind.lower()
    if kind not in SAFE_ROOTS:
        raise HTTPException(status_code=400, detail="kind doit être 'png' ou 'dicom'")
    base = SAFE_ROOTS[kind]
    base.mkdir(parents=True, exist_ok=True)
    p = (base / filename).resolve()
    if base.resolve() not in p.parents and p != base.resolve():
        raise HTTPException(status_code=400, detail="Chemin invalide")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    return p

def _iter_files(root: Path, pattern: Optional[str]=None) -> Iterable[Path]:
    if not root.exists():
        return []
    if pattern:
        yield from root.glob(pattern)
    else:
        yield from root.glob("*")

@router.get("/files", response_model=List[FileItem])
def list_files(
    kind: Optional[str] = Query(None, description="png | dicom (si None → les deux)"),
    q: Optional[str] = Query(None, description="pattern glob ex: '*.png' ou 'bfa0*.png'"),
    limit: int = Query(200, ge=1, le=5000),
    order: str = Query("desc", regex="^(asc|desc)$", description="tri par date de création"),
):
    items: List[FileItem] = []
    kinds = [kind] if kind in {"png", "dicom"} else ["png", "dicom"]
    for k in kinds:
        root = SAFE_ROOTS[k]
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

@router.get("/download/{kind}/{filename}")
def download_file(kind: str, filename: str):
    p = _safe_resolve(kind, filename)
    media_type = "application/octet-stream"
    if kind == "png":
        media_type = "image/png"
    elif kind == "dicom":
        media_type = "application/dicom"
    return FileResponse(path=str(p), media_type=media_type, filename=p.name)

@router.post("/export/zip")
def export_zip(
    filenames: Optional[List[str]] = Query(None, description="noms de fichiers, ex: filenames=bfa001.png&filenames=bfa002.png"),
    kind: str = Query("png", description="png | dicom"),
    all_files: bool = Query(False, description="si true → ignore 'filenames' et compresse tout"),
):
    root = SAFE_ROOTS.get(kind)
    if root is None:
        raise HTTPException(status_code=400, detail="kind doit être 'png' ou 'dicom'")
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
            for p in to_zip:
                zf.write(p, arcname=p.name)
        buf.seek(0)
        yield from buf
        buf.close()

    ts = int(time.time())
    filename = f"export_{kind}_{ts}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(stream(), media_type="application/zip", headers=headers)
