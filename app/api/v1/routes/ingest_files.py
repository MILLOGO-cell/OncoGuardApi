# app/api/v1/routes/ingest.py  (Files & export)
from __future__ import annotations

import os
import io
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Iterable, Dict

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from app.ingest.config import DERIVED_IMG_DIR, NORMALIZED_DICOM_DIR
from app.api.v1.schemas.files import FileItem

# Support optionnel du dossier "tagged" (overlays)
try:
    from app.ingest.overlay import TAGGED_DIR  # type: ignore
except Exception:
    TAGGED_DIR = None  # type: ignore[assignment]

router = APIRouter(prefix="/ingest", tags=["Fichiers et exportations"])

def _safe_roots() -> Dict[str, Path]:
    roots = {
        "png": DERIVED_IMG_DIR,
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
        # lève ValueError si 'p' n'est pas sous 'base'
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

@router.get("/files", response_model=List[FileItem])
def list_files(
    kind: Optional[str] = Query(None, description="png | dicom | tagged (si None → tous disponibles)"),
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
                    created_at=stat.st_mtime,  # timestamp (float)
                    download_url=f"/api/v1/ingest/download/{k}/{f.name}",
                ))

    items.sort(key=lambda x: x.created_at, reverse=(order == "desc"))
    return items[:limit]

@router.get("/download/{kind}/{filename}")
def download_file(kind: str, filename: str):
    p = _safe_resolve(kind, filename)
    kind = kind.lower()
    if kind == "png" or (kind == "tagged" and p.suffix.lower() == ".png"):
        media_type = "image/png"
    elif kind == "dicom":
        media_type = "application/dicom"
    else:
        media_type = "application/octet-stream"
    return FileResponse(path=str(p), media_type=media_type, filename=p.name)

@router.post("/export/zip")
def export_zip(
    filenames: Optional[List[str]] = Query(None, description="noms de fichiers, ex: filenames=bfa001.png&filenames=bfa002.png"),
    kind: str = Query("png", description="png | dicom | tagged"),
    all_files: bool = Query(False, description="si true → ignore 'filenames' et compresse tout"),
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
        # stream du buffer
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
