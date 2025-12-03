"""Routes API pour la gestion des fichiers (liste, téléchargement, export)."""
from __future__ import annotations

import os
import io
import time
import zipfile
import logging
from pathlib import Path
from typing import List, Optional, Iterable, Dict

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, Response
import cv2
import numpy as np

from app.core.config import DERIVED_IMG_DIR, NORMALIZED_DICOM_DIR, TAGGED_DIR
from app.api.v1.schemas.files import FileItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Fichiers et exportations"])


def _safe_roots() -> Dict[str, Path]:
    roots = {
        "png": DERIVED_IMG_DIR,
        "dicom": NORMALIZED_DICOM_DIR,
        "tagged": TAGGED_DIR,
    }
    return roots


def _safe_resolve(kind: str, filename: str) -> Path:
    kind = kind.lower()
    roots = _safe_roots()
    
    if kind not in roots:
        allowed = "', '".join(roots.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"kind doit être parmi : '{allowed}'"
        )
    
    base = roots[kind]
    
    if not base.exists():
        logger.error(f"Dossier {kind} non disponible : {base}")
        raise HTTPException(
            status_code=503,
            detail=f"Dossier '{kind}' temporairement indisponible"
        )
    
    base.mkdir(parents=True, exist_ok=True)
    
    p = (base / filename).resolve()
    
    try:
        p.relative_to(base.resolve())
    except ValueError:
        logger.warning(f"Tentative de path traversal détectée : {filename}")
        raise HTTPException(
            status_code=400, 
            detail="Chemin invalide (path traversal détecté)"
        )
    
    if not p.exists() or not p.is_file():
        raise HTTPException(
            status_code=404, 
            detail=f"Fichier introuvable : {filename}"
        )
    
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
    kind: Optional[str] = Query(
        None, 
        description="Type de fichiers : 'png', 'dicom', 'tagged' (None = tous)"
    ),
    q: Optional[str] = Query(
        None, 
        description="Pattern glob, ex: '*.png' ou 'bfa0*.dcm'"
    ),
    limit: int = Query(200, ge=1, le=5000, description="Nombre max de résultats"),
    order: str = Query(
        "desc", 
        regex="^(asc|desc)$", 
        description="Tri par date : 'asc' ou 'desc'"
    ),
):
    items: List[FileItem] = []
    roots = _safe_roots()
    
    if kind is not None:
        kind = kind.lower()
        if kind not in roots:
            allowed = "', '".join(roots.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"kind doit être parmi : '{allowed}'"
            )
        kinds = [kind]
    else:
        kinds = list(roots.keys())
    
    for k in kinds:
        root = roots[k]
        if not root.exists():
            logger.warning(f"Dossier {k} inexistant, ignoré : {root}")
            continue
        
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
    
    logger.info(f"Liste de {len(items)} fichiers (kind={kind}, limit={limit})")
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
    
    logger.info(f"Téléchargement : {kind}/{filename}")
    return FileResponse(path=str(p), media_type=media_type, filename=p.name)


@router.get("/preview/{kind}/{filename}")
def preview_file(kind: str, filename: str):
    p = _safe_resolve(kind, filename)
    kind = kind.lower()
    ext = p.suffix.lower()
    
    if ext == ".png":
        with open(p, "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    
    if ext in [".dcm", ".dicom"]:
        try:
            import pydicom
            dcm = pydicom.dcmread(p)
            img_array = dcm.pixel_array.astype(np.float32)
            
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            img_array = (img_array * 255).astype(np.uint8)
            
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            success, buffer = cv2.imencode('.png', img_array)
            if not success:
                raise ValueError("Échec encodage PNG")
            
            return Response(content=buffer.tobytes(), media_type="image/png")
        except Exception as e:
            logger.exception("Erreur conversion DICOM → PNG")
            raise HTTPException(status_code=500, detail=f"Conversion DICOM échouée: {e}")
    
    if ext == ".pgm":
        try:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Impossible de lire le PGM")
            
            success, buffer = cv2.imencode('.png', img)
            if not success:
                raise ValueError("Échec encodage PNG")
            
            return Response(content=buffer.tobytes(), media_type="image/png")
        except Exception as e:
            logger.exception("Erreur conversion PGM → PNG")
            raise HTTPException(status_code=500, detail=f"Conversion PGM échouée: {e}")
    
    raise HTTPException(status_code=400, detail=f"Type de fichier non supporté: {ext}")


@router.post("/export/zip")
def export_zip(
    filenames: Optional[List[str]] = Query(
        None, 
        description="Noms de fichiers à inclure"
    ),
    kind: str = Query("png", description="Type de fichiers : 'png', 'dicom', 'tagged'"),
    all_files: bool = Query(
        False, 
        description="Si true, ignore 'filenames' et compresse tout le dossier"
    ),
):
    roots = _safe_roots()
    kind = kind.lower()
    root = roots.get(kind)
    
    if root is None:
        allowed = "', '".join(roots.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"kind doit être parmi : '{allowed}'"
        )
    
    root.mkdir(parents=True, exist_ok=True)
    
    to_zip: List[Path] = []
    
    if all_files:
        to_zip = [p for p in root.glob("*") if p.is_file()]
        logger.info(f"Export ZIP complet de {kind} : {len(to_zip)} fichiers")
    else:
        if not filenames:
            raise HTTPException(
                status_code=400, 
                detail="Fournir 'filenames' ou activer 'all_files=true'"
            )
        
        for name in filenames:
            p = _safe_resolve(kind, name)
            to_zip.append(p)
        
        logger.info(f"Export ZIP sélectif de {kind} : {len(to_zip)} fichiers")
    
    if not to_zip:
        raise HTTPException(
            status_code=404, 
            detail="Aucun fichier à compresser"
        )
    
    def stream():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pth in to_zip:
                zf.write(pth, arcname=pth.name)
        
        buf.seek(0)
        
        chunk_size = 256 * 1024
        while True:
            chunk = buf.read(chunk_size)
            if not chunk:
                break
            yield chunk
        
        buf.close()
    
    ts = int(time.time())
    fname = f"export_{kind}_{ts}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    
    return StreamingResponse(stream(), media_type="application/zip", headers=headers)


@router.delete("/delete/{kind}/{filename}")
def delete_file(
    kind: str, 
    filename: str,
    delete_related: bool = Query(
        False,
        description="Si true, supprime aussi le fichier tagged associé"
    )
):
    p = _safe_resolve(kind, filename)
    
    deleted = []
    failed = []
    
    try:
        p.unlink()
        deleted.append(filename)
        logger.info(f"Fichier supprimé : {kind}/{filename}")
    except Exception as e:
        logger.exception(f"Erreur suppression {kind}/{filename}")
        raise HTTPException(
            status_code=500,
            detail=f"Impossible de supprimer le fichier : {e}"
        )
    
    if delete_related and kind in ("png", "dicom"):
        base_name = p.stem
        tagged_name = f"{base_name}__tag.png"
        tagged_path = TAGGED_DIR / tagged_name
        
        if tagged_path.exists():
            try:
                tagged_path.unlink()
                deleted.append(f"tagged/{tagged_name}")
                logger.info(f"Fichier tagged associé supprimé : {tagged_name}")
            except Exception as e:
                failed.append({
                    "filename": tagged_name,
                    "reason": str(e)
                })
                logger.warning(f"Impossible de supprimer le tagged : {e}")
    
    return {
        "deleted": deleted,
        "failed": failed,
        "total_deleted": len(deleted)
    }


@router.delete("/delete-batch")
def delete_files_batch(
    kind: str = Query(..., description="Type de fichiers : 'png', 'dicom', 'tagged'"),
    filenames: List[str] = Query(..., description="Liste des noms de fichiers à supprimer"),
    delete_related: bool = Query(
        False,
        description="Si true, supprime aussi les fichiers tagged associés"
    )
):
    if not filenames:
        raise HTTPException(
            status_code=400,
            detail="La liste 'filenames' ne peut pas être vide"
        )
    
    deleted = []
    failed = []
    
    for filename in filenames:
        try:
            p = _safe_resolve(kind, filename)
            
            try:
                p.unlink()
                deleted.append(filename)
                logger.info(f"Fichier supprimé : {kind}/{filename}")
            except Exception as e:
                failed.append({
                    "filename": filename,
                    "reason": f"Erreur suppression : {str(e)}"
                })
                logger.warning(f"Échec suppression {kind}/{filename}: {e}")
                continue
            
            if delete_related and kind in ("png", "dicom"):
                base_name = p.stem
                tagged_name = f"{base_name}__tag.png"
                tagged_path = TAGGED_DIR / tagged_name
                
                if tagged_path.exists():
                    try:
                        tagged_path.unlink()
                        deleted.append(f"tagged/{tagged_name}")
                        logger.info(f"Fichier tagged associé supprimé : {tagged_name}")
                    except Exception as e:
                        failed.append({
                            "filename": f"tagged/{tagged_name}",
                            "reason": f"Erreur suppression tagged : {str(e)}"
                        })
                        logger.warning(f"Impossible de supprimer le tagged : {e}")
        
        except HTTPException as he:
            failed.append({
                "filename": filename,
                "reason": he.detail
            })
            logger.warning(f"Échec résolution {kind}/{filename}: {he.detail}")
        
        except Exception as e:
            failed.append({
                "filename": filename,
                "reason": f"Erreur inattendue : {str(e)}"
            })
            logger.exception(f"Erreur inattendue pour {kind}/{filename}")
    
    logger.info(f"Suppression batch : {len(deleted)} réussis, {len(failed)} échecs")
    
    return {
        "deleted": deleted,
        "failed": failed,
        "total_deleted": len(deleted),
        "total_failed": len(failed),
        "total_requested": len(filenames)
    }