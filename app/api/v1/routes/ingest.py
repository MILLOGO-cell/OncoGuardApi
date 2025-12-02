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
from fastapi.responses import FileResponse, StreamingResponse

from app.ingest.config import DERIVED_IMG_DIR, NORMALIZED_DICOM_DIR
from app.api.v1.schemas.files import FileItem

import cv2
import numpy as np
from fastapi.responses import Response

logger = logging.getLogger(__name__)

# Import robuste du dossier tagged
try:
    from app.utils.image_annotation import TAGGED_DIR
except ImportError as e:
    logger.warning(f"Module image_annotation non disponible : {e}")
    TAGGED_DIR = None

router = APIRouter(prefix="/ingest", tags=["Fichiers et exportations"])


def _safe_roots() -> Dict[str, Path]:
    """
    Retourne les dossiers racines disponibles selon les modules chargés.
    
    Returns:
        Dictionnaire {kind: Path} des dossiers disponibles
    """
    roots = {
        "png": DERIVED_IMG_DIR,
        "dicom": NORMALIZED_DICOM_DIR,
    }
    
    if TAGGED_DIR is not None:
        roots["tagged"] = TAGGED_DIR
    
    return roots


def _safe_resolve(kind: str, filename: str) -> Path:
    """
    Résout un chemin de fichier de manière sécurisée.
    
    Protège contre les attaques par path traversal en vérifiant
    que le chemin résolu est bien sous le dossier racine.
    
    Args:
        kind: Type de fichier ("png", "dicom", "tagged")
        filename: Nom du fichier
    
    Returns:
        Chemin Path résolu et validé
    
    Raises:
        HTTPException 400: Si kind invalide ou path traversal détecté
        HTTPException 404: Si le fichier n'existe pas
        HTTPException 503: Si le dossier racine n'est pas disponible
    """
    kind = kind.lower()
    roots = _safe_roots()
    
    # Vérifier que le kind est valide
    if kind not in roots:
        allowed = "', '".join(roots.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"kind doit être parmi : '{allowed}'"
        )
    
    base = roots[kind]
    
    # Vérifier que le dossier racine existe
    if not base.exists():
        logger.error(f"Dossier {kind} non disponible : {base}")
        raise HTTPException(
            status_code=503,
            detail=f"Dossier '{kind}' temporairement indisponible"
        )
    
    base.mkdir(parents=True, exist_ok=True)
    
    # Résoudre le chemin complet
    p = (base / filename).resolve()
    
    # Protection contre path traversal
    try:
        p.relative_to(base.resolve())
    except ValueError:
        logger.warning(f"Tentative de path traversal détectée : {filename}")
        raise HTTPException(
            status_code=400, 
            detail="Chemin invalide (path traversal détecté)"
        )
    
    # Vérifier existence du fichier
    if not p.exists() or not p.is_file():
        raise HTTPException(
            status_code=404, 
            detail=f"Fichier introuvable : {filename}"
        )
    
    return p


def _iter_files(root: Path, pattern: Optional[str] = None) -> Iterable[Path]:
    """
    Itère sur les fichiers d'un dossier avec pattern optionnel.
    
    Args:
        root: Dossier racine
        pattern: Pattern glob optionnel (ex: "*.png", "bfa*.dcm")
    
    Yields:
        Chemins Path des fichiers correspondants
    """
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
    """
    Liste les fichiers disponibles avec filtrage et pagination.
    
    Permet de lister tous les fichiers d'un ou plusieurs types,
    avec recherche par pattern et tri par date de modification.
    
    Args:
        kind: Type de fichiers ("png", "dicom", "tagged") ou None pour tous
        q: Pattern de recherche glob (ex: "*.png", "patient_*.dcm")
        limit: Nombre maximum de résultats (1-5000)
        order: Ordre de tri ("asc"=chronologique, "desc"=antichronologique)
    
    Returns:
        Liste de FileItem avec métadonnées (nom, taille, date, URL)
    
    Raises:
        HTTPException 400: Si kind invalide
    
    Example:
        GET /api/v1/ingest/files?kind=png&q=bfa*.png&limit=50&order=desc
    """
    items: List[FileItem] = []
    roots = _safe_roots()
    
    # Déterminer les types de fichiers à lister
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
    
    # Parcourir tous les types demandés
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
                    created_at=stat.st_mtime,  # Timestamp UNIX
                    download_url=f"/api/v1/ingest/download/{k}/{f.name}",
                ))
    
    # Tri par date de modification
    items.sort(key=lambda x: x.created_at, reverse=(order == "desc"))
    
    logger.info(f"Liste de {len(items)} fichiers (kind={kind}, limit={limit})")
    return items[:limit]


@router.get("/download/{kind}/{filename}")
def download_file(kind: str, filename: str):
    """
    Télécharge un fichier par son type et nom.
    
    Args:
        kind: Type de fichier ("png", "dicom", "tagged")
        filename: Nom du fichier à télécharger
    
    Returns:
        FileResponse avec le fichier demandé
    
    Raises:
        HTTPException 400: Si kind invalide ou path traversal
        HTTPException 404: Si fichier introuvable
    
    Example:
        GET /api/v1/ingest/download/png/image_001.png
    """
    p = _safe_resolve(kind, filename)
    kind = kind.lower()
    
    # Déterminer le Content-Type
    if kind == "png" or (kind == "tagged" and p.suffix.lower() == ".png"):
        media_type = "image/png"
    elif kind == "dicom":
        media_type = "application/dicom"
    else:
        media_type = "application/octet-stream"
    
    logger.info(f"Téléchargement : {kind}/{filename}")
    return FileResponse(path=str(p), media_type=media_type, filename=p.name)


@router.post("/export/zip")
def export_zip(
    filenames: Optional[List[str]] = Query(
        None, 
        description="Noms de fichiers à inclure (ex: filenames=bfa001.png&filenames=bfa002.png)"
    ),
    kind: str = Query("png", description="Type de fichiers : 'png', 'dicom', 'tagged'"),
    all_files: bool = Query(
        False, 
        description="Si true, ignore 'filenames' et compresse tout le dossier"
    ),
):
    """
    Exporte des fichiers dans une archive ZIP.
    
    Deux modes d'export :
    - Mode sélectif : Spécifier une liste de fichiers
    - Mode complet : Tout le dossier (all_files=true)
    
    Args:
        filenames: Liste des noms de fichiers à inclure (mode sélectif)
        kind: Type de fichiers ("png", "dicom", "tagged")
        all_files: Si True, exporte tous les fichiers du dossier
    
    Returns:
        StreamingResponse avec archive ZIP
    
    Raises:
        HTTPException 400: Si kind invalide ou paramètres manquants
        HTTPException 404: Si aucun fichier trouvé
    
    Example:
        POST /api/v1/ingest/export/zip?kind=png&all_files=true
        POST /api/v1/ingest/export/zip?kind=png&filenames=img1.png&filenames=img2.png
    """
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
    
    # Déterminer les fichiers à zipper
    to_zip: List[Path] = []
    
    if all_files:
        # Mode complet : tous les fichiers du dossier
        to_zip = [p for p in root.glob("*") if p.is_file()]
        logger.info(f"Export ZIP complet de {kind} : {len(to_zip)} fichiers")
    else:
        # Mode sélectif : fichiers spécifiés
        if not filenames:
            raise HTTPException(
                status_code=400, 
                detail="Fournir 'filenames' ou activer 'all_files=true'"
            )
        
        for name in filenames:
            p = _safe_resolve(kind, name)
            to_zip.append(p)
        
        logger.info(f"Export ZIP sélectif de {kind} : {len(to_zip)} fichiers")
    
    # Vérifier qu'il y a des fichiers
    if not to_zip:
        raise HTTPException(
            status_code=404, 
            detail="Aucun fichier à compresser"
        )
    
    # Générateur de streaming
    def stream():
        """Stream le ZIP en chunks pour économiser la RAM."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pth in to_zip:
                zf.write(pth, arcname=pth.name)
        
        buf.seek(0)
        
        # Streamer par chunks de 256KB
        chunk_size = 256 * 1024
        while True:
            chunk = buf.read(chunk_size)
            if not chunk:
                break
            yield chunk
        
        buf.close()
    
    # Nom du fichier avec timestamp
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
        description="Si true, supprime aussi le fichier tagged associé (pour PNG/DICOM)"
    )
):
    """
    Supprime un fichier par son type et nom.
    
    Args:
        kind: Type de fichier ("png", "dicom", "tagged")
        filename: Nom du fichier à supprimer
        delete_related: Si True, supprime aussi le fichier tagged associé
    
    Returns:
        dict: Résultat de la suppression avec fichiers supprimés
    
    Raises:
        HTTPException 400: Si kind invalide ou path traversal
        HTTPException 404: Si fichier introuvable
        HTTPException 500: Si erreur lors de la suppression
    
    Example:
        DELETE /api/v1/ingest/delete/png/image_001.png
        DELETE /api/v1/ingest/delete/png/image_001.png?delete_related=true
    """
    # Résoudre le chemin du fichier principal
    p = _safe_resolve(kind, filename)
    
    deleted = []
    failed = []
    
    # Supprimer le fichier principal
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
    
    # Supprimer le fichier tagged associé si demandé
    if delete_related and TAGGED_DIR is not None and kind in ("png", "dicom"):
        # Déduire le nom du fichier tagged
        base_name = p.stem  # sans extension
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
    
@router.get("/preview/{kind}/{filename}")
def preview_file(kind: str, filename: str):
    """
    Retourne toujours un PNG pour aperçu (conversion à la volée si nécessaire).
    """
    p = _safe_resolve(kind, filename)
    kind = kind.lower()
    ext = p.suffix.lower()
    
    # PNG: retour direct
    if ext == ".png":
        with open(p, "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    
    # DICOM: conversion à la volée
    if ext in [".dcm", ".dicom"]:
        try:
            import pydicom
            dcm = pydicom.dcmread(p)
            img_array = dcm.pixel_array.astype(np.float32)
            
            # Normalisation
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            img_array = (img_array * 255).astype(np.uint8)
            
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Encodage PNG en mémoire
            success, buffer = cv2.imencode('.png', img_array)
            if not success:
                raise ValueError("Échec encodage PNG")
            
            return Response(content=buffer.tobytes(), media_type="image/png")
        except Exception as e:
            logger.exception("Erreur conversion DICOM → PNG")
            raise HTTPException(status_code=500, detail=f"Conversion DICOM échouée: {e}")
    
    # PGM: conversion à la volée
    if ext == ".pgm":
        try:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Impossible de lire le PGM")
            
            # Encodage PNG en mémoire
            success, buffer = cv2.imencode('.png', img)
            if not success:
                raise ValueError("Échec encodage PNG")
            
            return Response(content=buffer.tobytes(), media_type="image/png")
        except Exception as e:
            logger.exception("Erreur conversion PGM → PNG")
            raise HTTPException(status_code=500, detail=f"Conversion PGM échouée: {e}")
    
    raise HTTPException(status_code=400, detail=f"Type de fichier non supporté: {ext}")