"""
Routes API pour l'inférence et l'analyse d'images mammographiques.

Ce module fournit des endpoints pour :
- Prédire la classification d'images individuelles ou par lot
- Annoter les images avec les résultats (tagged images)
- Exporter les résultats et les images annotées
- Gérer l'historique des analyses en base de données

MODÈLE SUPPORTÉ :
Le modèle peut classifier en 2 ou 3 classes selon l'entraînement :
- Version 2 classes (CBIS-DDSM) : benign → BI-RADS 2, malignant → BI-RADS 5
- Version 3 classes (CBIS-DDSM + MIAS) : normal → BI-RADS 1, benign → BI-RADS 2, malignant → BI-RADS 5

Le nombre de classes est détecté automatiquement depuis le modèle chargé.
"""
import os
import io
import csv
import uuid
import shutil
import zipfile
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.v1.schemas.image_inference import (
    InferenceTaggedResponse,
    ResultsResponse,
    ResultItem,
)
from app.ml.predictor import predict as ml_predict, get_model_info
from app.db.database import get_db
from app.api.v1.models.image_analysis import ImageAnalysis
from app.api.v1.models.enums import BiradsCategory, AnalysisStatus

# Import conditionnel de pydicom
try:
    import pydicom
    _HAS_PYDICOM = True
except ImportError:
    _HAS_PYDICOM = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/image-inference", tags=["Analyse d'images"])

# ============================================================================
# CONFIGURATION
# ============================================================================

# Formats d'images acceptés
ALLOWED_IMG = {
    "image/jpeg",
    "image/png",
    "image/pgm",
    "application/dicom",
    "application/dicom+json",
}

# Extensions autorisées
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pgm", ".dcm"}

# Dossiers de travail
ROOT_DIR = Path(__file__).resolve().parents[5] if len(Path(__file__).parents) >= 6 else Path.cwd()
UPLOAD_DIR = ROOT_DIR / "uploads"
TAGGED_DIR = ROOT_DIR / "tagged"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TAGGED_DIR.mkdir(parents=True, exist_ok=True)

# Couleurs pour l'annotation selon le label (BGR)
LABEL_COLORS = {
    "normal": (0, 220, 0),      # Vert
    "benign": (0, 170, 255),    # Ambre/Orange
    "malignant": (38, 38, 220), # Rouge
    "default": (128, 128, 128)  # Gris
}


# ============================================================================
# SCHÉMAS PYDANTIC
# ============================================================================

class PredictFromUploadedRequest(BaseModel):
    """
    Schéma pour prédire depuis des fichiers déjà présents sur le serveur.
    
    Attributes:
        kind: Type de fichiers ("dicom" ou "pgm")
        filenames: Liste des noms de fichiers à analyser
    """
    kind: str = Field(..., description="Type de fichiers: 'dicom' ou 'pgm'")
    filenames: List[str] = Field(..., description="Liste des noms de fichiers")


class ModelInfoResponse(BaseModel):
    """
    Informations détaillées sur le modèle chargé.
    
    Attributes:
        classes: Liste des classes détectées par le modèle
        n_classes: Nombre de classes (2 ou 3)
        feature_dim: Dimension du vecteur de features
        model_type: Type de modèle (XGBoost)
        input_size: Taille d'entrée des images
        dataset: Dataset(s) d'entraînement
        note: Notes importantes sur les capacités du modèle
    """
    classes: List[str]
    n_classes: int
    feature_dim: str
    model_type: str
    input_size: str
    dataset: str
    note: str


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def _save_upload(file: UploadFile) -> str:
    """
    Sauvegarde un fichier uploadé dans le dossier UPLOAD_DIR avec un nom unique.
    
    Génère un nom UUID pour éviter les collisions et sanitize l'extension
    pour des raisons de sécurité.
    
    Args:
        file: Fichier FastAPI UploadFile
    
    Returns:
        Chemin absolu vers le fichier sauvegardé
    
    Raises:
        HTTPException 500: Si l'écriture du fichier échoue
    """
    # Déterminer l'extension (limitée à 10 caractères pour sécurité)
    ext = (os.path.splitext(file.filename or "")[1].lower() or ".png")[:10]
    
    # Validation de l'extension
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".png"
    
    # Générer un nom unique avec UUID
    fname = f"{uuid.uuid4()}{ext}"
    dest = UPLOAD_DIR / fname
    
    # Copier le fichier sur le disque
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.exception("Erreur lors de la sauvegarde du fichier")
        raise HTTPException(
            status_code=500,
            detail=f"Impossible de sauvegarder le fichier : {e}"
        )
    
    logger.debug(f"Fichier sauvegardé : {dest}")
    return str(dest)


def _dicom_to_png(dcm_path: str) -> str:
    """
    Convertit un fichier DICOM en PNG temporaire avec normalisation robuste.
    
    Applique une normalisation min-max pour convertir les valeurs DICOM
    (souvent 12/16 bits) en uint8 [0-255] compatible avec OpenCV.
    
    Args:
        dcm_path: Chemin vers le fichier DICOM
    
    Returns:
        Chemin vers le fichier PNG temporaire (suffixe .__tmp.png)
    
    Raises:
        HTTPException 415: Si pydicom n'est pas installé
        HTTPException 400: Si la lecture du DICOM échoue
        HTTPException 500: Si l'écriture du PNG échoue
    
    Warning:
        Le fichier PNG temporaire DOIT être supprimé après utilisation
        pour éviter la saturation du disque.
    """
    if not _HAS_PYDICOM:
        raise HTTPException(
            status_code=415, 
            detail="DICOM reçu mais pydicom non installé. Installez avec: pip install pydicom"
        )
    
    try:
        # Lecture du DICOM avec pydicom
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype("float32")
    except Exception as e:
        logger.exception("Échec lecture DICOM")
        raise HTTPException(
            status_code=400, 
            detail=f"Impossible de lire le DICOM: {e}"
        )
    
    # Normalisation robuste (min-max scaling)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-6:
        # Image uniforme (cas pathologique) : créer une image noire
        arr = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr = 255.0 * (arr - mn) / (mx - mn)
    
    arr = np.clip(arr, 0, 255).astype("uint8")
    
    # Sauvegarde en PNG temporaire avec suffixe spécial
    png_path = f"{dcm_path}.__tmp.png"
    if not cv2.imwrite(png_path, arr):
        raise HTTPException(
            status_code=500, 
            detail="Impossible d'écrire l'image PNG temporaire"
        )
    
    logger.debug(f"DICOM converti en PNG temporaire : {png_path}")
    return png_path


def _label_to_birads(label: Optional[str]) -> BiradsCategory:
    """
    Convertit un label de prédiction en enum BiradsCategory.
    
    Supporte les modèles 2 classes et 3 classes :
    - normal → BI-RADS 1 (Normal)
    - benign → BI-RADS 2 (Anomalie bénigne)
    - malignant → BI-RADS 5 (Évocateur de cancer)
    
    Args:
        label: Label prédit ("normal", "benign" ou "malignant")
    
    Returns:
        Enum BiradsCategory correspondant
    
    Note:
        Si le label est invalide ou None, retourne BI-RADS 0 (Examen incomplet).
        La comparaison utilise .startswith() pour tolérer les variations de casse.
    """
    if not label:
        return BiradsCategory.BI_RADS_0
    
    lab = label.strip().lower()
    
    # Mapping label → BI-RADS avec tolérance de préfixe
    if lab.startswith("norm"):
        return BiradsCategory.BI_RADS_1  # Normal
    elif lab.startswith("ben"):
        return BiradsCategory.BI_RADS_2  # Bénin
    elif lab.startswith("mal"):
        return BiradsCategory.BI_RADS_5  # Malin
    else:
        # Par défaut : examen incomplet
        logger.warning(f"Label inconnu '{label}', retour BI-RADS 0")
        return BiradsCategory.BI_RADS_0


def _extract_birads_code(birads_category: Optional[BiradsCategory]) -> Optional[str]:
    """
    Extrait le code numérique d'une catégorie BI-RADS de manière robuste.
    
    Gère le format "X - Description" et retourne uniquement le code numérique.
    Plus robuste qu'un simple split car gère les cas limites.
    
    Args:
        birads_category: Enum BiradsCategory ou None
    
    Returns:
        Code numérique ("1", "2", "5", etc.) ou None si entrée invalide
    
    Examples:
        >>> _extract_birads_code(BiradsCategory.BI_RADS_1)
        "1"
        >>> _extract_birads_code(BiradsCategory.BI_RADS_5)
        "5"
        >>> _extract_birads_code(None)
        None
    """
    if not birads_category:
        return None
    
    value = birads_category.value  # Ex: "1 - Normal"
    
    # Split avec limite pour éviter les problèmes si "-" apparaît dans la description
    parts = value.split(" - ", 1)
    
    # Retourner le premier élément (code) ou "0" en fallback
    return parts[0].strip() if parts else "0"


def _overlay_tag(
    src_path: str, 
    out_base: str, 
    label: str, 
    birads: str, 
    confidence: float
) -> str:
    """
    Crée une copie annotée de l'image avec les résultats de prédiction.
    
    Ajoute une barre d'information colorée en haut de l'image affichant :
    - Le label prédit (NORMAL, BENIGN, MALIGNANT)
    - La catégorie BI-RADS correspondante
    - Le score de confiance en pourcentage
    
    Couleurs selon le diagnostic :
    - Vert : Normal (BI-RADS 1)
    - Ambre/Orange : Benign (BI-RADS 2)
    - Rouge : Malignant (BI-RADS 5)
    - Gris : Inconnu/Erreur
    
    Args:
        src_path: Chemin vers l'image source
        out_base: Nom de base pour le fichier de sortie (sans extension)
        label: Label prédit ("normal", "benign", "malignant")
        birads: Catégorie BI-RADS (format "BI-RADS X")
        confidence: Score de confiance entre 0 et 1
    
    Returns:
        Chemin vers l'image annotée sauvegardée dans TAGGED_DIR
    
    Raises:
        HTTPException 500: Si l'image ne peut pas être lue ou écrite
    """
    # Lecture de l'image source en niveaux de gris
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(
            status_code=500, 
            detail=f"Impossible de lire l'image pour annotation : {src_path}"
        )
    
    # Conversion en BGR pour ajouter des couleurs
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Texte d'annotation formaté
    txt = f"{label.upper()} • {birads} • {confidence * 100:.1f}%"
    
    # Sélection de la couleur selon le label (BGR)
    label_lower = label.lower().strip()
    color = LABEL_COLORS.get(label_lower, LABEL_COLORS["default"])
    
    # Dimensions de la barre d'information
    pad_y = 8
    bar_h = 36 + pad_y * 2
    
    # Dessiner une barre noire semi-opaque en haut
    cv2.rectangle(img_rgb, (0, 0), (img_rgb.shape[1], bar_h), (0, 0, 0), -1)
    
    # Ajouter le texte coloré sur la barre
    cv2.putText(
        img_rgb, 
        txt, 
        (12, 28 + pad_y // 2), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        color, 
        2, 
        cv2.LINE_AA
    )
    
    # Sauvegarder l'image annotée dans TAGGED_DIR
    out_path = TAGGED_DIR / f"{out_base}__tag.png"
    if not cv2.imwrite(str(out_path), img_rgb):
        raise HTTPException(
            status_code=500, 
            detail="Impossible d'écrire l'image annotée"
        )
    
    logger.info(f"Image annotée créée : {out_path}")
    return str(out_path)


def _is_allowed_upload(file: UploadFile) -> bool:
    """
    Vérifie si un fichier uploadé est dans un format autorisé.
    
    Vérifie à la fois le Content-Type HTTP et l'extension du nom de fichier
    pour une validation robuste.
    
    Args:
        file: Fichier FastAPI UploadFile
    
    Returns:
        True si le format est autorisé, False sinon
    """
    # Vérifier le content-type HTTP
    if file.content_type in ALLOWED_IMG:
        return True
    
    # Vérifier l'extension du nom de fichier
    if file.filename:
        ext = file.filename.lower()
        return any(ext.endswith(e) for e in ALLOWED_EXTENSIONS)
    
    return False


def _cleanup_temp_file(file_path: str) -> None:
    """
    Supprime un fichier temporaire de manière sécurisée.
    
    Log un warning en cas d'échec mais ne lève pas d'exception pour
    ne pas interrompre le flux principal.
    
    Args:
        file_path: Chemin vers le fichier à supprimer
    """
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.debug(f"Fichier temporaire supprimé : {file_path}")
        except Exception as e:
            logger.warning(f"Impossible de supprimer le fichier temporaire {file_path} : {e}")


# ============================================================================
# ROUTES API
# ============================================================================

@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_information():
    """
    Retourne les informations détaillées sur le modèle de classification chargé.
    
    Endpoint utile pour :
    - Vérifier quel modèle est actuellement chargé (2 ou 3 classes)
    - Connaître les classes détectées par le modèle
    - Comprendre les limitations et capacités du modèle
    
    Returns:
        ModelInfoResponse: Métadonnées complètes du modèle incluant :
            - classes: Liste des labels détectables
            - n_classes: Nombre de classes (2 ou 3)
            - feature_dim: Dimension du vecteur de features
            - model_type: Type d'algorithme (XGBoost)
            - input_size: Résolution d'entrée attendue
            - dataset: Dataset(s) d'entraînement utilisé(s)
            - note: Notes importantes sur les capacités
    
    Raises:
        HTTPException 500: Si le modèle ne peut pas être chargé
    
    Example:
        GET /image-inference/model-info
        
        Response 200:
        {
            "classes": ["benign", "malignant", "normal"],
            "n_classes": 3,
            "feature_dim": "28542",
            "model_type": "XGBoost",
            "input_size": "160×160",
            "dataset": "CBIS-DDSM + MIAS",
            "note": "Modèle 3 classes : détecte normal, benign et malignant"
        }
    """
    try:
        info = get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.exception("Erreur récupération infos modèle")
        raise HTTPException(
            status_code=500, 
            detail=f"Impossible de charger les infos du modèle: {e}"
        )


@router.post("/predict", response_model=InferenceTaggedResponse)
async def predict_single_image(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """
    Prédit la classification d'une seule image mammographique avec workflow complet.
    
    Pipeline d'exécution :
    1. Validation du format de fichier (JPEG, PNG, PGM, DICOM)
    2. Sauvegarde sécurisée avec nom UUID unique
    3. Conversion DICOM→PNG temporaire si nécessaire
    4. Prédiction ML (HOG + LBP + Wavelets → XGBoost)
    5. Annotation visuelle avec overlay coloré
    6. Enregistrement en base de données PostgreSQL
    7. Nettoyage des fichiers temporaires
    8. Retour du résultat JSON
    
    Args:
        file: Fichier image uploadé (formats acceptés : JPEG, PNG, PGM, DICOM)
        db: Session SQLAlchemy injectée automatiquement par FastAPI
    
    Returns:
        InferenceTaggedResponse contenant :
            - label: Classe prédite ("normal", "benign", "malignant")
            - birads: Catégorie BI-RADS ("BI-RADS 1/2/5")
            - confidence: Score de confiance (0-1)
            - filename: Nom du fichier sauvegardé
            - tagged_filename: Nom du fichier annoté
    
    Raises:
        HTTPException 400: Format de fichier non autorisé
        HTTPException 415: DICOM reçu mais pydicom non installé
        HTTPException 500: Erreur lors du traitement ML ou de l'annotation
    
    Example:
        POST /image-inference/predict
        Content-Type: multipart/form-data
        
        Body:
            file: mammogram.png (binary)
        
        Response 200:
        {
            "label": "malignant",
            "birads": "BI-RADS 5",
            "confidence": 0.873,
            "filename": "abc123-xyz.png",
            "tagged_filename": "abc123-xyz__tag.png"
        }
    
    Note:
        - Le modèle 2 classes classifie TOUTES les images en benign OU malignant
        - Le modèle 3 classes peut aussi détecter les cas normaux
        - Les images annotées sont disponibles via GET /tagged/{filename}
        - En cas d'échec DB, le résultat est quand même retourné (rollback silencieux)
    """
    # 1. Vérification du format de fichier
    if not _is_allowed_upload(file):
        raise HTTPException(
            status_code=400, 
            detail="Formats autorisés: JPEG, PNG, PGM, DICOM"
        )
    
    # 2. Sauvegarde sécurisée du fichier
    saved_path = _save_upload(file)
    work_path = saved_path
    
    # 3. Détection et conversion DICOM si nécessaire
    is_dicom = (
        (file.content_type in {"application/dicom", "application/dicom+json"}) or 
        saved_path.lower().endswith(".dcm")
    )
    
    if is_dicom:
        work_path = _dicom_to_png(saved_path)
    
    # Variables pour stocker les résultats
    res = None
    tagged_path = None
    
    try:
        # 4. Prédiction ML avec le modèle XGBoost
        res = ml_predict(work_path)
        
        # 5. Annotation de l'image avec les résultats
        base = os.path.splitext(os.path.basename(saved_path))[0]
        tagged_path = _overlay_tag(
            work_path, 
            base, 
            res.label, 
            res.birads, 
            float(res.confidence)
        )
        
    except Exception as e:
        logger.exception("Erreur lors de la prédiction ou de l'annotation")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse de l'image : {e}"
        )
    
    finally:
        # 6. Nettoyage du fichier temporaire DICOM→PNG
        # CORRECTION: Nettoyage après annotation pour éviter les erreurs
        if is_dicom and work_path != saved_path:
            _cleanup_temp_file(work_path)
    
    # 7. Enregistrement en base de données
    try:
        birads_enum = _label_to_birads(res.label)
        row = ImageAnalysis(
            filename=os.path.basename(saved_path),
            result_class=birads_enum,
            confidence=float(res.confidence),
            description=f"Pred: {res.label} ({res.birads}) - Model: {res.model_type}",
            status=AnalysisStatus.COMPLETED,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        logger.info(f"Analyse enregistrée en DB : ID={row.id}, Label={res.label}, BI-RADS={birads_enum.value}")
    except Exception as e:
        db.rollback()
        logger.exception(f"Échec enregistrement DB: {e}")
        # On continue malgré l'erreur DB pour retourner le résultat au client
    
    # 8. Retour du résultat JSON
    return InferenceTaggedResponse(
        label=res.label,
        birads=res.birads,
        confidence=float(res.confidence),
        filename=os.path.basename(saved_path),
        tagged_filename=os.path.basename(tagged_path),
    )


@router.post("/predict-batch", response_model=List[InferenceTaggedResponse])
async def predict_multiple_images(
    files: List[UploadFile] = File(...), 
    persist: bool = Query(True, description="Enregistrer les résultats en DB"),
    db: Session = Depends(get_db)
):
    """
    Prédit la classification pour un lot d'images mammographiques en une seule requête.
    
    Workflow identique à /predict mais optimisé pour le traitement par lot.
    Les fichiers invalides sont ignorés silencieusement et n'interrompent pas
    le traitement des autres fichiers.
    
    Avantages du traitement par lot :
    - Chargement unique du modèle ML (optimisation mémoire)
    - Réduction de la latence réseau (une seule requête HTTP)
    - Possibilité de traitement parallèle futur
    - Progression globale trackable
    
    Args:
        files: Liste de fichiers images à analyser
        persist: Si True, enregistre tous les résultats en DB (défaut: True)
        db: Session SQLAlchemy injectée automatiquement
    
    Returns:
        Liste de InferenceTaggedResponse dans le même ordre que les fichiers uploadés.
        Les fichiers invalides sont omis de la liste de résultats.
    
    Raises:
        HTTPException 400: Si TOUS les fichiers sont invalides ou échouent
    
    Example:
        POST /image-inference/predict-batch?persist=true
        Content-Type: multipart/form-data
        
        Body:
            files: img1.png (binary)
            files: img2.png (binary)
            files: img3.dcm (binary)
        
        Response 200:
        [
            {"label": "benign", "birads": "BI-RADS 2", "confidence": 0.912, ...},
            {"label": "malignant", "birads": "BI-RADS 5", "confidence": 0.856, ...},
            {"label": "normal", "birads": "BI-RADS 1", "confidence": 0.943, ...}
        ]
    
    Note:
        - Les fichiers sont traités séquentiellement (pas de parallélisation)
        - En cas d'erreur sur un fichier, il est ignoré et les autres sont traités
        - Plus efficace que d'appeler /predict N fois
        - Chaque fichier peut avoir un format différent (mix DICOM/PNG/PGM)
    """
    results: List[InferenceTaggedResponse] = []
    
    for idx, file in enumerate(files):
        # Ignorer les fichiers non valides sans lever d'exception
        if not _is_allowed_upload(file):
            logger.warning(f"Fichier {idx+1}/{len(files)} ignoré (format non autorisé)")
            continue
        
        saved_path = None
        work_path = None
        
        try:
            # Sauvegarde du fichier
            saved_path = _save_upload(file)
            work_path = saved_path
            
            # Conversion DICOM si nécessaire
            is_dicom = (
                (file.content_type in {"application/dicom", "application/dicom+json"}) or 
                saved_path.lower().endswith(".dcm")
            )
            
            if is_dicom:
                work_path = _dicom_to_png(saved_path)
            
            # Prédiction ML
            res = ml_predict(work_path)
            
            # Annotation visuelle
            base = os.path.splitext(os.path.basename(saved_path))[0]
            tagged_path = _overlay_tag(
                work_path, 
                base, 
                res.label, 
                res.birads, 
                float(res.confidence)
            )
            
            # Nettoyage fichier temporaire DICOM
            if is_dicom and work_path != saved_path:
                _cleanup_temp_file(work_path)
            
            # Ajouter aux résultats
            results.append(
                InferenceTaggedResponse(
                    label=res.label,
                    birads=res.birads,
                    confidence=float(res.confidence),
                    filename=os.path.basename(saved_path),
                    tagged_filename=os.path.basename(tagged_path),
                )
            )
            
            # Persistence optionnelle en base de données
            if persist:
                try:
                    birads_enum = _label_to_birads(res.label)
                    row = ImageAnalysis(
                        filename=os.path.basename(saved_path),
                        result_class=birads_enum,
                        confidence=float(res.confidence),
                        description=f"Batch pred: {res.label} ({res.birads})",
                        status=AnalysisStatus.COMPLETED,
                    )
                    db.add(row)
                    db.commit()
                except Exception as e:
                    db.rollback()
                    logger.exception(f"Échec enregistrement DB (batch, fichier {idx+1}): {e}")
        
        except Exception as e:
            logger.exception(f"Erreur traitement fichier {idx+1}/{len(files)}: {e}")
            # Continuer avec les autres fichiers malgré l'erreur
            continue
    
    # Vérifier qu'au moins un fichier a été traité avec succès
    if not results:
        raise HTTPException(
            status_code=400,
            detail="Aucun fichier valide n'a pu être traité"
        )
    
    logger.info(f"Batch prediction terminé : {len(results)}/{len(files)} fichiers traités avec succès")
    return results


@router.get("/tagged/{filename}")
def download_tagged_image(filename: str):
    """
    Télécharge une image annotée par son nom de fichier.
    
    Les images annotées sont générées automatiquement lors de la prédiction
    et stockées dans le dossier TAGGED_DIR avec le suffixe "__tag.png".
    
    Args:
        filename: Nom du fichier annoté (ex: "abc123-xyz__tag.png")
    
    Returns:
        FileResponse: Image PNG annotée avec headers appropriés
    
    Raises:
        HTTPException 404: Fichier introuvable dans TAGGED_DIR
    
    Example:
        GET /image-inference/tagged/abc123-xyz__tag.png
        
        Response 200:
            Content-Type: image/png
            Content-Disposition: inline; filename="abc123-xyz__tag.png"
            [binary image data]
    """
    p = TAGGED_DIR / filename
    if not p.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Image annotée introuvable : {filename}"
        )
    
    return FileResponse(
        str(p), 
        media_type="image/png", 
        filename=filename
    )


@router.get("/results", response_model=ResultsResponse)
def list_analysis_results(
    limit: int = Query(200, ge=1, le=5000, description="Nombre max de résultats"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Ordre de tri"),
    db: Session = Depends(get_db)
):
    """
    Liste tous les résultats d'analyse stockés en base de données avec pagination.
    
    Retourne une liste paginée des analyses avec leurs métadonnées complètes.
    Utile pour :
    - Afficher l'historique complet des analyses
    - Générer des statistiques et rapports
    - Exporter des données pour analyse externe
    
    Args:
        limit: Nombre maximum de résultats à retourner (1-5000, défaut: 200)
        order: Ordre de tri temporel ("asc"=chronologique, "desc"=antichronologique)
        db: Session SQLAlchemy injectée automatiquement
    
    Returns:
        ResultsResponse contenant :
            - items: Liste des analyses avec métadonnées
            - total: Nombre total d'analyses en DB (avant pagination)
    
    Example:
        GET /image-inference/results?limit=50&order=desc
        
        Response 200:
        {
            "items": [
                {
                    "id": 42,
                    "filename": "abc123.png",
                    "label": "2 - Bénin",
                    "birads": "2",
                    "confidence": 0.912,
                    "tagged_filename": "abc123__tag.png",
                    "submitted_at": "2025-01-15T10:30:00Z"
                },
                ...
            ],
            "total": 150
        }
    """
    # Récupérer toutes les analyses depuis la base de données
    rows = db.query(ImageAnalysis).all()
    
    # Trier par date de soumission selon l'ordre demandé
    rows_sorted = sorted(
        rows, 
        key=lambda r: r.submitted_at or datetime.min, 
        reverse=(order == "desc")
    )
    
    # Construire la liste des résultats avec extraction robuste du code BI-RADS
    items: List[ResultItem] = []
    for r in rows_sorted[:limit]:
        # Déduire le nom du fichier annoté correspondant
        base = os.path.splitext(os.path.basename(r.filename or ""))[0]
        tagged = f"{base}__tag.png" if base else None
        tagged_path = TAGGED_DIR / tagged if tagged else None
        
        # Vérifier l'existence du fichier annoté
        tagged_exists = tagged_path and tagged_path.exists()
        
        items.append(
            ResultItem(
                id=r.id,
                filename=r.filename,
                label=(r.result_class.value if r.result_class else None),
                birads=_extract_birads_code(r.result_class),  # Extraction robuste
                confidence=(
                    float(r.confidence) 
                    if r.confidence is not None else None
                ),
                tagged_filename=(tagged if tagged_exists else None),
                submitted_at=(
                    r.submitted_at.isoformat() 
                    if r.submitted_at else None
                ),
            )
        )
    
    logger.info(f"Retour de {len(items)} résultats sur {len(rows_sorted)} total (limit={limit})")
    return ResultsResponse(items=items, total=len(rows_sorted))


@router.get("/report.csv")
def export_results_csv(db: Session = Depends(get_db)):
    """
    Exporte tous les résultats d'analyse en fichier CSV pour analyse externe.
    
    Format CSV standardisé :
    - Délimiteur : virgule
    - Encodage : UTF-8
    - En-têtes : id, filename, birads, confidence, submitted_at
    
    Args:
        db: Session SQLAlchemy injectée automatiquement
    
    Returns:
        StreamingResponse: Fichier CSV streamé en chunks pour économiser la mémoire
    
    Example:
        GET /image-inference/report.csv
        
        Response 200:
            Content-Type: text/csv; charset=utf-8
            Content-Disposition: attachment; filename="report.csv"
            
            id,filename,birads,confidence,submitted_at
            1,img1.png,BI-RADS 2,0.9120,2025-01-15T10:30:00
            2,img2.png,BI-RADS 5,0.8560,2025-01-15T10:31:15
            ...
    
    Note:
        - Le streaming permet de gérer de gros exports sans saturer la RAM
        - Compatible Excel, Google Sheets, R, Python pandas
        - Utile pour générer des rapports statistiques
    """
    rows = db.query(ImageAnalysis).all()
    
    def stream():
        """Générateur pour streamer le CSV en chunks."""
        buf = io.StringIO()
        w = csv.writer(buf)
        
        # En-tête CSV
        w.writerow(["id", "filename", "birads", "confidence", "submitted_at"])
        
        # Données ligne par ligne
        for r in rows:
            w.writerow([
                r.id,
                r.filename,
                r.result_class.value if r.result_class else "",
                f"{float(r.confidence):.4f}" if r.confidence is not None else "",
                r.submitted_at.isoformat() if r.submitted_at else "",
            ])
        
        buf.seek(0)
        yield buf.read()
        buf.close()
    
    headers = {"Content-Disposition": 'attachment; filename="report.csv"'}
    logger.info(f"Export CSV de {len(rows)} résultats")
    return StreamingResponse(
        stream(), 
        media_type="text/csv; charset=utf-8", 
        headers=headers
    )


@router.get("/export-tagged.zip")
def export_all_tagged_images():
    """
    Exporte toutes les images annotées dans une archive ZIP téléchargeable.
    
    Streaming en chunks de 256KB pour économiser la mémoire.
    Idéal pour :
    - Télécharger toutes les annotations en une seule fois
    - Créer des backups des résultats visuels
    - Partager les résultats avec des collègues
    
    Returns:
        StreamingResponse: Archive ZIP streamée en chunks
    
    Raises:
        HTTPException 404: Aucune image annotée disponible
    
    Example:
        GET /image-inference/export-tagged.zip
        
        Response 200:
            Content-Type: application/zip
            Content-Disposition: attachment; filename="tagged_images.zip"
            [binary ZIP data containing all *.png files from TAGGED_DIR]
    
    Note:
        - Seules les images PNG sont incluses
        - Compression ZIP_DEFLATED pour réduire la taille
        - Ne charge pas tout en mémoire grâce au streaming
        - Peut prendre du temps si beaucoup d'images (>1000)
    """
    # Lister toutes les images PNG du dossier tagged
    files = [f for f in os.listdir(TAGGED_DIR) if f.lower().endswith(".png")]
    
    if not files:
        raise HTTPException(
            status_code=404, 
            detail="Aucune image annotée disponible"
        )
    
    def stream():
        """Générateur pour streamer le ZIP en chunks."""
        mem = io.BytesIO()
        
        # Créer le ZIP en mémoire avec compression
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            for name in files:
                full_path = os.path.join(TAGGED_DIR, name)
                z.write(full_path, arcname=name)
        
        mem.seek(0)
        
        # Streamer par chunks de 256KB
        chunk_size = 256 * 1024
        for chunk in iter(lambda: mem.read(chunk_size), b""):
            yield chunk
        
        mem.close()
    
    headers = {"Content-Disposition": 'attachment; filename="tagged_images.zip"'}
    logger.info(f"Export ZIP de {len(files)} images annotées")
    return StreamingResponse(
        stream(), 
        media_type="application/zip", 
        headers=headers
    )