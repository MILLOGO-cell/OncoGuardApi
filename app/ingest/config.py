"""Configuration du système d'ingestion des images mammographiques."""
from __future__ import annotations
from pathlib import Path
import os
import shutil
import logging

logger = logging.getLogger(__name__)

# ==========================
# Dossiers & fichiers
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.getenv("DATASET_DIR", BASE_DIR / "dataset_bfa")).resolve()

# Entrées principales (DICOM/PGM uniquement)
RAW_DICOM_DIR = DATASET_DIR / "raw_dicom"
RAW_PGM_DIR = DATASET_DIR / "raw_pgm"

# Sorties (normalisation/standardisation)
PROCESSED_DIR = DATASET_DIR / "processed"
NORMALIZED_DICOM_DIR = PROCESSED_DIR / "dicom"
DERIVED_IMG_DIR = PROCESSED_DIR / "derived_images"

# Métadonnées / labels
INFO_TXT = DATASET_DIR / "info.txt"
LABELS_CSV = DATASET_DIR / "labels.csv"
AGES_CSV = DATASET_DIR / "ages.csv"

# Création arborescence
for p in (RAW_DICOM_DIR, RAW_PGM_DIR, NORMALIZED_DICOM_DIR, DERIVED_IMG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Initialisation des fichiers CSV avec en-têtes
if not INFO_TXT.exists():
    INFO_TXT.write_text("", encoding="utf-8")

if not LABELS_CSV.exists():
    LABELS_CSV.write_text(
        "filename,label,birads,confidence,created_at\n", 
        encoding="utf-8"
    )
    logger.info(f"Fichier labels.csv initialisé : {LABELS_CSV}")

if not AGES_CSV.exists():
    AGES_CSV.write_text(
        "id,age,age_bin,source\n", 
        encoding="utf-8"
    )
    logger.info(f"Fichier ages.csv initialisé : {AGES_CSV}")

# ==========================
# Formats & règles d'ingest
# ==========================

# Formats d'entrée acceptés (DICOM et PGM uniquement)
ACCEPTED_INPUT_EXTS = {".dcm", ".dicom", ".pgm"}

# OCR : uniquement si BurnedInAnnotation == "YES" ou tags manquants
USE_OCR_FALLBACK_BY_DEFAULT = False

# ==========================
# Bins d'âge & OCR
# ==========================

# Bins d'âge pour classification (format : min, max, valeur_représentative)
AGE_BINS = [
    (20, 30, 25),
    (30, 40, 35),
    (40, 50, 45),
    (50, 60, 55),
    (60, 70, 65),
    (70, 80, 75)
]

# Vérification disponibilité Tesseract OCR
try:
    import pytesseract  # noqa: F401
    TESS_AVAILABLE = bool(shutil.which("tesseract"))
except ImportError:
    TESS_AVAILABLE = False

# ==========================
# Fonctions utilitaires
# ==========================

def should_use_ocr(burned_in_annotation: str | None, dicom_age_present: bool) -> bool:
    """
    Décide si l'OCR doit être utilisé pour extraire des informations.
    
    Logique de décision :
    - True si BurnedInAnnotation == 'YES' (texte brûlé dans l'image)
    - True si l'âge DICOM est absent ou invalide
    - Sinon, utilise la valeur par défaut (False)
    
    Args:
        burned_in_annotation: Valeur du tag DICOM BurnedInAnnotation
        dicom_age_present: True si l'âge est présent et valide dans les tags
    
    Returns:
        True si l'OCR doit être tenté, False sinon
    """
    # Vérifier si annotation brûlée
    bia_yes = (burned_in_annotation or "").strip().upper() == "YES"
    if bia_yes:
        logger.info("OCR activé : BurnedInAnnotation == 'YES'")
        return True
    
    # Vérifier si âge DICOM manquant
    if not dicom_age_present:
        logger.info("OCR activé : âge DICOM absent ou invalide")
        return True
    
    # Sinon, utiliser la config par défaut
    return USE_OCR_FALLBACK_BY_DEFAULT


def is_input_file_supported(path: Path) -> bool:
    """
    Vérifie si un fichier a une extension supportée.
    
    Args:
        path: Chemin du fichier à vérifier
    
    Returns:
        True si l'extension est dans ACCEPTED_INPUT_EXTS, False sinon
    
    Example:
        >>> is_input_file_supported(Path("image.dcm"))
        True
        >>> is_input_file_supported(Path("image.jpg"))
        False
    """
    return path.suffix.lower() in ACCEPTED_INPUT_EXTS