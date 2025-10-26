# ingest/config.py
from __future__ import annotations
from pathlib import Path
import os
import shutil

# ==========================
# Dossiers & fichiers
# ==========================
# Tu peux surcharger DATASET_DIR via l'ENV DATASET_DIR
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.getenv("DATASET_DIR", BASE_DIR / "dataset_bfa")).resolve()

# Entrées principales (DICOM/PGM uniquement)
RAW_DICOM_DIR = DATASET_DIR / "raw_dicom"          # .dcm en priorité
RAW_PGM_DIR   = DATASET_DIR / "raw_pgm"            # .pgm si tu en as (optionnel)

# Sorties (normalisation/standardisation, PAS d'anonymisation)
PROCESSED_DIR        = DATASET_DIR / "processed"           # racine de sorties
NORMALIZED_DICOM_DIR = PROCESSED_DIR / "dicom"             # DICOM réécrits/normalisés si besoin
DERIVED_IMG_DIR      = PROCESSED_DIR / "derived_images"    # rendus PGM/PNG pour visu interne si utile

# Métadonnées / labels
INFO_TXT   = DATASET_DIR / "info.txt"
LABELS_CSV = DATASET_DIR / "labels.csv"
AGES_CSV   = DATASET_DIR / "ages.csv"

# Création arborescence
for p in (
    RAW_DICOM_DIR,
    RAW_PGM_DIR,
    NORMALIZED_DICOM_DIR,
    DERIVED_IMG_DIR,
):
    p.mkdir(parents=True, exist_ok=True)

if not INFO_TXT.exists():
    INFO_TXT.write_text("", encoding="utf-8")

if not AGES_CSV.exists():
    AGES_CSV.write_text("id,age,age_bin,source\n", encoding="utf-8")

# ==========================
# Formats & règles d’ingest
# ==========================
# On ne traite PAS de JPEG/PNG en entrée.
ACCEPTED_INPUT_EXTS = {".dcm", ".dicom", ".pgm"}

# DICOM → utiliser les tags en priorité.
# OCR uniquement si:
#   - BurnedInAnnotation == "YES"
#   - OU tags d'âge manquants/inexploitables
USE_OCR_FALLBACK_BY_DEFAULT = False  # on ne fait pas d'OCR si tout est correct côté DICOM

# ==========================
# Réglages “overlay/bordures”
# ==========================
# Les appareils exportent déjà des images anonymisées ; on évite de recadrer par défaut.
# Si tu détectes des bordures UI ou overlays brûlés, active ces marges (proportions).
CROP_TOP = float(os.getenv("CROP_TOP", 0.00))
CROP_BOTTOM = float(os.getenv("CROP_BOTTOM", 0.00))
CROP_LEFT = float(os.getenv("CROP_LEFT", 0.00))
CROP_RIGHT = float(os.getenv("CROP_RIGHT", 0.00))

# ==========================
# Bins d’âge & OCR
# ==========================
AGE_BINS = [(20,30,25), (30,40,35), (40,50,45), (50,60,55), (60,70,65), (70,80,75)]

try:
    import pytesseract  # noqa: F401
    TESS_AVAILABLE = bool(shutil.which("tesseract"))
except Exception:
    TESS_AVAILABLE = False

# ==========================
# Petits utilitaires
# ==========================
def should_use_ocr(burned_in_annotation: str | None, dicom_age_present: bool) -> bool:
    """
    Décide si l’on doit tenter l’OCR.
    - True si BurnedInAnnotation == 'YES'
    - True si l’âge DICOM est absent/illisible
    - Sinon False
    """
    bia_yes = (burned_in_annotation or "").strip().upper() == "YES"
    if bia_yes:
        return True
    if not dicom_age_present:
        return True
    return USE_OCR_FALLBACK_BY_DEFAULT

def is_input_file_supported(path: Path) -> bool:
    return path.suffix.lower() in ACCEPTED_INPUT_EXTS
