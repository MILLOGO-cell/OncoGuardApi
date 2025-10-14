from pathlib import Path
import shutil

# --- Dossiers principaux ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset_bfa"
RAW_DICOM_DIR = DATASET_DIR / "raw_dicom"
RAW_IMG_DIR = DATASET_DIR / "raw_images"
ANON_DICOM_DIR = DATASET_DIR / "anonymized_dicom"
IMG_DIR = DATASET_DIR / "images"
INFO_TXT = DATASET_DIR / "info.txt"
LABELS_CSV = DATASET_DIR / "labels.csv"
AGES_CSV = DATASET_DIR / "ages.csv"

# Création
for p in (RAW_DICOM_DIR, RAW_IMG_DIR, ANON_DICOM_DIR, IMG_DIR):
    p.mkdir(parents=True, exist_ok=True)
if not INFO_TXT.exists():
    INFO_TXT.write_text("", encoding="utf-8")
if not AGES_CSV.exists():
    AGES_CSV.write_text("id,age,age_bin,source\n", encoding="utf-8")

# --- Réglages redaction (recadrage UI) ---
CROP_TOP = 0.05
CROP_BOTTOM = 0.10
CROP_LEFT = 0.12
CROP_RIGHT = 0.02

# --- Bins d'âge ---
AGE_BINS = [(20,30,25),(30,40,35),(40,50,45),(50,60,55),(60,70,65),(70,80,75)]

# --- OCR: Tesseract dispo ? ---
try:
    import pytesseract  # noqa: F401
    TESS_AVAILABLE = bool(shutil.which("tesseract"))
except Exception:
    TESS_AVAILABLE = False
