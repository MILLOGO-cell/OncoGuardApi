import glob
from pathlib import Path
from typing import List
from .config import RAW_IMG_DIR, RAW_DICOM_DIR, IMG_DIR, ANON_DICOM_DIR, INFO_TXT, AGES_CSV
from .io_utils import read_labels
from .photo_proc import process_one_photo

# DICOM support optionnel (seulement si pydicom installé)
try:
    from .dicom_proc import process_one_dicom
    HAS_PYDICOM = True
except Exception:
    HAS_PYDICOM = False

def _collect_images() -> List[Path]:
    photos = []
    for folder in (RAW_IMG_DIR, RAW_DICOM_DIR):
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
            photos += [Path(p) for p in glob.glob(str(folder / "**" / ext), recursive=True)]
    return sorted(photos)

def _collect_dicoms() -> List[Path]:
    if not HAS_PYDICOM: return []
    dcm = []
    for ext in ("*.dcm","*.DCM"):
        dcm += [Path(p) for p in glob.glob(str(RAW_DICOM_DIR / "**" / ext), recursive=True)]
    return sorted(dcm)

def run() -> None:
    labels = read_labels()

    # Photos
    photo_paths = _collect_images()
    count_photo = 0
    for p in photo_paths:
        try:
            process_one_photo(p, labels)
            count_photo += 1
        except Exception as e:
            print(f"[WARN] Photo skipped {p.name}: {e}")

    # Dicoms
    count_dicom = 0
    if HAS_PYDICOM:
        for d in _collect_dicoms():
            try:
                process_one_dicom(d, labels)
                count_dicom += 1
            except Exception as e:
                print(f"[WARN] DICOM skipped {d.name}: {e}")

    total = count_photo + count_dicom
    if total == 0:
        print(f"[INFO] Aucune image trouvée.")
        print(f" - Place des JPEG/PNG dans: {RAW_IMG_DIR}")
        print(f" - Ou des DICOM dans     : {RAW_DICOM_DIR}")
        return

    print(f"OK: {total} images traitées ({count_photo} photos, {count_dicom} dicom).")
    print(f"PNG images       -> {IMG_DIR}")
    print(f"Anonymized DICOM -> {ANON_DICOM_DIR}")
    print(f"info.txt         -> {INFO_TXT}")
    print(f"ages.csv         -> {AGES_CSV}")
