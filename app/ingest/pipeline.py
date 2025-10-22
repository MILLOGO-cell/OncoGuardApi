# app/ingest/pipeline.py
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from .config import IMG_DIR, ANON_DICOM_DIR

# read_labels peut ne pas exister : on gère proprement
try:
    from .io_utils import read_labels  # lit LABELS_CSV -> Dict[str, Dict[str, str]]
except Exception:
    def read_labels() -> Dict[str, Dict[str, str]]:  # fallback
        return {}

from .photo_proc import process_one_photo

# DICOM optionnel
try:
    from .dicom_proc import process_one_dicom
    HAS_PYDICOM = True
except Exception:
    HAS_PYDICOM = False

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DICOM_EXTS = {".dcm", ".DCM"}


def _proc_photo_one(p: Path, labels, out_img_dir: Path) -> Tuple[str, Path, Optional[Path]]:
    img_id, png_out = process_one_photo(p, labels, out_img_dir=out_img_dir)
    return img_id, png_out, None


def _proc_dicom_one(p: Path, labels, out_img_dir: Path, out_dicom_dir: Path) -> Tuple[str, Path, Optional[Path]]:
    assert HAS_PYDICOM, "pydicom manquant"
    img_id, png_out, dcm_out = process_one_dicom(
        p, labels, out_img_dir=out_img_dir, out_dicom_dir=out_dicom_dir
    )
    return img_id, png_out, dcm_out


def run(
    input_paths: List[Path],
    out_img_dir: Path = IMG_DIR,
    out_dicom_dir: Path = ANON_DICOM_DIR,
    max_workers: Optional[int] = None,
) -> List[Tuple[str, Path, Optional[Path]]]:
    """
    Traite UNIQUEMENT les fichiers fournis.
    Retourne: [(image_id, png_path, anonymized_dicom_path|None)].
    """
    if not input_paths:
        return []

    # parallélisme borné
    workers = max(1, int(os.getenv("INGEST_WORKERS", "2")))
    if max_workers:
        workers = max(1, min(workers, max_workers))

    labels = read_labels() or {}
    results: List[Tuple[str, Path, Optional[Path]]] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for p in input_paths:
            ext = p.suffix.lower()
            if ext in PHOTO_EXTS:
                futures.append(ex.submit(_proc_photo_one, p, labels, out_img_dir))
            elif ext in DICOM_EXTS and HAS_PYDICOM:
                futures.append(ex.submit(_proc_dicom_one, p, labels, out_img_dir, out_dicom_dir))
            else:
                continue

        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"[WARN] Fichier ignoré: {e}")

    return results
