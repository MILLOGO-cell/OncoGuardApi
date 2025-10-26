# app/ingest/pipeline.py
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from PIL import Image
import numpy as np

from .config import DERIVED_IMG_DIR, NORMALIZED_DICOM_DIR
# read_labels peut ne pas exister : on gère proprement
try:
    from .io_utils import read_labels  # Dict[str, Dict[str, str]]
except Exception:
    def read_labels() -> Dict[str, Dict[str, str]]:  # fallback
        return {}

# DICOM (optionnel au runtime)
try:
    from .dicom_proc import process_one_dicom
    HAS_PYDICOM = True
except Exception:
    HAS_PYDICOM = False

# utilitaires communs
from .io_utils import (
    next_image_id, resize_pad_to_square, append_info_line,
    load_existing_ages, save_age_row, impute_age_from_bins,
    parse_side_view_from_name,
)
from .age_ocr import extract_age  # gère DICOM (si fourni) ou OCR sur pixels

# Extensions prises en charge
PGM_EXTS = {".pgm"}
DICOM_EXTS = {".dcm", ".dicom"}

def _proc_pgm_one(p: Path, labels: Dict[str, Dict[str, str]], out_img_dir: Path) -> Tuple[str, Path, Optional[Path]]:
    """
    Traitement minimal d'un PGM:
    - lecture pixels -> grayscale
    - resize + pad carré
    - âge via OCR (tags inexistants)
    - append info + ages.csv
    """
    if not p.exists():
        raise FileNotFoundError(str(p))

    # Lecture PGM -> PIL -> Numpy
    pil = Image.open(p).convert("L")
    arr = np.array(pil)

    # Âge (pas de DICOM ici) : OCR direct sur pixels
    age = extract_age(gray_img=arr, dicom_ds=None)
    source = "ocr" if age is not None else ""

    # Heuristique côté/vue depuis le nom de fichier
    side, view = parse_side_view_from_name(p.stem)

    # Prépare image dérivée
    pil_out = resize_pad_to_square(pil, 1024)
    img_id = next_image_id("bfa")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_img_dir / f"{img_id}.png"
    pil_out.save(png_path)

    # Info + âge
    tissue = "U"
    if img_id in (labels or {}):
        a = labels[img_id]
        append_info_line(img_id, side, view, tissue, a["abn"], a["sev"], a["x"], a["y"], a["r"])
    else:
        append_info_line(img_id, side, view, tissue, "NORM")

    if age is None:
        age = impute_age_from_bins(load_existing_ages())
        save_age_row(img_id, age, "imputed" if age is not None else "none")
    else:
        save_age_row(img_id, age, source)

    return img_id, png_path, None


def _proc_dicom_one(p: Path, labels: Dict[str, Dict[str, str]], out_img_dir: Path, out_dicom_dir: Path) -> Tuple[str, Path, Optional[Path]]:
    assert HAS_PYDICOM, "pydicom manquant"
    return process_one_dicom(p, labels, out_img_dir=out_img_dir, out_dicom_dir=out_dicom_dir)


def run(
    input_paths: List[Path],
    out_img_dir: Path = DERIVED_IMG_DIR,
    out_dicom_dir: Path = NORMALIZED_DICOM_DIR,
    max_workers: Optional[int] = None,
) -> List[Tuple[str, Path, Optional[Path]]]:
    """
    Traite UNIQUEMENT les fichiers fournis.
    Retourne: [(image_id, png_path, dicom_path|None)].
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
            if ext in PGM_EXTS:
                futures.append(ex.submit(_proc_pgm_one, p, labels, out_img_dir))
            elif ext in DICOM_EXTS and HAS_PYDICOM:
                futures.append(ex.submit(_proc_dicom_one, p, labels, out_img_dir, out_dicom_dir))
            else:
                # extension non supportée ou pydicom absent
                continue

        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"[WARN] Fichier ignoré: {e}")

    return results
