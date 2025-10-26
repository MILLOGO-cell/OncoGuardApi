# app/ingest/dicom_proc.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING
import uuid
import numpy as np
from PIL import Image

# --- Typage 'pydicom' sans casser l'analyse statique ---
if TYPE_CHECKING:
    # Visible seulement pour l'analyseur de type
    from pydicom.dataset import Dataset as DicomDataset
else:
    # Au runtime, on alias vers Any pour éviter les erreurs si pydicom manque
    DicomDataset = Any  # type: ignore[misc]

# Import runtime optionnel de pydicom
try:
    import pydicom  # type: ignore
except Exception:
    pydicom = None  # type: ignore[assignment]

from .config import (
    CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT,
)
from .io_utils import (
    next_image_id, resize_pad_to_square, append_info_line,
    load_existing_ages, save_age_row, impute_age_from_bins,
)
from .age_ocr import extract_age


# ----------------------------
# Utilitaires image / DICOM
# ----------------------------
def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        # WindowCenter/Width peuvent être MultiValue
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            return float(list(v)[0])  # type: ignore[arg-type]
        return float(v)
    except Exception:
        return None


def laterality(ds: DicomDataset) -> str:
    lat = (getattr(ds, "Laterality", "") or "").strip().upper()
    return lat if lat in {"L", "R"} else "U"


def view_position(ds: DicomDataset) -> str:
    vp = (getattr(ds, "ViewPosition", "") or "").strip().upper()
    return vp or "UNK"


def to_uint8_dcm(ds: DicomDataset) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)

    # MONOCHROME1 → inverser
    if (getattr(ds, "PhotometricInterpretation", "") or "").upper() == "MONOCHROME1":
        arr = np.max(arr) - arr

    wc = _to_float(getattr(ds, "WindowCenter", None))
    ww = _to_float(getattr(ds, "WindowWidth", None))
    if wc is not None and ww is not None and ww > 1:
        low, high = wc - ww / 2.0, wc + ww / 2.0
        arr = np.clip(arr, low, high)
    else:
        lo, hi = np.percentile(arr, (1, 99))
        arr = np.clip(arr, lo, hi)

    denom = (arr.max() - arr.min()) or 1.0
    arr = (arr - arr.min()) / denom
    return (arr * 255.0).round().astype(np.uint8)


def _optional_border_crop(pil: Image.Image) -> Image.Image:
    """Recadre légèrement selon les ratios définis en config (par défaut 0)."""
    if all(v == 0 for v in (CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT)):
        return pil
    w, h = pil.size
    left = int(w * CROP_LEFT)
    right = w - int(w * CROP_RIGHT)
    top = int(h * CROP_TOP)
    bottom = h - int(h * CROP_BOTTOM)
    if right <= left or bottom <= top:
        return pil
    return pil.crop((left, top, right, bottom))


# ----------------------------
# Traitement principal
# ----------------------------
def process_one_dicom(
    dcm_path: Path,
    labels: Dict[str, Dict[str, str]],  # conservé pour compat signature
    out_img_dir: Path,
    out_dicom_dir: Path,
) -> Tuple[str, Path, Path]:
    """
    - NE PAS anonymiser (les appareils exportent déjà anonymisé).
    - Âge: lire via tags DICOM (PatientAge / BirthDate + Study/Content/AcquisitionDate),
      sinon OCR en secours si nécessaire.
    - Pas de 'redact' des pixels; recadrage léger des bordures UI (optionnel via config).
    - Génère un PNG dérivé pour la visu interne (pipeline).
    """
    if pydicom is None:
        raise RuntimeError("pydicom manquant — installe-le pour traiter des DICOMs.")

    # Lire le dataset complet (pixels inclus)
    ds: DicomDataset = pydicom.dcmread(str(dcm_path), force=True)

    # Copie “normalisée” (pas d’anonymisation), juste pour centraliser la sortie
    out_dicom_dir.mkdir(parents=True, exist_ok=True)
    norm_name = f"{uuid.uuid4().hex}.dcm"
    dcm_out = out_dicom_dir / norm_name
    ds.save_as(str(dcm_out), write_like_original=False)

    # Métadonnées utiles
    side = laterality(ds)          # L / R / U
    view = view_position(ds)       # MLO / CC / etc.

    # Image 8 bits pour dérivé PNG
    arr = to_uint8_dcm(ds)
    pil = Image.fromarray(arr, mode="L")

    # Flip horizontal si tu veux uniformiser la latéralité
    if side == "R":
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)

    # Recadrage optionnel (bordures UI), AUCUNE censure de contenu clinique
    pil = _optional_border_crop(pil)

    # Redimensionnement/padding carré (pour pipeline/visu)
    pil = resize_pad_to_square(pil, 1024)

    # Sauvegarde image dérivée
    img_id = next_image_id("bfa")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_img_dir / f"{img_id}.png"
    pil.save(png_path)

    # Info ligne (ex. pour cataloguer)
    append_info_line(img_id, side, view, "U", "NORM")

    # ----- ÂGE -----
    # Priorité DICOM -> OCR fallback géré par extract_age
    age = extract_age(gray_img=np.array(pil), dicom_ds=ds)
    if age is not None:
        source = "dicom_or_ocr"
    else:
        # dernier recours: imputation par bins existants
        age = impute_age_from_bins(load_existing_ages())
        source = "imputed" if age is not None else "none"

    save_age_row(img_id, age, source)

    return img_id, png_path, dcm_out
