# app/ingest/dicom_proc.py
from pathlib import Path
from typing import Dict, Optional, Tuple
import uuid
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    from pydicom.multival import MultiValue
except Exception:
    pydicom = None
    MultiValue = tuple  # type: ignore

from .io_utils import next_image_id, resize_pad_to_square, append_info_line, \
                      load_existing_ages, save_age_row, impute_age_from_bins
from .redact import redact_burned_text

TO_CLEAR = [
    (0x0010,0x0010),(0x0010,0x0020),(0x0010,0x0030),(0x0010,0x0040),
    (0x0008,0x0020),(0x0008,0x0030),(0x0008,0x0080),(0x0008,0x0090),
    (0x0008,0x1010),(0x0008,0x1040),(0x0018,0x1000),
    (0x0020,0x000D),(0x0020,0x000E),(0x0008,0x0018),
]

def _to_float(v: Optional[object]) -> Optional[float]:
    try:
        if v is None: return None
        if isinstance(v, MultiValue): return float(v[0])
        return float(v)
    except Exception:
        return None

def anonymize_dataset(ds):
    ds.remove_private_tags()
    for tag in TO_CLEAR:
        if tag in ds:
            ds[tag].value = ""
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    # normalise le meta
    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    return ds

def laterality(ds) -> str:
    lat = (getattr(ds, "Laterality", "") or "").strip().upper()
    return lat if lat in {"L","R"} else "U"

def view_position(ds) -> str:
    vp = (getattr(ds, "ViewPosition", "") or "").strip().upper()
    return vp or "UNK"

def to_uint8_dcm(ds) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    if (getattr(ds, "PhotometricInterpretation", "") or "").upper() == "MONOCHROME1":
        arr = np.max(arr) - arr
    wc = _to_float(getattr(ds, "WindowCenter", None))
    ww = _to_float(getattr(ds, "WindowWidth", None))
    if wc is not None and ww is not None and ww > 1:
        low, high = wc - ww/2.0, wc + ww/2.0
        arr = np.clip(arr, low, high)
    else:
        lo, hi = np.percentile(arr, (1, 99))
        arr = np.clip(arr, lo, hi)
    denom = (arr.max() - arr.min()) or 1.0
    arr = (arr - arr.min()) / denom
    return (arr * 255.0).round().astype(np.uint8)

def process_one_dicom(
    dcm_path: Path,
    labels: Dict[str, Dict[str, str]],
    out_img_dir: Path,
    out_dicom_dir: Path,
) -> Tuple[str, Path, Path]:
    assert pydicom is not None, "pydicom manquant"

    # lecture légère des en-têtes; les pixels ne sont chargés que quand on accède à pixel_array
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    # pour l’export PNG on recharge avec pixels (chargement à la demande)
    ds_full = pydicom.dcmread(str(dcm_path), force=True)

    ds = anonymize_dataset(ds_full)

    out_dicom_dir.mkdir(parents=True, exist_ok=True)
    anon_name = f"{uuid.uuid4().hex}.dcm"
    dcm_out = out_dicom_dir / anon_name
    ds.save_as(str(dcm_out), write_like_original=False)

    side = laterality(ds)
    view = view_position(ds)

    arr = to_uint8_dcm(ds)
    pil = Image.fromarray(arr, mode="L")
    if side == "R":
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
    pil = redact_burned_text(pil)
    pil = resize_pad_to_square(pil, 1024)

    img_id = next_image_id("bfa")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_img_dir / f"{img_id}.png"
    pil.save(png_path)

    append_info_line(img_id, side, view, "U", "NORM")

    age = impute_age_from_bins(load_existing_ages())
    save_age_row(img_id, age, "imputed" if age is not None else "none")
    return img_id, png_path, dcm_out
