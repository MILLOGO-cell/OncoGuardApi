from pathlib import Path
from typing import Dict, Optional
import uuid
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.uid import generate_uid
    from pydicom.multival import MultiValue
except Exception:
    pydicom = None
    MultiValue = tuple  # type: ignore

from .config import ANON_DICOM_DIR
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
        if tag in ds: ds[tag].value = ""
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    return ds

def laterality(ds) -> str:
    lat = (ds.get("Laterality") or "").strip().upper()
    return lat if lat in {"L","R"} else "U"

def view_position(ds) -> str:
    vp = (ds.get("ViewPosition") or "").strip().upper()
    return vp or "UNK"

def to_uint8_dcm(ds) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    if (ds.get("PhotometricInterpretation") or "").upper() == "MONOCHROME1":
        arr = np.max(arr) - arr
    wc = _to_float(ds.get("WindowCenter")); ww = _to_float(ds.get("WindowWidth"))
    if wc is not None and ww is not None and ww > 1:
        low, high = wc - ww/2.0, wc + ww/2.0
        arr = np.clip(arr, low, high)
    else:
        lo, hi = np.percentile(arr, (1,99))
        arr = np.clip(arr, lo, hi)
    denom = (arr.max()-arr.min()) or 1.0
    arr = (arr - arr.min())/denom
    return (arr*255.0).round().astype(np.uint8)

def process_one_dicom(dcm_path: Path, labels: Dict[str, Dict[str, str]]) -> str:
    assert pydicom is not None, "pydicom manquant"
    ds = pydicom.dcmread(str(dcm_path))
    ds = anonymize_dataset(ds)
    (ANON_DICOM_DIR / f"{uuid.uuid4().hex}.dcm").write_bytes(ds.to_json().encode("utf-8"))

    side = laterality(ds)
    view = view_position(ds)

    pil = Image.fromarray(to_uint8_dcm(ds), mode="L")
    if side == "R":
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
    pil = redact_burned_text(pil)
    pil = resize_pad_to_square(pil, 1024)

    img_id = next_image_id("bfa")
    pil.save( (dcm_path.parent.parent / "images" / f"{img_id}.png") )

    append_info_line(img_id, side, view, "U", "NORM")

    age = impute_age_from_bins(load_existing_ages())
    save_age_row(img_id, age, "imputed" if age is not None else "none")
    return img_id
