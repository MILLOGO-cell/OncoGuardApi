# app/ingest/photo_proc.py
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from PIL import Image
from .io_utils import next_image_id, resize_pad_to_square, append_info_line, \
                      parse_side_view_from_name, load_existing_ages, save_age_row, impute_age_from_bins
from .redact import redact_burned_text
from .age_ocr import extract_age_from_pixels

def process_one_photo(
    img_path: Path,
    labels: Dict[str, Dict[str, str]],
    out_img_dir: Path,
) -> Tuple[str, Path]:
    orig = Image.open(img_path)
    side, view = parse_side_view_from_name(img_path.stem)
    pil = orig.transpose(Image.FLIP_LEFT_RIGHT) if side == "R" else orig

    # OCR âge AVANT rédaction
    age = extract_age_from_pixels(np.array(orig.convert("L")))
    source = "ocr" if age is not None else ""

    pil = redact_burned_text(pil)
    pil = resize_pad_to_square(pil, 1024)

    img_id = next_image_id("bfa")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_img_dir / f"{img_id}.png"
    pil.save(png_path)

    tissue = "U"
    if img_id in labels:
        a = labels[img_id]
        append_info_line(img_id, side, view, tissue, a["abn"], a["sev"], a["x"], a["y"], a["r"])
    else:
        append_info_line(img_id, side, view, tissue, "NORM")

    if age is None:
        age = impute_age_from_bins(load_existing_ages())
        save_age_row(img_id, age, "imputed" if age is not None else "none")
    else:
        save_age_row(img_id, age, source)

    return img_id, png_path
