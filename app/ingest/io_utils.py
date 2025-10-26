# app/ingest/io_utils.py
from __future__ import annotations

import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageOps

# On garde le même nom IMG_DIR dans ce module, mais il pointe vers DERIVED_IMG_DIR
from .config import (
    DERIVED_IMG_DIR as IMG_DIR,
    INFO_TXT,
    LABELS_CSV,
    AGE_BINS,
    AGES_CSV,
    DATASET_DIR,
)

# ---------- ID incrementeur (cache process) ----------
_id_counter: int | None = None
_id_regex = re.compile(r"^([a-zA-Z]+)(\d+)\.png$")


def _init_id_counter(prefix: str) -> int:
    IMG_DIR.mkdir(parents=True, exist_ok=True)  # s'assurer que le dossier existe
    max_idx = 0
    for p in IMG_DIR.glob("*.png"):
        m = _id_regex.match(p.name)
        if not m:
            continue
        pfx, num = m.group(1), m.group(2)
        if pfx.lower() == prefix.lower():
            try:
                max_idx = max(max_idx, int(num))
            except Exception:
                pass
    return max_idx


def next_image_id(prefix: str = "bfa") -> str:
    global _id_counter
    if _id_counter is None:
        _id_counter = _init_id_counter(prefix)
    _id_counter += 1
    return f"{prefix}{_id_counter:03d}"


# ---------- Labels ----------
def read_labels() -> Dict[str, Dict[str, str]]:
    if not LABELS_CSV.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with LABELS_CSV.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rid = (row.get("id") or "").strip()
            if not rid:
                continue
            out[rid] = {
                "abn": (row.get("abnormality") or "NORM").strip().upper(),
                "sev": (row.get("severity") or "").strip().upper(),
                "x": (row.get("x") or "").strip(),
                "y": (row.get("y") or "").strip(),
                "r": (row.get("r") or "").strip(),
            }
    return out


# ---------- Helpers image ----------
def resize_pad_to_square(img: Image.Image, size: int = 1024) -> Image.Image:
    """
    Redimensionne en conservant le ratio, puis pad pour obtenir un carré 'size'x'size'.
    Sortie en niveaux de gris ("L").
    """
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("L", "I;16", "I", "F"):
        img = img.convert("L")

    w, h = img.size
    if w == 0 or h == 0:
        # évite division par zéro si image vide
        canvas = Image.new("L", (size, size), 0)
        return canvas

    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    # Utiliser l’API moderne de Pillow pour l’interpolation
    img = img.resize((nw, nh), Image.Resampling.BILINEAR)

    canvas = Image.new("L", (size, size), 0)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def parse_side_view_from_name(name: str) -> Tuple[str, str]:
    """
    Heuristique basée sur le nom de fichier (utile quand les tags DICOM sont absents).
    """
    n = name.lower()
    side = "UNKNOWN"
    view = "UNK"
    if any(k in n for k in ["_l_", "-l-", " left", " left.", " left_", "l-"]):
        side = "L"
    if any(k in n for k in ["_r_", "-r-", " right", " right.", " right_", "r-"]):
        side = "R"
    if "mlo" in n:
        view = "MLO"
    elif "cc" in n:
        view = "CC"
    return side, view


def append_info_line(
    img_id: str,
    side: str,
    view: str,
    tissue: str,
    abn: str,
    sev: str = "",
    x: str = "",
    y: str = "",
    r: str = "",
) -> None:
    """
    Ecrit une ligne dans info.txt (id, côté, vue, tissu, anomalie, ...).
    """
    side = side.upper()
    side = {"L": "LEFT", "R": "RIGHT"}.get(
        side, side if side in {"LEFT", "RIGHT"} else "UNKNOWN"
    )
    view = (view or "UNK").upper()
    tissue = (tissue or "U").upper()
    abn = (abn or "NORM").upper()

    parts = [img_id, side, view, tissue, abn]
    if abn != "NORM" and sev:
        parts.append(sev.upper())
    if x and y and r:
        try:
            parts.extend(
                [str(int(float(x))), str(int(float(y))), str(int(float(r)))]
            )
        except Exception:
            # ignore si x/y/r ne sont pas numériques
            pass

    INFO_TXT.parent.mkdir(parents=True, exist_ok=True)
    with INFO_TXT.open("a", encoding="utf-8") as f:
        f.write(" ".join(parts) + "\n")


# ---------- Ages ----------
def load_existing_ages() -> List[int]:
    if not AGES_CSV.exists():
        return []
    vals: List[int] = []
    with AGES_CSV.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                a = int(row.get("age") or "")
                if 10 < a < 120:
                    vals.append(a)
            except Exception:
                continue
    return vals


def age_to_bin(age: int) -> str:
    for lo, hi, _mid in AGE_BINS:
        if lo <= age < hi:
            return f"{lo}-{hi-1}"
    return "80+"


def save_age_row(img_id: str, age: Optional[int], source: str) -> None:
    """
    Append (id, age, age_bin, source) dans ages.csv.
    Si ages.csv est ouvert/locké (Excel), écrire dans ages_buffer.csv en attendant.
    """
    out_file = AGES_CSV
    if age is None:
        age_bin, val = "", ""
    else:
        age_bin, val = age_to_bin(age), str(age)

    line = f"{img_id},{val},{age_bin},{source}\n"

    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("a", newline="", encoding="utf-8") as f:
            f.write(line)
    except PermissionError:
        buf = DATASET_DIR / "ages_buffer.csv"
        buf.parent.mkdir(parents=True, exist_ok=True)
        with buf.open("a", newline="", encoding="utf-8") as f:
            if buf.stat().st_size == 0:
                f.write("id,age,age_bin,source\n")
            f.write(line)
        print(
            f"[WARN] ages.csv verrouillé : écrit dans {buf.name}. "
            "Fermez Excel puis fusionnez."
        )


def impute_age_from_bins(existing: List[int]) -> Optional[int]:
    if not existing:
        return None
    hist: List[Tuple[int, int]] = []
    for lo, hi, mid in AGE_BINS:
        c = sum(1 for a in existing if lo <= a < hi)
        hist.append((c, mid))
    hist.sort(reverse=True)
    return hist[0][1] if hist and hist[0][0] > 0 else None
