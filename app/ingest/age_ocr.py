# ingest/age_ocr.py
import re
from typing import Optional, List, Tuple, Union, TYPE_CHECKING, Any
import numpy as np
import cv2

if TYPE_CHECKING:
    from pydicom.dataset import Dataset
else:
    Dataset = Any  # type: ignore

try:
    # pydicom est optionnel mais recommandé car tes données sont en DICOM
    import pydicom
    DICOM_AVAILABLE = True
except Exception:
    DICOM_AVAILABLE = False

from .config import TESS_AVAILABLE
if TESS_AVAILABLE:
    import pytesseract  # type: ignore

# -----------------------
# Regex / constantes OCR
# -----------------------
AGE_WORDS = r"(age|âge|yrs?|years?|ans?)"
AGE_REGEX = re.compile(rf"{AGE_WORDS}\s*[:=\-]?\s*(\d{{1,3}})\s*(y|yr|yrs|years|ans)?", re.IGNORECASE)

# Exemples de DOB: 07/10/1980, 7-10-1980, 07 Oct 1980, DOB: 07-10-1980
DOB_PATTERNS = [
    re.compile(r"(\d{1,2})[-/\s](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/\s](\d{4})", re.IGNORECASE),
    re.compile(r"(\d{1,2})[-/\s](\d{1,2})[-/\s](\d{4})"),
    re.compile(r"DOB\s*[:=\-]?\s*(\d{1,2})[-/\s](\d{1,2})[-/\s](\d{4})", re.IGNORECASE),
]
MONTH_MAP = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
)}

# -----------------------
# Utilitaires DICOM
# -----------------------
def _safe_str(ds: "Dataset", tag: str) -> Optional[str]:
    try:
        val = getattr(ds, tag, None)
        if val is None:
            return None
        s = str(val).strip()
        return s if s else None
    except Exception:
        return None

def _parse_patient_age_tag(value: str) -> Optional[int]:
    """
    PatientAge (0010,1010) typiquement '045Y', '032Y', '018M', '200D', etc.
    Convertit en années (arrondi inférieur raisonnable).
    """
    try:
        v = value.strip().upper()
        m = re.match(r"(\d{1,3})([YMWD])", v)
        if not m:
            # parfois juste '45' sans suffixe
            if v.isdigit():
                iv = int(v)
                return iv if 0 < iv < 130 else None
            return None
        num = int(m.group(1))
        unit = m.group(2)
        if unit == "Y":
            years = num
        elif unit == "M":
            years = num / 12.0
        elif unit == "W":
            years = num / 52.0
        else:  # 'D'
            years = num / 365.25
        iy = int(years)  # âge en années entières
        return iy if 0 < iy < 130 else None
    except Exception:
        return None

def _parse_yyyymmdd(s: str) -> Optional[Tuple[int, int, int]]:
    # DICOM date format: YYYYMMDD
    try:
        s = s.strip()
        if len(s) != 8 or not s.isdigit():
            return None
        y = int(s[0:4]); m = int(s[4:6]); d = int(s[6:8])
        if not (1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31):
            return None
        return y, m, d
    except Exception:
        return None

def _age_from_dates(birth: Tuple[int, int, int], ref: Tuple[int, int, int]) -> Optional[int]:
    try:
        by, bm, bd = birth
        ry, rm, rd = ref
        # âge entier : soustraire, puis corriger si l'anniversaire n'est pas encore passé
        age = ry - by - ((rm, rd) < (bm, bd))
        return age if 0 <= age < 130 else None
    except Exception:
        return None

def _age_from_dicom(ds: "Dataset") -> Optional[int]:
    # 1) PatientAge direct
    pa = _safe_str(ds, "PatientAge")
    if pa:
        age = _parse_patient_age_tag(pa)
        if age is not None:
            return age

    # 2) BirthDate + Study/Content/Acquisition date
    dob = _safe_str(ds, "PatientBirthDate")
    ref_date = _safe_str(ds, "StudyDate") or _safe_str(ds, "ContentDate") or _safe_str(ds, "AcquisitionDate")
    if dob and ref_date:
        b = _parse_yyyymmdd(dob)
        r = _parse_yyyymmdd(ref_date)
        if b and r:
            return _age_from_dates(b, r)

    # 3) BirthDate seul (peu fiable — on évite d'utiliser la date système ici)
    # -> si pas de ref_date, on ne calcule pas pour éviter un âge "aujourd'hui" non pertinent.
    return None

# -----------------------
# OCR fallback
# -----------------------
def _calc_age_from_dob(day: int, month: int, year: int, current_year: int = 2025) -> Optional[int]:
    try:
        if not (1900 <= year <= current_year): return None
        if not (1 <= month <= 12): return None
        if not (1 <= day <= 31): return None
        age = current_year - year
        return age if 10 <= age < 130 else None
    except Exception:
        return None

def _ocr_age_from_mat(gray: np.ndarray) -> Optional[int]:
    if not TESS_AVAILABLE:
        return None

    # prétraitements robustes caractères fins (overlays mammographie)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)
    thr1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    mats = [thr1, thr2]

    ages: List[int] = []
    tesseract_cfgs = [
        "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz:/- ",
        "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz:/- ",
        "--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz:/- ",
    ]

    for th in mats:
        # léger morph pour recoller caractères fragmentés
        kernel = np.ones((2, 2), np.uint8)
        thm = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

        for cfg in tesseract_cfgs:
            try:
                text = pytesseract.image_to_string(thm, config=cfg)
            except Exception:
                continue
            if not text:
                continue

            # DOB → âge
            for pat in DOB_PATTERNS:
                for m in pat.findall(text):
                    try:
                        if len(m) == 3:
                            d, mth, y = m
                            if isinstance(mth, str) and mth.lower()[:3] in MONTH_MAP:
                                month = MONTH_MAP[mth.lower()[:3]]
                            else:
                                month = int(mth)
                            age = _calc_age_from_dob(int(d), month, int(y))
                            if age is not None:
                                ages.append(age)
                    except Exception:
                        pass

            # AGE explicite
            m = AGE_REGEX.search(text)
            if m:
                try:
                    val = int(m.group(1))
                    if 0 < val < 130:
                        ages.append(val)
                except Exception:
                    pass

            # cas compacts: "45Y", "56yrs", "60ans"
            for compact in re.findall(r"(\d{1,3})\s*(y|yr|yrs|years|ans)\b", text, flags=re.IGNORECASE):
                try:
                    v = int(compact[0])
                    if 0 < v < 130:
                        ages.append(v)
                except Exception:
                    pass

    return max(set(ages), key=ages.count) if ages else None

# -----------------------
# API principale
# -----------------------
def extract_age(
    *,
    gray_img: Optional[np.ndarray] = None,
    dicom_ds: Optional["Dataset"] = None
) -> Optional[int]:
    """
    Tente de retourner l'âge de la patiente:
    1) Tags DICOM (PatientAge, PatientBirthDate + Study/Content/AcquisitionDate)
    2) OCR des overlays si BurnedInAnnotation=YES ou si tags manquants

    Params:
        gray_img: image 2D (PGM/NumPy) – utile pour l'OCR si nécessaire
        dicom_ds: pydicom Dataset – priorité pour lecture de l'âge

    Returns:
        âge en années (int) ou None.
    """
    # (A) DICOM prioritaire si dispo
    if DICOM_AVAILABLE and isinstance(dicom_ds, pydicom.dataset.Dataset):
        age = _age_from_dicom(dicom_ds)
        if age is not None:
            return age

        # Si le producteur signale des annotations brulées dans l'image
        bia = _safe_str(dicom_ds, "BurnedInAnnotation")
        if bia and bia.upper() == "YES" and gray_img is not None:
            ocr_age = _ocr_age_from_mat(gray_img)
            if ocr_age is not None:
                return ocr_age
        # Sinon, si pas d'info DICOM et image fournie → tenter quand même OCR
        if gray_img is not None:
            ocr_age = _ocr_age_from_mat(gray_img)
            if ocr_age is not None:
                return ocr_age
        return None

    # (B) Pas de DICOM dispo → OCR si image fournie
    if gray_img is not None:
        return _ocr_age_from_mat(gray_img)

    return None


def extract_age_from_pixels(gray_img: np.ndarray) -> Optional[int]:
    """
    Compatibilité rétro (signature identique à l'ancienne fonction).
    Conserve une logique de ROIs + OCR, mais **à utiliser seulement en fallback**
    si DICOM non disponible.
    """
    if gray_img is None or gray_img.ndim != 2:
        return None

    h, w = gray_img.shape[:2]
    rois: List[Tuple[int, int, int, int]] = [
        (0, int(0.50 * w), int(0.50 * h), h),
        (0, int(0.50 * w), 0, int(0.30 * h)),
        (0, w, int(0.70 * h), h),
        (int(0.50 * w), w, 0, int(0.30 * h)),
        (int(0.50 * w), w, int(0.70 * h), h),
    ]
    for x1, x2, y1, y2 in rois:
        roi = gray_img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        age = _ocr_age_from_mat(roi)
        if age:
            return age
    return _ocr_age_from_mat(gray_img)