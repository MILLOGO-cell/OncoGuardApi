import re
from typing import Optional, List, Tuple
import numpy as np
import cv2

from .config import TESS_AVAILABLE
if TESS_AVAILABLE:
    import pytesseract  # type: ignore

AGE_REGEX = re.compile(r"age\s*[:\-]?\s*(\d{1,3})\s*[yY]?", re.IGNORECASE)
DOB_PATTERNS = [
    re.compile(r"(\d{1,2})[-/\s](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/\s](\d{4})", re.IGNORECASE),
    re.compile(r"(\d{1,2})[-/\s](\d{1,2})[-/\s](\d{4})"),
    re.compile(r"DOB\s*[:\-]?\s*(\d{1,2})[-/\s](\d{1,2})[-/\s](\d{4})", re.IGNORECASE),
]

MONTH_MAP = {m: i+1 for i, m in enumerate(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
)}

def _calc_age_from_dob(day: int, month: int, year: int, current_year: int = 2025) -> Optional[int]:
    try:
        if not (1900 <= year <= current_year): return None
        if not (1 <= month <= 12): return None
        if not (1 <= day <= 31): return None
        age = current_year - year
        return age if 10 < age < 120 else None
    except Exception:
        return None

def _ocr_age_from_mat(gray: np.ndarray) -> Optional[int]:
    if not TESS_AVAILABLE:
        return None
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (3,3), 0)
    mats = [
        cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
        cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1],
    ]
    ages: List[int] = []
    for th in mats:
        for psm in (6,7,11):
            try:
                text = pytesseract.image_to_string(th, config=f"--psm {psm}")
            except Exception:
                continue
            # DOB
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
                            if age: ages.append(age)
                    except Exception:
                        pass
            # AGE:
            m = AGE_REGEX.search(text)
            if m:
                try:
                    val = int(m.group(1))
                    if 10 < val < 120: ages.append(val)
                except Exception:
                    pass
    return max(set(ages), key=ages.count) if ages else None

def extract_age_from_pixels(gray_img: np.ndarray) -> Optional[int]:
    """gray_img: image numpy 2D (L). ROI multiple pour maximiser la détection."""
    h, w = gray_img.shape[:2]
    rois: List[Tuple[int,int,int,int]] = [
        (0, int(0.50*w), int(0.50*h), h),
        (0, int(0.50*w), 0, int(0.30*h)),
        (0, w, int(0.70*h), h),
        (int(0.50*w), w, 0, int(0.30*h)),
        (int(0.50*w), w, int(0.70*h), h),
    ]
    for x1,x2,y1,y2 in rois:
        roi = gray_img[y1:y2, x1:x2]
        if roi.size == 0: continue
        age = _ocr_age_from_mat(roi)
        if age: return age
    return _ocr_age_from_mat(gray_img)
