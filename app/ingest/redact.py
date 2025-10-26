# app/ingest/redact.py
from __future__ import annotations

import numpy as np
from PIL import Image
import cv2

from .config import CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT


def _clean_roi(crop_gray: np.ndarray) -> np.ndarray:
    """
    Détection robuste de textes/overlays dans la ROI uniquement (pas de pixels cliniques).
    Combine blackhat et tophat, puis inpainting TELEA.
    """
    # Lissage léger
    blur = cv2.GaussianBlur(crop_gray, (3, 3), 0)

    # Kernels morpho (adaptés aux textes fins des overlays)
    k_black = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    k_top   = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))

    # Emphase des tracés sombres sur fond clair (blackhat) et clairs sur fond sombre (tophat)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, k_black)
    tophat   = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT,   k_top)

    # Seuils Otsu
    _, th_b = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_t = cv2.threshold(tophat,   0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Masque combiné
    mask = cv2.bitwise_or(th_b, th_t)

    # Nettoyage morpho du masque
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    mask = cv2.dilate(mask,      cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)

    # Inpainting uniquement dans la ROI
    cleaned = cv2.inpaint(crop_gray, mask, 3, cv2.INPAINT_TELEA)
    return cleaned


def redact_burned_text(pil_img: Image.Image) -> Image.Image:
    """
    Nettoie les overlays/écritures UNIQUEMENT dans la zone de marge définie par CROP_*
    (bandeaux UI), sans jamais modifier l’aire clinique.
    - Si CROP_* == 0 → no-op (retourne l'image d'origine).
    - Retourne une image dans le même mode (L ou RGB) que l'entrée.
    """
    # No-op si pas de marges configurées
    if all(v == 0 for v in (CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT)):
        return pil_img

    # Travail en niveaux de gris
    orig_mode = pil_img.mode
    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape[:2]

    # Définition de la ROI (marges) ; si invalides → no-op
    y0 = int(h * CROP_TOP)
    y1 = int(h * (1.0 - CROP_BOTTOM))
    x0 = int(w * CROP_LEFT)
    x1 = int(w * (1.0 - CROP_RIGHT))
    if y1 <= y0 or x1 <= x0:
        return pil_img

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return pil_img

    # Nettoyage de la ROI
    try:
        cleaned_roi = _clean_roi(roi)
    except Exception:
        # En cas d'échec (OpenCV absent, etc.), on ne modifie rien
        return pil_img

    # Réinsertion de la ROI nettoyée
    out_gray = gray.copy()
    out_gray[y0:y1, x0:x1] = cleaned_roi

    # Retourne dans le mode d'origine
    if orig_mode == "L":
        return Image.fromarray(out_gray, mode="L")
    else:
        # Recompose en RGB en gardant la luminance nettoyée
        # (évite d'altérer les teintes si entrée RGB)
        rgb = np.array(pil_img.convert("RGB"))
        # Remplace chaque canal par la luminance nettoyée dans la ROI pour cohérence visuelle
        rgb[y0:y1, x0:x1, 0] = out_gray[y0:y1, x0:x1]
        rgb[y0:y1, x0:x1, 1] = out_gray[y0:y1, x0:x1]
        rgb[y0:y1, x0:x1, 2] = out_gray[y0:y1, x0:x1]
        return Image.fromarray(rgb, mode="RGB")
