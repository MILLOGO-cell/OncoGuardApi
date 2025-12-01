"""
Module unifié pour l'annotation d'images mammographiques.

Fournit une fonction unique pour créer des overlays visuels cohérents
à travers toute l'application (ingestion + inférence ML).
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Dossier de sortie par défaut
TAGGED_DIR = Path(__file__).resolve().parent.parent / "dataset_bfa" / "tagged"
TAGGED_DIR.mkdir(parents=True, exist_ok=True)

# Couleurs par label (BGR pour OpenCV)
LABEL_COLORS = {
    "normal": (0, 220, 0),       # Vert
    "benign": (0, 191, 255),     # Ambre/Orange
    "malignant": (38, 38, 220),  # Rouge
    "unknown": (128, 128, 128),  # Gris
}

ColorMode = Literal["adaptive", "amber"]


def overlay_tag(
    src_path: str,
    out_base: str,
    label: str,
    birads: str,
    confidence: float,
    color_mode: ColorMode = "adaptive",
    output_dir: Path | None = None,
) -> str:
    """
    Crée une image annotée avec bandeau d'information coloré.
    
    Fonction unifiée pour toute l'application :
    - Mode "adaptive" : couleur selon label (vert/orange/rouge)
    - Mode "amber" : couleur ambre fixe (pour ingestion)
    
    Args:
        src_path: Chemin vers l'image source (PNG)
        out_base: Nom de base du fichier de sortie (sans extension)
        label: Label de classification ("normal", "benign", "malignant")
        birads: Catégorie BI-RADS (ex: "BI-RADS 2")
        confidence: Score de confiance entre 0 et 1
        color_mode: "adaptive" (couleur selon label) ou "amber" (orange fixe)
        output_dir: Dossier de sortie personnalisé (défaut: TAGGED_DIR)
    
    Returns:
        Chemin absolu vers l'image annotée (suffixe __tag.png)
    
    Raises:
        FileNotFoundError: Si l'image source n'existe pas
        ValueError: Si color_mode est invalide
    
    Examples:
        >>> # Mode adaptatif pour ML inference
        >>> path = overlay_tag("img.png", "abc123", "malignant", "BI-RADS 5", 0.87)
        >>> # Résultat : abc123__tag.png avec texte ROUGE
        
        >>> # Mode ambre pour ingestion
        >>> path = overlay_tag("img.png", "xyz789", "benign", "BI-RADS 2", 0.91, 
        ...                    color_mode="amber")
        >>> # Résultat : xyz789__tag.png avec texte ORANGE
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Image non trouvée : {src_path}")
    
    # Lecture image avec conversion RGB si nécessaire
    try:
        pil_img = Image.open(src)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img = np.array(pil_img)
    except Exception as e:
        raise ValueError(f"Impossible de lire l'image {src_path} : {e}")
    
    h, w = img.shape[:2]
    
    # Construction du texte affiché
    label_upper = label.strip().upper()
    birads_upper = birads.strip().upper()
    conf_txt = f"{confidence * 100:.1f}%"
    text = f"{label_upper} • {birads_upper} • {conf_txt}"
    
    # Sélection de la couleur selon le mode
    if color_mode == "adaptive":
        color = LABEL_COLORS.get(label.lower(), LABEL_COLORS["unknown"])
    elif color_mode == "amber":
        color = LABEL_COLORS["benign"]  # Orange/Ambre
    else:
        raise ValueError(f"color_mode invalide : '{color_mode}' (attendu : 'adaptive' ou 'amber')")
    
    # Paramètres de police adaptés à la largeur de l'image
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = max(0.6, min(1.6, w / 1024.0))
    thickness = max(2, int(round(base_scale * 2)))
    
    # Ajustement automatique si texte trop large
    font_scale = _fit_font_scale(text, w, base_scale, thickness, font)
    
    # Mesure finale du texte
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Dimensions du bandeau noir
    pad_y = max(8, int(round(10 * font_scale)))
    band_h = text_h + baseline + 2 * pad_y
    band_h = max(band_h, int(0.06 * h))  # Au moins 6% de la hauteur
    
    # Dessiner bandeau noir plein en haut
    cv2.rectangle(img, (0, 0), (w, band_h), (0, 0, 0), thickness=-1)
    
    # Dessiner texte coloré aligné à gauche
    pad_x = max(16, int(round(20 * font_scale)))
    text_org = (pad_x, pad_y + text_h)
    cv2.putText(img, text, text_org, font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Sauvegarde avec suffixe unifié __tag.png
    out_dir = output_dir or TAGGED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_base}__tag.png"
    
    try:
        Image.fromarray(img).save(out_path)
        logger.info(f"Image annotée créée : {out_path}")
    except Exception as e:
        raise IOError(f"Impossible de sauvegarder l'image annotée : {e}")
    
    return str(out_path)


def _fit_font_scale(
    text: str, 
    img_w: int, 
    base_scale: float, 
    thickness: int, 
    font
) -> float:
    """
    Réduit fontScale si le texte dépasse la largeur disponible.
    
    Itère jusqu'à 12 fois en réduisant la taille de 10% à chaque fois,
    avec une taille minimale de 0.4 pour garantir la lisibilité.
    
    Args:
        text: Texte à mesurer
        img_w: Largeur de l'image en pixels
        base_scale: Échelle de police de base
        thickness: Épaisseur du trait
        font: Police OpenCV
    
    Returns:
        Échelle de police ajustée (>= 0.4)
    """
    margin = max(16, int(0.02 * img_w))
    max_w = img_w - 2 * margin
    scale = base_scale
    
    for _ in range(12):
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_w:
            break
        scale *= 0.9
    
    return max(scale, 0.4)