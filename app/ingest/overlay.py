# app/ingest/overlay.py
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Dossier de sortie des images taguées
TAGGED_DIR = Path(__file__).resolve().parent.parent / "dataset_bfa" / "tagged"
TAGGED_DIR.mkdir(parents=True, exist_ok=True)

# Couleur par classe (peut servir plus tard si besoin),
# mais on force le texte en "ambre" comme demandé.
COLOR_MAP = {
    "normal": (0, 255, 0),       # BGR vert
    "benign": (255, 255, 0),     # BGR jaune
    "malignant": (0, 0, 255),    # BGR rouge
    "unknown": (128, 128, 128),  # BGR gris
}

# Couleurs (BGR) pour le bandeau et le texte
BAND_COLOR = (0, 0, 0)           # noir plein
TEXT_COLOR_AMBER = (0, 191, 255) # "ambre" en BGR (RGB 255,191,0)

def _fit_font_scale(text: str, img_w: int, base_scale: float, thickness: int, font) -> float:
    """
    Réduit le fontScale si le texte dépasse la largeur disponible (avec marges).
    """
    margin = max(16, int(0.02 * img_w))
    max_w = img_w - 2 * margin
    scale = base_scale
    for _ in range(12):  # 12 itérations max
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_w:
            break
        scale *= 0.9
    return max(scale, 0.4)

def overlay_tag(png_path: str, base_name: str, label: str, birads: str, confidence: float) -> str:
    """
    Crée une image annotée avec un bandeau noir en haut et un texte aligné à gauche (ambre).
    - png_path: chemin de l'image source (PNG)
    - base_name: nom de base pour le fichier de sortie
    - label: 'normal' | 'benign' | 'malignant' | 'unknown'
    - birads: ex. 'BI-RADS 2'
    - confidence: score (0.0 - 1.0)
    Retourne le chemin du fichier tagué (str).
    """
    src = Path(png_path)
    if not src.exists():
        raise FileNotFoundError(f"Image non trouvée: {png_path}")

    # Lecture image (assure 3 canaux)
    pil_img = Image.open(src)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    img = np.array(pil_img)

    h, w = img.shape[:2]

    # --- Texte à afficher (forme compacte et claire) ---
    # Exemple: "BENIGN — BI-RADS 2 — 45.8%"
    label_up = (label or "unknown").strip().upper()
    birads_up = (birads or "").strip().upper()
    conf_txt = f"{confidence * 100:.1f}%"
    text = f"{label_up} — {birads_up} — {conf_txt}" if birads_up else f"{label_up} — {conf_txt}"

    # --- Paramètres de police ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Base scale proportionnelle à la largeur
    base_scale = max(0.6, min(1.6, w / 1024.0))
    thickness = max(2, int(round(base_scale * 2)))

    # Ajuste la taille pour que ça tienne avec des marges
    font_scale = _fit_font_scale(text, w, base_scale, thickness, font)

    # Mesure finale
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # --- Bandeau noir haut, pleine largeur ---
    # hauteur du bandeau = texte + marge haut/bas
    pad_y = max(8, int(round(10 * font_scale)))
    band_h = text_h + baseline + 2 * pad_y
    band_h = max(band_h, int(0.06 * h))  # au moins ~6% de la hauteur

    # Dessin du bandeau noir
    cv2.rectangle(img, (0, 0), (w, band_h), BAND_COLOR, thickness=-1)

    # --- Positionnement du texte (aligné à gauche) ---
    pad_x = max(16, int(round(20 * font_scale)))
    text_org = (pad_x, pad_y + text_h)  # baseline

    # --- Texte en "ambre" ---
    cv2.putText(
        img,
        text,
        text_org,
        font,
        font_scale,
        TEXT_COLOR_AMBER,
        thickness,
        cv2.LINE_AA,
    )

    # Sauvegarde
    out_path = TAGGED_DIR / f"{base_name}_tagged.png"
    Image.fromarray(img).save(out_path)
    return str(out_path)
