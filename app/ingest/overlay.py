# app/ingest/overlay.py
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# dossier de sortie des images taguées
TAGGED_DIR = Path(__file__).resolve().parent.parent / "dataset_bfa" / "tagged"
TAGGED_DIR.mkdir(parents=True, exist_ok=True)

# couleur par classe
COLOR_MAP = {
    "normal": (0, 255, 0),      # vert
    "benign": (255, 255, 0),    # jaune
    "malignant": (255, 0, 0),   # rouge
    "unknown": (128, 128, 128), # gris
}

def overlay_tag(png_path: str, base_name: str, label: str, birads: str, confidence: float) -> str:
    """
    Crée une image annotée avec la prédiction.
    - png_path: chemin de l'image source (PNG)
    - base_name: nom de base pour le fichier de sortie
    - label: 'normal' | 'benign' | 'malignant'
    - birads: 'BI-RADS 1-5'
    - confidence: score (0.0 - 1.0)
    Retourne le chemin du fichier tagué.
    """
    src = Path(png_path)
    if not src.exists():
        raise FileNotFoundError(f"Image non trouvée: {png_path}")

    # Lecture image
    img = np.array(Image.open(src).convert("RGB"))

    color = COLOR_MAP.get(label.lower(), (128, 128, 128))
    text = f"{label.upper()} ({birads})  {confidence*100:.1f}%"

    # préparation du texte (OpenCV)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size

    # dessiner un fond semi-transparent
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (20 + text_w, 20 + text_h + 5),
        color,
        -1,
    )
    # fusion (alpha)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # écrire le texte en blanc ou noir selon contraste
    cv2.putText(
        img,
        text,
        (15, 25 + text_h // 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # sauvegarde
    out_path = TAGGED_DIR / f"{base_name}_tagged.png"
    Image.fromarray(img).save(out_path)
    return str(out_path)
