import os
import cv2
import numpy as np
from skimage.feature import hog
from joblib import load
from app.api.v1.models.enums import BiradsCategory
from app.core.config import MIAS_DATA_DIR

# Chemin du modèle SVM entraîné (supposé sauvegardé ici)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "svm_birads_model.joblib")

def load_svm_model():
    """
    Charge le modèle SVM pré-entraîné.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable à {MODEL_PATH}")
    return load(MODEL_PATH)

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Charge une image PGM/PNG et la prétraite (gris, redimensionnement).
    
    Args:
        image_path: chemin vers l'image
    Returns:
        Image numpy 2D prétraitée.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image non trouvée : {image_path}")
    image = cv2.resize(image, (256, 256))
    # Optionnel : filtrage, normalisation...
    return image

def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Extrait des caractéristiques HOG de l'image.
    
    Args:
        image: image numpy 2D en niveaux de gris
    Returns:
        Vecteur de caractéristiques HOG.
    """
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )
    return features

def predict_birads(image_path: str) -> BiradsCategory:
    """
    Prédit la catégorie BI-RADS pour une image donnée.
    
    Args:
        image_path: chemin vers l'image mammographique
    
    Returns:
        Une valeur BiradsCategory enum.
    """
    model = load_svm_model()
    image = preprocess_image(image_path)
    features = extract_hog_features(image)
    pred_class_int = model.predict([features])[0]

    # Mapping des classes prédites vers enums BI-RADS (adapter selon entraînement)
    mapping = {
        0: BiradsCategory.BI_RADS_0,
        1: BiradsCategory.BI_RADS_1,
        2: BiradsCategory.BI_RADS_2,
        3: BiradsCategory.BI_RADS_3,
        4: BiradsCategory.BI_RADS_4A,
        5: BiradsCategory.BI_RADS_4B,
        6: BiradsCategory.BI_RADS_4C,
        7: BiradsCategory.BI_RADS_5,
        8: BiradsCategory.BI_RADS_6,
    }
    return mapping.get(pred_class_int, BiradsCategory.BI_RADS_0)
