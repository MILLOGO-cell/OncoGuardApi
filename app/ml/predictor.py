# app/ml/predictor.py
"""
Module de prédiction pour la classification de mammographies - VERSION 3 CLASSES.

Ce module charge un modèle XGBoost pré-entraîné et permet de classifier
des images mammographiques en TROIS catégories :
- normal (sans anomalie) → BI-RADS 1
- benign (bénin) → BI-RADS 2  
- malignant (malin) → BI-RADS 5

IMPORTANT : Ce module supporte DEUX versions de modèles :
1. Modèle 2 classes (CBIS-DDSM seul) : benign, malignant
2. Modèle 3 classes (CBIS-DDSM + MIAS) : normal, benign, malignant

Le nombre de classes est détecté automatiquement depuis le label_encoder.

LOGIQUE DE CLASSIFICATION "NORMAL" :
- Si confiance < seuil → normal
- Si incertitude entre benign/malignant → normal
- Sinon → prédiction du modèle
"""
import os
import cv2
import numpy as np
import pywt
import logging
from dataclasses import dataclass
from joblib import load
from skimage.feature import hog, local_binary_pattern
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Chemins vers les artefacts du modèle
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
XGB_PATH = os.path.join(MODEL_DIR, "xgb_birads_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# Cache global des modèles chargés
_model = None
_scaler = None
_encoder = None


@dataclass
class PredictionResult:
    """
    Résultat d'une prédiction de classification mammographique.
    
    Attributes:
        label: Label prédit ("normal", "benign" ou "malignant")
        confidence: Score de confiance entre 0 et 1
        birads: Catégorie BI-RADS correspondante
        probabilities: Dictionnaire des probabilités par classe (optionnel)
        model_type: Type de modèle ("2-class" ou "3-class")
        reclassified_as_normal: True si reclassifié en normal par logique d'incertitude
    """
    label: str
    confidence: float
    birads: str
    probabilities: Optional[dict] = None
    model_type: Optional[str] = None
    reclassified_as_normal: bool = False


def _load_artifacts():
    """
    Charge les artefacts du modèle (modèle, scaler, encoder) en mémoire.
    
    Détecte automatiquement si le modèle est à 2 ou 3 classes.
    
    Raises:
        FileNotFoundError: Si un des fichiers du modèle est manquant
    """
    global _model, _scaler, _encoder
    
    if _model is not None:
        return
    
    if not os.path.exists(XGB_PATH):
        raise FileNotFoundError(
            f"Modèle XGBoost introuvable : {XGB_PATH}\n"
            f"Assurez-vous d'avoir exécuté l'entraînement avec train_eval_mias.py"
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler introuvable : {SCALER_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"LabelEncoder introuvable : {ENCODER_PATH}")
    
    _model = load(XGB_PATH)
    _scaler = load(SCALER_PATH)
    _encoder = load(ENCODER_PATH)
    
    logging.info(f"✅ Modèle chargé : {len(_encoder.classes_)} classes - {_encoder.classes_}")


def preprocess_mammogram(image: np.ndarray, target_size: int = 160) -> np.ndarray:
    """
    Applique le prétraitement standard sur une mammographie.
    
    Pipeline identique à celui utilisé durant l'entraînement :
    1. Égalisation d'histogramme adaptative (CLAHE)
    2. Normalisation
    3. Redimensionnement à la taille cible
    
    Args:
        image: Image en niveaux de gris (numpy array)
        target_size: Dimension du carré de sortie (défaut: 160)
    
    Returns:
        Image prétraitée de taille (target_size, target_size)
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    
    image = image.astype('float32') / 255.0
    image = (image * 255).astype('uint8')
    
    image = cv2.resize(
        image, 
        (target_size, target_size), 
        interpolation=cv2.INTER_CUBIC
    )
    
    return image


def extract_features(image_gray: np.ndarray) -> np.ndarray:
    """
    Extrait les descripteurs de texture multi-échelles d'une image.
    
    Descripteurs combinés :
    - HOG (Histogram of Oriented Gradients)
    - LBP (Local Binary Patterns)
    - Ondelettes de Haar
    
    Args:
        image_gray: Image en niveaux de gris prétraitée
    
    Returns:
        Vecteur de caractéristiques 1D (environ 28542 features)
    """
    hog_feat = hog(
        image_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image_gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )
    lbp_hist = lbp_hist.astype("float32")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    cA, (cH, cV, cD) = pywt.dwt2(image_gray, 'haar')
    wavelet_feat = np.hstack((
        cA.ravel(), 
        cH.ravel(), 
        cV.ravel(), 
        cD.ravel()
    )).astype("float32")
    
    return np.hstack([hog_feat, lbp_hist, wavelet_feat]).astype("float32")


def map_to_birads(label: str) -> str:
    """
    Convertit un label de classification en catégorie BI-RADS.
    
    Mapping pour 3 classes (CBIS-DDSM + MIAS) :
    - normal → BI-RADS 1 (Normal)
    - benign → BI-RADS 2 (Anomalie bénigne)
    - malignant → BI-RADS 5 (Évocateur de cancer)
    
    Args:
        label: Label prédit par le modèle
    
    Returns:
        Catégorie BI-RADS correspondante
    """
    label_lower = label.lower().strip()
    
    if label_lower.startswith("norm"):
        return "1 - Normal"
    elif label_lower.startswith("ben"):
        return "2 - Bénin"
    elif label_lower.startswith("mal"):
        return "5 - Évocateur de cancer"
    else:
        return "0 - Examen incomplet"


def predict(
    image_path: str, 
    confidence_threshold: float = 0.60,
    uncertainty_threshold: float = 0.35
) -> PredictionResult:
    """
    Prédit la catégorie d'une image mammographique avec détection de "normal".
    
    Pipeline complet :
    1. Chargement de l'image en niveaux de gris
    2. Prétraitement (CLAHE + normalisation + resize)
    3. Extraction de caractéristiques (HOG + LBP + Wavelets)
    4. Normalisation des features
    5. Prédiction avec XGBoost
    6. Calcul des probabilités
    7. Logique de reclassification en "normal" si incertitude
    8. Mapping vers BI-RADS
    
    Logique de classification "NORMAL" :
    - Si confiance < confidence_threshold → NORMAL
    - Si incertitude entre benign/malignant (écart < uncertainty_threshold) → NORMAL
    - Sinon → prédiction du modèle
    
    Args:
        image_path: Chemin vers l'image à classifier
        confidence_threshold: Seuil de confiance minimum (défaut: 0.60)
        uncertainty_threshold: Seuil d'écart min entre benign/malignant (défaut: 0.35)
    
    Returns:
        PredictionResult avec label, confidence, birads et probabilités
    
    Raises:
        FileNotFoundError: Si l'image ou les modèles sont introuvables
        ValueError: Si l'image ne peut pas être lue
    
    Example:
        >>> result = predict("mammogram.png")
        >>> print(f"{result.label} ({result.birads}) - {result.confidence:.2%}")
        normal (BI-RADS 1) - 45.2%
        
        >>> # Ajuster les seuils
        >>> result = predict("mammogram.png", confidence_threshold=0.70, uncertainty_threshold=0.40)
    """
    _load_artifacts()
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")
    
    img_preprocessed = preprocess_mammogram(img, target_size=160)
    features = extract_features(img_preprocessed).reshape(1, -1)
    features_scaled = _scaler.transform(features)
    
    probas = _model.predict_proba(features_scaled)[0]
    classes = _encoder.classes_
    
    proba_dict = {cls: float(prob) for cls, prob in zip(classes, probas)}
    
    y_pred = np.argmax(probas)
    original_label = _encoder.inverse_transform([y_pred])[0]
    label = original_label
    confidence = float(np.max(probas))
    reclassified = False
    
    n_classes = len(classes)
    
    # LOGIQUE DE DÉCISION POUR "NORMAL"
    if n_classes == 2:
        # Modèle 2 classes (benign, malignant uniquement)
        logging.info(f"Modèle 2 classes détecté - Confiance: {confidence:.2%}")
        
        if confidence < confidence_threshold:
            logging.warning(f"⚠️ Confiance faible ({confidence:.2%}) → Reclassé NORMAL")
            label = "normal"
            birads = "1 - Normal"
            reclassified = True
        elif abs(probas[0] - probas[1]) < uncertainty_threshold:
            logging.warning(
                f"⚠️ Incertitude élevée (écart: {abs(probas[0] - probas[1]):.2%}) → Reclassé NORMAL"
            )
            label = "normal"
            birads = "1 - Normal"
            reclassified = True
        else:
            birads = map_to_birads(label)
    
    elif n_classes == 3:
        # Modèle 3 classes (normal, benign, malignant)
        lower = [c.lower() for c in classes]
        
        try:
            benign_idx = next(i for i, c in enumerate(lower) if c.startswith("ben"))
            malignant_idx = next(i for i, c in enumerate(lower) if c.startswith("mal"))
        except StopIteration:
            birads = map_to_birads(label)
        else:
            benign_prob = probas[benign_idx]
            malignant_prob = probas[malignant_idx]
            
            logging.info(
                f"Modèle 3 classes - Confiance: {confidence:.2%} | "
                f"Benign: {benign_prob:.2%} | Malignant: {malignant_prob:.2%}"
            )
            
            if confidence < confidence_threshold:
                logging.warning(f"⚠️ Confiance faible ({confidence:.2%}) → Reclassé NORMAL")
                label = "normal"
                birads = "1 - Normal"
                reclassified = True
            elif (
                abs(benign_prob - malignant_prob) < uncertainty_threshold 
                and max(benign_prob, malignant_prob) < 0.7
            ):
                logging.warning(
                    f"⚠️ Incertitude benign/malignant (écart: {abs(benign_prob - malignant_prob):.2%}) "
                    f"→ Reclassé NORMAL"
                )
                label = "normal"
                birads = "1 - Normal"
                reclassified = True
            else:
                birads = map_to_birads(label)
    else:
        birads = map_to_birads(label)
    
    model_type = f"{n_classes}-class"
    
    if reclassified:
        logging.info(f"✅ Résultat final: {label.upper()} (reclassifié depuis {original_label})")
    else:
        logging.info(f"✅ Résultat final: {label.upper()}")
    
    return PredictionResult(
        label=label,
        confidence=confidence,
        birads=birads,
        probabilities=proba_dict,
        model_type=model_type,
        reclassified_as_normal=reclassified
    )


def predict_batch(
    image_paths: list[str],
    confidence_threshold: float = 0.60,
    uncertainty_threshold: float = 0.35
) -> list[PredictionResult]:
    """
    Prédit les catégories pour un lot d'images.
    
    Plus efficace que d'appeler predict() en boucle car :
    - Les modèles sont chargés une seule fois
    - Possibilité d'optimisations futures (batch processing)
    
    Args:
        image_paths: Liste de chemins vers les images
        confidence_threshold: Seuil de confiance minimum
        uncertainty_threshold: Seuil d'incertitude maximum
    
    Returns:
        Liste de PredictionResult dans le même ordre
    """
    _load_artifacts()
    
    results = []
    for path in image_paths:
        try:
            result = predict(path, confidence_threshold, uncertainty_threshold)
            results.append(result)
        except Exception as e:
            logging.error(f"❌ Erreur sur {path}: {e}")
            results.append(PredictionResult(
                label="error",
                confidence=0.0,
                birads="0 - Examen incomplet",
                probabilities=None,
                model_type="unknown",
                reclassified_as_normal=False
            ))
    
    return results


def get_model_info() -> dict:
    """
    Retourne des informations sur le modèle chargé.
    
    Détecte automatiquement le nombre de classes et fournit
    des informations appropriées.
    
    Returns:
        Dictionnaire avec les métadonnées du modèle
    
    Example:
        >>> info = get_model_info()
        >>> print(f"Classes: {info['classes']}")
        Classes: ['benign', 'malignant', 'normal']
    """
    _load_artifacts()
    
    classes = list(_encoder.classes_)
    n_classes = len(classes)
    
    if n_classes == 2:
        note = (
            "Modèle 2 classes (benign/malignant). "
            "Les cas incertains sont automatiquement classés comme 'normal'."
        )
        dataset = "CBIS-DDSM"
    elif n_classes == 3:
        note = (
            "Modèle 3 classes (normal/benign/malignant). "
            "Détection automatique de 'normal' en cas d'incertitude."
        )
        dataset = "CBIS-DDSM + MIAS"
    else:
        note = f"Modèle {n_classes} classes"
        dataset = "Inconnu"
    
    return {
        "classes": classes,
        "n_classes": n_classes,
        "feature_dim": _scaler.n_features_in_ if hasattr(_scaler, 'n_features_in_') else "unknown",
        "model_type": "XGBoost",
        "input_size": "160×160",
        "dataset": dataset,
        "note": note
    }


if __name__ == "__main__":
    """Point d'entrée pour tests rapides."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predictor.py <image_path> [confidence_threshold] [uncertainty_threshold]")
        print("\nInfo sur le modèle:")
        info = get_model_info()
        for k, v in info.items():
            print(f"  {k}: {v}")
        print("\nExemples:")
        print("  python predictor.py mammogram.png")
        print("  python predictor.py mammogram.png 0.70 0.40  # Seuils personnalisés")
        sys.exit(1)
    
    img_path = sys.argv[1]
    conf_thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.60
    uncert_thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    
    result = predict(img_path, conf_thresh, uncert_thresh)
    
    print(f"\n🔍 Résultat de la prédiction")
    print(f"=" * 60)
    print(f"Label:       {result.label.upper()}")
    print(f"BI-RADS:     {result.birads}")
    print(f"Confiance:   {result.confidence:.2%}")
    print(f"Type modèle: {result.model_type}")
    
    if result.reclassified_as_normal:
        print(f"⚠️  Reclassifié en NORMAL (incertitude détectée)")
    
    if result.probabilities:
        print(f"\nProbabilités par classe:")
        for cls in sorted(result.probabilities.keys()):
            prob = result.probabilities[cls]
            bar = "█" * int(prob * 40)
            print(f"  {cls:10} : {prob:.2%} {bar}")
    
    print(f"\n💡 Ajuster les seuils:")
    print(f"   python predictor.py {img_path} 0.70 0.40  # Plus conservateur")
    print(f"   python predictor.py {img_path} 0.50 0.25  # Moins conservateur")