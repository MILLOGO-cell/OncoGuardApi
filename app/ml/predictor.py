# app/ml/predictor.py
import os
import cv2
import numpy as np
import pywt
from dataclasses import dataclass
from joblib import load
from skimage.feature import hog, local_binary_pattern

# Dossier des artefacts produits par train.py
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
XGB_PATH = os.path.join(MODEL_DIR, "xgb_birads_model.joblib")  # <- créé par train.py
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

_model = None
_scaler = None
_encoder = None

@dataclass
class PredictionResult:
    label: str         # "Normal" | "Benign" | "Malignant"
    confidence: float  # 0..1 si proba dispo, sinon 0.0
    birads: str        # "BI-RADS 1" | "BI-RADS 2" | "BI-RADS 5"

def _load_artifacts():
    global _model, _scaler, _encoder
    if _model is not None:
        return
    if not os.path.exists(XGB_PATH):
        raise FileNotFoundError(f"Modèle XGBoost introuvable : {XGB_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler introuvable : {SCALER_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"LabelEncoder introuvable : {ENCODER_PATH}")
    _model = load(XGB_PATH)
    _scaler = load(SCALER_PATH)
    _encoder = load(ENCODER_PATH)

def _preprocess(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image non trouvée : {path}")
    return cv2.resize(img, (128, 128))  # même taille que train.py

def _extract_features(image_gray: np.ndarray) -> np.ndarray:
    # HOG
    hog_feat = hog(
        image_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    # LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image_gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(),
                               bins=np.arange(0, n_points + 3),
                               range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float32")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    # Wavelets (haar)
    cA, (cH, cV, cD) = pywt.dwt2(image_gray, 'haar')
    wavelet_feat = np.hstack((cA.ravel(), cH.ravel(), cV.ravel(), cD.ravel())).astype("float32")
    return np.hstack([hog_feat, lbp_hist, wavelet_feat]).astype("float32")

def _map_to_birads(label: str) -> str:
    l = label.lower()
    if l.startswith("norm"): return "BI-RADS 1"
    if l.startswith("ben"):  return "BI-RADS 2"
    if l.startswith("mal"):  return "BI-RADS 5"
    return "BI-RADS 0"

def predict(image_path: str) -> PredictionResult:
    _load_artifacts()
    img = _preprocess(image_path)
    feats = _extract_features(img).reshape(1, -1)
    feats = _scaler.transform(feats)

    y_pred = _model.predict(feats)[0]
    label = _encoder.inverse_transform([y_pred])[0]

    try:
        proba = _model.predict_proba(feats)[0]
        conf = float(np.max(proba))
    except Exception:
        conf = 0.0

    return PredictionResult(label=label, confidence=conf, birads=_map_to_birads(label))
