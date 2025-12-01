import os
import cv2
import numpy as np
import pandas as pd
import pywt

from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from joblib import dump

from app.core.config import MIAS_DATA_DIR



def augment_image(image):
    # Trois versions : originale, retournée, tournée
    flipped = cv2.flip(image, 1)
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return [image, flipped, rotated]


def extract_features(image):
    # HOG
    hog_feat = hog(image, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)

    # LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    (lbp_hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, n_points + 3),
                                 range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # Wavelet
    coeffs2 = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs2
    wavelet_feat = np.hstack((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))

    # Combinaison
    features = np.hstack([hog_feat, lbp_hist, wavelet_feat])
    return features


def train_model():
    labels_csv_path = os.path.join(MIAS_DATA_DIR, "mias_annotations.csv")
    df = pd.read_csv(labels_csv_path)

    features = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(MIAS_DATA_DIR, row['filename'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[❌] Image non trouvée : {img_path}")
            continue

        # Redimensionne l’image
        image_resized = cv2.resize(image, (128, 128))

        # Augmentations : original + flipped + rotated
        for aug_img in augment_image(image_resized):
            feats = extract_features(aug_img)
            features.append(feats)
            labels.append(row['label'])

    print(f"[✅] Total exemples générés : {len(features)}")

    # Encodage des labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entraînement XGBoost
    clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    clf.fit(X_train, y_train)

    # Évaluation
    y_pred = clf.predict(X_test)
    print("\n[📊] Rapport de classification :")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"[🎯] Accuracy : {clf.score(X_test, y_test):.2f}")

    # Sauvegarde modèle
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "xgb_birads_model.joblib")
    dump(clf, model_file)

    # Sauvegarde scaler + encoder
    dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    dump(le, os.path.join(model_dir, "label_encoder.joblib"))

    print(f"[💾] Modèle sauvegardé : {model_file}")


if __name__ == "__main__":
    train_model()
