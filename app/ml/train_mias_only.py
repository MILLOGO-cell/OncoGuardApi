from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import pywt
from joblib import dump
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    precision_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parents[2]
MIAS_DATA_DIR = BASE_DIR / "app" / "data" / "mias"
MODELS_DIR = BASE_DIR / "app" / "ml" / "models"
ARTIFACTS_DIR = BASE_DIR / "app" / "ml" / "artifacts_mias_only"


@dataclass
class EvalResult:
    accuracy: float
    recall_malignant: float
    precision_malignant: float
    f1_malignant: float
    specificity_malignant: float
    npv_malignant: float
    auc_ovr: float
    classes: List[str]
    confusion: np.ndarray
    report: dict


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)


def preprocess_mammogram(image: np.ndarray, target_size: int = 160) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = image.astype("float32") / 255.0
    image = (image * 255).astype("uint8")
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return image


def augment_image(image: np.ndarray) -> List[np.ndarray]:
    augmented = [image, cv2.flip(image, 1)]
    for angle in [90, 180, 270]:
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented.append(cv2.warpAffine(image, M, (image.shape[1], image.shape[0])))
    augmented.append(cv2.convertScaleAbs(image, alpha=1.15, beta=5))
    return augmented


def extract_features(image_gray: np.ndarray) -> np.ndarray:
    h = hog(
        image_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype("float32")

    radius, n_points = 3, 24
    lbp = local_binary_pattern(image_gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2),
    )
    hist = (hist / (hist.sum() + 1e-6)).astype("float32")

    cA, (cH, cV, cD) = pywt.dwt2(image_gray, "haar")
    w = np.hstack((cA.ravel(), cH.ravel(), cV.ravel(), cD.ravel())).astype("float32")

    return np.hstack([h, hist, w]).astype("float32")


def load_mias_data(csv_path: Path, data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    df["label"] = df["label"].str.strip().str.lower()
    
    df = df.drop_duplicates(subset=["filename"])
    df = df[df["label"].isin(["normal", "benign", "malignant"])]
    df = df.reset_index(drop=True)
    
    logging.info(f"📊 Distribution MIAS :")
    logging.info(f"\n{df['label'].value_counts()}\n")
    
    return df


def build_matrix(
    rows: pd.DataFrame,
    root: Path,
    use_augment: bool,
    resize: int = 160,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[str] = []

    for _, r in rows.iterrows():
        fp = root / str(r["filename"])

        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.warning(f"Impossible de charger : {fp}")
            continue

        img = preprocess_mammogram(img, target_size=resize)
        imgs = augment_image(img) if use_augment else [img]

        for im in imgs:
            feats.append(extract_features(im))
            labels.append(str(r["label"]))

    if not feats:
        raise RuntimeError("Aucune feature générée")

    X = np.vstack(feats).astype("float32")
    y = np.array(labels)
    logging.info(f"✅ Matrice : {X.shape[0]} exemples, {X.shape[1]} features")
    return X, y


def fit_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_class_weights: bool = True,
    seed: int = 42,
) -> XGBClassifier:
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    logging.info(f"📊 Distribution train : {class_distribution}")

    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=seed,
        tree_method="hist",
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )

    if use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        
        logging.info(f"⚖️ Poids des classes : {dict(zip(classes, weights))}")
        
        sw = np.array([weights[np.where(classes == yi)[0][0]] for yi in y_train])
        clf.fit(X_train, y_train, sample_weight=sw, verbose=False)
    else:
        clf.fit(X_train, y_train, verbose=False)

    return clf


def evaluate(
    clf: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: List[str],
) -> EvalResult:
    y_proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    lower = [c.lower() for c in classes]
    mal_idx = next(i for i, c in enumerate(lower) if c.startswith("mal"))

    acc = float(accuracy_score(y_test, y_pred))
    y_true_mal = y_test == mal_idx
    y_pred_mal = y_pred == mal_idx

    recall_mal = float(recall_score(y_true_mal, y_pred_mal, zero_division=0))
    precision_mal = float(precision_score(y_true_mal, y_pred_mal, zero_division=0))
    f1_mal = float(f1_score(y_true_mal, y_pred_mal, zero_division=0))

    tn = np.sum((~y_true_mal) & (~y_pred_mal))
    fp = np.sum((~y_true_mal) & y_pred_mal)
    fn = np.sum(y_true_mal & (~y_pred_mal))

    specificity = float(tn / (tn + fp + 1e-10))
    npv = float(tn / (tn + fn + 1e-10))

    y_test_bin = label_binarize(y_test, classes=np.arange(len(classes)))
    auc_ovr = float(roc_auc_score(y_test_bin, y_proba, multi_class="ovr"))

    conf = confusion_matrix(y_test, y_pred)
    rep = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    return EvalResult(
        acc,
        recall_mal,
        precision_mal,
        f1_mal,
        specificity,
        npv,
        auc_ovr,
        classes,
        conf,
        rep,
    )


def export_artifacts(out_dir: Path, result: EvalResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        result.confusion,
        index=result.classes,
        columns=result.classes,
    ).to_csv(out_dir / "confusion_matrix.csv", index=True)
    
    pd.DataFrame(result.report).to_csv(
        out_dir / "classification_report.csv",
        index=True,
    )

    metrics = {
        "accuracy": result.accuracy,
        "recall_malignant": result.recall_malignant,
        "precision_malignant": result.precision_malignant,
        "f1_malignant": result.f1_malignant,
        "specificity_malignant": result.specificity_malignant,
        "npv_malignant": result.npv_malignant,
        "auc_ovr": result.auc_ovr,
        "classes": result.classes,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logging.info(f"📁 Artefacts sauvegardés dans : {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Entraînement XGBoost sur MIAS uniquement (3 classes)")
    parser.add_argument("--csv", type=str, default=None, help="Chemin CSV MIAS")
    parser.add_argument("--data-dir", type=str, default=None, help="Dossier images MIAS")
    parser.add_argument("--cv", action="store_true", help="Validation croisée 5-fold")
    parser.add_argument("--resize", type=int, default=160, help="Taille redimensionnement")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    args = parser.parse_args()

    set_seeds(args.seed)

    csv_path = Path(args.csv) if args.csv else MIAS_DATA_DIR / "mias_annotations.csv"
    data_dir = Path(args.data_dir) if args.data_dir else MIAS_DATA_DIR

    logging.info(f"📂 CSV : {csv_path}")
    logging.info(f"📂 Images : {data_dir}")

    df = load_mias_data(csv_path, data_dir)

    if args.cv:
        logging.info("🔄 Validation croisée 5-fold...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        cv_results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["label"]), 1):
            logging.info(f"\n📊 Fold {fold}/5")

            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            X_train, y_train = build_matrix(train_df, data_dir, True, args.resize)
            X_test, y_test = build_matrix(test_df, data_dir, False, args.resize)

            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)

            scaler = StandardScaler()
            X_train_std = scaler.fit_transform(X_train)
            X_test_std = scaler.transform(X_test)

            clf = fit_classifier(X_train_std, y_train_enc, True, args.seed)
            result = evaluate(clf, X_test_std, y_test_enc, [str(c) for c in le.classes_])

            cv_results.append(result)
            logging.info(
                f"✅ Acc={result.accuracy:.3f} | Recall_mal={result.recall_malignant:.3f} | "
                f"F1_mal={result.f1_malignant:.3f} | AUC={result.auc_ovr:.3f}"
            )

        logging.info("\n" + "="*60)
        logging.info("📊 MOYENNES VALIDATION CROISÉE")
        logging.info("="*60)
        logging.info(f"Accuracy       : {np.mean([r.accuracy for r in cv_results]):.3f} ± {np.std([r.accuracy for r in cv_results]):.3f}")
        logging.info(f"Recall (malin) : {np.mean([r.recall_malignant for r in cv_results]):.3f} ± {np.std([r.recall_malignant for r in cv_results]):.3f}")
        logging.info(f"F1 (malin)     : {np.mean([r.f1_malignant for r in cv_results]):.3f} ± {np.std([r.f1_malignant for r in cv_results]):.3f}")
        logging.info(f"AUC            : {np.mean([r.auc_ovr for r in cv_results]):.3f} ± {np.std([r.auc_ovr for r in cv_results]):.3f}")

    logging.info("\n🚀 Entraînement final...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=df["label"],
    )

    X_train, y_train = build_matrix(train_df.reset_index(drop=True), data_dir, True, args.resize)
    X_test, y_test = build_matrix(test_df.reset_index(drop=True), data_dir, False, args.resize)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    clf = fit_classifier(X_train_std, y_train_enc, True, args.seed)
    result = evaluate(clf, X_test_std, y_test_enc, [str(c) for c in le.classes_])

    export_artifacts(ARTIFACTS_DIR, result)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(clf, MODELS_DIR / "xgb_birads_model.joblib")
    dump(scaler, MODELS_DIR / "scaler.joblib")
    dump(le, MODELS_DIR / "label_encoder.joblib")

    logging.info("\n" + "="*60)
    logging.info("🎯 RÉSULTATS FINAUX")
    logging.info("="*60)
    logging.info(f"✅ Accuracy        : {result.accuracy:.3f}")
    logging.info(f"✅ Recall (malin)  : {result.recall_malignant:.3f}")
    logging.info(f"✅ Precision (malin): {result.precision_malignant:.3f}")
    logging.info(f"✅ F1 (malin)      : {result.f1_malignant:.3f}")
    logging.info(f"✅ Specificity     : {result.specificity_malignant:.3f}")
    logging.info(f"✅ NPV             : {result.npv_malignant:.3f}")
    logging.info(f"✅ AUC-OVR         : {result.auc_ovr:.3f}")
    logging.info(f"\n📁 Modèle sauvegardé : {MODELS_DIR}")


if __name__ == "__main__":
    main()