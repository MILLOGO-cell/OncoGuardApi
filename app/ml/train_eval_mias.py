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

from app.core.config import (
    BASE_DIR,
    MIAS_DATA_DIR,
    MODELS_DIR,
    ARTIFACTS_DIR,
    UNIFIED_CSV_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class EvalResult:
    """Structure pour stocker les métriques d'évaluation du modèle."""

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
    """
    Fixe les graines aléatoires pour garantir la reproductibilité.
    
    Args:
        seed: Graine aléatoire à utiliser
    """
    np.random.seed(seed)
    random.seed(seed)


def preprocess_mammogram(image: np.ndarray, target_size: int = 160) -> np.ndarray:
    """
    Applique le prétraitement standard sur une mammographie.
    
    Étapes :
    1. Égalisation d'histogramme adaptative (CLAHE)
    2. Normalisation des valeurs
    3. Redimensionnement
    
    Args:
        image: Image en niveaux de gris
        target_size: Taille de sortie (carré)
    
    Returns:
        Image prétraitée et redimensionnée
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = image.astype("float32") / 255.0
    image = (image * 255).astype("uint8")
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return image


def augment_image(image: np.ndarray) -> List[np.ndarray]:
    """
    Génère des variantes augmentées d'une image pour l'entraînement.
    
    Augmentations appliquées :
    - Image originale
    - Flip horizontal
    - Rotations (90°, 180°, 270°)
    - Ajustement de luminosité/contraste
    
    Args:
        image: Image source
    
    Returns:
        Liste de 6 variantes de l'image
    """
    augmented = [image, cv2.flip(image, 1)]
    for angle in [90, 180, 270]:
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented.append(cv2.warpAffine(image, M, (image.shape[1], image.shape[0])))
    augmented.append(cv2.convertScaleAbs(image, alpha=1.15, beta=5))
    return augmented


def extract_features(image_gray: np.ndarray) -> np.ndarray:
    """
    Extrait les descripteurs de texture multi-échelles d'une image.
    
    Descripteurs extraits :
    - HOG (Histogram of Oriented Gradients) : texture orientée
    - LBP (Local Binary Patterns) : texture locale
    - Ondelettes de Haar : décomposition fréquentielle
    
    Args:
        image_gray: Image en niveaux de gris
    
    Returns:
        Vecteur de caractéristiques concaténé
    """
    # Extraction HOG
    h = hog(
        image_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype("float32")

    # Extraction LBP
    radius, n_points = 3, 24
    lbp = local_binary_pattern(image_gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2),
    )
    hist = (hist / (hist.sum() + 1e-6)).astype("float32")

    # Extraction ondelettes
    cA, (cH, cV, cD) = pywt.dwt2(image_gray, "haar")
    w = np.hstack((cA.ravel(), cH.ravel(), cV.ravel(), cD.ravel())).astype("float32")

    return np.hstack([h, hist, w]).astype("float32")


def read_index(csv_path: Path) -> pd.DataFrame:
    """
    Charge un fichier CSV d'annotations avec gestion flexible des colonnes.
    
    Gère automatiquement différents formats de CSV en cherchant les colonnes
    appropriées pour les chemins de fichiers et les labels.
    
    Args:
        csv_path: Chemin vers le fichier CSV
    
    Returns:
        DataFrame avec colonnes 'filepath'/'filename' et 'label'
    
    Raises:
        ValueError: Si la colonne 'label' est absente
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    if "filepath" in df.columns and "label" in df.columns:
        return df[["filepath", "label"]].dropna().reset_index(drop=True)

    if "filename" not in df.columns and "file" in df.columns:
        df["filename"] = df["file"]

    if "label" not in df.columns:
        if "severity" in df.columns:
            df["label"] = df["severity"]
        elif "class" in df.columns:
            df["label"] = df["class"]
        else:
            raise ValueError("Colonne 'label' absente dans le CSV fourni.")

    return df[["filename", "label"]].dropna().reset_index(drop=True)


def build_matrix(
    rows: pd.DataFrame,
    root: Path,
    use_augment: bool,
    resize: int = 160,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit la matrice de caractéristiques X et le vecteur de labels y.
    
    Pour chaque image :
    1. Charge l'image depuis le disque
    2. Applique le prétraitement
    3. Applique l'augmentation (optionnel, pour entraînement seulement)
    4. Extrait les descripteurs
    
    Args:
        rows: DataFrame avec les annotations
        root: Dossier racine pour les images
        use_augment: Activer l'augmentation de données (x6)
        resize: Taille de redimensionnement
    
    Returns:
        Tuple (X, y) avec X = matrice de features, y = labels
    
    Raises:
        RuntimeError: Si aucune feature n'a pu être générée
    """
    feats: List[np.ndarray] = []
    labels: List[str] = []

    use_filepath = "filepath" in rows.columns
    label_col = "label"

    for _, r in rows.iterrows():
        if use_filepath:
            fp = Path(str(r["filepath"]))
            if not fp.is_absolute():
                fp = root / fp
        else:
            fp = root / str(r["filename"])

        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = preprocess_mammogram(img, target_size=resize)
        imgs = augment_image(img) if use_augment else [img]

        for im in imgs:
            feats.append(extract_features(im))
            labels.append(str(r[label_col]))

    if not feats:
        raise RuntimeError("Aucune feature générée à partir des annotations fournies.")

    X = np.vstack(feats).astype("float32")
    y = np.array(labels)
    logging.info("Matrice: %d exemples, %d features", X.shape[0], X.shape[1])
    return X, y


def fit_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_class_weights: bool = True,
    seed: int = 42,
) -> XGBClassifier:
    """
    Entraîne un classifieur XGBoost avec pondération de classes optionnelle.
    
    Configuration XGBoost optimisée pour la classification médicale :
    - 300 arbres pour une bonne capacité de généralisation
    - Learning rate faible (0.05) pour éviter le surapprentissage
    - Subsampling pour améliorer la robustesse
    
    Args:
        X_train: Matrice de features d'entraînement
        y_train: Labels d'entraînement
        use_class_weights: Pondérer les classes pour gérer le déséquilibre
        seed: Graine aléatoire
    
    Returns:
        Classifieur XGBoost entraîné
    """
    try:
        clf = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=seed,
            use_label_encoder=False,
            tree_method="hist",
        )
    except TypeError:
        # Fallback pour versions XGBoost plus anciennes
        clf = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=seed,
            tree_method="hist",
        )

    if use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        sw = np.array([weights[np.where(classes == yi)[0][0]] for yi in y_train])
        clf.fit(X_train, y_train, sample_weight=sw, verbose=False)
    else:
        clf.fit(X_train, y_train, verbose=False)

    return clf


def predict_with_threshold(y_proba: np.ndarray, mal_idx: int, tau: float) -> np.ndarray:
    """
    Applique un seuil de décision personnalisé favorisant la détection des malignités.
    
    En médecine, on préfère souvent minimiser les faux négatifs (malignités non détectées)
    au prix d'augmenter les faux positifs.
    
    Args:
        y_proba: Probabilités prédites
        mal_idx: Index de la classe malignant
        tau: Seuil de probabilité (ex: 0.3 au lieu de 0.5)
    
    Returns:
        Prédictions ajustées
    """
    y_hat = np.argmax(y_proba, axis=1)
    mask = y_proba[:, mal_idx] >= float(tau)
    y_hat[mask] = mal_idx
    return y_hat


def evaluate(
    clf: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: List[str],
    tau: Optional[float] = None,
) -> Tuple[EvalResult, Optional[EvalResult]]:
    """
    Calcule les métriques d'évaluation complètes du modèle.
    
    Métriques calculées :
    - Accuracy, Recall, Precision, F1 (focus sur classe malignant)
    - Specificity, NPV (Negative Predictive Value)
    - AUC-ROC (adapté pour binaire ou multi-classe)
    - Matrices de confusion
    
    CORRECTION: Gère correctement les problèmes binaires (2 classes) vs multi-classes
    
    Args:
        clf: Classifieur entraîné
        X_test: Features de test
        y_test: Labels de test
        classes: Liste des noms de classes
        tau: Seuil optionnel pro-malin
    
    Returns:
        Tuple (résultats standard, résultats avec seuil optionnel)
    """
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

    # CORRECTION: Calcul AUC adapté pour binaire vs multi-classe
    if len(classes) == 2:
        # Problème binaire : utiliser directement les probabilités de la classe positive
        auc_ovr = float(roc_auc_score(y_test, y_proba[:, mal_idx]))
    else:
        # Problème multi-classe : utiliser label_binarize et multi_class="ovr"
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

    std_res = EvalResult(
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

    thr_res: Optional[EvalResult] = None
    if tau is not None:
        y_pred_tau = predict_with_threshold(y_proba, mal_idx, tau)
        acc_t = float(accuracy_score(y_test, y_pred_tau))
        y_pred_tau_mal = y_pred_tau == mal_idx

        rec_t = float(recall_score(y_true_mal, y_pred_tau_mal, zero_division=0))
        prec_t = float(precision_score(y_true_mal, y_pred_tau_mal, zero_division=0))
        f1_t = float(f1_score(y_true_mal, y_pred_tau_mal, zero_division=0))

        tn_t = np.sum((~y_true_mal) & (~y_pred_tau_mal))
        fp_t = np.sum((~y_true_mal) & y_pred_tau_mal)
        fn_t = np.sum(y_true_mal & (~y_pred_tau_mal))

        spec_t = float(tn_t / (tn_t + fp_t + 1e-10))
        npv_t = float(tn_t / (tn_t + fn_t + 1e-10))

        conf_t = confusion_matrix(y_test, y_pred_tau)
        rep_t = classification_report(
            y_test,
            y_pred_tau,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )

        thr_res = EvalResult(
            acc_t,
            rec_t,
            prec_t,
            f1_t,
            spec_t,
            npv_t,
            auc_ovr,  # Même AUC que standard
            classes,
            conf_t,
            rep_t,
        )

    return std_res, thr_res


def export_artifacts(out_dir: Path, std: EvalResult, thr: Optional[EvalResult]) -> None:
    """
    Exporte tous les artefacts d'évaluation sur disque.
    
    Fichiers créés :
    - confusion_mias.csv : Matrice de confusion
    - classification_report_mias.csv : Rapport détaillé sklearn
    - metrics_mias.json : Toutes les métriques en JSON
    - mias_results_table.tex : Tableau LaTeX pour publication
    
    Args:
        out_dir: Dossier de sortie
        std: Résultats avec seuil standard
        thr: Résultats avec seuil personnalisé (optionnel)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        std.confusion,
        index=std.classes,
        columns=std.classes,
    ).to_csv(out_dir / "confusion_mias.csv", index=True)
    
    pd.DataFrame(std.report).to_csv(
        out_dir / "classification_report_mias.csv",
        index=True,
    )

    metrics = {
        "standard": {
            "accuracy": std.accuracy,
            "recall_malignant": std.recall_malignant,
            "precision_malignant": std.precision_malignant,
            "f1_malignant": std.f1_malignant,
            "specificity_malignant": std.specificity_malignant,
            "npv_malignant": std.npv_malignant,
            "auc_ovr": std.auc_ovr,
            "classes": std.classes,
        }
    }

    if thr is not None:
        metrics["threshold"] = {
            "accuracy": thr.accuracy,
            "recall_malignant": thr.recall_malignant,
            "precision_malignant": thr.precision_malignant,
            "f1_malignant": thr.f1_malignant,
            "specificity_malignant": thr.specificity_malignant,
            "npv_malignant": thr.npv_malignant,
            "auc_ovr": thr.auc_ovr,
            "classes": thr.classes,
        }
    else:
        metrics["threshold"] = None

    with open(out_dir / "metrics_mias.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Tableau LaTeX pour publication
    tex = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Résultats sur le jeu de test unifié.}\n"
        "\\label{tab:mias-results}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "\\textbf{Modèle} & \\textbf{Acc.} & \\textbf{Rappel (malin)} & "
        "\\textbf{F1 (malin)} & \\textbf{AUC} \\\\\n\\midrule\n"
        f"XGBoost (traits multi-échelles) & {std.accuracy:.2f} & "
        f"{std.recall_malignant:.2f} & {std.f1_malignant:.2f} & {std.auc_ovr:.2f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out_dir / "mias_results_table.tex").write_text(tex, encoding="utf-8")

    if thr is not None:
        tex_thr = (
            "\\begin{table}[H]\n\\centering\n"
            "\\caption{Résultats sur le jeu de test avec seuil pro-malin.}\n"
            "\\label{tab:mias-results-threshold}\n"
            "\\begin{tabular}{lccc}\n\\toprule\n"
            "\\textbf{Modèle} & \\textbf{Acc.} & "
            "\\textbf{Rappel (malin)} & \\textbf{F1 (malin)} \\\\\n\\midrule\n"
            f"XGBoost (seuil) & {thr.accuracy:.2f} & "
            f"{thr.recall_malignant:.2f} & {thr.f1_malignant:.2f} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )
        (out_dir / "mias_results_table_threshold.tex").write_text(
            tex_thr,
            encoding="utf-8",
        )


def train_and_evaluate(
    csv_path: Path,
    root: Path,
    out_dir: Path,
    use_class_weights: bool,
    tau: Optional[float],
    use_cv: bool,
    n_folds: int,
    resize: int,
    seed: int = 42,
) -> None:
    """
    Pipeline complet d'entraînement et d'évaluation.
    
    Workflow :
    1. Chargement et affichage des statistiques des données
    2. Validation croisée stratifiée (optionnel)
    3. Entraînement final sur 80% des données
    4. Évaluation sur 20% de test
    5. Sauvegarde des modèles et artefacts
    
    Args:
        csv_path: Chemin vers le CSV d'annotations
        root: Dossier racine des images
        out_dir: Dossier de sortie pour les artefacts
        use_class_weights: Pondération automatique des classes
        tau: Seuil pro-malin optionnel
        use_cv: Activer la validation croisée
        n_folds: Nombre de folds pour la CV
        resize: Taille de redimensionnement des images
        seed: Graine aléatoire pour reproductibilité
    """
    set_seeds(seed)
    idx = read_index(csv_path)

    logging.info("Total images: %d", len(idx))
    logging.info("Distribution:\n%s", idx["label"].value_counts())
    logging.info("Résolution: %d×%d", resize, resize)

    if use_cv:
        logging.info("Validation croisée %d-fold...", n_folds)
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        cv_results: List[EvalResult] = []

        for fold, (train_idx, test_idx) in enumerate(
            skf.split(idx, idx["label"]),
            1,
        ):
            logging.info("Fold %d/%d", fold, n_folds)

            train_df = idx.iloc[train_idx].reset_index(drop=True)
            test_df = idx.iloc[test_idx].reset_index(drop=True)

            X_train, y_train = build_matrix(train_df, root, True, resize)
            X_test, y_test = build_matrix(test_df, root, False, resize)

            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)

            scaler = StandardScaler()
            X_train_std = scaler.fit_transform(X_train)
            X_test_std = scaler.transform(X_test)
            del X_train, X_test

            clf = fit_classifier(X_train_std, y_train_enc, use_class_weights, seed)
            std_res, _ = evaluate(
                clf,
                X_test_std,
                y_test_enc,
                [str(c) for c in le.classes_],
                None,
            )

            cv_results.append(std_res)
            logging.info(
                "Acc=%.3f | Recall_mal=%.3f | F1_mal=%.3f | AUC=%.3f",
                std_res.accuracy,
                std_res.recall_malignant,
                std_res.f1_malignant,
                std_res.auc_ovr,
            )

            del clf, X_train_std, X_test_std, scaler

        logging.info("=" * 60)
        logging.info("MOYENNES VALIDATION CROISÉE")
        logging.info("=" * 60)
        logging.info(
            "Accuracy:       %.3f ± %.3f",
            np.mean([r.accuracy for r in cv_results]),
            np.std([r.accuracy for r in cv_results]),
        )
        logging.info(
            "Recall (malin): %.3f ± %.3f",
            np.mean([r.recall_malignant for r in cv_results]),
            np.std([r.recall_malignant for r in cv_results]),
        )
        logging.info(
            "F1 (malin):     %.3f ± %.3f",
            np.mean([r.f1_malignant for r in cv_results]),
            np.std([r.f1_malignant for r in cv_results]),
        )
        logging.info(
            "AUC:            %.3f ± %.3f",
            np.mean([r.auc_ovr for r in cv_results]),
            np.std([r.auc_ovr for r in cv_results]),
        )

    logging.info("Entraînement final...")
    train_idx_df, test_idx_df = train_test_split(
        idx,
        test_size=0.2,
        random_state=seed,
        stratify=idx["label"],
    )

    X_train, y_train = build_matrix(
        train_idx_df.reset_index(drop=True),
        root,
        True,
        resize,
    )
    X_test, y_test = build_matrix(
        test_idx_df.reset_index(drop=True),
        root,
        False,
        resize,
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    clf = fit_classifier(X_train_std, y_train_enc, use_class_weights, seed)
    std_res, thr_res = evaluate(
        clf,
        X_test_std,
        y_test_enc,
        [str(c) for c in le.classes_],
        tau,
    )

    export_artifacts(out_dir, std_res, thr_res)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(clf, MODELS_DIR / "xgb_birads_model.joblib")
    dump(scaler, MODELS_DIR / "scaler.joblib")
    dump(le, MODELS_DIR / "label_encoder.joblib")

    logging.info("=" * 60)
    logging.info("RÉSULTATS FINAUX")
    logging.info("=" * 60)
    logging.info(
        "Standard  Acc=%.3f | R_malin=%.3f | F1_malin=%.3f | AUC=%.3f",
        std_res.accuracy,
        std_res.recall_malignant,
        std_res.f1_malignant,
        std_res.auc_ovr,
    )
    if thr_res:
        logging.info(
            "Seuil     Acc=%.3f | R_malin=%.3f | F1_malin=%.3f",
            thr_res.accuracy,
            thr_res.recall_malignant,
            thr_res.f1_malignant,
        )
    logging.info("Artefacts: %s", out_dir)


def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Entraînement et évaluation de modèle XGBoost pour mammographies"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                      help="Dossier racine des données")
    parser.add_argument("--csv", type=str, default=None,
                      help="Chemin vers le CSV d'annotations")
    parser.add_argument("--out", type=str, default=None,
                      help="Dossier de sortie pour les artefacts")
    parser.add_argument("--use-class-weights", action="store_true", default=True,
                      help="Utiliser la pondération de classes (défaut: True)")
    parser.add_argument("--no-class-weights", action="store_false", 
                      dest="use_class_weights",
                      help="Désactiver la pondération de classes")
    parser.add_argument("--tau", type=float, default=None,
                      help="Seuil pro-malin pour la classification (ex: 0.3)")
    parser.add_argument("--cv", action="store_true",
                      help="Activer la validation croisée")
    parser.add_argument("--n-folds", type=int, default=5,
                      help="Nombre de folds pour la CV (défaut: 5)")
    parser.add_argument("--resize", type=int, default=160,
                      help="Taille de redimensionnement (défaut: 160)")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée principal du script."""
    args = parse_args()

    if args.data_dir:
        root = Path(args.data_dir)
        if not root.is_absolute():
            root = (BASE_DIR / root).resolve()
    else:
        root = BASE_DIR

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = (BASE_DIR / csv_path).resolve()
    else:
        if UNIFIED_CSV_PATH.exists():
            csv_path = UNIFIED_CSV_PATH
        else:
            candidates = [
                MIAS_DATA_DIR / "Breast_cancer_dataset.csv",
                MIAS_DATA_DIR / "mias_annotations.csv",
                MIAS_DATA_DIR / "annotations.csv",
            ]
            csv_path = next((p for p in candidates if p.exists()), None)
            if csv_path is None:
                raise FileNotFoundError(
                    f"Aucun CSV trouvé dans {MIAS_DATA_DIR} ni CSV unifié."
                )
            root = MIAS_DATA_DIR

    out_dir = Path(args.out) if args.out else ARTIFACTS_DIR
    if not out_dir.is_absolute():
        out_dir = (BASE_DIR / out_dir).resolve()

    train_and_evaluate(
        csv_path=csv_path,
        root=root,
        out_dir=out_dir,
        use_class_weights=args.use_class_weights,
        tau=args.tau,
        use_cv=args.cv,
        n_folds=args.n_folds,
        resize=args.resize,
        seed=42,
    )


if __name__ == "__main__":
    main()