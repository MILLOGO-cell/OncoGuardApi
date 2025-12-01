# eval_prevalidation_locale.py
import os, json, math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, accuracy_score
from app.ml.predictor import predict  # adapter l’import au chemin réel

@dataclass
class ConcordanceResult:
    concordance: float
    ci95_low: float
    ci95_high: float
    confusion: np.ndarray
    classes: list

def wilson_ci(k: int, n: int, z: float = 1.96):
    """
    Calcule l’intervalle de confiance de Wilson pour une proportion k/n.
    """
    if n == 0: 
        return (0.0, 0.0)
    p = k / n
    den = 1 + z**2 / n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / den
    return (center - half, center + half)

def evaluate_local(csv_path: str, image_root: str, out_dir: str) -> ConcordanceResult:
    """
    Évalue la concordance système–consensus sur un jeu local et exporte les résultats.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    y_true, y_pred = [], []
    for _, r in df.iterrows():
        truth = str(r["consensus"]).strip()
        path = os.path.join(image_root, r["filename"])
        res = predict(path)
        y_true.append(truth)
        y_pred.append(res.label)
    acc = accuracy_score(y_true, y_pred)
    cm_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    k = int(acc * len(df))
    low, high = wilson_ci(k, len(df), 1.96)

    pd.DataFrame(cm, index=cm_labels, columns=cm_labels).to_csv(os.path.join(out_dir, "confusion_local.csv"))
    meta = {
        "concordance": acc,
        "ci95_low": low,
        "ci95_high": high,
        "n": int(len(df)),
        "labels": cm_labels
    }
    with open(os.path.join(out_dir, "preval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    tex = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Prévalidation clinique locale (100 cas).}\n\\label{tab:local-preval}\n"
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "\\textbf{Indicateur} & \\textbf{Valeur} & \\textbf{Détails} \\\\\n\\midrule\n"
        f"Concordance globale & {acc:.2f} & IC\\,95\\,\\%\\, [{low*100:.1f}\\,\\%; {high*100:.1f}\\,\\%] \\\\\n"
        f"Cas concordants & {k} & sur {len(df)} \\\\\n"
        f"Cas discordants & {len(df)-k} &  \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open(os.path.join(out_dir, "local_preval_table.tex"), "w", encoding="utf-8") as f:
        f.write(tex)

    return ConcordanceResult(acc, low, high, cm, cm_labels)

if __name__ == "__main__":
    """
    Exemple d’exécution: adapter CSV et dossier d’images.
    """
    CSV = "./local_preval.csv"
    IMG = "./images_locales"
    OUT = "./artifacts_local"
    evaluate_local(CSV, IMG, OUT)
